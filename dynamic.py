"""
================================================================================
dynamic.py -- Gradient-trainable hyperparameters and structural knobs.
================================================================================

THE IDEA:
    "Fully dynamic parameters that self-adjust" is a great instinct. The
    auto_optimizer.py file does it via heuristics outside the loss. This
    file does it the principled way: turn things that are usually fixed
    hyperparameters into LEARNABLE PARAMETERS that gradient descent tunes
    along with everything else.

    This is a strictly stronger formulation than heuristic auto-tuning,
    BECAUSE the gradient knows what's actually helping the loss. Heuristics
    only know what's correlated with helping.

WHAT'S IN HERE:
    LearnableLayerDrop
        -- per-layer skip probability that's learned (via concrete relaxation
           so it's differentiable).
    ConcreteDropout
        -- learnable dropout probability (Gal et al. 2017). Replaces nn.Dropout.
    LearnableResidualGate
        -- learnable scalar (or vector) gate on each block's residual
           contribution: x = x + g * sublayer(x). g starts at 1 (so it's a
           no-op) but can shrink to ignore unhelpful sublayers, or grow to
           amplify helpful ones. Like a mini "learn how much each layer
           contributes" mechanism.
    LearnableRouterTemperature
        -- temperature on the MoE router softmax, learned.
        -- low temp = harder routing (sharper top-k); high temp = softer.

HOW THESE COMBINE WITH THE REST OF THE SYSTEM:
    auto_optimizer.py: outside-the-loss heuristics for things you CAN'T
                       backprop through (like LR -- you can't differentiate
                       through your own optimizer step easily).
    dynamic.py:        inside-the-loss learnable params for things you CAN
                       backprop through. Use this when possible -- gradients
                       are smarter than heuristics.

WHEN TO PREFER WHICH:
    LR / WD             -> auto_optimizer (no clean differentiation)
    Layer dropout       -> dynamic (concrete relaxation)
    Activation dropout  -> dynamic (Concrete Dropout)
    Residual gates      -> dynamic (just a scalar parameter)
    Router temperature  -> dynamic (just a scalar parameter)
    Loss weights        -> dynamic with care (need a meta-objective else they
                           collapse to zero -- see "uncertainty weighting"
                           Kendall & Gal 2018 for the right way to do it)
"""

import torch
import torch.nn as nn

from architecture import RMSNorm, GroupedQueryAttention


# ============================================================================
# LearnableLayerDrop
# ============================================================================
class LearnableLayerDrop(nn.Module):
    """
    A drop-in replacement for the boolean LayerDrop in TransformerBlock.
    Stores a learnable "raw" parameter, sigmoid'd to a probability in (0,1),
    and uses a Gumbel-Sigmoid relaxation to make the discrete skip / keep
    decision differentiable.

    USAGE:
        drop = LearnableLayerDrop(init_prob=0.0)
        keep = drop.sample_keep_mask(training=True)   # scalar tensor in {0,1}
        sublayer_output = sublayer_output * keep      # 0 = dropped, 1 = kept

    During training, the gradient flows back to drop.raw via the straight-
    through estimator, so SGD learns the per-layer drop rate.

    Note: in practice, learning a per-layer drop rate is not always stable.
    A simpler alternative is just a learnable scalar GATE on the layer
    output (see LearnableResidualGate below) -- often works better.
    """
    def __init__(self, init_prob: float = 0.0):
        super().__init__()
        # Clamp init_prob away from 0 / 1 to keep logit finite.
        eps = 1e-4
        p = max(eps, min(1.0 - eps, init_prob))
        self.raw = nn.Parameter(torch.log(torch.tensor(p) / (1.0 - p)))

    @property
    def prob(self):
        return torch.sigmoid(self.raw)

    def sample_keep_mask(self, training: bool):
        """
        Return a scalar in [0, 1].
            value 1.0 with probability (1 - drop_prob) -- KEEP
            value 0.0 with probability drop_prob       -- DROP

        Forward pass is hard (0 or 1). Backward pass uses the soft
        probability via the straight-through estimator, so the gradient flows
        back to self.raw.

        At eval time we always keep (no stochastic dropping).
        """
        if not training:
            return torch.tensor(1.0, device=self.raw.device)

        # We want to sample Bernoulli(keep_prob) where keep_prob = 1 - drop_prob.
        # self.raw is logit(drop_prob), so logit(keep_prob) = -self.raw.
        # The Gumbel-Sigmoid relaxation samples Bernoulli(sigmoid(logit)) via:
        #     z = sigmoid((logit + log(u) - log(1-u)) / temp)
        # Plug logit = -self.raw to get a soft KEEP indicator.
        u = torch.rand_like(self.raw).clamp(1e-7, 1 - 1e-7)
        keep_logit = -self.raw
        gate_soft = torch.sigmoid(
            (keep_logit + torch.log(u) - torch.log(1 - u)) / 0.1
        )
        # Straight-through estimator: hard decision in forward, soft gradient
        # in backward. The detach() on gate_soft makes the difference cancel
        # in the forward pass (so gate_hard wins) while still letting the
        # gradient w.r.t. gate_soft flow.
        gate_hard = (gate_soft > 0.5).float()
        return gate_hard - gate_soft.detach() + gate_soft


# ============================================================================
# ConcreteDropout
# ============================================================================
class ConcreteDropout(nn.Module):
    """
    Concrete Dropout (Gal, Hron, Kendall 2017).

    Standard nn.Dropout has a fixed dropout probability. Concrete Dropout
    learns it. The key trick: relax the binary dropout mask using a continuous
    distribution (the "concrete" distribution) so that the dropout probability
    becomes differentiable and can be optimized by gradient descent.

    Forward pass: applies a soft dropout mask sampled from a relaxed Bernoulli.
    Backward pass: gradient flows through to self.raw_p, the logit of the
                   dropout probability.

    Drop-in replacement for nn.Dropout(p=...). Same shape semantics.

    Practical note: the relaxation temperature is a hyperparameter; we use
    0.1 which is the value from the paper. Lower temp = closer to hard
    Bernoulli but higher gradient variance.
    """
    def __init__(self, init_p: float = 0.1, temperature: float = 0.1):
        super().__init__()
        eps = 1e-4
        p = max(eps, min(1.0 - eps, init_p))
        self.raw_p = nn.Parameter(torch.log(torch.tensor(p) / (1.0 - p)))
        self.temperature = temperature

    @property
    def p(self):
        return torch.sigmoid(self.raw_p)

    def forward(self, x):
        # Eval mode: dropout is disabled (standard convention). We don't
        # short-circuit on small p values here -- avoiding self.p.item()
        # is more important than the trivial extra compute, since .item()
        # forces CPU/GPU sync.
        if not self.training:
            return x
        p = self.p  # cache to avoid re-computing sigmoid
        # Sample uniform noise, transform to "soft Bernoulli" with reparam trick.
        u = torch.rand_like(x).clamp(1e-7, 1 - 1e-7)
        # Concrete distribution sample: in (0, 1), shape matches x.
        # Closer to 1 = "keep this activation"; closer to 0 = "drop it".
        # logit(1-p) + log(u/(1-u)) gives a soft Bernoulli with keep prob = (1-p).
        mask = torch.sigmoid(
            (
                torch.log(1.0 - p + 1e-7)
                - torch.log(p + 1e-7)
                + torch.log(u)
                - torch.log(1.0 - u)
            )
            / self.temperature
        )
        # Inverse-scale to keep expected value the same (just like regular dropout).
        return x * mask / (1.0 - p + 1e-7)


# ============================================================================
# LearnableResidualGate
# ============================================================================
class LearnableResidualGate(nn.Module):
    """
    A scalar (or per-channel) gate on a sublayer's residual contribution.

        x = x + g * sublayer(x)         instead of    x = x + sublayer(x)

    Initialized to 1.0 so the model starts as if the gate weren't there.
    SGD can shrink g toward 0 if the sublayer isn't helping (effectively
    pruning it without removing it) or grow g if it should contribute more.

    This is similar in spirit to the "per-layer alpha" parameters in some
    NAS papers. It's a simple, robust mechanism for "learn which layers
    matter" without all the machinery of true NAS.

    `per_channel=True` gives one gate per dim (more expressive); False gives
    a scalar (more conservative).
    """
    def __init__(self, dim: int, per_channel: bool = False, init: float = 1.0):
        super().__init__()
        # Cast init explicitly to float -- torch.tensor and torch.full infer
        # dtype from the input, so passing init=1 (int) gives an integer
        # parameter which silently breaks AdamW.
        init_f = float(init)
        if per_channel:
            self.gate = nn.Parameter(torch.full((dim,), init_f))
        else:
            self.gate = nn.Parameter(torch.tensor(init_f))

    def forward(self, residual_contribution):
        # Multiply the sublayer's output by the gate before adding to residual.
        return self.gate * residual_contribution


# ============================================================================
# LearnableRouterTemperature
# ============================================================================
# NOTE: this module is provided as a stand-alone building block. The current
# MoEFeedForward in moe.py does NOT integrate it -- you would need to insert
# `logits = self.router_temp(self.router(x_flat))` in MoEFeedForward.forward
# and add `self.router_temp = LearnableRouterTemperature(1.0)` in its __init__
# to wire it in. Left as-is so users can opt in without touching the default
# MoE path.
class LearnableRouterTemperature(nn.Module):
    """
    A learnable temperature for the MoE router softmax.

    Standard MoE: probs = softmax(router_logits)
    With learnable temp: probs = softmax(router_logits / T)
        T < 1 -> sharper -> harder routing (more confident, top-k more decisive)
        T > 1 -> softer -> more even spread of probabilities

    Initialized to 1.0 (standard softmax). The model learns whether to
    sharpen or soften over time. We parameterize T = exp(raw) to keep T > 0.
    """
    def __init__(self, init: float = 1.0):
        super().__init__()
        self.raw = nn.Parameter(torch.log(torch.tensor(float(init))))

    @property
    def T(self):
        return torch.exp(self.raw)

    def forward(self, logits):
        return logits / self.T


# ============================================================================
# Drop-in TransformerBlock that wires up the learnable hyperparameters.
# ============================================================================
class DynamicTransformerBlock(nn.Module):
    """
    Like architecture.TransformerBlock but with:
        - LearnableResidualGate on both attn and ff residuals
        - LearnableLayerDrop instead of static layer_drop

    Use this in place of TransformerBlock when you want maximum trainable
    structure. Note: more learnable knobs = more places for things to go
    wrong. Start with just gates before turning on learnable layer drop.

    Return contract matches TransformerBlock: (x, aux, kv_cache).
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_module: nn.Module,
        attn_dropout: float,
        init_layer_drop: float = 0.0,
        learnable_layer_drop: bool = True,
        learnable_gates: bool = True,
        per_channel_gates: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, attn_dropout)
        self.ff = ffn_module

        if learnable_layer_drop:
            self.layer_drop = LearnableLayerDrop(init_prob=init_layer_drop)
        else:
            self.layer_drop = None

        if learnable_gates:
            self.attn_gate = LearnableResidualGate(dim, per_channel=per_channel_gates)
            self.ff_gate = LearnableResidualGate(dim, per_channel=per_channel_gates)
        else:
            self.attn_gate = None
            self.ff_gate = None

    def forward(self, x, key_padding_mask=None, kv_cache=None):
        # Compute the keep/skip decision once -- applies to both attn and ff
        # so the layer-drop is for the WHOLE block, not just one sub-layer.
        if self.layer_drop is not None and self.training:
            keep = self.layer_drop.sample_keep_mask(self.training)
        else:
            keep = None

        # ---- Attention ----
        a = self.attn_norm(x)
        if kv_cache is not None:
            attn_out, new_cache = self.attn(a, key_padding_mask, kv_cache=kv_cache)
        else:
            attn_out = self.attn(a, key_padding_mask)
            new_cache = None

        if self.attn_gate is not None:
            attn_out = self.attn_gate(attn_out)
        if keep is not None:
            attn_out = attn_out * keep
        x = x + attn_out

        # ---- Feed-Forward ----
        f = self.ff_norm(x)
        # Forward valid_mask to ff so PlasticSwiGLU can mask Hebbian updates
        # on padding. SwiGLU and MoEFeedForward accept and ignore it.
        valid_mask = (~key_padding_mask) if key_padding_mask is not None else None
        ff_out = self.ff(f, valid_mask=valid_mask)
        if isinstance(ff_out, tuple):
            ff_tensor, aux = ff_out
        else:
            ff_tensor, aux = ff_out, None

        if self.ff_gate is not None:
            ff_tensor = self.ff_gate(ff_tensor)
        if keep is not None:
            ff_tensor = ff_tensor * keep
        x = x + ff_tensor

        return x, aux, new_cache
