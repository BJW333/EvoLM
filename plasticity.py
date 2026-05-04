"""
================================================================================
plasticity.py -- Differentiable Hebbian plasticity (Miconi et al. 2018, 2020).
================================================================================

WHAT'S IN HERE:
    PlasticLinear   -- linear layer with a fast Hebbian trace
    PlasticSwiGLU   -- SwiGLU FFN whose down-projection is plastic

CONCEPT:
    A plastic linear layer has THREE things, not just one:
        W       -- "slow" weights, learned by SGD as usual
        alpha   -- "plasticity coefficients", per-synapse, ALSO learned by SGD
        Hebb    -- "fast" Hebbian trace, NOT learned by SGD; updates locally
                   based on activity correlations between pre- and post-
                   synaptic neurons

    The effective weight at any moment is:
        W_eff = W + alpha * Hebb

    The Hebbian trace updates as:
        Hebb_{t+1} = clip(Hebb_t + eta * outer(post_t, pre_t), -1, 1)

    Where post_t = the layer's output at time t and pre_t = the input at time t.
    The clip keeps the trace bounded so it can't run away.

WHY DOES THIS MATTER FOR YOUR "SELF-ORGANIZING" GOAL?
    This is the closest thing to "synaptic plasticity in a deep net" that
    actually trains end-to-end. The trace gives the network within-sequence
    memory at the synaptic level -- separate from the weight memory that SGD
    is updating, and separate from the activations.

    Concretely: the network can adapt its computation on the fly based on
    correlations it sees DURING a sequence, without waiting for a gradient
    update. Miconi has shown this helps on small meta-learning, RL, and
    sequence tasks. Whether it helps for language modeling at scale is open.

PRACTICAL CAVEATS:
    1. Sequential through tokens. The trace at time t depends on times
       1..t-1, so we can't parallelize the time dimension. This makes it
       SLOW. Use sparingly -- only on the last few layers of the model.

    2. Memory cost is the real killer. Because the trace at step t is in
       the autograd graph alongside the trace at step t-1, autograd has to
       keep ALL S intermediate trace tensors alive until backward(). Each
       trace is shape (B, out, in). Concrete example:
           B=8, S=256, hidden_dim=3456, dim=1280
           one trace tensor = 8 * 1280 * 3456 * 4 bytes = ~140 MB
           total across S = 256 * 140 MB = ~36 GB
       That's per plastic layer. So plastic layers + long sequences will
       OOM you fast. Mitigations, in order of how much they help:
         (a) only put plasticity on 1-2 layers
         (b) keep sequence length short within a "plastic chunk" (256-512 max)
         (c) wrap PlasticLinear.forward with torch.utils.checkpoint
         (d) periodically detach the trace mid-sequence (sacrifices long-
             range gradient flow for memory)
       This is the bottleneck, not compute. Plan around it.

    3. The trace is sequence-state. You should reset it at the start of
       each sequence (model.reset_plasticity) and detach it from the graph
       between sequences (model.detach_plasticity) to avoid BPTT through
       all of history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasticLinear(nn.Module):
    """
    Plastic linear layer. Stateful: the Hebbian trace persists across calls
    until you call .reset_trace().

    Designed to be called on whole sequences -- it folds the within-sequence
    trace internally by looping over the time dimension.

    The pre/post-synaptic identification:
        pre_t  = input  to the layer at time t   (shape (B, in_features))
        post_t = output of the layer at time t   (shape (B, out_features))
    The Hebbian update is the OUTER PRODUCT of post and pre, scaled by eta.
    Hebbian = "neurons that fire together, wire together."
    """

    def __init__(self, in_features: int, out_features: int, eta: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ---- Slow weights (W) ----
        # Standard Linear weight matrix, learned by SGD.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)

        # ---- Plasticity coefficients (alpha) ----
        # One scalar per synapse. Initialized to 0 so plasticity starts off
        # fully -- the network must explicitly learn to use it. Without this
        # init choice, the network has random plasticity at step 0 and
        # training is much harder.
        self.alpha = nn.Parameter(torch.zeros(out_features, in_features))

        # ---- Hebbian learning rate (eta) ----
        # Stored as a logit so sigmoid(eta_raw) is in (0, 1). Clamp the user-
        # supplied eta to this open interval to avoid logit(0) = -inf and
        # logit(1) = +inf.
        eps = 1e-4
        eta_clamped = max(eps, min(1.0 - eps, float(eta)))
        # logit(p) = log(p / (1 - p))
        eta_init = torch.log(torch.tensor(eta_clamped) / (1.0 - eta_clamped))
        self.eta_raw = nn.Parameter(eta_init)

        # ---- Hebbian trace (Hebb) ----
        # Lazily allocated when batch size is first known, since (B, out, in)
        # depends on B. Not a Parameter (no gradient), not a buffer (we
        # explicitly manage device/state).
        self.hebb = None

    def reset_trace(self, batch_size: int, device):
        """Zero the Hebbian trace. Call this at the start of each sequence."""
        self.hebb = torch.zeros(
            batch_size, self.out_features, self.in_features, device=device
        )

    @property
    def eta(self):
        """Effective Hebbian learning rate, in (0, 1)."""
        return torch.sigmoid(self.eta_raw)

    def forward(self, x, valid_mask=None):
        """
        x:          (B, S, in_features) or (B, in_features)
        valid_mask: optional (B, S) bool, True where the token is real
                    (NOT a pad token). When provided, the Hebbian trace is
                    only updated on real tokens, so padding doesn't poison
                    the fast weights. The OUTPUT for pad positions is still
                    computed (it's harmless and keeps shapes consistent).
        """
        # Allow calling on a single time step.
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        B, S, _ = x.shape

        # Allocate trace if needed. Reset if batch size changed.
        if self.hebb is None or self.hebb.size(0) != B:
            self.reset_trace(B, x.device)

        eta = self.eta  # cached scalar tensor

        # We loop over the time dimension because the trace at step t depends
        # on all earlier steps. There's no clean way to parallelize this.
        outputs = []
        for t in range(S):
            x_t = x[:, t, :]                                    # (B, in)

            # ---- Compute output with effective weights ----
            # W_eff = W + alpha * Hebb is per-batch-element. alpha is shared
            # across the batch but Hebb is not. Use einsum to do the per-
            # batch matmul.
            #   "boi,bi->bo"  means W_eff[b, o, i] * x[b, i] summed over i
            W_eff = self.weight + self.alpha * self.hebb       # (B, out, in)
            y_t = torch.einsum("boi,bi->bo", W_eff, x_t)        # (B, out)

            # ---- Hebbian update ----
            # Outer product of post and pre, gated by valid_mask if given.
            outer = y_t.unsqueeze(2) * x_t.unsqueeze(1)         # (B, out, in)
            if valid_mask is not None:
                # Zero the update for pad positions. mask shape: (B,) at this t.
                m = valid_mask[:, t].to(outer.dtype).view(B, 1, 1)
                outer = outer * m

            # Update + clip. The clip is essential -- without it, repeated
            # positive correlations make the trace explode.
            self.hebb = torch.clamp(self.hebb + eta * outer, -1.0, 1.0)

            outputs.append(y_t)

        out = torch.stack(outputs, dim=1)                       # (B, S, out)
        if squeeze:
            out = out.squeeze(1)
        return out

    def detach_trace(self):
        """Detach the Hebbian trace from the autograd graph. Call between
        sequences so we don't backprop through the entire history."""
        if self.hebb is not None:
            self.hebb = self.hebb.detach()


class PlasticSwiGLU(nn.Module):
    """
    SwiGLU FFN where the DOWN projection (w3) is a PlasticLinear.

    Why w3 and not w1/w2? The down projection is where the FFN compresses
    its hidden representation back to model dim. That's where richer
    representations live, so it's the most useful place for fast adaptation.
    Also: keeping w1 and w2 standard means we don't pay the sequential cost
    on the wider matrices.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, eta: float = 0.01):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)        # gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)        # up
        self.w3 = PlasticLinear(hidden_dim, dim, eta=eta)       # plastic down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None):
        h = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(h, valid_mask=valid_mask))

    def reset_plasticity(self, batch_size: int, device):
        """Reset Hebbian state. Call at the start of each new sequence."""
        self.w3.reset_trace(batch_size, device)

    def detach_plasticity(self):
        """Detach trace from graph. Call between sequences."""
        self.w3.detach_trace()
