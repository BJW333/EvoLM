"""
================================================================================
model.py -- The EvoLM model: assembles all the pieces.
================================================================================

WHAT'S IN HERE:
    EvoLM             -- the full language model
    small_config      -- ~110M params, prototyping
    medium_config     -- ~500M params, the default
    large_config      -- ~1.5B params

CONFIGURATION KNOBS:
    moe_layers      -- which layer indices use MoE FFN. None or []
                       means no MoE. Recommended: every other layer.
    plastic_layers  -- which layer indices use PlasticSwiGLU. None or []
                       means no plasticity. Recommended: only the LAST few
                       (1-3) layers because plasticity is sequential and slow.
    use_dynamic     -- if True, build with DynamicTransformerBlock (learnable
                       layer drop, residual gates). False uses the basic
                       TransformerBlock.

A given layer can use AT MOST ONE special FFN type. If a layer is in both
moe_layers and plastic_layers, MoE wins (by design -- you'd rarely want both
at the same depth).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from architecture import (
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    llama_init,
    scale_residual_projections,
)
from moe import MoEFeedForward
from plasticity import PlasticSwiGLU
from dynamic import DynamicTransformerBlock


class EvoLM(nn.Module):
    """
    Decoder-only transformer LM with:
      - GQA attention with RoPE
      - SwiGLU FFNs by default
      - Optional MoE FFNs at chosen layer indices
      - Optional Plastic FFNs at chosen layer indices
      - Optional learnable layer drop / residual gates (DynamicTransformerBlock)
      - LLaMA-style init with depth-aware residual scaling
      - Tied input/output embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq: int,
        pad_token_id: int,
        # ----- core architecture (sane aspect ratio) -----
        dim: int = 1280,
        n_heads: int = 20,
        n_kv_heads: int = 4,                # GQA: 5x KV cache reduction
        num_layers: int = 24,
        ff_dim: int = 3456,                 # ~2.7x dim, GPU-friendly
        # ----- regularization -----
        dropout: float = 0.05,
        attn_dropout: float = 0.05,
        layer_drop: float = 0.0,
        # ----- MoE config -----
        moe_layers: Optional[List[int]] = None,
        moe_n_experts: int = 8,
        moe_top_k: int = 2,
        # ----- plasticity config -----
        plastic_layers: Optional[List[int]] = None,
        plastic_eta: float = 0.01,
        # ----- dynamic block (learnable layer drop + residual gates) -----
        use_dynamic: bool = False,
        # ----- memory / compute trade-offs -----
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.max_seq = max_seq                 # informational; checked in forward()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.attn_dropout = attn_dropout
        self.layer_drop = layer_drop
        self.dropout = dropout
        # Gradient checkpointing trades ~30% extra compute (recomputing block
        # activations during backward) for ~50% less activation memory. Off by
        # default; enable via constructor arg or set_gradient_checkpointing().
        # Skipped automatically for PlasticSwiGLU blocks (their Hebbian-trace
        # side effect can't be safely re-executed during backward).
        self.gradient_checkpointing = gradient_checkpointing

        # ---- Resolve special-layer assignments ----
        # MoE wins ties with plasticity (by design).
        moe_layers = set(moe_layers or [])
        plastic_layers = set(plastic_layers or [])
        plastic_layers = plastic_layers - moe_layers
        self.moe_layers = moe_layers
        self.plastic_layers = plastic_layers

        # ---- Token embedding ----
        # padding_idx ensures pad row is zeroed at construction. Keep this
        # in mind: re-init via llama_init will overwrite it; we re-zero
        # below after init.
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)

        # ---- Build the stack of transformer blocks ----
        blocks = []
        for i in range(num_layers):
            # Decide which FFN this layer gets.
            if i in moe_layers:
                ffn = MoEFeedForward(
                    dim=dim,
                    hidden_dim=ff_dim,
                    n_experts=moe_n_experts,
                    top_k=moe_top_k,
                    dropout=dropout,
                )
            elif i in plastic_layers:
                ffn = PlasticSwiGLU(dim, ff_dim, dropout, eta=plastic_eta)
            else:
                ffn = SwiGLU(dim, ff_dim, dropout)

            # Decide which block type.
            if use_dynamic:
                block = DynamicTransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    ffn_module=ffn,
                    attn_dropout=attn_dropout,
                    init_layer_drop=layer_drop,
                    learnable_layer_drop=True,
                    learnable_gates=True,
                )
            else:
                block = TransformerBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    ffn_module=ffn,
                    attn_dropout=attn_dropout,
                    layer_drop=layer_drop,
                )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.ln_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # ---- Initialization ----
        # Step 1: standard LLaMA-style init for all linears and embeddings.
        self.apply(lambda m: llama_init(m, num_layers))
        # Step 2: rescale residual projections by 1/sqrt(2N).
        scale_residual_projections(self, num_layers)
        # Step 3: re-zero the embedding pad row (llama_init clobbered it).
        if self.pad_token_id is not None:
            with torch.no_grad():
                self.token_emb.weight[self.pad_token_id].zero_()
        # Step 4: tie the lm_head to the embedding (saves ~vocab_size * dim
        # params and tends to improve quality slightly).
        self.lm_head.weight = self.token_emb.weight

    # ------------------------------------------------------------------
    # Plasticity helpers (delegated to PlasticSwiGLU layers)
    # ------------------------------------------------------------------
    def reset_plasticity(self, batch_size: int, device):
        """Call at the start of each new sequence. Zeros the Hebbian trace."""
        for block in self.blocks:
            if isinstance(block.ff, PlasticSwiGLU):
                block.ff.reset_plasticity(batch_size, device)

    def detach_plasticity(self):
        """Call between sequences. Stops backprop through the trace's history."""
        for block in self.blocks:
            if isinstance(block.ff, PlasticSwiGLU):
                block.ff.detach_plasticity()

    # ------------------------------------------------------------------
    # MoE helpers
    # ------------------------------------------------------------------
    def evolve_moe(self, device, **kwargs):
        """Run grow/prune on every MoE layer. Returns dict of per-layer events."""
        events = {}
        for i, block in enumerate(self.blocks):
            if isinstance(block.ff, MoEFeedForward):
                events[i] = block.ff.maybe_grow_or_prune(device, **kwargs)
        return events

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def set_gradient_checkpointing(self, enabled: bool):
        """Toggle gradient checkpointing at runtime."""
        self.gradient_checkpointing = enabled

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        kv_caches=None,
    ):
        """
        input_ids:      (B, S) long tensor. During cached decode, S is just
                        the number of NEW tokens (typically 1).
        attention_mask: (B, S) bool tensor, True at REAL tokens (not pad).
                        If None, derived from input_ids != pad_token_id.
        labels:         (B, S) long tensor for next-token CE loss. Pad
                        positions get -100 (ignored). If None, no loss.
        kv_caches:      Pass to enable cached inference. Conventions:
                          None         -- no caching (training / non-cached eval).
                          []           -- prefill: caching pathway, no past tokens.
                                          We'll build initial caches from input.
                          [c0, c1, ..] -- decode: per-layer caches from prior call.
                                          Length must equal len(model.blocks).
                                          Each entry is either None (prefill that
                                          layer) or a (k, v) tuple from a prior call.

        Returns:
            no caching, no labels:    logits
            no caching, with labels:  (logits, loss, parts_dict)
            caching, no labels:       (logits, new_kv_caches)
            caching, with labels:     (logits, loss, parts_dict, new_kv_caches)
        """
        B, S = input_ids.size()
        caching = kv_caches is not None

        # ---- Length validation (caching-aware) ----
        # Without caching: S itself must fit. With caching: total context
        # (past + new) must fit. Past length is read from any layer's cache.
        if caching:
            if len(kv_caches) == 0:
                # Prefill: no past, normalize to a list of None placeholders.
                kv_caches = [None] * len(self.blocks)
            elif len(kv_caches) != len(self.blocks):
                raise ValueError(
                    f"kv_caches has {len(kv_caches)} entries but model has "
                    f"{len(self.blocks)} blocks"
                )
            past_len = 0
            for c in kv_caches:
                if c is not None and c[0] is not None:
                    past_len = c[0].size(2)
                    break
            if past_len + S > self.max_seq:
                raise ValueError(
                    f"cached length {past_len} + new {S} = {past_len + S} "
                    f"exceeds max_seq={self.max_seq}"
                )
        else:
            if S > self.max_seq:
                raise ValueError(
                    f"sequence length {S} exceeds model max_seq={self.max_seq}"
                )

        # Build the attention mask if not given.
        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = input_ids != self.pad_token_id

        # Token embedding -> initial residual stream.
        x = self.token_emb(input_ids)

        # Blocks expect key_padding_mask with True = pad (mask out). Our
        # attention_mask is True = real token, so invert.
        key_padding_mask = ~attention_mask if attention_mask is not None else None

        # Aggregate aux losses (from MoE layers) into a single scalar.
        aux_total = x.new_zeros(())

        # ---- Stack of transformer blocks ----
        new_kv_caches = [] if caching else None

        for i, block in enumerate(self.blocks):
            # Decide what cache pointer to pass into this block.
            #   not caching       -> kv_cache=None        (block returns aux=None for cache)
            #   caching, prefill  -> kv_cache=(None,None) (block builds a new cache)
            #   caching, decode   -> kv_cache=(k, v)      (block extends the cache)
            if caching:
                cache_entry = kv_caches[i]
                block_cache = cache_entry if cache_entry is not None else (None, None)
            else:
                block_cache = None

            # Decide whether to gradient-checkpoint this block. Skip when:
            #   - feature is off or we're in eval mode
            #   - we're in caching pathway (memory's not the bottleneck there)
            #   - the FFN has side effects that can't safely be re-executed
            #     during backward (PlasticSwiGLU mutates self.hebb).
            use_ckpt = (
                self.gradient_checkpointing
                and self.training
                and not caching
                and not isinstance(block.ff, PlasticSwiGLU)
            )

            if use_ckpt:
                # checkpoint() re-runs the block during backward instead of
                # storing all its intermediate activations. ~30% extra compute
                # for ~50% less activation memory. use_reentrant=False is the
                # modern non-reentrant version, recommended by PyTorch.
                x, aux, _ = _checkpoint(
                    block, x, key_padding_mask, None,
                    use_reentrant=False,
                )
            else:
                x, aux, new_cache = block(
                    x,
                    key_padding_mask=key_padding_mask,
                    kv_cache=block_cache,
                )
                if caching:
                    new_kv_caches.append(new_cache)

            if aux is not None:
                aux_total = aux_total + aux

            # If we checkpointed (training, non-caching), we don't have a
            # new_cache to record -- caching is False so the list isn't used.

        # Final norm and language modeling head.
        x = self.ln_f(x)
        logits = self.lm_head(x)            # (B, S, vocab_size)

        # ---- Loss (when labels given) ----
        if labels is not None:
            labels = labels.clone()
            labels[labels == self.pad_token_id] = -100
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = ce + aux_total
            parts = {"ce": ce.detach(), "aux": aux_total.detach()}
            if caching:
                return logits, loss, parts, new_kv_caches
            return logits, loss, parts

        if caching:
            return logits, new_kv_caches
        return logits

    # ------------------------------------------------------------------
    def num_params(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Decay vs no-decay classification (used by get_param_groups AND by
    # the helpers that add NEW params from grown layers / spawned experts)
    # ------------------------------------------------------------------
    _NO_DECAY_KEYWORDS = (
        "norm",          # RMSNorm gains
        "gate",          # LearnableResidualGate
        "alpha",         # PlasticLinear plasticity coefficients
        "eta_raw",       # PlasticLinear hebbian rate
        "raw",           # LearnableLayerDrop / ConcreteDropout / RouterTemp
        "raw_p",         # ConcreteDropout
    )

    @classmethod
    def should_skip_decay(cls, name: str, param) -> bool:
        """True if this param should NOT get weight decay."""
        lname = name.lower()
        return (
            param.ndim < 2
            or any(kw in lname for kw in cls._NO_DECAY_KEYWORDS)
        )

    # ------------------------------------------------------------------
    def get_param_groups(self, weight_decay: float = 0.1):
        """
        Split parameters into two groups: those that get weight decay and
        those that don't. Standard transformer recipe.

        NO weight decay for:
          - 1D parameters (biases, RMSNorm gains, learnable scalars)
          - LearnableResidualGate.gate (would drag toward 0, killing the layer)
          - PlasticLinear.alpha and .eta_raw (special learned scalars)
          - LearnableLayerDrop.raw, ConcreteDropout.raw_p (special scalars)
          - LearnableRouterTemperature.raw

        WD for everything else (the actual weight matrices).

        Returns a list of two dicts suitable for passing to AdamW(...).
        Each group is tagged with a "name" so helpers like
        add_new_params_to_optimizer can find the right group when growing
        the model later.

        Why this matters: AdamW applies weight decay as a multiplicative
        shrinkage, theta -= lr * wd * theta. For matrices this regularizes
        weight magnitudes (good). For learned scalars and norm gains, it
        drags them toward 0 even when the loss doesn't want that, which can
        kill carefully-set parameters like residual gates that should sit
        near 1.0.
        """
        decay = []
        no_decay = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if self.should_skip_decay(name, p) else decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay, "name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
        ]


# ============================================================================
# Helper for adding NEW params (e.g. from grown layers or spawned experts)
# to an existing optimizer with proper decay/no-decay grouping.
# ============================================================================
def add_new_params_to_optimizer(named_params, optimizer):
    """
    Given an iterable of (name, param) pairs, classify each by the same
    decay rules as get_param_groups and append to the matching existing
    optimizer group (decay or no_decay).

    This is what evolve_moe_with_optimizer and grow_layer_with_optimizer
    should use, so newly-created RMSNorm gains and learnable scalars don't
    accidentally get weight decay applied to them.

    Falls back to creating new groups if the optimizer doesn't already
    have decay / no_decay groups (e.g. someone passed model.parameters()
    directly without using get_param_groups).
    """
    new_decay = []
    new_no_decay = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if EvoLM.should_skip_decay(name, p):
            new_no_decay.append(p)
        else:
            new_decay.append(p)

    # Find existing groups by name first, fall back to weight_decay value.
    decay_group = None
    no_decay_group = None
    for g in optimizer.param_groups:
        if g.get("name") == "decay":
            decay_group = g
        elif g.get("name") == "no_decay":
            no_decay_group = g
    if decay_group is None or no_decay_group is None:
        for g in optimizer.param_groups:
            if g.get("weight_decay", 0.0) > 0 and decay_group is None:
                decay_group = g
            elif g.get("weight_decay", 0.0) == 0.0 and no_decay_group is None:
                no_decay_group = g

    # Append to existing groups, or create them if missing.
    default_lr = optimizer.param_groups[0]["lr"]
    default_wd = max(
        (g.get("weight_decay", 0.0) for g in optimizer.param_groups),
        default=0.1,
    )
    if new_decay:
        if decay_group is not None:
            decay_group["params"].extend(new_decay)
        else:
            optimizer.add_param_group({
                "params": new_decay, "lr": default_lr,
                "weight_decay": default_wd, "name": "decay",
            })
    if new_no_decay:
        if no_decay_group is not None:
            no_decay_group["params"].extend(new_no_decay)
        else:
            optimizer.add_param_group({
                "params": new_no_decay, "lr": default_lr,
                "weight_decay": 0.0, "name": "no_decay",
            })


def restore_topology(model, saved_topology: dict, device: str, optimizer=None):
    """
    Before loading a state_dict, grow the model so its architecture matches
    the saved checkpoint. Two cases:

    1. **Layer growth.** The saved model had more transformer blocks than the
       freshly-built model. Append vanilla TransformerBlocks until the count
       matches. Residual projections are zeroed so the appended block is
       initially a no-op (matches grow_layer's function-preserving init).

    2. **MoE expert growth.** A saved MoE layer has more experts than the
       freshly-built one. Spawn experts (cloning expert 0's structure --
       weights get overwritten by the load_state_dict that follows).

    If `optimizer` is provided, the new params are also registered with the
    optimizer in their proper decay / no-decay groups via
    add_new_params_to_optimizer. Without this, optimizer.load_state_dict
    will likely fail because the param group sizes wouldn't match.

    Returns: True if anything was grown, False if topology was already a match.

    LIMITATIONS:
    - Doesn't handle layer SHRINKAGE (saved had fewer layers than current).
    - Doesn't handle expert PRUNING (saved had fewer experts than current).
    - Doesn't preserve the exact block class -- always appends TransformerBlock,
      even if the original blocks were DynamicTransformerBlock. Inference works
      because the GQA/RMSNorm/FFN layout is the same.
    """
    grew_anything = False
    new_named = []

    # ---- Layer count ----
    target_n = saved_topology.get("num_layers", len(model.blocks))
    while len(model.blocks) < target_n:
        new_block = TransformerBlock(
            dim=model.dim,
            n_heads=model.n_heads,
            n_kv_heads=model.n_kv_heads,
            ffn_module=SwiGLU(model.dim, model.ff_dim, model.dropout),
            attn_dropout=model.attn_dropout,
            layer_drop=model.layer_drop,
        ).to(device)
        # Match grow_layer's function-preserving zeroing of residuals.
        with torch.no_grad():
            for name, p in new_block.named_parameters():
                if "out_proj.weight" in name or "w3.weight" in name:
                    p.zero_()
        idx = len(model.blocks)
        model.blocks.append(new_block)
        if optimizer is not None:
            for name, p in new_block.named_parameters():
                new_named.append((f"blocks.{idx}.{name}", p))
        grew_anything = True
    if hasattr(model, "num_layers"):
        model.num_layers = len(model.blocks)

    # ---- MoE expert counts ----
    for layer_idx, target_count in saved_topology.get("moe_expert_counts", {}).items():
        layer_idx = int(layer_idx)
        if layer_idx >= len(model.blocks):
            continue
        block = model.blocks[layer_idx]
        if not isinstance(block.ff, MoEFeedForward):
            continue
        moe = block.ff
        grew_experts_here = False
        while moe.n_experts < target_count:
            clone = SwiGLU(moe.dim, moe.hidden_dim, dropout=0.0).to(device)
            clone.load_state_dict(moe.experts[0].state_dict())
            moe.experts.append(clone)
            # _extend_router rebuilds moe.router as a fresh nn.Linear, so
            # the OLD router weight in the optimizer becomes a dead ref --
            # we'll filter those out after this loop.
            moe._extend_router(clone_from=0, device=device, perturb_std=0.0)
            if optimizer is not None:
                expert_idx = moe.n_experts - 1
                for name, p in clone.named_parameters():
                    new_named.append(
                        (f"blocks.{layer_idx}.ff.experts.{expert_idx}.{name}", p)
                    )
            grew_anything = True
            grew_experts_here = True
        # ONLY register the new router if growth actually happened in this
        # layer. Otherwise the existing router is already in the optimizer
        # and re-registering it would create a duplicate reference (Adam
        # would then update the same param twice per step).
        if optimizer is not None and grew_experts_here:
            new_named.append(
                (f"blocks.{layer_idx}.ff.router.weight", moe.router.weight)
            )

    if optimizer is not None and new_named:
        add_new_params_to_optimizer(new_named, optimizer)
        # Clean up dead Parameter references (old routers replaced by
        # _extend_router are no longer in the model but may still be
        # held by optimizer.param_groups).
        current_param_ids = {id(p) for p in model.parameters()}
        for group in optimizer.param_groups:
            group["params"] = [
                p for p in group["params"] if id(p) in current_param_ids
            ]
        # Also drop any orphaned optimizer state.
        for state_key in list(optimizer.state.keys()):
            if id(state_key) not in current_param_ids:
                del optimizer.state[state_key]

    return grew_anything


# ============================================================================
# Convenience configs
# ============================================================================
def small_config():
    """~110M params -- prototyping, debugging, fast iteration."""
    return dict(
        dim=768, n_heads=12, n_kv_heads=4,
        num_layers=12, ff_dim=2048,
    )


def medium_config():
    """~500M params -- the default. Healthy aspect ratio for serious training."""
    return dict(
        dim=1280, n_heads=20, n_kv_heads=4,
        num_layers=24, ff_dim=3456,
    )


def large_config():
    """~1.5B params -- needs serious GPU memory."""
    return dict(
        dim=2048, n_heads=16, n_kv_heads=4,
        num_layers=32, ff_dim=5632,
    )
