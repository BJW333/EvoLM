"""
================================================================================
evolution.py -- Architecture-level self-modification.
================================================================================

WHAT'S IN HERE:
    ArchitectureManager
        observe_grads()                  -- accumulate per-layer grad norm stats
                                            (forward stats are auto-tracked by
                                            forward hooks installed at __init__)
        layer_importance()               -- combined score per layer
        prune_dead_heads()               -- mask attention heads with near-zero Q
        grow_layer()                     -- insert a new transformer block
                                            (Net2Net-style function-preserving)
        grow_layer_with_optimizer()      -- grow + register new params atomically
        evolve()                         -- one-shot scheduled call

PHILOSOPHY:
    True NAS (DARTS, ENAS) optimizes over a discrete architecture space using
    a continuous relaxation. Implementing that properly is its own multi-month
    project and the gains over a well-designed fixed architecture are usually
    modest.

    What this module does instead is the lightweight "grow what works, prune
    what's dead" version. It's not as principled as DARTS but it's tractable
    and it doesn't slow down training the way relaxation-based NAS does.

WHAT "IMPORTANCE" MEANS HERE:
    For each block we track:
        act_delta = ||x_after - x_before|| (EMA over batches)
                    -- how much does this block change its input?
        grad_norm = mean ||grad of block params|| (EMA over batches)
                    -- how much is SGD updating this block?
    A block with both small is dead weight. A block with both large is
    pulling its weight. The product (act_delta * grad_norm) is the score.

WHAT GETS PRUNED VS. GROWN:
    - Heads inside attention layers can be MASKED (zeroed). We don't
      physically remove them because the GQA shape constraints would break.
      A masked head still costs compute -- it's a quality lever, not speed.
    - New transformer blocks can be APPENDED (or inserted) using Net2Net
      function-preserving init: the new block's residual contributions are
      zeroed, so adding it doesn't change the model's output. SGD then lets
      the new block specialize.
"""

from typing import List, Dict, Optional
import torch
import torch.nn as nn

from architecture import (
    GroupedQueryAttention,
    llama_init,
)
from model import add_new_params_to_optimizer


class ArchitectureManager:
    """
    Tracks running statistics on the model and exposes mutation operations.
    Construct once, keep around for the lifetime of training.

    Lifecycle:
        mgr = ArchitectureManager(model)
        for step in training:
            ...
            loss.backward()
            mgr.observe_grads()                    # snapshot grads
            optimizer.step()
            if step % EVOLVE == 0:
                mgr.evolve(prune_heads=True)
            if step % GROW == 0:
                mgr.evolve(
                    grow=True,
                    block_factory=make_block,
                    optimizer=optimizer,           # so new params get registered
                    grow_device=device,
                )
    """

    def __init__(self, model: nn.Module, ema_decay: float = 0.99):
        self.model = model
        self.ema_decay = ema_decay
        # Per-layer stats. Keys are integer indices into model.blocks. Values
        # are 0-dim tensors held on the model's device. We deliberately do NOT
        # call .item() in the hot path -- forcing CPU/GPU sync on every
        # forward+backward step burns ~10ms/step. We only sync when
        # layer_importance() is called (rare: at evolve events).
        self.layer_act_norm: Dict[int, torch.Tensor] = {}
        self.layer_grad_norm: Dict[int, torch.Tensor] = {}

        self._hooks = []
        self._install_hooks()

    # ------------------------------------------------------------------
    # Forward hooks: track how much each block changes its input.
    # ------------------------------------------------------------------
    def _install_hooks(self):
        """Attach forward hooks that update layer_act_norm in place."""
        blocks = self._get_blocks()
        for i, block in enumerate(blocks):
            # We need to bind `i` at hook-registration time using a default arg,
            # else the closure would capture the loop variable and they'd all
            # log to the same key.
            def fwd_hook(mod, inp, out, idx=i):
                # block.forward returns (x, aux, kv_cache). The input is just x.
                x_in = inp[0]
                x_out = out[0] if isinstance(out, tuple) else out
                with torch.no_grad():
                    # RMS magnitude of the per-layer change. Stays on device.
                    delta = (x_out - x_in).pow(2).mean().sqrt()
                    prev = self.layer_act_norm.get(idx)
                    if prev is None:
                        self.layer_act_norm[idx] = delta
                    else:
                        self.layer_act_norm[idx] = (
                            self.ema_decay * prev + (1 - self.ema_decay) * delta
                        )

            self._hooks.append(block.register_forward_hook(fwd_hook))

    def remove_hooks(self):
        """Remove all forward hooks. Call before grow/prune that replaces blocks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def refresh(self):
        """
        Reinstall forward hooks against the current model.blocks list. Call
        this after any operation that changes the block list out-of-band --
        most notably, after load_checkpoint with topology restoration. Resets
        per-layer stats since the indices may now refer to different blocks.
        """
        self.remove_hooks()
        self.layer_act_norm = {}
        self.layer_grad_norm = {}
        self._install_hooks()

    def _get_blocks(self) -> List[nn.Module]:
        return list(self.model.blocks)

    # ------------------------------------------------------------------
    # Backward stats: call after loss.backward() and before optimizer.step()
    # ------------------------------------------------------------------
    def observe_grads(self):
        """
        Update layer_grad_norm with EMA of mean per-param grad norm. Stays
        on device until layer_importance() is called -- no .item() here.
        """
        for i, block in enumerate(self._get_blocks()):
            norms = []
            for p in block.parameters():
                if p.grad is not None:
                    # 0-dim tensor on device; gradient tensors are leaves so
                    # this doesn't build any autograd graph.
                    norms.append(p.grad.norm())
            if not norms:
                continue
            gn = torch.stack(norms).mean()
            prev = self.layer_grad_norm.get(i)
            if prev is None:
                self.layer_grad_norm[i] = gn
            else:
                self.layer_grad_norm[i] = (
                    self.ema_decay * prev + (1 - self.ema_decay) * gn
                )

    # ------------------------------------------------------------------
    # Importance scoring
    # ------------------------------------------------------------------
    def layer_importance(self) -> Dict[int, float]:
        """
        Score per layer. Combines forward contribution and gradient magnitude.
        Higher = more important.

        A block where both signals are small is "dead": it isn't changing the
        residual stream much AND SGD isn't updating it much. Either it's not
        useful or it's saturated. Either way, prune candidate.

        This is the only place we sync the on-device EMA tensors back to
        Python floats -- the .item() calls force a CPU/GPU sync, but it
        only happens when the importance scores are actually queried
        (typically only at evolve events).
        """
        out = {}
        for i in self.layer_act_norm:
            a = self.layer_act_norm.get(i)
            g = self.layer_grad_norm.get(i)
            if a is not None and g is not None:
                out[i] = (a * g).item()
            else:
                out[i] = 0.0
        return out

    # ------------------------------------------------------------------
    # PRUNING: zero out the lowest-importance attention heads.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prune_dead_heads(self, threshold_ratio: float = 0.1):
        """
        For each attention layer:
            1. Compute the Frobenius norm of each Q head's weight.
            2. Find heads with norm < threshold_ratio * median_norm.
            3. Zero those heads in q_proj and the corresponding columns of
               out_proj.

        Why use Q-projection norm as the importance proxy?
        It's noisy but cheap. A head whose Q weights have decayed to zero is
        mathematically not contributing to attention scores -- it's effectively
        already pruned, we're just making it explicit so we can track it.

        Why not also prune K and V?
        Under GQA, K and V are shared across multiple Q heads. Pruning K/V
        of one shared group affects all the Qs in that group. Doing it cleanly
        requires knowing which group is fully dead. Skipped for simplicity.

        Returns: dict {layer_idx: [list of pruned head indices]}
        """
        pruned = {}
        for i, block in enumerate(self._get_blocks()):
            attn: GroupedQueryAttention = block.attn
            n_heads = attn.n_heads
            d_head = attn.d_head

            # q_proj.weight shape: (n_heads * d_head, dim)
            # Reshape to group the OUTPUT dim by head.
            qw = attn.q_proj.weight.view(n_heads, d_head, -1)
            head_norms = qw.norm(dim=(1, 2))                  # (n_heads,)
            median = head_norms.median()
            dead = (
                head_norms < threshold_ratio * median
            ).nonzero(as_tuple=True)[0].tolist()

            if dead:
                # Zero the Q rows for the dead heads.
                qw_view = attn.q_proj.weight.view(n_heads, d_head, -1)
                for h in dead:
                    qw_view[h].zero_()
                # Zero the corresponding columns of out_proj. out_proj has
                # shape (dim, dim) = (out, n_heads * d_head). Columns
                # h*d_head : (h+1)*d_head are head h's contribution.
                ow = attn.out_proj.weight
                for h in dead:
                    ow[:, h * d_head:(h + 1) * d_head].zero_()
                pruned[i] = dead
        return pruned

    # ------------------------------------------------------------------
    # GROWING: insert a new transformer block.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def grow_layer(
        self,
        block_factory,
        insert_at: Optional[int] = None,
        device=None,
    ):
        """
        Insert a new transformer block at position `insert_at` (default: append).

        block_factory: a callable returning a fully-constructed block (e.g.,
                       TransformerBlock or DynamicTransformerBlock with all
                       its FFN, gates, etc. wired in). This lets you grow
                       blocks of whatever shape your model uses -- you're not
                       locked into the vanilla TransformerBlock the way the
                       previous version of this method was.

        Net2Net-style function-preserving init:
            1. block_factory builds the block using its own init.
            2. We re-apply LLaMA-style init for consistency with the rest of
               the model (the factory might use PyTorch defaults).
            3. We zero the residual projections (out_proj of attn, w3 of FFN).
               This makes the block contribute exactly 0 to the residual
               stream, so inserting it doesn't change the model's output.
            4. SGD's first gradient updates will move the residual projections
               away from zero, letting the block start contributing.

        IMPORTANT: After this returns, the new block's parameters are NOT in
        the optimizer yet. Either:
            (a) call optimizer.add_param_group(...) yourself, or
            (b) use grow_layer_with_optimizer below, which does it atomically.
        Skipping this step means SGD silently does nothing to the new layer.
        """
        block = block_factory()
        if device is not None:
            block = block.to(device)

        # Apply LLaMA init for consistency. num_layers=1 here is a placeholder
        # because llama_init's per-module logic doesn't actually use it; the
        # depth-aware scaling is handled separately and we're skipping it
        # because we're about to zero the residuals anyway.
        block.apply(lambda m: llama_init(m, num_layers=1))

        # Zero residual projections (the function-preserving step).
        for name, p in block.named_parameters():
            if "out_proj.weight" in name or "w3.weight" in name:
                p.zero_()

        # Insert into the model's block list.
        blocks = list(self.model.blocks)
        if insert_at is None:
            insert_at = len(blocks)
        blocks.insert(insert_at, block)
        self.model.blocks = nn.ModuleList(blocks)

        # Reinstall hooks because block list identity changed.
        self.remove_hooks()
        self.layer_act_norm = {}
        self.layer_grad_norm = {}
        self._install_hooks()

        # Keep the model's bookkeeping current.
        if hasattr(self.model, "num_layers"):
            self.model.num_layers = len(self.model.blocks)

        return insert_at, block

    @torch.no_grad()
    def grow_layer_with_optimizer(
        self,
        block_factory,
        optimizer,
        insert_at: Optional[int] = None,
        device=None,
    ):
        """
        Same as grow_layer but also adds the new block's parameters to the
        optimizer with proper decay / no-decay classification (so RMSNorm
        gains and learnable scalars don't get weight decay).

        Returns: (insert_at_idx, new_block)
        """
        idx, block = self.grow_layer(block_factory, insert_at=insert_at, device=device)
        # Classify each new param and append to the matching existing group.
        # Names are prefixed with "blocks.{idx}." for clarity but only the
        # leaf-name pattern matters for classification.
        named = [
            (f"blocks.{idx}.{name}", p) for name, p in block.named_parameters()
        ]
        add_new_params_to_optimizer(named, optimizer)
        return idx, block

    # ------------------------------------------------------------------
    # SCHEDULED EVOLUTION STEP
    # ------------------------------------------------------------------
    def evolve(
        self,
        prune_heads: bool = True,
        head_threshold: float = 0.1,
        grow: bool = False,
        block_factory=None,
        grow_insert_at: Optional[int] = None,
        grow_device=None,
        optimizer=None,
    ):
        """
        Convenience wrapper: optionally prune dead heads and/or grow a layer.
        Returns a report dict you can log.

        If `optimizer` is provided alongside `grow=True`, the new block's
        parameters get added to the optimizer atomically.
        """
        report = {"pruned_heads": {}, "grew_layer_at": None}
        if prune_heads:
            report["pruned_heads"] = self.prune_dead_heads(head_threshold)
        if grow and block_factory is not None:
            if optimizer is not None:
                idx, _ = self.grow_layer_with_optimizer(
                    block_factory,
                    optimizer,
                    insert_at=grow_insert_at,
                    device=grow_device,
                )
            else:
                idx, _ = self.grow_layer(
                    block_factory,
                    insert_at=grow_insert_at,
                    device=grow_device,
                )
            report["grew_layer_at"] = idx
        return report
