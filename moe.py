"""
================================================================================
moe.py -- Mixture of Experts FFN with dynamic expert add/prune.
================================================================================

WHAT'S IN HERE:
    MoEFeedForward        -- top-k MoE FFN, drop-in replacement for SwiGLU
    .maybe_grow_or_prune  -- mutate expert count between training steps

CONCEPT IN ONE PARAGRAPH:
    Instead of routing every token through one big FFN, we have N small
    "experts" (each is a SwiGLU). A learned router decides which experts each
    token goes to (top-k of them, k typically 2). Outputs are weighted by the
    router's softmax probabilities. This means each token only activates
    k/N of the FFN params -- you can scale up TOTAL params (more experts)
    without scaling up per-token compute. Mixtral 8x7B has 47B params total
    but only ~13B active per token.

THE PARTS THAT ARE STANDARD AND WORK WELL:
    - Top-k routing with softmax gates
    - Capacity factor: each expert can only handle so many tokens per batch
    - Switch Transformer load-balancing aux loss: discourage degenerate
      "everyone routes to expert 0" solutions
    - Router z-loss: penalize the magnitude of router logits to prevent the
      softmax from saturating (a numerical stability trick from PaLM/ST-MoE)

THE EXPERIMENTAL PART (clearly marked):
    - Dynamic expert spawning: clone a hot expert with small perturbation
    - Dynamic expert pruning: drop a cold expert
    These mutate module structure between steps. Be careful -- they invalidate
    optimizer state for the affected parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture import SwiGLU


class MoEFeedForward(nn.Module):
    """
    Top-k MoE FFN. Returns (output, aux_loss) where aux_loss combines:
        - load-balancing loss (encourage uniform expert utilization)
        - z-loss (keep router logits well-conditioned)

    INPUTS:
        dim:             model hidden size
        hidden_dim:      FFN hidden size of each expert
        n_experts:       starting number of experts
        top_k:           how many experts each token gets routed to
        capacity_factor: experts can take up to ceil(top_k * tokens / n_experts
                         * capacity_factor) tokens per batch. >1 means slack
                         for unbalanced batches; 1.0 is tight.
        dropout:         dropout inside each expert
        lb_loss_weight:  weight on load-balancing aux loss
        z_loss_weight:   weight on router z-loss

    OUTPUT (per forward call):
        output:    (B, S, D)
        aux_loss:  scalar tensor, add this to your main loss
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
        lb_loss_weight: float = 0.01,
        z_loss_weight: float = 1e-3,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.lb_loss_weight = lb_loss_weight
        self.z_loss_weight = z_loss_weight

        # The router: a linear layer mapping x -> per-expert logits.
        # No bias is the Switch Transformer convention; biases let the model
        # learn a static preference for some expert which screws up balancing.
        self.router = nn.Linear(dim, n_experts, bias=False)

        # Experts as a ModuleList so we can append/delete them at runtime.
        self.experts = nn.ModuleList(
            [SwiGLU(dim, hidden_dim, dropout) for _ in range(n_experts)]
        )

        # Utilization tracking: an EMA of "what fraction of tokens went to
        # each expert?" across recent batches. Used by maybe_grow_or_prune to
        # decide who's hot and who's cold. We register it as a buffer so it
        # gets moved with .to(device) and saved with state_dict.
        self.register_buffer("utilization", torch.ones(n_experts) / n_experts)
        self.util_decay = 0.99  # bigger = slower-changing stats

    @property
    def n_experts(self):
        # Always read from the ModuleList so it's correct after add/prune.
        return len(self.experts)

    # ------------------------------------------------------------------
    # FORWARD PASS
    # ------------------------------------------------------------------
    def forward(self, x, valid_mask=None):
        """
        x: (B, S, D). Returns (out, aux_loss).
        Computation is done on the flattened view (N=B*S, D), then reshaped
        back at the end.

        `valid_mask` is accepted for signature compatibility with PlasticSwiGLU
        but ignored here -- MoE routing doesn't need to know about padding.
        """
        del valid_mask
        B, S, D = x.shape
        x_flat = x.view(-1, D)              # (N, D)
        N = x_flat.size(0)
        n_e = self.n_experts

        # ---- ROUTER ----
        # Per-token logits over experts.
        logits = self.router(x_flat)        # (N, n_experts)

        # Router z-loss: this penalizes the LOG SUM EXP magnitude of the
        # logits. It keeps the router from blowing up logits to make hard
        # decisions, which causes numerical issues in fp16. Cheap insurance.
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean() * self.z_loss_weight

        # Softmax gives per-expert probabilities. We need this for both the
        # load-balancing loss (uses full distribution) and the dispatch
        # weights (uses only top-k).
        probs = F.softmax(logits, dim=-1)   # (N, n_experts)

        # ---- TOP-K SELECTION ----
        # For each token, pick its top_k experts and their probabilities.
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)   # both (N, top_k)
        # Renormalize so the kept probabilities sum to 1 -- standard Mixtral
        # convention. Without this, dropping all-but-top-k makes the per-token
        # output systematically smaller.
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # ---- LOAD-BALANCING LOSS ----
        # Switch Transformer eq. 4:  loss = n_experts * sum_i (f_i * P_i)
        # where:
        #   f_i = fraction of dispatch slots routed to expert i (counts)
        #   P_i = mean router probability for expert i over batch (probs)
        # Why does minimizing this balance the load? sum_i f_i * P_i is
        # minimized (subject to f_i, P_i being valid distributions) when both
        # are uniform. Multiplying by n_experts makes the optimum exactly 1
        # regardless of n_experts.
        #
        # Counting convention: for top-k routing with k>=2 we count all
        # k slots (not just the top-1 choice). This matches Mixtral / GShard
        # and means "f_i = fraction of total dispatched compute that goes
        # through expert i". Just counting top-1 leaves k>=2 partly unbalanced.
        with torch.no_grad():
            # Flatten the (N, top_k) index tensor to (N * top_k,) and count.
            f = torch.zeros(n_e, device=x.device).scatter_add_(
                0,
                topk_idx.flatten(),
                torch.ones(N * self.top_k, device=x.device, dtype=torch.float),
            ) / (N * self.top_k)
        P = probs.mean(dim=0)               # (n_experts,)
        lb_loss = n_e * (f * P).sum() * self.lb_loss_weight

        # Update the EMA utilization buffer (no grad). Same convention as f.
        # Used downstream by maybe_grow_or_prune to decide who's hot/cold.
        with torch.no_grad():
            self.utilization.mul_(self.util_decay).add_(f, alpha=1 - self.util_decay)

        # ---- CAPACITY ----
        # Each expert handles at most `capacity` tokens per batch. Excess
        # tokens are DROPPED (their contribution is just zero). The slack
        # factor > 1 absorbs uneven routing across mini-batches.
        capacity = max(
            1, int(math.ceil(self.top_k * N * self.capacity_factor / n_e))
        )

        # ---- DISPATCH ----
        # Loop over experts. For each, find the tokens routed to it, drop the
        # excess, run them through, and scatter-add the results back.
        # This is O(n_experts) sequential calls -- fine up to ~32 experts.
        # Beyond that you want a fused implementation (Megablocks etc).
        output = torch.zeros_like(x_flat)

        for e_idx in range(n_e):
            # Build a mask of (token, slot) pairs that picked this expert.
            mask = topk_idx == e_idx                   # (N, top_k) bool
            if not mask.any():
                continue

            # Get the (token_index, slot_index) coordinates of all hits.
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue

            # Per-assignment weight is the router prob for that token's
            # choice of this expert.
            weights = topk_probs[token_idx, slot_idx]   # (k,)

            # ----- CAPACITY DROPPING -----
            # If too many tokens picked this expert, keep the HIGHEST-WEIGHT
            # ones (the router was most confident about them). This is what
            # Switch / GShard / Mixtral do. Previously we kept the first ones
            # in arbitrary order which biases against late-position tokens.
            if token_idx.numel() > capacity:
                # Sort by weight descending, take top `capacity`.
                _, keep = torch.topk(weights, k=capacity, largest=True)
                token_idx = token_idx[keep]
                slot_idx = slot_idx[keep]
                weights = weights[keep]

            # Run the expert on the kept tokens, weight, scatter-add.
            expert_in = x_flat[token_idx]
            expert_out = self.experts[e_idx](expert_in) * weights.unsqueeze(-1)
            output.index_add_(0, token_idx, expert_out)

        output = output.view(B, S, D)
        aux_loss = lb_loss + z_loss
        return output, aux_loss

    # ------------------------------------------------------------------
    # DYNAMIC EXPERT MANAGEMENT
    # ------------------------------------------------------------------
    # Call between training steps, NOT inside the forward pass.
    # These methods mutate the ModuleList -- if you call them mid-forward
    # while another thread is using the model, you'll have a bad time.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def maybe_grow_or_prune(
        self,
        device,
        max_experts: int = 32,
        min_experts: int = 2,
        spawn_threshold: float = 3.0,    # spawn if util > spawn_threshold * mean_util
        prune_threshold: float = 0.1,    # prune if util < prune_threshold * mean_util
        perturb_std: float = 0.01,
    ):
        """
        Look at the utilization EMA and decide whether to:
        (a) clone an over-utilized expert (so it can be split via SGD), or
        (b) drop an under-utilized expert (free up capacity).

        Returns a dict {"spawned": [old_idx, ...], "pruned": [old_idx, ...]}.

        WHY perturbation when cloning? If we just duplicated the expert exactly
        AND copied its router row, the two clones would be functionally
        identical and SGD would update them identically -- they'd never
        diverge. A small Gaussian perturbation breaks that symmetry; SGD then
        pulls them apart based on which tokens each happens to win.

        WARNING: This rebuilds the router as a fresh nn.Linear. The optimizer
        is still holding stale moment buffers for the OLD router weights.
        For Adam this means the new router starts with no momentum, which is
        usually fine but mildly suboptimal. If you care, call:
            evolve_moe_with_optimizer(model, optimizer, device, **kwargs)
        from the helper at the bottom of this file -- it scrubs optimizer
        state for the rebuilt parameters.
        """
        u = self.utilization
        mean_u = u.mean().item()
        events = {"spawned": [], "pruned": []}

        # ---- SPAWN over-utilized ----
        if self.n_experts < max_experts:
            hot_idx = (u > spawn_threshold * mean_u).nonzero(as_tuple=True)[0].tolist()
            for h in hot_idx:
                if self.n_experts >= max_experts:
                    break
                # Build a clone with same architecture.
                clone = SwiGLU(self.dim, self.hidden_dim, dropout=0.0).to(device)
                clone.load_state_dict(self.experts[h].state_dict())
                # Perturb to break symmetry.
                for p in clone.parameters():
                    p.add_(torch.randn_like(p) * perturb_std)
                self.experts.append(clone)
                # Extend the router with a copy of the hot expert's row,
                # also perturbed. Without this, the new expert has zero
                # logit weight and never gets routed to.
                self._extend_router(clone_from=h, device=device, perturb_std=perturb_std)
                events["spawned"].append(h)

        # ---- PRUNE under-utilized ----
        if self.n_experts > min_experts:
            cold_idx = (u < prune_threshold * mean_u).nonzero(as_tuple=True)[0].tolist()
            # Sort descending so we delete from the end first; otherwise
            # earlier deletions invalidate later indices.
            for c in sorted(cold_idx, reverse=True):
                if self.n_experts <= min_experts:
                    break
                del self.experts[c]
                self._shrink_router(c, device=device)
                events["pruned"].append(c)

        return events

    def _extend_router(self, clone_from: int, device, perturb_std: float):
        """Append a row to the router matrix, initialized as a perturbed copy
        of the row for `clone_from`."""
        old_w = self.router.weight.data    # (n_experts, dim) BEFORE the new row
        new_row = (
            old_w[clone_from:clone_from + 1]
            + torch.randn_like(old_w[clone_from:clone_from + 1]) * perturb_std
        )
        new_w = torch.cat([old_w, new_row], dim=0)
        # Replace the router. Note: this loses optimizer state for the router.
        self.router = nn.Linear(self.dim, new_w.size(0), bias=False).to(device)
        self.router.weight.data.copy_(new_w)
        # Extend the utilization buffer with the mean (a reasonable prior
        # for a fresh expert).
        new_util = torch.cat(
            [self.utilization, self.utilization.mean().unsqueeze(0)], dim=0
        )
        self.utilization = new_util

    def _shrink_router(self, idx: int, device):
        """Drop a row from the router and the corresponding utilization entry."""
        old_w = self.router.weight.data
        keep = [i for i in range(old_w.size(0)) if i != idx]
        new_w = old_w[keep]
        self.router = nn.Linear(self.dim, new_w.size(0), bias=False).to(device)
        self.router.weight.data.copy_(new_w)
        self.utilization = self.utilization[keep]


# ============================================================================
# Optimizer-aware helper
# ============================================================================
def evolve_moe_with_optimizer(model, optimizer, device, **grow_prune_kwargs):
    """
    Helper that runs maybe_grow_or_prune on every MoE layer, then fixes the
    optimizer:
      - removes optimizer state for parameters that no longer exist
      - removes dead Parameter refs from param_groups (otherwise they leak)
      - adds new params to the right decay/no-decay group via
        model.add_new_params_to_optimizer.

    This is the "do it right" version. The simpler alternative is to ignore
    the optimizer state issue -- Adam will rebuild momentum after a few steps
    and the brief inefficiency is rarely material.
    """
    from model import add_new_params_to_optimizer

    # Snapshot the set of param ids currently in the optimizer.
    old_params = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            old_params.add(id(p))

    all_events = {}
    for i, mod in enumerate(model.modules()):
        if isinstance(mod, MoEFeedForward):
            ev = mod.maybe_grow_or_prune(device, **grow_prune_kwargs)
            if ev["spawned"] or ev["pruned"]:
                all_events[i] = ev

    # Find new (name, param) pairs from the model. We use named_parameters
    # so add_new_params_to_optimizer can classify them correctly.
    new_named = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in old_params
    ]
    if new_named:
        add_new_params_to_optimizer(new_named, optimizer)

    # Drop optimizer state AND param_group references for params no longer
    # in the model. Without the param_group filtering, dead parameter
    # objects stay alive (held by param_groups), leaking memory across
    # many evolve cycles.
    current_param_ids = {id(p) for p in model.parameters()}
    for state_key in list(optimizer.state.keys()):
        if id(state_key) not in current_param_ids:
            del optimizer.state[state_key]
    for group in optimizer.param_groups:
        group["params"] = [p for p in group["params"] if id(p) in current_param_ids]

    return all_events
