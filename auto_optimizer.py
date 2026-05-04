"""
================================================================================
auto_optimizer.py -- Heuristic auto-tuning of LR, weight decay, and dropout.
================================================================================

WHAT THIS DOES (in plain language):
    Watches the training loss curve and a few internal stats. Periodically
    nudges hyperparameters in the direction that the heuristics suggest will
    help.

THE THREE LEVERS:
    1. Learning-rate scale (multiplier on the LR schedule)
       - If loss is decreasing AND variance is low -> scale up (try faster)
       - If loss stalled or variance high -> scale down (training is rough)
       Bounded by [lr_scale_min, lr_scale_max]. Note: this is a MULTIPLIER
       on top of train.py's cosine schedule, NOT an absolute LR. Train.py
       owns the schedule; the tuner adjusts how fast we ride it.

    2. Weight decay (per param group, but only for the "decay" group)
       - If grad norms are large relative to param norms -> bump WD up
       - If grad norms are tiny relative -> ease WD off
       Bounded by [wd_min, wd_max]. The "no_decay" group (norm gains,
       gates, learnable scalars) is left alone -- those parameters should
       never get weight decay.

    3. Dropout (per pattern)
       - If train_loss << val_loss -> overfitting, raise dropout
       - If train_loss ~= val_loss -> not overfitting, can lower dropout
       Bounded by [do_min, do_max]. Per-pattern matching so attention
       dropout, FFN dropout, and MoE dropout can be tuned independently.

HONEST CAVEATS:
    These are HEURISTICS. They will work on some setups and hurt on others.
    For production runs, use:
        - Cosine LR schedule with linear warmup (well-tuned, monotonic)
        - Prodigy / D-Adaptation if you want adaptive LR (theoretically grounded)
        - Fixed dropout chosen via ablation (not auto-tuned)
    Use this for research / exploration, not for your final big run.
"""

from collections import deque
from typing import Optional, List
import math
import torch


class AutoTuner:
    def __init__(
        self,
        model,
        optimizer,
        # ---- LR scale tuning ----
        lr_scale_min: float = 0.05,      # absolute floor: 5% of scheduled LR
        lr_scale_max: float = 5.0,       # absolute ceiling: 5x scheduled LR
        lr_grow_factor: float = 1.05,    # multiplicative bump up
        lr_shrink_factor: float = 0.7,   # multiplicative cut on instability
        # ---- WD tuning ----
        wd_min: float = 0.0,
        wd_max: float = 0.5,
        # ---- Dropout tuning ----
        do_min: float = 0.0,
        do_max: float = 0.4,
        do_step: float = 0.02,
        do_patterns: Optional[List[str]] = None,  # name fragments to tune
        # ---- Cadence ----
        loss_window: int = 100,
        apply_every: int = 200,
    ):
        """
        do_patterns: list of substrings to match parameter names against.
                     Each Dropout module is grouped by which pattern matches
                     its name. Default: ["attn", "ff"] groups attention and
                     feed-forward dropouts separately.

        lr_scale: train.py reads this every step and applies it as a
                  multiplier on top of the cosine LR. Starts at 1.0 (no
                  change) and the tuner walks it up/down based on heuristics.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scale = 1.0                  # the multiplier train.py reads
        self.lr_scale_min = lr_scale_min
        self.lr_scale_max = lr_scale_max
        self.lr_grow, self.lr_shrink = lr_grow_factor, lr_shrink_factor
        self.wd_min, self.wd_max = wd_min, wd_max
        self.do_min, self.do_max, self.do_step = do_min, do_max, do_step
        self.do_patterns = do_patterns if do_patterns is not None else ["attn", "ff"]
        self.apply_every = apply_every

        # Rolling windows for loss tracking.
        self.train_losses = deque(maxlen=loss_window)
        self.val_losses = deque(maxlen=loss_window)
        self.steps_since_apply = 0

    # ------------------------------------------------------------------
    # Call every step with the latest losses.
    # ------------------------------------------------------------------
    def step(self, train_loss: float, val_loss: Optional[float] = None):
        if math.isfinite(train_loss):
            self.train_losses.append(train_loss)
        if val_loss is not None and math.isfinite(val_loss):
            self.val_losses.append(val_loss)

        self.steps_since_apply += 1
        if self.steps_since_apply >= self.apply_every:
            self.apply()
            self.steps_since_apply = 0

    # ------------------------------------------------------------------
    # The actual mutation. Called every `apply_every` steps.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def apply(self):
        if len(self.train_losses) < 10:
            # Not enough data yet.
            return {}

        report = {}
        losses = torch.tensor(list(self.train_losses))

        # ---------- LR SCALE HEURISTIC ----------
        # Compare the mean loss in the first vs second half of the window.
        # If the second half is lower, training is making progress.
        # Also check variance -- volatile loss = bad, steady loss = good.
        # We adjust lr_scale (a MULTIPLIER) rather than pg["lr"] directly,
        # because train.py overwrites pg["lr"] every step from the cosine
        # schedule. The multiplier persists across steps.
        half = len(losses) // 2
        first_mean = losses[:half].mean().item()
        second_mean = losses[half:].mean().item()
        var = losses.var().item()
        improvement = first_mean - second_mean
        # Normalize variance by mean^2 so the threshold is scale-free.
        relative_var = var / (losses.mean().item() ** 2 + 1e-8)

        if improvement > 0 and relative_var < 0.01:
            # Smooth descent: try going a bit faster.
            self.lr_scale = min(self.lr_scale * self.lr_grow, self.lr_scale_max)
        elif improvement < 0 or relative_var > 0.05:
            # Stagnation or volatility: pull back.
            self.lr_scale = max(self.lr_scale * self.lr_shrink, self.lr_scale_min)
        report["lr_scale"] = self.lr_scale

        # ---------- WEIGHT DECAY HEURISTIC ----------
        # Look at average gradient norm vs average param norm.
        # If grads are large compared to params, params are moving fast
        # relative to their size -> regularize more. If grads are tiny,
        # ease off. Skip the no_decay group (norm gains, gates, learnable
        # scalars) -- those should never get weight decay.
        total_param_norm, total_grad_norm, n = 0.0, 0.0, 0
        for p in self.model.parameters():
            if p.requires_grad:
                total_param_norm += p.detach().norm().item()
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item()
                n += 1
        if n > 0:
            avg_param = total_param_norm / n
            avg_grad = total_grad_norm / n
            ratio = avg_grad / (avg_param + 1e-8)
            for pg in self.optimizer.param_groups:
                # Skip the no_decay group entirely.
                if pg.get("name") == "no_decay" or pg.get("weight_decay", 0.0) == 0.0:
                    continue
                old_wd = pg["weight_decay"]
                if ratio > 0.1:
                    new_wd = min(old_wd * 1.1 + 1e-4, self.wd_max)
                elif ratio < 0.001:
                    new_wd = max(old_wd * 0.9, self.wd_min)
                else:
                    new_wd = old_wd
                pg["weight_decay"] = new_wd
            # Report the WD on the decay group (or the first group if none named).
            for pg in self.optimizer.param_groups:
                if pg.get("name") == "decay" or pg.get("weight_decay", 0.0) > 0:
                    report["wd"] = pg["weight_decay"]
                    break

        # ---------- DROPOUT HEURISTIC ----------
        # Compare recent train and val losses. Big gap = overfitting.
        # Each name pattern (e.g., "attn", "ff") gets its own dropout level.
        if len(self.val_losses) >= 5:
            recent_train = sum(list(self.train_losses)[-20:]) / min(20, len(self.train_losses))
            recent_val = sum(list(self.val_losses)[-20:]) / min(20, len(self.val_losses))
            gap = recent_val - recent_train
            ref = max(abs(recent_val), 1e-4)
            rel_gap = gap / ref

            # Group standard nn.Dropout modules by name pattern. Note: this
            # only matches torch.nn.Dropout. dynamic.ConcreteDropout has its
            # own learnable rate that the *gradient* tunes, so it's handled
            # by SGD rather than this heuristic -- which is the intended split
            # between auto_optimizer.py and dynamic.py.
            by_pattern = {pat: [] for pat in self.do_patterns}
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    for pat in self.do_patterns:
                        if pat in name:
                            by_pattern[pat].append(module)
                            break
                    # Modules that don't match any pattern are left alone.
                    # If you want them tuned, add an empty string "" to
                    # do_patterns -- it matches every name.

            # Apply the same delta to all dropouts in a group.
            new_drops = {}
            for pat, mods in by_pattern.items():
                if not mods:
                    continue
                if rel_gap > 0.1:
                    new_p = min(mods[0].p + self.do_step, self.do_max)
                elif rel_gap < 0.02:
                    new_p = max(mods[0].p - self.do_step, self.do_min)
                else:
                    new_p = mods[0].p
                for m in mods:
                    m.p = new_p
                new_drops[pat] = new_p

            report["dropout"] = new_drops

        return report
