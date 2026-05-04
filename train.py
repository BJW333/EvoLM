"""
================================================================================
train.py -- Production training script for EvoLM.
================================================================================

Trains an EvoLM model on a pre-tokenized binary file of token IDs.
Includes the full self-organizing pipeline:
    - Mixed precision (bf16 by default; fp32 fallback if no CUDA)
    - Linear warmup + cosine LR schedule (always applies, even with auto-tuner)
    - Gradient accumulation for large effective batch sizes
    - Gradient clipping
    - Atomic checkpointing with topology save/restore for resume
    - Real validation pass on held-out tokens
    - JSON metric log for plotting later
    - MoE expert evolution, dead-head pruning, layer growth
    - Heuristic auto-tuner for LR/WD/dropout adjustment
    - --auto preset: turns on dynamic blocks + auto-tuner + MoE evolve

DATA FORMAT:
    Expects two files under --data-dir:
        train.bin  -- np.uint16 array of token IDs (one big concatenated stream)
        val.bin    -- same, for validation
    See prepare_data.py for how to make these from raw text.

QUICK START:
    python prepare_data.py --input-file my_corpus.txt --output-dir data/
    python train.py --data-dir data/ --config small --steps 10000

LIFECYCLE PER OPTIMIZER STEP:
    1. Update LR per cosine schedule (warmup + cosine decay)
    2. for each grad-accum micro-batch:
         - reset Hebbian fast-weights (per sequence)
         - forward in autocast(bf16)
         - loss.backward() (loss already includes MoE aux)
         - detach Hebbian trace
    3. Observe per-layer grad norms
    4. Clip gradient norm
    5. optimizer.step()
    6. Maybe evaluate, evolve, grow, save, log

LIFECYCLE PER N STEPS:
    eval_every:    compute val loss, possibly save best.pt
    evolve_every:  MoE expert add/prune + dead-head pruning
    grow_every:    insert a new transformer block (function-preserving)
    save_every:    write latest.pt
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from architecture import SwiGLU, TransformerBlock
from model import (
    EvoLM, small_config, medium_config, large_config,
    restore_topology,
)
from evolution import ArchitectureManager
from auto_optimizer import AutoTuner
from moe import evolve_moe_with_optimizer, MoEFeedForward


# ============================================================================
# Data loading
# ============================================================================
class TokenDataLoader:
    """
    Memory-maps a .bin file of token IDs (uint16 numpy) and yields random
    fixed-length chunks. This is the nanoGPT-style approach: instead of
    epochs over a finite dataset, you sample random offsets into a giant
    token stream. Works great for LM pretraining and is fast (mmap means
    we don't load the file into RAM).

    Why uint16? Standard tokenizers have vocab sizes <= 65535. Storing as
    uint16 halves disk space vs int32.
    """
    def __init__(self, bin_path: str, batch_size: int, seq_len: int, device: str):
        self.path = bin_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        # mmap: tokens stay on disk; OS pages them in lazily.
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if len(self.data) < seq_len + 1:
            raise ValueError(
                f"{bin_path} has {len(self.data)} tokens but we need at "
                f"least seq_len + 1 = {seq_len + 1}"
            )

    def __len__(self):
        return len(self.data)

    def get_batch(self):
        # Pick batch_size random starting offsets such that we can read
        # seq_len + 1 tokens (one extra for the next-token target).
        starts = np.random.randint(
            0, len(self.data) - self.seq_len - 1, size=(self.batch_size,)
        )
        # nn.Embedding needs int64.
        x = np.stack([
            self.data[s : s + self.seq_len].astype(np.int64) for s in starts
        ])
        y = np.stack([
            self.data[s + 1 : s + 1 + self.seq_len].astype(np.int64) for s in starts
        ])
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        y = torch.from_numpy(y).to(self.device, non_blocking=True)
        return {"input_ids": x, "labels": y}


# ============================================================================
# Learning-rate schedule: linear warmup + cosine decay
# ============================================================================
def get_lr(step: int, warmup: int, cosine_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Standard LLM LR schedule:
      - Steps [0, warmup):           linear ramp from 0 to max_lr.
      - Steps [warmup, warmup + cosine_steps]: cosine decay max_lr -> min_lr.
      - After that:                  stay at min_lr.

    Why warmup? At step 0 the model's params are random, so gradients are
    noisy. Hitting them with full LR causes destabilization. The linear
    ramp lets Adam's running statistics settle before we go full speed.

    Why cosine? Empirically beats linear or exponential decay for LLMs.
    """
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step > warmup + cosine_steps:
        return min_lr
    progress = (step - warmup) / max(1, cosine_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================================================
# Checkpointing
# ============================================================================
def _model_topology(model) -> dict:
    """Capture architectural state that may have changed since model
    construction (number of blocks, MoE expert counts per layer)."""
    moe_expert_counts = {}
    for i, block in enumerate(model.blocks):
        if isinstance(block.ff, MoEFeedForward):
            moe_expert_counts[i] = block.ff.n_experts
    return {
        "num_layers": len(model.blocks),
        "moe_expert_counts": moe_expert_counts,
    }


def save_checkpoint(path: Path, model, optimizer, step: int, best_val: float, args):
    """
    Save model + optimizer + step counter + architecture topology.
    Atomic write via tmp+rename so a crash mid-save can't corrupt the
    existing checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "args": vars(args),
            "topology": _model_topology(model),
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(path: Path, model, optimizer, device):
    """
    Load model + optimizer state. Returns (start_step, best_val_loss).
    Calls restore_topology first so grown architectures load cleanly.
    """
    state = torch.load(path, map_location=device)

    # Step 1: rebuild architecture to match saved topology.
    saved_topology = state.get("topology", {})
    grew = False
    if saved_topology:
        grew = restore_topology(model, saved_topology, device, optimizer=optimizer)
        if grew:
            print(f"[load_ckpt] grew model to match saved topology: "
                  f"{len(model.blocks)} layers, {model.num_params() / 1e6:.1f}M params")

    # Step 2: load weights.
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"[load_ckpt] {len(missing)} missing keys (will keep init), "
              f"sample: {missing[:3]}")
    if unexpected:
        print(f"[load_ckpt] {len(unexpected)} unexpected keys (ignored), "
              f"sample: {unexpected[:3]}")

    # Step 3: load optimizer state.
    #
    # IMPORTANT: We deliberately SKIP the optimizer state if topology was
    # regrown. Reason: during training, layer growth and MoE expert growth
    # are interleaved across many steps in arbitrary order. restore_topology
    # always does layers-then-experts, so the parameter ordering in
    # optimizer.param_groups[*]["params"] won't match the saved order.
    # PyTorch's load_state_dict matches state to params by INDEX within
    # each group, so a mismatch would silently put Adam moments on the
    # wrong parameters -- corrupting the resumed training run.
    #
    # Trade-off: with grown topology we lose ~hundreds of steps' worth of
    # Adam momentum buildup. Adam re-warms quickly so this is fine.
    if grew:
        print(f"[load_ckpt] topology was regrown -- skipping optimizer state "
              f"load (Adam moments will rebuild over the next few hundred steps)")
    else:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError) as e:
            print(f"[load_ckpt] optimizer state did not match current groups: {e}")
            print(f"[load_ckpt] continuing with fresh optimizer state")

    return state["step"], state.get("best_val", float("inf"))


# ============================================================================
# Validation
# ============================================================================
@torch.no_grad()
def evaluate(model, val_loader: TokenDataLoader, n_batches: int, device, autocast_ctx):
    """
    Mean CE loss over n_batches random validation chunks.
    """
    model.eval()
    losses = []
    for _ in range(n_batches):
        batch = val_loader.get_batch()
        # Reset plasticity each eval batch so traces don't leak across
        # unrelated random chunks.
        model.reset_plasticity(batch_size=batch["input_ids"].size(0), device=device)
        with autocast_ctx:
            _, _, parts = model(batch["input_ids"], labels=batch["labels"])
        losses.append(parts["ce"].item())
        model.detach_plasticity()
    model.train()
    return float(np.mean(losses))


# ============================================================================
# JSON logger
# ============================================================================
class JSONLogger:
    """
    Append-only JSON-lines (jsonl) logger. One line = one event. Use jq /
    pandas to load and plot afterwards.
    """
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # Open in append mode so resume keeps previous logs.
        self._fp = open(path, "a", buffering=1)  # line buffered

    def log(self, **kwargs):
        self._fp.write(json.dumps(kwargs) + "\n")

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass


# ============================================================================
# Setup helpers
# ============================================================================
def make_model_from_args(args, device: str):
    """Build an EvoLM with config matching args."""
    cfg_fn = {
        "small": small_config, "medium": medium_config, "large": large_config,
    }[args.config]
    cfg = cfg_fn()

    # Decide which layer indices get MoE / plasticity.
    moe_layers = (
        list(range(1, cfg["num_layers"], args.moe_every_n))
        if args.moe_every_n > 0 else []
    )
    plastic_layers = (
        list(range(cfg["num_layers"] - args.plastic_last_n, cfg["num_layers"]))
        if args.plastic_last_n > 0 else []
    )

    model = EvoLM(
        vocab_size=args.vocab_size,
        max_seq=args.max_seq,
        pad_token_id=args.pad_token_id,
        moe_layers=moe_layers,
        plastic_layers=plastic_layers,
        moe_n_experts=args.moe_n_experts,
        moe_top_k=args.moe_top_k,
        plastic_eta=args.plastic_eta,
        use_dynamic=args.use_dynamic,
        dropout=args.dropout,
        attn_dropout=args.dropout,
        layer_drop=args.layer_drop,
        gradient_checkpointing=args.grad_checkpoint,
        **cfg,
    ).to(device)
    return model


def make_block_factory(model):
    """Return a factory for grow_layer that matches the model's block type."""
    def factory():
        return TransformerBlock(
            dim=model.dim,
            n_heads=model.n_heads,
            n_kv_heads=model.n_kv_heads,
            ffn_module=SwiGLU(model.dim, model.ff_dim, model.dropout),
            attn_dropout=model.attn_dropout,
            layer_drop=model.layer_drop,
        )
    return factory


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # ----- Data -----
    parser.add_argument("--data-dir", type=str, required=True,
                        help="dir containing train.bin and val.bin")
    parser.add_argument("--vocab-size", type=int, default=50258,
                        help="must match the tokenizer used by prepare_data.py")
    parser.add_argument("--pad-token-id", type=int, default=50257)

    # ----- Architecture -----
    parser.add_argument("--config", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--max-seq", type=int, default=1024)
    parser.add_argument("--moe-every-n", type=int, default=2,
                        help="MoE FFN every N layers; 0 = no MoE")
    parser.add_argument("--moe-n-experts", type=int, default=8)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--plastic-last-n", type=int, default=2,
                        help="last N layers use PlasticSwiGLU; 0 = no plasticity")
    parser.add_argument("--plastic-eta", type=float, default=0.01)
    parser.add_argument("--use-dynamic", action="store_true",
                        help="use DynamicTransformerBlock (learnable gates + layer drop)")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--layer-drop", type=float, default=0.0)
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="enable gradient checkpointing -- ~30%% slower per "
                             "step but ~50%% less activation memory. Useful for "
                             "fitting medium/large configs on 16-24GB GPUs. "
                             "Plastic layers are auto-skipped (their Hebbian "
                             "side-effects can't be safely re-executed during "
                             "backward).")

    # ----- Optimization -----
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="effective batch = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=3e-4, help="peak LR after warmup")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="LR floor")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--cosine-steps", type=int, default=None,
                        help="defaults to steps - warmup_steps")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    # ----- Self-organization schedule -----
    parser.add_argument("--evolve-every", type=int, default=1000,
                        help="MoE add/prune + head-prune cadence; 0 = disabled")
    parser.add_argument("--grow-every", type=int, default=0,
                        help="layer growth cadence; 0 = disabled (recommended)")
    parser.add_argument("--use-auto-tuner", action="store_true",
                        help="enable heuristic LR/WD/dropout tuning. Cosine LR still "
                             "applies as the baseline; auto-tuner adjusts on top.")
    parser.add_argument("--auto", action="store_true",
                        help="convenience preset: --use-dynamic --use-auto-tuner "
                             "with all evolution mechanisms on. Equivalent to a "
                             "'maximum self-organization' run.")

    # ----- I/O -----
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-file", type=str, default="train_log.jsonl",
                        help="JSON-lines metric log; relative paths go under --checkpoint-dir")
    parser.add_argument("--resume", action="store_true",
                        help="resume from <checkpoint-dir>/latest.pt if it exists")

    # ----- Hardware / dtype -----
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Apply --auto preset.
    if args.auto:
        args.use_dynamic = True
        args.use_auto_tuner = True
        if args.evolve_every == 0:
            args.evolve_every = 1000

    return args


def main():
    args = parse_args()

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Device + dtype ----
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device

    # bf16 needs Ampere+; fall back gracefully.
    if args.dtype == "bf16" and (
        device == "cpu" or not torch.cuda.is_bf16_supported()
    ):
        print("[setup] bf16 unavailable, falling back to fp32")
        args.dtype = "fp32"
    if args.dtype == "bf16":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif args.dtype == "fp16":
        # Note: no GradScaler here. fp16 LM training is finicky -- bf16 is
        # strongly preferred. This branch exists only for older GPUs.
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # ---- Data ----
    data_dir = Path(args.data_dir)
    train_loader = TokenDataLoader(
        str(data_dir / "train.bin"), args.batch_size, args.max_seq, device,
    )
    val_loader = TokenDataLoader(
        str(data_dir / "val.bin"), args.batch_size, args.max_seq, device,
    )
    print(f"[data] train tokens: {len(train_loader):,}  "
          f"val tokens: {len(val_loader):,}")

    # ---- Model ----
    model = make_model_from_args(args, device)
    print(f"[model] {model.num_params() / 1e6:.1f}M params  "
          f"({len(model.blocks)}L x {model.dim}D)  use_dynamic={args.use_dynamic}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.get_param_groups(weight_decay=args.weight_decay),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    arch_mgr = ArchitectureManager(model)
    tuner = AutoTuner(model, optimizer) if args.use_auto_tuner else None

    # ---- Resume ----
    start_step = 0
    best_val = float("inf")
    ckpt_dir = Path(args.checkpoint_dir)
    latest_path = ckpt_dir / "latest.pt"
    if args.resume and latest_path.exists():
        start_step, best_val = load_checkpoint(latest_path, model, optimizer, device)
        # If load_checkpoint grew the model to match saved topology, the
        # arch_mgr's forward hooks are still on the OLD block list. Refresh
        # so the new blocks get tracked too.
        arch_mgr.refresh()
        print(f"[resume] from step {start_step}, best_val={best_val:.4f}, "
              f"now {len(model.blocks)} layers, {model.num_params() / 1e6:.1f}M params")

    cosine_steps = args.cosine_steps or max(1, args.steps - args.warmup_steps)

    # ---- JSON logger ----
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = ckpt_dir / log_path
    logger = JSONLogger(log_path)
    logger.log(event="start", step=start_step, args=vars(args))

    # ---- Block factory for layer growth ----
    block_factory = make_block_factory(model)

    # ============================================================
    # Training loop
    # ============================================================
    model.train()
    t0 = time.time()

    # Pre-initialize so the `finally` block always has a defined step,
    # even if the for-loop never executes (e.g., resuming a completed run).
    step = start_step

    try:
        for step in range(start_step, args.steps):
            # ---- LR schedule (always applies). The cosine schedule sets
            # the BASE LR; the auto-tuner (if enabled) maintains a multiplier
            # in tuner.lr_scale that we apply on top. The multiplier
            # persists across steps even though the base gets reset each
            # iteration. This is the correct way to compose the two.
            base_lr = get_lr(
                step, args.warmup_steps, cosine_steps, args.lr, args.min_lr,
            )
            lr_scale = tuner.lr_scale if tuner is not None else 1.0
            effective_lr = base_lr * lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = effective_lr

            # ---- Forward + backward over micro-batches ----
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            ce_accum = 0.0
            aux_accum = 0.0

            for micro in range(args.grad_accum):
                batch = train_loader.get_batch()
                model.reset_plasticity(
                    batch_size=batch["input_ids"].size(0), device=device,
                )
                with autocast_ctx:
                    _, loss, parts = model(batch["input_ids"], labels=batch["labels"])
                    # Scale by 1/grad_accum so the accumulated gradient is
                    # the mean across micro-batches (not the sum).
                    loss = loss / args.grad_accum
                loss.backward()
                model.detach_plasticity()

                loss_accum += loss.item() * args.grad_accum  # un-scale for logging
                ce_accum += parts["ce"].item()
                aux_accum += parts["aux"].item()

            loss_accum /= args.grad_accum
            ce_accum /= args.grad_accum
            aux_accum /= args.grad_accum

            # ---- Stats / clip / step ----
            arch_mgr.observe_grads()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # ---- Validation (and "best" save) ----
            val_loss = None
            if step > 0 and step % args.eval_every == 0:
                val_loss = evaluate(
                    model, val_loader, args.eval_batches, device, autocast_ctx,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        ckpt_dir / "best.pt", model, optimizer, step, best_val, args,
                    )

            # ---- Auto-tuner: ONE call per step, with optional val_loss.
            # This was a real bug in the previous version (calling step
            # twice on eval steps double-counted losses). ----
            if tuner is not None:
                tuner.step(train_loss=loss_accum, val_loss=val_loss)

            # ---- Logging ----
            if step % args.log_every == 0:
                dt = time.time() - t0
                tok_per_step = args.batch_size * args.grad_accum * args.max_seq
                tok_per_sec = tok_per_step * args.log_every / max(dt, 1e-6)
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"step {step:>6}  loss={loss_accum:.4f}  "
                    f"ce={ce_accum:.4f}  aux={aux_accum:.4f}  "
                    f"lr={lr_now:.2e}  tok/s={tok_per_sec:.0f}"
                )
                logger.log(
                    event="step", step=step, loss=loss_accum,
                    ce=ce_accum, aux=aux_accum, lr=lr_now,
                    tok_per_sec=tok_per_sec,
                )
                t0 = time.time()

            if val_loss is not None:
                print(f"  [eval @ step {step}] val_loss={val_loss:.4f}"
                      f"{'  *NEW BEST*' if val_loss == best_val else ''}")
                logger.log(event="eval", step=step, val_loss=val_loss, best_val=best_val)

            # ---- MoE evolution + head pruning ----
            if (
                args.evolve_every > 0
                and step > 0
                and step % args.evolve_every == 0
            ):
                moe_events = evolve_moe_with_optimizer(model, optimizer, device)
                arch_events = arch_mgr.evolve(prune_heads=True, head_threshold=0.1)
                if moe_events or any(arch_events.values()):
                    print(f"  [evolve @ step {step}] moe={moe_events}  arch={arch_events}")
                    logger.log(
                        event="evolve", step=step,
                        moe=moe_events, arch=arch_events,
                    )

            # ---- Layer growth ----
            if (
                args.grow_every > 0
                and step > 0
                and step % args.grow_every == 0
            ):
                arch_mgr.evolve(
                    prune_heads=False,
                    grow=True,
                    block_factory=block_factory,
                    grow_device=device,
                    optimizer=optimizer,
                )
                n_layers_now = len(model.blocks)
                params_now = model.num_params() / 1e6
                print(f"  [grow @ step {step}] now {n_layers_now} layers, "
                      f"{params_now:.1f}M params")
                logger.log(
                    event="grow", step=step,
                    num_layers=n_layers_now, num_params_M=params_now,
                )

            # ---- Checkpoint ----
            if step > 0 and step % args.save_every == 0:
                save_checkpoint(latest_path, model, optimizer, step, best_val, args)

    except KeyboardInterrupt:
        print("\n[interrupt] saving and exiting...")
    finally:
        # Always save on exit so unexpected crashes don't lose work.
        save_checkpoint(latest_path, model, optimizer, step, best_val, args)
        logger.log(event="end", step=step, best_val=best_val)
        logger.close()
        print(f"[done] step {step}, best_val={best_val:.4f}, latest={latest_path}")


if __name__ == "__main__":
    main()
