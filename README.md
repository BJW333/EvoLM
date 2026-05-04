# EvoLM

A transformer language model with self-organizing components: dynamic
mixture-of-experts, neural plasticity (Hebbian fast weights), architecture
evolution (head pruning + layer growth), and gradient-trainable structural
hyperparameters.

## Files

| File | What it does |
|------|--------------|
| `architecture.py` | Core transformer: GQA, RMSNorm, SwiGLU, LLaMA init, KV cache support |
| `moe.py` | Top-k MoE FFN with load balancing + dynamic expert add/prune |
| `plasticity.py` | Differentiable Hebbian plasticity (Miconi 2018) |
| `evolution.py` | Layer/head importance, head pruning, layer growth (Net2Net) |
| `dynamic.py` | Learnable layer drop, residual gates, Concrete Dropout |
| `auto_optimizer.py` | Heuristic LR / weight-decay / dropout tuning |
| `model.py` | `EvoLM` class (with KV cache + gradient checkpointing) + `restore_topology` |
| `tokenizer.py` | Shared GPT-2 tokenizer wrapper (tiktoken) |
| `prepare_data.py` | Tokenize raw text → `train.bin` / `val.bin` |
| `train.py` | Production training script |
| `generate.py` | KV-cached inference / sampling |
| `smoke_test.py` | End-to-end correctness check (run before any real training run) |

## Quick start

```bash
pip install torch numpy tiktoken rotary_embedding_torch

# 0. Always run the smoke test first -- catches integration issues in seconds.
python smoke_test.py

# 1. Tokenize your corpus
python prepare_data.py --input-file my_corpus.txt --output-dir data/

# 2. Train (small config = ~110M params; runs on a single 16-24GB GPU)
python train.py --data-dir data/ --config small --steps 10000 \
                --batch-size 4 --grad-accum 4

# 3. Resume later (architecture + topology restored automatically)
python train.py --data-dir data/ --config small --steps 50000 --resume

# 4. Generate text from a trained checkpoint
python generate.py --checkpoint checkpoints/best.pt \
                   --prompt "Once upon a time" \
                   --max-new-tokens 200 --temperature 0.8 --top-p 0.9
```

## Configuration knobs you'll actually touch

In `train.py` (all CLI args):

| Flag | What it does |
|------|---------|
| `--config {small,medium,large}` | ~110M / ~500M / ~1.5B params |
| `--max-seq` | Sequence length per training chunk |
| `--batch-size` × `--grad-accum` | Effective batch (~tokens per optimizer step) |
| `--lr`, `--min-lr`, `--warmup-steps` | LR schedule (warmup + cosine) |
| `--moe-every-n` | MoE FFN every Nth layer (0 = disabled) |
| `--plastic-last-n` | Last N layers use plastic FFNs (0 = disabled) |
| `--use-dynamic` | Learnable residual gates + layer drop |
| `--evolve-every` | MoE expert add/prune cadence (0 = disabled) |
| `--grow-every` | Layer growth cadence (0 = disabled, default) |
| `--use-auto-tuner` | Heuristic LR/WD/dropout tuner (layered on top of cosine) |
| `--auto` | Convenience preset: enables dynamic blocks + auto-tuner + evolution |
| `--grad-checkpoint` | Trade ~30% slower steps for ~50% less activation memory |

## What `--auto` does

`--auto` is shorthand for "turn on all the self-organizing machinery":

```
--use-dynamic            # learnable residual gates + layer drop
--use-auto-tuner         # heuristic LR/WD/dropout adjustment
--evolve-every 1000      # MoE expert spawn/prune + dead-head pruning
```

Layer growth (`--grow-every`) is NOT included — it's the most disruptive
mechanism and best enabled deliberately, late in training, after the
fixed-architecture loss has converged.

## Recommended baselines

**Conservative (highest probability of working well):**
```bash
python train.py --data-dir data/ --config small --steps 50000 \
    --batch-size 8 --grad-accum 4 --max-seq 1024 \
    --moe-every-n 2 --plastic-last-n 0 --evolve-every 2000 \
    --warmup-steps 500
```
Plain transformer + MoE + cosine LR. Everything else off.

**Aggressive (research-grade, more variance):**
```bash
python train.py --data-dir data/ --config small --steps 50000 \
    --batch-size 8 --grad-accum 4 --max-seq 1024 --auto
```

**Memory-constrained (medium config on a 16GB card):**
```bash
python train.py --data-dir data/ --config medium --steps 50000 \
    --batch-size 1 --grad-accum 16 --grad-checkpoint
```

## What's saved in checkpoints

Every checkpoint contains:
- `model` — full state_dict (including grown layers and spawned MoE experts)
- `optimizer` — AdamW moment buffers
- `step` — for resuming the LR schedule and step counter
- `best_val` — so resume tracks improvement correctly
- `args` — the CLI args used at training start (used by `generate.py` to
  rebuild the architecture before loading weights)
- `topology` — `{num_layers, moe_expert_counts}`. On resume, the model is
  grown to match this BEFORE loading weights. Without this, layers and
  experts added by evolution would be silently dropped.

## Inference performance

Generation now uses the KV cache:
- **Prefill**: one forward pass over the entire prompt to build per-layer caches.
- **Decode**: each new token reuses cached keys/values; only the new token
  goes through attention. ~10-50× faster than the naive approach for long
  prompts.
- **Limitation**: stops at `max_seq` total length (rolling-window cache
  would invalidate the RoPE rotations baked into the cached keys).

## Known practical gotchas

1. **Plasticity is sequential per token + memory-hungry.** The Hebbian trace
   keeps S copies of a (B, out, in) tensor in the autograd graph. For
   B=8, S=256, hidden_dim=3456, dim=1280 → ~36 GB per plastic layer. Use
   only on the last 1–2 layers. **Recommendation**: set `--plastic-last-n 0`
   for your first real training runs. The plasticity machinery is research-
   grade for language modeling and adds memory pressure for unproven gains.

2. **`use_dynamic=True` adds learnable knobs that can be unstable.** If
   training diverges with dynamic blocks, turn off `learnable_layer_drop`
   first while keeping residual gates — gates are the simpler, more robust
   knob.

3. **`--grow-every 0` (the default) disables layer growth.** Growing layers
   from random init is a real transient hit to the loss curve. Recommended:
   train at fixed depth first, only enable growth in late training if you
   need more capacity.

4. **bf16 needs Ampere or newer (A100 / H100 / RTX 30+).** Older GPUs auto-
   fall back to fp32. fp16 is supported via `--dtype fp16` but isn't
   recommended — bf16 is always preferred when available.

5. **Gradient checkpointing skips plastic blocks.** PlasticSwiGLU mutates
   its Hebbian trace as a side effect, which can't be safely re-executed
   during backward. So gradient checkpointing only saves memory on the
   non-plastic blocks. Plan accordingly.

## What's NOT in here yet

- Full evolutionary search over architecture (Sakana-style population
  training). Single-model surgery only.
- Distributed training (DDP/FSDP). Single-GPU only.
- Rolling-window KV cache for indefinite-length generation.
- Real downstream evaluation harness (perplexity is computed during
  training; downstream tasks would be a separate script).
