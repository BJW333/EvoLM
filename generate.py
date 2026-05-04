"""
================================================================================
generate.py -- Inference / text generation from a trained EvoLM checkpoint.
================================================================================

WHAT THIS DOES:
    Load a checkpoint, take a text prompt, and generate a continuation.
    Supports the standard sampling controls:
        --max-new-tokens N       how much to generate
        --temperature T          0 = greedy (deterministic, picks argmax)
                                 1 = unmodified softmax
                                 >1 = more random (flatter distribution)
                                 <1 = more focused (sharper distribution)
        --top-k K                only sample from the top K logits (0 = disabled)
        --top-p P                nucleus sampling: smallest set whose cumulative
                                 probability >= P (0 = disabled)
        --seed N                 reproducible randomness

USAGE:
    python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"
    python generate.py --checkpoint checkpoints/latest.pt --prompt-file prompt.txt \\
                       --max-new-tokens 200 --temperature 0.8 --top-p 0.9

NOTES:
    - The checkpoint stores the args used to build the model, so we
      reconstruct the exact same architecture before loading weights.
      MoE expert growth and layer growth are handled via restore_topology.
    - Generation uses the KV cache: prefill processes the full prompt once
      to build per-layer caches, then decoding feeds one new token at a
      time and reuses cached keys/values. This is O(prefill + new_tokens)
      attention work, vs. O(S^2 * new_tokens) without the cache.
    - Plasticity is reset once at the start of generation. The Hebbian
      trace accumulates naturally across prefill and decode (each token
      enters the model exactly once, so no double-counting).
    - Generation stops when EOS is sampled or when the cache + prompt fill
      max_seq (no rolling-window cache logic -- truncating the cache would
      invalidate the RoPE rotations baked into the cached keys).
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from model import (
    EvoLM, small_config, medium_config, large_config,
    restore_topology,
)
from tokenizer import encode, decode, EOT_TOKEN_ID


# ============================================================================
# Architecture reconstruction from saved args
# ============================================================================
def _build_model_from_saved_args(saved_args: dict, device: str) -> EvoLM:
    """
    Recreate the model with the same configuration that was used at training
    time. The training args dict is saved inside each checkpoint (see
    train.save_checkpoint). This function mirrors train.make_model_from_args
    but reads from the saved args dict.
    """
    cfg_fn = {
        "small": small_config, "medium": medium_config, "large": large_config,
    }[saved_args["config"]]
    cfg = cfg_fn()

    moe_layers = (
        list(range(1, cfg["num_layers"], saved_args["moe_every_n"]))
        if saved_args["moe_every_n"] > 0 else []
    )
    plastic_layers = (
        list(range(cfg["num_layers"] - saved_args["plastic_last_n"], cfg["num_layers"]))
        if saved_args["plastic_last_n"] > 0 else []
    )

    model = EvoLM(
        vocab_size=saved_args["vocab_size"],
        max_seq=saved_args["max_seq"],
        pad_token_id=saved_args["pad_token_id"],
        moe_layers=moe_layers,
        plastic_layers=plastic_layers,
        moe_n_experts=saved_args["moe_n_experts"],
        moe_top_k=saved_args["moe_top_k"],
        plastic_eta=saved_args["plastic_eta"],
        use_dynamic=saved_args["use_dynamic"],
        dropout=saved_args["dropout"],
        attn_dropout=saved_args["dropout"],
        layer_drop=saved_args["layer_drop"],
        **cfg,
    ).to(device)
    return model


# ============================================================================
# Sampling
# ============================================================================
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """
    Given (B, vocab) raw logits for the next token, sample one token id per
    batch element. Returns a (B,) long tensor.

    Sampling pipeline (applied in order):
      1. Greedy if temperature == 0 (just argmax, return immediately).
      2. Otherwise scale logits by 1/temperature.
      3. Optional top-k filtering (set everything outside top-k to -inf).
      4. Optional top-p (nucleus) filtering.
      5. Softmax + multinomial sample.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-7)

    # Top-k: keep only the K highest-logit tokens.
    if top_k > 0:
        # Get the k-th largest logit per row, mask everything below it.
        kth = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
        logits = torch.where(
            logits < kth,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    # Top-p (nucleus): keep the smallest set of tokens whose cumulative prob
    # exceeds p. This is the most common "natural-sounding" sampler.
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Tokens to remove: those whose cumulative probability EXCEEDS p.
        # We keep the first one that crosses p (so the kept set has
        # probability mass >= p). Standard convention.
        mask_remove = cumulative > top_p
        # Always keep the top-1 token (shift mask by one to the right).
        mask_remove[..., 1:] = mask_remove[..., :-1].clone()
        mask_remove[..., 0] = False
        # Scatter the mask back to the un-sorted logits ordering.
        remove_unsorted = torch.zeros_like(mask_remove)
        remove_unsorted.scatter_(-1, sorted_idx, mask_remove)
        logits = torch.where(
            remove_unsorted, torch.full_like(logits, float("-inf")), logits,
        )

    # Multinomial sampling. Cast to float32 first -- under autocast(bf16),
    # logits arrive as bf16, and torch.multinomial is finicky about
    # low-precision floats across PyTorch versions. fp32 is always safe.
    probs = F.softmax(logits.float(), dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ============================================================================
# Main generation loop -- uses KV cache for O(prefill_len + new_tokens) attn
# ============================================================================
@torch.no_grad()
def generate(
    model: EvoLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: int = None,
):
    """
    prompt_ids: (B, S) long tensor on the model's device.
    Returns: (B, S + new) long tensor.

    Implementation:
      1. Truncate prompt to model.max_seq (we can't generate beyond that
         anyway; longer prompts get their oldest tokens dropped).
      2. PREFILL: one forward pass over the full prompt with kv_caches=[].
         This builds initial per-layer caches and gives us the next-token
         logits at the last position.
      3. DECODE LOOP: feed only the new token, with the running caches.
         Each iteration is O(seq_len_so_far) attention instead of O(S^2),
         a huge speedup vs the non-cached version.
      4. Stop on EOS or hitting max_seq (cache can't grow further).

    Plasticity: we reset the Hebbian trace ONCE at the start. The trace
    then accumulates correctly across prefill + decode without
    double-counting, since each token enters the model exactly once.
    """
    model.eval()
    device = prompt_ids.device

    # Truncate the prompt if it's longer than max_seq -- we can't usefully
    # carry context beyond what the model was trained for.
    if prompt_ids.size(1) > model.max_seq:
        prompt_ids = prompt_ids[:, -model.max_seq:]
    out = prompt_ids
    B = out.size(0)

    # Reset plasticity once at the start of generation. Hebbian state will
    # accumulate naturally across prefill and decode.
    model.reset_plasticity(batch_size=B, device=device)

    # ---- PREFILL ----
    # Pass kv_caches=[] to signal "build new caches". Returns (logits, caches).
    logits, kv_caches = model(out, kv_caches=[])
    next_logits = logits[:, -1, :]
    next_id = sample_next_token(
        next_logits, temperature=temperature, top_k=top_k, top_p=top_p,
    )
    out = torch.cat([out, next_id.unsqueeze(-1)], dim=1)

    if eos_token_id is not None and (next_id == eos_token_id).all():
        return out

    # ---- DECODE LOOP ----
    # We've already emitted 1 token; loop max_new_tokens-1 more times.
    for step in range(1, max_new_tokens):
        if out.size(1) >= model.max_seq:
            # Hit context limit. Cache can't grow further; stop here rather
            # than introduce rolling-window logic that would invalidate the
            # RoPE rotations baked into the cached keys.
            break

        # Pass only the newest token, with the running caches.
        logits, kv_caches = model(
            next_id.unsqueeze(-1),
            kv_caches=kv_caches,
        )
        next_logits = logits[:, -1, :]
        next_id = sample_next_token(
            next_logits, temperature=temperature, top_k=top_k, top_p=top_p,
        )
        out = torch.cat([out, next_id.unsqueeze(-1)], dim=1)

        if eos_token_id is not None and (next_id == eos_token_id).all():
            break

    return out


# ============================================================================
# Main CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str, help="text prompt (inline)")
    src.add_argument("--prompt-file", type=str, help="read prompt from a .txt file")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="generate this many continuations from the same prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-eos", action="store_true",
                        help="don't stop generation at the EOS / endoftext token")
    args = parser.parse_args()

    # ---- Setup ----
    torch.manual_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device

    # Pick autocast dtype.
    if args.dtype == "bf16" and (device == "cpu" or not torch.cuda.is_bf16_supported()):
        args.dtype = "fp32"
    if args.dtype == "bf16":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif args.dtype == "fp16":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # ---- Load checkpoint ----
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    saved_args = state.get("args", {})
    if not saved_args:
        sys.exit("Checkpoint does not contain a saved 'args' dict; cannot reconstruct architecture.")

    # ---- Reconstruct model ----
    model = _build_model_from_saved_args(saved_args, device)

    # If the checkpoint had grown layers OR spawned MoE experts, grow the
    # model to match before loading weights -- otherwise those weights get
    # silently dropped by strict=False. We pass optimizer=None because
    # inference doesn't have one; restore_topology handles that.
    saved_topology = state.get("topology", {})
    if saved_topology:
        restore_topology(model, saved_topology, device, optimizer=None)

    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        print(
            f"[load] WARNING: {len(unexpected)} unexpected keys -- "
            f"likely from grown MoE experts. Generation will work but the "
            f"checkpoint and model don't fully match. Sample: {unexpected[:3]}",
            file=sys.stderr,
        )
    if missing:
        # Missing keys with strict=False mean those params keep their init.
        # Most often this happens if the checkpoint was saved with a smaller
        # moe expert count than the current build.
        print(
            f"[load] note: {len(missing)} keys at init values (e.g. {missing[:3]})",
            file=sys.stderr,
        )

    print(f"[load] {model.num_params() / 1e6:.1f}M params from {ckpt_path}",
          file=sys.stderr)

    # ---- Read prompt and tokenize ----
    if args.prompt is not None:
        prompt_text = args.prompt
    else:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    prompt_ids = encode(prompt_text)
    if not prompt_ids:
        sys.exit("Empty prompt after tokenization.")
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    # Replicate to (num_samples, S) so all samples are generated in parallel.
    prompt_tensor = prompt_tensor.unsqueeze(0).expand(args.num_samples, -1).contiguous()

    eos = None if args.no_eos else EOT_TOKEN_ID

    # ---- Generate ----
    with autocast_ctx:
        out = generate(
            model,
            prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos,
        )

    # ---- Decode and print ----
    # Print to stdout so the result can be piped/redirected. Logs go to stderr.
    for i in range(args.num_samples):
        text = decode(out[i].cpu())
        print(f"=== sample {i + 1} / {args.num_samples} ===", file=sys.stderr)
        print(text)
        print()


if __name__ == "__main__":
    main()
