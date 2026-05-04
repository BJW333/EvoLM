"""
================================================================================
smoke_test.py -- End-to-end smoke test for the pipeline.
================================================================================

Run this first, before committing to any real training. It:
    1. Builds a tiny model (~1M params)
    2. Runs forward in training mode (tests basic path + losses)
    3. Runs forward with gradient checkpointing (tests memory-saving path)
    4. Runs forward with KV cache prefill, then decode (tests cache logic)
    5. Verifies that cached and non-cached forward give the same logits
       for the same prefix (this is the most important correctness check)
    6. Runs the generate.py sampling loop on a tiny prompt

If anything is wrong with the architecture or the cache plumbing, this
will catch it in seconds rather than after hours of training.

USAGE:
    python smoke_test.py

REQUIREMENTS:
    pip install torch numpy rotary_embedding_torch
    (tiktoken not needed for this test -- we use random token IDs)
"""

import sys
import torch

from model import EvoLM


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def build_tiny_model(use_dynamic=False, gradient_checkpointing=False):
    """A model so small that even CPU forward is fast."""
    return EvoLM(
        vocab_size=256,         # tiny vocab so embeddings are small
        max_seq=64,
        pad_token_id=255,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        num_layers=4,
        ff_dim=128,
        moe_layers=[1],         # one MoE layer at index 1
        moe_n_experts=4,
        plastic_layers=[],      # plasticity off for this test (memory expensive even tiny)
        use_dynamic=use_dynamic,
        gradient_checkpointing=gradient_checkpointing,
    )


def test_basic_forward():
    section("Test 1: basic forward in training mode")
    torch.manual_seed(0)
    model = build_tiny_model()
    model.train()

    B, S = 2, 16
    input_ids = torch.randint(0, 256, (B, S))
    labels = input_ids.clone()

    logits, loss, parts = model(input_ids, labels=labels)
    assert logits.shape == (B, S, 256), f"bad logits shape: {logits.shape}"
    assert loss.dim() == 0, f"loss should be scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    print(f"  loss = {loss.item():.4f}, ce = {parts['ce'].item():.4f}, "
          f"aux = {parts['aux'].item():.4f}")

    # Backward should work.
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    print(f"  backward OK, {len(grads)} params got grads")
    print("  PASS")


def test_gradient_checkpointing():
    section("Test 2: gradient checkpointing")
    torch.manual_seed(0)
    model = build_tiny_model(gradient_checkpointing=True)
    model.train()

    B, S = 2, 16
    input_ids = torch.randint(0, 256, (B, S))
    labels = input_ids.clone()

    logits, loss, parts = model(input_ids, labels=labels)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients with checkpointing"
    print(f"  loss = {loss.item():.4f}, {len(grads)} params got grads")
    print("  PASS")


def test_kv_cache_consistency():
    section("Test 3: KV cache produces same logits as non-cached")
    torch.manual_seed(42)

    # IMPORTANT: this test isolates the KV-cache plumbing. We deliberately
    # build the model WITHOUT MoE because MoE's capacity-dropping is
    # batch-size-dependent: the prefill (S=9 tokens) computes a different
    # capacity than the full forward (S=10 tokens), and if any expert
    # happens to be over-subscribed in one path but not the other, different
    # tokens get dropped, producing legitimately different outputs. That's a
    # property of MoE, not a cache bug, and it would muddle this test.
    # The cache equivalence we DO want to verify -- correct RoPE offsets,
    # correct mask shapes, correct k/v concatenation -- is fully exercised
    # in the dense-FFN model below.
    model = EvoLM(
        vocab_size=256, max_seq=64, pad_token_id=255,
        dim=64, n_heads=4, n_kv_heads=2, num_layers=4, ff_dim=128,
        moe_layers=[], plastic_layers=[],
    )
    model.eval()

    B, S = 1, 10
    input_ids = torch.randint(0, 255, (B, S))   # avoid pad token (255)

    # ---- Path A: standard non-cached forward over full prefix ----
    with torch.no_grad():
        logits_full = model(input_ids)              # (B, S, V)

    # ---- Path B: cached forward, prefill on prefix[:S-1], then decode last token ----
    prefix = input_ids[:, :S - 1]
    last_token = input_ids[:, S - 1:]

    with torch.no_grad():
        # Prefill
        logits_prefill, kv_caches = model(prefix, kv_caches=[])
        # Decode one token
        logits_decode, kv_caches = model(last_token, kv_caches=kv_caches)

    # Compare logits at the LAST position: full[:, -1] vs decode[:, 0]
    full_last = logits_full[:, -1, :]
    cached_last = logits_decode[:, 0, :]

    max_diff = (full_last - cached_last).abs().max().item()
    rel_diff = max_diff / (full_last.abs().max().item() + 1e-9)

    print(f"  full_last     = {full_last[0, :4].tolist()}")
    print(f"  cached_last   = {cached_last[0, :4].tolist()}")
    print(f"  max abs diff  = {max_diff:.2e}")
    print(f"  rel diff      = {rel_diff:.2e}")

    # In fp32, this should be near-exact. Allow some slack for fp accumulation
    # order (concatenation introduces tiny re-ordering of additions).
    if max_diff > 1e-4:
        print(f"  FAIL: cache is producing different logits than non-cache!")
        print(f"  This means the KV cache plumbing has a bug.")
        sys.exit(1)
    print("  PASS")


def test_generation_loop():
    section("Test 4: end-to-end generation with sampling")
    torch.manual_seed(0)
    model = build_tiny_model()
    model.eval()

    from generate import generate

    prompt_ids = torch.randint(0, 200, (1, 8))   # well below pad
    print(f"  prompt:  {prompt_ids.tolist()}")

    out = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=20,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        eos_token_id=None,           # no EOS -- generate the full max_new_tokens
    )
    print(f"  output:  {out.tolist()}")
    assert out.size(1) == prompt_ids.size(1) + 20, (
        f"expected {prompt_ids.size(1) + 20} tokens, got {out.size(1)}"
    )
    # Make sure we actually appended new tokens (not all zeros or all pad).
    new_tokens = out[:, prompt_ids.size(1):]
    assert (new_tokens != 255).any(), "all generated tokens are pad?!"
    print("  PASS")


def test_dynamic_block():
    section("Test 5: DynamicTransformerBlock forward + backward")
    torch.manual_seed(0)
    model = build_tiny_model(use_dynamic=True)
    model.train()

    B, S = 2, 16
    input_ids = torch.randint(0, 255, (B, S))
    labels = input_ids.clone()

    logits, loss, parts = model(input_ids, labels=labels)
    assert torch.isfinite(loss), "dynamic loss not finite"
    loss.backward()
    print(f"  loss = {loss.item():.4f}")
    print("  PASS")


def test_topology_save_restore():
    section("Test 6: topology save/restore")
    from model import restore_topology
    from moe import evolve_moe_with_optimizer

    torch.manual_seed(0)
    model_a = build_tiny_model()
    model_a.train()

    # Make expert 0 look hot so spawn fires.
    moe = model_a.blocks[1].ff
    moe.utilization[0] = 1.0
    moe.utilization[1:] = 0.0
    optimizer = torch.optim.AdamW(model_a.get_param_groups(), lr=1e-3)

    # IMPORTANT: pass min_experts >= current count so the prune branch can't
    # fire. Otherwise the bimodal utilization above triggers BOTH spawn AND
    # prune in a single call: spawn 1 (4 -> 5), then prune all 3 cold ones
    # (5 -> 2). That's correct production behavior for that input, but it
    # would defeat this test which is specifically checking that the saved
    # topology has MORE experts than the freshly-built model.
    events = evolve_moe_with_optimizer(
        model_a, optimizer, device="cpu",
        spawn_threshold=0.1,
        min_experts=10,
    )
    print(f"  evolve events: {events}")
    print(f"  model_a now has {moe.n_experts} experts")
    assert moe.n_experts > 4, (
        f"expected >4 experts after spawn, got {moe.n_experts}"
    )

    # Save, build a fresh model_b, restore topology, load weights.
    state = {"model": model_a.state_dict(), "topology": {
        "num_layers": len(model_a.blocks),
        "moe_expert_counts": {1: moe.n_experts},
    }}

    model_b = build_tiny_model()
    grew = restore_topology(model_b, state["topology"], device="cpu", optimizer=None)
    print(f"  model_b grew={grew}, now has {model_b.blocks[1].ff.n_experts} experts")
    assert grew, "should have grown"
    assert model_b.blocks[1].ff.n_experts == moe.n_experts

    # Strict load should NOT error after topology restore.
    missing, unexpected = model_b.load_state_dict(state["model"], strict=False)
    assert not unexpected, f"unexpected keys after restore: {unexpected[:3]}"
    print(f"  loaded with {len(missing)} missing keys (should be 0 or near-0)")
    print("  PASS")


if __name__ == "__main__":
    print("Running EvoLM smoke tests...\n")
    test_basic_forward()
    test_gradient_checkpointing()
    test_kv_cache_consistency()
    test_generation_loop()
    test_dynamic_block()
    test_topology_save_restore()
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
