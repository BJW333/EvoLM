"""
================================================================================
architecture.py -- Core transformer building blocks.
================================================================================

WHAT'S IN HERE:
    RMSNorm                     -- Root-mean-square LayerNorm (LLaMA-style)
    GroupedQueryAttention       -- Multi-head attn with grouped KV heads
    SwiGLU                      -- Gated FFN (also LLaMA-style)
    TransformerBlock            -- Pre-norm block holding attn + ffn
    llama_init                  -- Standard LLaMA-style weight init
    scale_residual_projections  -- Depth-aware rescaling of out_proj / w3

DESIGN PRINCIPLES:
    1. Pre-norm everywhere. Apply norm BEFORE attn/ffn, not after. This is the
       modern recipe -- post-norm is unstable at depth.
    2. RMSNorm not LayerNorm. Drops the mean-subtraction; cheaper and works just
       as well in practice.
    3. RoPE for positions. No learned positional embeddings. Rotary embeddings
       generalize better to longer contexts and avoid an extra parameter table.
    4. SwiGLU not ReLU. Gated linear unit with SiLU activation. Better than ReLU
       in nearly every benchmark, costs ~33% more FFN params (which is why
       hidden_dim is usually ~2.7x model_dim instead of 4x).
    5. No biases on linear layers in the FFN/attn. Empirically slightly better
       and saves tiny amount of params.
    6. GQA over MHA. Multiple query heads share a single KV head. Reduces KV
       cache size at inference time without hurting quality much.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding as RotaryEmbeddingLib


# ============================================================================
# RMSNorm
# ============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm.

    Standard LayerNorm:  y = (x - mean) / sqrt(var + eps) * gamma + beta
    RMSNorm:             y = x / sqrt(mean(x^2) + eps) * gamma

    Why drop the mean subtraction? Empirically the centering is doing very
    little, and removing it saves a small bit of compute and one set of stats.
    Also: no learnable bias (gamma only). LLaMA, PaLM, T5 all use this.

    eps choice: 1e-6 is the standard for 16/32-bit training. The previous
    1e-4 was way too high -- it changes the normalization meaningfully when
    activations are small.
    """
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable gain (gamma). Initialized to 1 so the layer starts as identity.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute root mean square along the last dim, then take its reciprocal.
        # rsqrt is fused into a single op on most hardware -- faster than 1/sqrt.
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================================================
# Grouped-Query Attention
# ============================================================================
class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention where multiple query heads SHARE the same key/value
    heads. This is the main attention variant in LLaMA-2/3, Mistral, Qwen, etc.

    Setup:
        n_heads     -- number of query heads (full attention parallelism)
        n_kv_heads  -- number of key/value heads (must divide n_heads)
        n_rep       -- n_heads // n_kv_heads, how many Qs share each KV

    Why GQA?
        At inference, the bottleneck for long contexts is the KV cache, not
        compute. KV cache size scales with n_kv_heads. By going from
        n_kv_heads = n_heads (vanilla MHA) to n_kv_heads = n_heads/4 or smaller,
        you cut KV cache by 4x or more, with very little quality loss.

    Special cases:
        n_kv_heads == n_heads  -> regular Multi-Head Attention
        n_kv_heads == 1        -> Multi-Query Attention (max sharing)
        anything in between    -> Grouped-Query Attention

    During training, GQA is mathematically equivalent to expanding KV heads
    via repeat_interleave so they match Q head count. We do exactly that here.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0, "dim must divide n_heads evenly"
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads      # KV repeat factor for GQA
        self.d_head = dim // n_heads
        self.dropout_p = dropout

        # Three separate projections rather than one fused QKV. Fused is fine
        # for plain MHA but gets fiddly with GQA because Q has more heads than
        # K/V. Splitting is cleaner and cost is identical (matrix mults fuse
        # at the kernel level on modern GPUs anyway).
        self.q_proj = nn.Linear(dim, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Rotary positional embedding. Applied to Q and K (NOT V).
        # The library handles the heavy lifting; we just give it head-dim.
        self.rotary = RotaryEmbeddingLib(self.d_head)

    def forward(self, x, key_padding_mask=None, kv_cache=None):
        """
        x:                shape (B, S, D) -- batch, sequence, dim
        key_padding_mask: shape (B, S) bool, True at positions to MASK OUT
                          (this matches the convention from the previous code:
                           True = pad token = ignore)
        kv_cache:         (past_k, past_v) tuple if using cache, else None

        Returns: (output, new_cache) if kv_cache was given, else just output.
        """
        B, S, D = x.size()

        # Project to Q, K, V. Reshape to (B, n_heads, S, d_head) for sdp.
        # Note Q has n_heads but K/V have n_kv_heads. Will broadcast later.
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        # All shapes now: (B, heads, S, d_head)

        # Compute the position offset for RoPE. When using a cache, the NEW
        # tokens are at sequence positions [past_len, past_len + S), not
        # [0, S). Without this offset, the new q/k get rotated as if they
        # were at the start of the sequence, producing incorrect relative
        # positions in the attention dot product. The past_k in the cache
        # was already rotated with its own (smaller) offset when it was
        # produced, so we don't re-rotate it.
        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0].size(2)

        # Rotary embeddings on Q and K (NOT V). RoPE rotates pairs of dims
        # by an angle determined by position, so q.k_T encodes RELATIVE
        # position. V is left alone -- only the attention pattern needs
        # position info, the values themselves don't.
        q = self.rotary.rotate_queries_or_keys(q, offset=past_len)
        k = self.rotary.rotate_queries_or_keys(k, offset=past_len)

        # KV cache handling for inference. The past tensors already carry
        # their RoPE rotation from when they were produced, so we just
        # concatenate the (already-rotated) new k/v onto them.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # GQA expansion: each KV head is shared by n_rep query heads. We
        # explicitly repeat them so the attention math is plain MHA after this.
        # repeat_interleave duplicates along dim=1 (the head axis).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        S_kv = k.size(2)  # may be larger than S if cache is non-empty

        # Build the attention mask:
        # - During training (no cache): standard causal mask, plus pad mask if any.
        # - During incremental/mixed decoding (cache present, S < S_kv): build an
        #   OFFSET causal mask so each new query sees all earlier keys.
        if kv_cache is not None and S != S_kv:
            # Mixed prefill / decode case: S new queries against S_kv keys
            # where S_kv > S (the cache holds older tokens too).
            #
            # Build an OFFSET causal mask. Query i within the new S corresponds
            # to global position (S_kv - S + i), and may attend to keys at
            # positions 0 .. (S_kv - S + i).
            #
            # When S == 1 this collapses to "see everything", which is exactly
            # what we want for single-token autoregressive decoding.
            # When S == S_kv (cache empty, first call) the else-branch handles
            # it as a square causal mask.
            #
            # Padding inside the cache is NOT masked here -- assumed the cache
            # holds only real tokens. If you generate from padded prefixes,
            # track valid_lengths separately.
            q_offset = S_kv - S
            rows = torch.arange(S, device=x.device).unsqueeze(1)        # (S, 1)
            cols = torch.arange(S_kv, device=x.device).unsqueeze(0)      # (1, S_kv)
            causal_mask = cols <= (rows + q_offset)                      # (S, S_kv)
            attn_mask = ~causal_mask
            is_causal = False
        else:
            # Causal mask: True at (i, j) means "query i can see key j".
            # Lower triangular (including diagonal) = each pos sees itself + prior.
            causal_mask = torch.tril(
                torch.ones(S, S_kv, dtype=torch.bool, device=x.device)
            )
            if key_padding_mask is not None:
                # Expand the per-batch padding mask to broadcast across heads
                # and queries. Shape: (B, n_heads, S, S_kv).
                pad_mask = key_padding_mask[:, None, None, :].expand(
                    -1, self.n_heads, S, -1
                )
                # PyTorch's sdp expects True = MASKED OUT (i.e. zero attention).
                # We have True = keep for causal, True = pad for pad_mask.
                # So masked-out = (NOT causal) OR pad.
                attn_mask = ~causal_mask.unsqueeze(0).unsqueeze(0) | pad_mask
            else:
                attn_mask = ~causal_mask  # (S, S_kv) -- broadcasts over B, heads
            is_causal = False  # we built the mask manually

        # Scaled dot-product attention. PyTorch picks Flash/efficient kernels
        # automatically when the inputs allow. Dropout is only active in train.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        # Back to (B, S, D) and project.
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        out = self.out_proj(attn_out)

        if kv_cache is not None:
            return out, new_cache
        return out


# ============================================================================
# SwiGLU Feed-Forward
# ============================================================================
class SwiGLU(nn.Module):
    """
    SwiGLU FFN: gated linear unit with SiLU (a.k.a. swish) activation.

        gate = SiLU(x @ W1)            -- the "gate"
        up   = x @ W2                  -- the "value"
        h    = gate * up               -- elementwise
        y    = h @ W3                  -- project back to dim

    Compared to a vanilla 2-layer FFN with ReLU:
        y = ReLU(x @ W_up) @ W_down
    SwiGLU adds a multiplicative gate, which gives finer control over what
    information passes through. Cost: 1.5x parameters in the FFN block (three
    matrices instead of two), so we shrink hidden_dim from 4*dim to ~2.7*dim
    to stay parameter-equivalent.

    No biases. Empirically marginally better, saves a tiny amount of params,
    keeps things consistent with the rest of the modern stack.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # up projection
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None):
        # `valid_mask` is unused here -- only PlasticSwiGLU uses it. Accepted
        # so all FFN types share the same forward signature, which lets
        # TransformerBlock pass it uniformly without isinstance checks.
        del valid_mask
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ============================================================================
# Transformer Block
# ============================================================================
class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:

        x = x + attn(norm(x))     -- residual around attention
        x = x + ffn(norm(x))      -- residual around feed-forward

    The FFN module is INJECTED, not constructed inside. This lets the parent
    model swap in MoE FFNs, plastic FFNs, or anything else with a compatible
    interface (takes a tensor of shape (B, S, D), returns one of the same
    shape OR a tuple of (tensor, aux_loss)).

    LayerDrop: stochastically skip the entire block during training. Acts
    like very heavy dropout at the layer level. Useful for very deep models;
    typically keep it at 0 for normal-sized models.

    RETURN CONTRACT (always 3-tuple):
        (x_out, aux_loss_or_None, new_kv_cache_or_None)
    Always 3 values, even when the layer is dropped or no MoE is used. Fixed
    from previous version where dropped layers returned a 2-tuple and crashed
    the model's forward loop.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_module: nn.Module,
        attn_dropout: float,
        layer_drop: float,
    ):
        super().__init__()
        self.layer_drop = layer_drop
        self.attn_norm = RMSNorm(dim)
        self.ff_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, attn_dropout)
        self.ff = ffn_module

    def forward(self, x, key_padding_mask=None, kv_cache=None):
        # LayerDrop: roll a die, maybe skip the whole block. Only at training.
        # We use a CPU tensor so .item() doesn't force a GPU sync. This keeps
        # the decision deterministic under torch.manual_seed but cheap.
        if (
            self.training
            and self.layer_drop > 0
            and torch.rand(1).item() < self.layer_drop
        ):
            # Skip both attn and ffn. Pass through unchanged.
            return x, None, kv_cache  # consistent 3-tuple shape

        # ---- Attention sub-block ----
        a = self.attn_norm(x)
        if kv_cache is not None:
            attn_out, new_cache = self.attn(a, key_padding_mask, kv_cache=kv_cache)
        else:
            attn_out = self.attn(a, key_padding_mask)
            new_cache = None
        x = x + attn_out

        # ---- Feed-forward sub-block ----
        f = self.ff_norm(x)
        # valid_mask: True where the token is real (not pad). Used by
        # PlasticSwiGLU to suppress Hebbian updates on padding; ignored by
        # SwiGLU / MoEFeedForward.
        valid_mask = (~key_padding_mask) if key_padding_mask is not None else None
        ff_out = self.ff(f, valid_mask=valid_mask)

        # Plain FFNs return a tensor; MoE returns (tensor, aux_loss).
        # Handle both cases.
        if isinstance(ff_out, tuple):
            ff_tensor, aux = ff_out
            x = x + ff_tensor
        else:
            ff_tensor = ff_out
            aux = None
            x = x + ff_tensor

        return x, aux, new_cache


# ============================================================================
# Initialization helpers
# ============================================================================
def llama_init(module: nn.Module, num_layers: int):
    """
    LLaMA-style weight initialization, applied via model.apply(...).

    Linear and Embedding weights -> Normal(0, 0.02).

    Why this rather than Xavier/Kaiming?
    For pre-norm transformers, the activations entering each layer are
    already roughly unit-variance because RMSNorm normalizes them. So we
    don't need the variance-preserving inits that Xavier/Kaiming compute
    based on fan_in/fan_out. A simple Normal(0, 0.02) does the job and
    matches every modern LLM I'm aware of.

    The 0.02 number comes from GPT-2 and has stuck around because it works.

    NOTE: The padding-token row of nn.Embedding gets clobbered by this init.
    The caller (e.g. model.py) should re-zero the padding row AFTER calling
    this -- nn.Embedding.padding_idx zeros it on construction but doesn't
    re-zero it after re-init.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def scale_residual_projections(model: nn.Module, num_layers: int):
    """
    Rescale the output projection of attention (out_proj) and the down
    projection of SwiGLU (w3) by 1 / sqrt(2 * num_layers).

    Why? In a residual network with N blocks, each block adds something to
    the residual stream. If we don't control the scale of each addition,
    activations grow as O(sqrt(N)) and the model becomes hard to train at
    depth. The 1/sqrt(2N) scaling keeps the variance of the residual stream
    approximately constant regardless of depth. The factor of 2 accounts for
    each block adding TWO residual contributions (one from attn, one from ffn).

    This trick has many names in the literature: "DeepNorm" (with a different
    formula), "GPT-2 init", "fixup-style scaling". The 1/sqrt(2N) version is
    what GPT-2 / LLaMA / Mistral all use.

    IMPORTANT: We mutate weights in-place. Call AFTER llama_init.
    Note: if you grow new layers post-init via evolution.py, those layers
    use this same scale (computed from the ORIGINAL num_layers, since the
    other layers were scaled with that). Re-running this on a grown model
    would shrink already-trained layers -- don't do that.
    """
    scale = 1.0 / math.sqrt(2 * num_layers)
    for name, p in model.named_parameters():
        # Match attention out_proj and SwiGLU's w3 (the "down" projection
        # in MoE experts also has w3 in its name, so they get scaled too).
        if any(s in name for s in ["out_proj.weight", "w3.weight"]):
            with torch.no_grad():
                p.mul_(scale)
