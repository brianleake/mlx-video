"""Validate the StreamFrame Phase 2 causal self-attention (causal.py).

Block-by-block generation through the KV-cache must be numerically identical to a
single full-sequence attention pass with a block-causal mask (same weights). Also
checks that the sink + sliding-window cache stays bounded.

    python tests/test_causal_attention.py
"""
import mlx.core as mx
import numpy as np

from mlx_video.models.wan_2.causal import CausalKVCache, WanCausalSelfAttention
from mlx_video.models.wan_2.rope import rope_apply, rope_params


def test_causal_matches_block_causal_reference():
    mx.random.seed(0)
    dim, heads = 64, 4
    F, H, W, fb = 9, 2, 2, 3
    fsl = H * W
    S = F * fsl

    attn = WanCausalSelfAttention(dim, heads, sink_size=0, local_attn_size=-1)
    mx.eval(attn.parameters())
    freqs = rope_params(1024, dim // heads)
    x = mx.random.normal((1, S, dim)).astype(mx.float32)

    cache = CausalKVCache(0, -1, fsl)
    outs = []
    for b in range(F // fb):
        sf = b * fb
        outs.append(attn(x[:, sf * fsl:(sf + fb) * fsl], [(fb, H, W)], freqs, cache, sf, commit=True))
    block_out = mx.concatenate(outs, axis=1)

    n, d = heads, dim // heads
    q = attn.norm_q(attn.q(x)).reshape(1, S, n, d)
    k = attn.norm_k(attn.k(x)).reshape(1, S, n, d)
    v = attn.v(x).reshape(1, S, n, d)
    q = rope_apply(q, [(F, H, W)], freqs).astype(x.dtype)
    k = rope_apply(k, [(F, H, W)], freqs).astype(x.dtype)
    blk = (np.arange(S) // fsl) // fb
    mask = mx.array(np.where(blk[None, :] <= blk[:, None], 0.0, -1e30).astype(np.float32))[None, None]
    ref = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
        scale=attn.scale, mask=mask)
    ref = attn.o(ref.transpose(0, 2, 1, 3).reshape(1, S, dim))

    mx.eval(block_out, ref)
    rel = float(mx.max(mx.abs(block_out - ref))) / float(mx.max(mx.abs(ref)))
    assert rel < 1e-4, f"causal KV-cache diverges from block-causal reference: rel={rel:.3e}"


def test_bounded_cache_evicts():
    fsl = 4
    cache = CausalKVCache(sink_size=1, local_attn_size=2, frame_seqlen=fsl)
    for b in range(3):
        blk = mx.zeros((1, 3 * fsl, 4, 16))
        cache.commit(blk, blk)
    assert cache.k.shape[1] <= (1 + 2) * fsl


if __name__ == "__main__":
    test_causal_matches_block_causal_reference()
    test_bounded_cache_evicts()
    print("causal attention: all checks PASS")
