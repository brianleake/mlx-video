"""Causal (autoregressive, KV-cached) Wan2.1 self-attention for StreamFrame Phase 2.

Causal-Forcing (thu-ml) reuses the *exact* Wan2.1-1.3B weights; causality lives
entirely in the forward pass:

  * RoPE with a temporal frame-offset (`causal_rope_apply`) so each generated
    block is positioned at its absolute frame index in the stream.
  * A bounded per-layer KV-cache = `sink_size` anchor frames (always kept) plus a
    sliding `local_attn_size`-frame window. This is FramePack-style bounded
    memory, trained into the model -> O(1) context, drift-free long video.
  * Block-causal attention: a block attends to its own tokens + everything in the
    cache (all valid past), so no token-level mask is needed.

This mirrors `WanSelfAttention` in attention.py (same q/k/v/o + QK-norm) but swaps
the bidirectional full-sequence attention for the cached causal path.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import WanRMSNorm, _linear_dtype
from .wan_2 import WanModel


def causal_rope_apply(
    x: mx.array, grid_sizes: list, freqs: mx.array, start_frame: int = 0
) -> mx.array:
    """3-way factorized RoPE with the temporal axis offset by `start_frame`.

    Identical to `rope_apply` (non-precomputed path) except the temporal
    frequencies are indexed `[start_frame : start_frame + f]` so a block emitted
    partway through the stream gets its absolute temporal positions.

    Args:
        x: [B, L, num_heads, head_dim]
        grid_sizes: list of (F, H, W) per batch element (F = frames in this block)
        freqs: [max_len, head_dim // 2, 2]
        start_frame: absolute frame index of this block's first frame
    """
    b, s, n, d = x.shape
    half_d = d // 2
    if freqs.dtype != x.dtype:
        freqs = freqs.astype(x.dtype)

    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3
    freqs_t = freqs[:, :d_t]
    freqs_h = freqs[:, d_t : d_t + d_h]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]

    outputs = []
    for i in range(b):
        f, h, w = grid_sizes[i]
        seq_len = f * h * w
        x_i = x[i, :seq_len].reshape(seq_len, n, half_d, 2)

        ft = mx.broadcast_to(
            freqs_t[start_frame : start_frame + f].reshape(f, 1, 1, d_t, 2),
            (f, h, w, d_t, 2),
        )
        fh = mx.broadcast_to(freqs_h[:h].reshape(1, h, 1, d_h, 2), (f, h, w, d_h, 2))
        fw = mx.broadcast_to(freqs_w[:w].reshape(1, 1, w, d_w, 2), (f, h, w, d_w, 2))
        freqs_i = mx.concatenate([ft, fh, fw], axis=3).reshape(seq_len, 1, half_d, 2)

        cos_f = freqs_i[..., 0]
        sin_f = freqs_i[..., 1]
        x_real = x_i[..., 0]
        x_imag = x_i[..., 1]
        out_real = x_real * cos_f - x_imag * sin_f
        out_imag = x_real * sin_f + x_imag * cos_f
        x_rot = mx.stack([out_real, out_imag], axis=-1).reshape(seq_len, n, d)

        if seq_len < s:
            x_rot = mx.concatenate([x_rot, x[i, seq_len:]], axis=0)
        outputs.append(x_rot)

    return mx.stack(outputs)


class CausalKVCache:
    """Per-layer bounded KV-cache: `sink` anchor tokens + sliding-window recent.

    Token counts are in *latent tokens* (frames * H_latent * W_latent). With
    `local_attn_size == -1` the window is unbounded (cache grows with the clip).
    """

    def __init__(self, sink_size: int, local_attn_size: int, frame_seqlen: int):
        self.sink_tokens = sink_size * frame_seqlen
        self.window_tokens = -1 if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.k: mx.array | None = None
        self.v: mx.array | None = None

    def reset(self) -> None:
        self.k = None
        self.v = None

    def view_with(self, k: mx.array, v: mx.array,
                  window_k: mx.array | None = None,
                  window_v: mx.array | None = None) -> tuple[mx.array, mx.array]:
        """Return persistent cache ++ [window-so-far] ++ this block's K/V WITHOUT
        persisting.

        Used during the multi-step denoise: the block attends to clean past
        (persistent) + its own (noisy, this step) K/V, but the noisy intermediates
        must not pollute the cache. Only the clean re-run commits (see `commit`).

        `window_k/v` carry the K/V of EARLIER sub-blocks of a diagonal denoising
        window (Rolling Forcing), which sit at different noise levels and must be
        visible to later sub-blocks without persisting either. See `window_kvs`
        in CausalWanModel.generate_block.
        """
        ks = [a for a in (self.k, window_k, k) if a is not None]
        vs = [a for a in (self.v, window_v, v) if a is not None]
        if len(ks) == 1:
            return ks[0], vs[0]
        return mx.concatenate(ks, axis=1), mx.concatenate(vs, axis=1)

    def commit(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """Persist a block's clean K/V ([B, s, n, d]); return the full windowed cache."""
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = mx.concatenate([self.k, k], axis=1)
            self.v = mx.concatenate([self.v, v], axis=1)

        if self.window_tokens != -1:
            limit = self.sink_tokens + self.window_tokens
            if self.k.shape[1] > limit:
                # Keep the sink anchor + the most-recent window; evict the middle.
                sink_k, sink_v = self.k[:, : self.sink_tokens], self.v[:, : self.sink_tokens]
                rec_k, rec_v = self.k[:, -self.window_tokens :], self.v[:, -self.window_tokens :]
                self.k = mx.concatenate([sink_k, rec_k], axis=1)
                self.v = mx.concatenate([sink_v, rec_v], axis=1)
        return self.k, self.v


class WanCausalSelfAttention(nn.Module):
    """Causal self-attention: same weights as WanSelfAttention, cached forward."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sink_size: int = 0,
        local_attn_size: int = -1,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sink_size = sink_size
        self.local_attn_size = local_attn_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else None

    def __call__(
        self,
        x: mx.array,
        grid_sizes: list,
        freqs: mx.array,
        cache: CausalKVCache,
        start_frame: int,
        commit: bool = False,
        window_kv: list | None = None,
    ) -> mx.array:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim

        w_dtype = _linear_dtype(self.q)
        x_w = x.astype(w_dtype)

        q = self.q(x_w)
        k = self.k(x_w)
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
        q = q.reshape(b, s, n, d)
        k = k.reshape(b, s, n, d)
        v = self.v(x_w).reshape(b, s, n, d)

        q = causal_rope_apply(q.astype(mx.float32), grid_sizes, freqs, start_frame)
        k = causal_rope_apply(k.astype(mx.float32), grid_sizes, freqs, start_frame)
        q = q.astype(w_dtype)
        k = k.astype(w_dtype)

        if commit:
            full_k, full_v = cache.commit(k, v)
        elif window_kv is None:
            full_k, full_v = cache.view_with(k, v)
        else:
            # Diagonal window: see the cache + earlier sub-blocks of this window
            # + self, then publish our K/V for the sub-blocks that follow. Nothing
            # persists — the window is noisy at every level.
            wk, wv = window_kv
            full_k, full_v = cache.view_with(k, v, wk, wv)
            window_kv[0] = k if wk is None else mx.concatenate([wk, k], axis=1)
            window_kv[1] = v if wv is None else mx.concatenate([wv, v], axis=1)

        out = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            full_k.transpose(0, 2, 1, 3),
            full_v.transpose(0, 2, 1, 3),
            scale=self.scale,
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return self.o(out)


class CausalWanModel(WanModel):
    """Wan2.1 backbone in causal/autoregressive mode (Causal-Forcing weights).

    Same parameters as WanModel (so the converted Causal-Forcing weights load
    unchanged) but each block's bidirectional self-attention is replaced with the
    cached WanCausalSelfAttention, and generation runs one block (`num_frame_per_block`
    latent frames) at a time through `generate_block`, carrying per-layer KV-caches.
    """

    def __init__(
        self,
        config,
        sink_size: int = 1,
        local_attn_size: int = -1,
        num_frame_per_block: int = 3,
    ):
        super().__init__(config)
        for blk in self.blocks:
            blk.self_attn = WanCausalSelfAttention(
                config.dim,
                config.num_heads,
                sink_size=sink_size,
                local_attn_size=local_attn_size,
                qk_norm=config.qk_norm,
                eps=config.eps,
            )
        self.sink_size = sink_size
        self.local_attn_size = local_attn_size
        self.num_frame_per_block = num_frame_per_block
        # Evaluate the residual stream every N layers so a single forward isn't one
        # giant Metal command buffer (trips the macOS GPU watchdog at native res).
        # 0 = off (small resolutions don't need it).
        self.eval_every = 0

    def make_self_caches(self, frame_seqlen: int) -> list:
        """One bounded KV-cache per transformer layer."""
        return [
            CausalKVCache(self.sink_size, self.local_attn_size, frame_seqlen)
            for _ in self.blocks
        ]

    def make_window_kvs(self) -> list:
        """Per-layer scratch K/V for ONE diagonal denoising window (Rolling
        Forcing). Make a fresh one per window; feed it to consecutive
        `generate_block` calls, cleanest sub-block first, so each sub-block
        attends to the sub-blocks before it. Never persists."""
        return [[None, None] for _ in self.blocks]

    def generate_block(
        self,
        x_block: mx.array,
        t_block: mx.array,
        context: mx.array,
        self_caches: list,
        cross_kv_caches: list,
        start_frame: int,
        commit: bool = False,
        window_kvs: list | None = None,
    ) -> mx.array:
        """Denoise one block of latent frames (or, with commit=True, write its
        clean K/V into the KV-cache).

        Args:
            x_block: latent [C, F, H, W] for this block (F = num_frame_per_block)
            t_block: per-frame integer timestep [F] (or scalar)
            context: pre-embedded text [1, text_len, dim] (see embed_text)
            self_caches: per-layer CausalKVCache (mutated in place only if commit)
            cross_kv_caches: per-layer (k, v) from prepare_cross_kv
            start_frame: absolute latent-frame index of this block's first frame
            commit: if True, persist this block's (clean) K/V into the cache;
                if False, attend to past + this block without persisting
            window_kvs: from `make_window_kvs` — non-persisting K/V of earlier
                sub-blocks in the same diagonal window (Rolling Forcing), so this
                sub-block sees them but NOT the noisier sub-blocks that follow.
                Mutated in place. None = plain block (CF/CF++), unchanged.
        Returns:
            model output (flow/velocity) latent [C, F, H, W]
        """
        p, gs = self._patchify(x_block)  # [1, S, dim], gs = (F, H, W)
        f, h, w = gs
        fsl = h * w
        S = f * fsl
        x = p

        # Per-frame timestep -> per-token, then the standard sinusoidal time MLP.
        if t_block.ndim == 0:
            t_block = mx.broadcast_to(t_block[None], (f,))
        t_tok = mx.repeat(t_block, fsl)[None]  # [1, S]
        sinusoid = t_tok[..., None].astype(mx.float32) * self._inv_freq
        sin_emb = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)
        e = self.time_embedding_1(self.time_embedding_act(self.time_embedding_0(sin_emb)))
        e0 = self.time_projection(self.time_projection_act(e)).reshape(1, S, 6, self.dim)

        for i, blk in enumerate(self.blocks):
            mod = blk.modulation + e0  # [1, S, 6, dim] (float32)
            x_mod = blk.norm1(x) * (1 + mod[:, :, 1, :]) + mod[:, :, 0, :]
            y = blk.self_attn(x_mod, [gs], self.freqs, self_caches[i], start_frame,
                              commit, window_kvs[i] if window_kvs is not None else None)
            x = x + y * mod[:, :, 2, :]

            x_cross = blk.norm3(x) if blk.norm3 is not None else x
            x = x + blk.cross_attn(x_cross, context, None, kv_cache=cross_kv_caches[i])

            x_mod = blk.norm2(x) * (1 + mod[:, :, 4, :]) + mod[:, :, 3, :]
            x = x + blk.ffn(x_mod) * mod[:, :, 5, :]

            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.eval(x, self_caches[i].k, self_caches[i].v)

        x = self.head(x, e)
        return self.unpatchify(x, [gs])[0].astype(mx.float32)
