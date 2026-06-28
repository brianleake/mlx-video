"""Integration smoke test for CausalWanModel (StreamFrame Phase 2).

Requires the converted Causal-Forcing MLX model at
~/.cache/streamframe/mlx/CausalForcing-Wan2.1-1.3B-MLX (skipped if absent).
Checks: the converted weights load 1:1, and block-by-block forward produces the
right shape with a growing bounded KV-cache and no NaNs.

    python tests/test_causal_model.py
"""
from pathlib import Path

import mlx.core as mx
import mlx.utils as mu

from mlx_video.models.wan_2.causal import CausalWanModel
from mlx_video.models.wan_2.config import WanModelConfig

MODEL = Path.home() / ".cache/streamframe/mlx/CausalForcing-Wan2.1-1.3B-MLX"


def test_causal_model_loads_and_forwards():
    if not (MODEL / "model.safetensors").exists():
        print("skip: converted Causal-Forcing model not present")
        return

    cfg = WanModelConfig.wan21_t2v_1_3b()
    model = CausalWanModel(cfg, sink_size=1, local_attn_size=-1, num_frame_per_block=3)
    w = mx.load(str(MODEL / "model.safetensors"))
    params = dict(mu.tree_flatten(model.parameters()))
    assert sum(1 for k in w if k in params) == len(w) == 825
    model.update(mu.tree_unflatten(list(w.items())))
    mx.eval(model.parameters())

    C, F, Hl, Wl = 16, 3, 32, 32
    fsl = (Hl // 2) * (Wl // 2)
    context = (mx.random.normal((1, 512, cfg.dim)) * 0.1).astype(mx.bfloat16)
    cross = model.prepare_cross_kv(context)
    caches = model.make_self_caches(frame_seqlen=fsl)
    x = mx.random.normal((C, F, Hl, Wl)).astype(mx.float32)
    t = mx.array([900, 900, 900])

    out = model.generate_block(x, t, context, caches, cross, start_frame=0)
    mx.eval(out)
    assert out.shape == (C, F, Hl, Wl)
    assert not bool(mx.any(mx.isnan(out)))

    model.generate_block(x, t, context, caches, cross, start_frame=F)
    assert caches[0].k.shape[1] == 2 * F * fsl


if __name__ == "__main__":
    test_causal_model_loads_and_forwards()
    print("causal model: checks PASS")
