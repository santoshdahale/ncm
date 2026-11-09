"""Layout and kernel helpers that bridge PyTorch semantics to MLX.

Three things differ between the two frameworks and are centralised here:

1. **Convolution layout.** ``torch.nn.Conv1d`` takes ``(B, C, L)`` and stores
   weights as ``(out, in, k)``. ``mlx.nn.Conv1d`` takes ``(B, L, C)`` and stores
   weights as ``(out, k, in)``. ``conv1d_bcl`` / ``conv_transpose1d_bcl`` keep
   the *PyTorch* calling convention so the model code reads like the reference
   implementation; ``weights.py`` handles the weight permutation.
2. **Interpolation.** MLX has no ``F.interpolate``. ``interpolate1d`` reproduces
   PyTorch's ``mode="area"`` (== ``adaptive_avg_pool1d``) and ``mode="linear"``
   with ``align_corners=False`` exactly, which the multiscale quantizer relies
   on to move between levels.
3. **float64.** MLX only has 16/32-bit floats, so anything the reference code
   did in ``float64`` is done in ``float32`` here.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

# ---------------------------------------------------------------------------
# Convolution helpers (PyTorch (B, C, L) convention)
# ---------------------------------------------------------------------------


def conv1d_bcl(conv, x_bcl):
    """Run an ``mlx.nn.Conv1d`` on a ``(B, C, L)`` tensor."""
    return conv(x_bcl.transpose(0, 2, 1)).transpose(0, 2, 1)


def conv_transpose1d_bcl(conv, x_bcl):
    """Run an ``mlx.nn.ConvTranspose1d`` on a ``(B, C, L)`` tensor."""
    return conv(x_bcl.transpose(0, 2, 1)).transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# Interpolation (works on the last axis, so callers stay in (B, C, L))
# ---------------------------------------------------------------------------

_POOL_W_CACHE: dict = {}
_LINEAR_IDX_CACHE: dict = {}


def _adaptive_pool_matrix(in_len: int, out_len: int) -> mx.array:
    """Averaging matrix of PyTorch's ``adaptive_avg_pool1d``.

    Output element ``a`` is the mean of ``in[floor(a*in/out):ceil((a+1)*in/out)]``,
    exactly the bounds PyTorch uses (``start_index`` / ``end_index``).
    """
    key = (in_len, out_len)
    hit = _POOL_W_CACHE.get(key)
    if hit is not None:
        return hit
    a = np.arange(out_len, dtype=np.int64)
    start = (a * in_len) // out_len
    end = -((-(a + 1) * in_len) // out_len)  # ceil((a+1) * in_len / out_len)
    w = np.zeros((out_len, in_len), dtype=np.float32)
    for i in range(out_len):
        w[i, start[i]:end[i]] = 1.0 / float(end[i] - start[i])
    _POOL_W_CACHE[key] = mx.array(w)
    return _POOL_W_CACHE[key]


def _linear_upsample_index(in_len: int, out_len: int):
    key = (in_len, out_len)
    hit = _LINEAR_IDX_CACHE.get(key)
    if hit is not None:
        return hit
    scale = in_len / out_len
    dst = np.arange(out_len, dtype=np.float64)
    src = np.maximum(scale * (dst + 0.5) - 0.5, 0.0)  # align_corners=False
    i0 = np.clip(np.floor(src).astype(np.int64), 0, in_len - 1)
    i1 = np.where(i0 < in_len - 1, i0 + 1, i0)
    w1 = (src - np.floor(src)).astype(np.float32)
    hit = (mx.array(i0), mx.array(i1), mx.array(1.0 - w1), mx.array(w1))
    _LINEAR_IDX_CACHE[key] = hit
    return hit


def area_pool1d(x, out_len: int):
    """``F.interpolate(x, size=out_len, mode='area')`` for ``(B, C, L)``."""
    in_len = x.shape[-1]
    if in_len == out_len:
        return x
    w = _adaptive_pool_matrix(in_len, out_len).astype(x.dtype)
    return mx.matmul(x, w.T)


def linear_upsample1d(x, out_len: int):
    """``F.interpolate(x, size=out_len, mode='linear', align_corners=False)``."""
    in_len = x.shape[-1]
    if in_len == out_len:
        return x
    i0, i1, w0, w1 = _linear_upsample_index(in_len, out_len)
    w0 = w0.astype(x.dtype)
    w1 = w1.astype(x.dtype)
    return mx.take(x, i0, axis=-1) * w0 + mx.take(x, i1, axis=-1) * w1


def interpolate1d(x, out_len: int, mode: str):
    if mode == "area":
        return area_pool1d(x, out_len)
    if mode == "linear":
        return linear_upsample1d(x, out_len)
    raise ValueError(f"unsupported interpolate mode for MLX port: {mode!r}")


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

_CAUSAL_CACHE: dict = {}


def causal_additive_mask(q_len: int, kv_len: int, dtype=mx.float32):
    """Additive ``(1, 1, q_len, kv_len)`` causal mask, bottom-right aligned.

    Mirrors ``is_causal=True`` in ``torch.nn.functional.scaled_dot_product_attention``:
    query ``i`` may attend to keys ``j <= i + (kv_len - q_len)``.
    """
    key = (q_len, kv_len, str(dtype))
    hit = _CAUSAL_CACHE.get(key)
    if hit is not None:
        return hit
    q = np.arange(q_len)[:, None]
    kv = np.arange(kv_len)[None, :]
    keep = kv <= (q + (kv_len - q_len))
    mask = np.where(keep, 0.0, -np.inf).astype(np.float32)
    _CAUSAL_CACHE[key] = mx.array(mask).astype(dtype).reshape(1, 1, q_len, kv_len)
    return _CAUSAL_CACHE[key]


def additive_causal_mask(t: int, dtype=mx.float32):
    return causal_additive_mask(t, t, dtype)


def causal_window_mask(t: int, window: int, dtype=mx.float32):
    """Additive ``(t, t)`` mask keeping ``|i - j| <= window`` (used by the VQ-VAE encoder)."""
    i = np.arange(t)[:, None]
    j = np.arange(t)[None, :]
    keep = np.abs(i - j) <= window
    return mx.array(np.where(keep, 0.0, -np.inf).astype(np.float32)).astype(dtype)


# ---------------------------------------------------------------------------
# Parameter introspection
# ---------------------------------------------------------------------------


def flatten_params(model):
    """``{dotted.path: array}`` for every parameter of an ``mlx.nn.Module``.

    Mirrors the naming of ``torch.nn.Module.named_parameters()`` (children are
    joined with ``.``, list elements use their index) which is what makes the
    checkpoint conversion a pure rename.
    """
    flat = {}

    def walk(value, prefix):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                walk(v, f"{prefix}{i}.")
        elif isinstance(value, mx.array):
            flat[prefix[:-1]] = value

    walk(model.parameters(), "")
    return flat

