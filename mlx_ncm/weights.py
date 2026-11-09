"""PyTorch ``state_dict`` <-> MLX weights.

The MLX port keeps every module path, attribute name and tensor name identical to
the PyTorch implementation, so conversion is only about *layout*:

* ``torch.nn.Linear`` / ``nn.Embedding`` weights are already ``(out, in)`` /
  ``(vocab, dim)`` and are copied verbatim.
* ``torch.nn.Conv1d`` stores ``(out, in, k)`` and ``torch.nn.ConvTranspose1d``
  stores ``(in, out, k)``; ``mlx.nn`` stores ``(out, k, in)`` for both, so the
  weights are permuted with ``(0, 2, 1)`` and ``(1, 2, 0)`` respectively.

Torch buffers registered with ``persistent=False`` (RoPE tables, the tied-codebook
distance terms, ...) never appear in a checkpoint: the MLX models recompute them
at construction and ``is_derived`` marks them as expected-missing.

This module is the only place that touches PyTorch, and only when converting.
"""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .ops import flatten_params

# Suffixes/substrings of buffers the PyTorch model does not persist.
_DERIVED_SUFFIXES = (".E", ".E_sq", ".d_scale", "_fd_step")
_DERIVED_SUBSTRINGS = ("rope_cos", "rope_sin")


def is_derived(name: str) -> bool:
    """True for parameters the MLX model rebuilds instead of loading."""
    return name.endswith(_DERIVED_SUFFIXES) or any(s in name for s in _DERIVED_SUBSTRINGS)


def _permutation_for(module):
    if isinstance(module, nn.ConvTranspose1d):
        return (1, 2, 0)
    if isinstance(module, nn.Conv1d):
        return (0, 2, 1)
    return None


def to_mx(value):
    """Convert a torch tensor / numpy array / MLX array to an ``mx.array``."""
    if isinstance(value, mx.array):
        return value
    if isinstance(value, np.ndarray):
        return mx.array(value)
    dtype = str(getattr(value, "dtype", ""))
    tensor = value.detach().cpu() if hasattr(value, "detach") else value
    if "bfloat16" in dtype:
        # numpy has no bfloat16: round-trip through float32, which is exact.
        return mx.array(tensor.float().numpy()).astype(mx.bfloat16)
    return mx.array(tensor.numpy())


def convert_state_dict(sd, model):
    """Map a PyTorch ``state_dict`` onto ``model``'s parameter names/layouts.

    Returns ``(weights, shape_mismatches)`` where ``weights`` is a
    ``{name: mx.array}`` dict and ``shape_mismatches`` lists any converted tensor
    whose shape does not match the MLX parameter it is meant for.
    """
    modules = dict(model.named_modules())
    expected = flatten_params(model)

    weights = {}
    for key, value in sd.items():
        arr = to_mx(value)
        if key.endswith("weight"):
            owner = key.rsplit(".", 1)[0] if "." in key else ""
            perm = _permutation_for(modules.get(owner))
            if perm is not None:
                arr = arr.transpose(perm)
        weights[key] = arr

    mismatches = [(k, tuple(v.shape), tuple(expected[k].shape))
                  for k, v in weights.items()
                  if k in expected and tuple(v.shape) != tuple(expected[k].shape)]
    return weights, mismatches


def apply_weights(model, weights, strict=True):
    """Load ``weights`` into ``model`` and report missing/unexpected names."""
    expected = flatten_params(model)
    missing = sorted(k for k in expected if k not in weights)
    unexpected = sorted(k for k in weights if k not in expected)

    usable = [(k, v) for k, v in weights.items() if k in expected]
    model.load_weights(usable, strict=False)
    mx.eval(model.parameters())

    if strict:
        real_missing = [k for k in missing if not is_derived(k)]
        if real_missing or unexpected:
            raise ValueError(
                "checkpoint does not match the MLX model:\n"
                f"  missing: {real_missing}\n  unexpected: {unexpected}")
    return missing, unexpected


def load_torch_state_dict(model, sd, strict=True):
    """Convert and load a PyTorch ``state_dict`` (dict of tensors into ``model``)."""
    weights, mismatches = convert_state_dict(sd, model)
    if mismatches:
        detail = "\n".join(f"  {k}: converted {a} vs model {b}" for k, a, b in mismatches[:20])
        raise ValueError(f"{len(mismatches)} converted tensors have the wrong shape:\n{detail}")
    return apply_weights(model, weights, strict=strict)


def load_torch_checkpoint(model, path, key="model", strict=True):
    """Load ``torch.load(path)[key]`` into ``model`` (requires torch installed)."""
    import torch  # noqa: F401  (only needed for conversion)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt[key] if key is not None and key in ckpt else ckpt
    return load_torch_state_dict(model, sd, strict=strict)


def save_module_weights(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    model.save_weights(path)
    return path


def load_module_weights(model, path, strict=True):
    """Load an MLX ``.safetensors`` file into ``model``."""
    weights = dict(mx.load(path))
    return apply_weights(model, weights, strict=strict)


def load_any(model, path, strict=True, key="model"):
    """Load either an MLX safetensors file or a PyTorch checkpoint into ``model``.

    ``.safetensors`` / ``.npz`` use the native MLX loader; anything else goes
    through ``torch.load`` (so the released ``.pt`` files work unchanged when
    torch is installed).
    """
    if path.endswith((".safetensors", ".npz")):
        return load_module_weights(model, path, strict=strict)
    if path.endswith((".pt", ".pth", ".bin")):
        return load_torch_checkpoint(model, path, key=key, strict=strict)
    raise ValueError(f"unsupported checkpoint format: {path!r}")



def save_state_dict(weights, path):
    """Save a ``{name: mx.array}`` dict as safetensors."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    mx.save_safetensors(path, {k: mx.array(v) for k, v in weights.items()})
    return path


def flatten_inference(ckpt):
    """Flatten a checkpoint's ``inference`` sub-dict into ``inference.<key>`` entries."""
    out = {}
    if isinstance(ckpt, dict) and "inference" in ckpt:
        for k, v in ckpt["inference"].items():
            out[f"inference.{k}"] = v
    return out
