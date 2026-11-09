"""Evaluation utilities (port of the ``evaluation/`` package)."""

from .gen_ppl import (  # noqa: F401
    GenPPLResult,
    generation_perplexity,
    mlx_ref_fn,
    torch_ref_fn,
)

__all__ = ["GenPPLResult", "generation_perplexity", "mlx_ref_fn", "torch_ref_fn"]
