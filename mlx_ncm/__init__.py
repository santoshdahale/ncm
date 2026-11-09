"""NCM (Next-Concept Modeling) -- MLX port.

This package mirrors the PyTorch reference implementation in ``models/``,
``tokenization/``, ``evaluation/`` and ``scripts/`` module-for-module, but runs
on Apple silicon through `MLX <https://github.com/ml-explore/mlx>`_.

Module names, attribute names and (therefore) checkpoint keys are kept
identical to the PyTorch code so that ``mlx_ncm.convert`` can map a PyTorch
``state_dict`` onto the MLX model with a single layout fix-up for convolutions.

Typical use::

    python -m mlx_ncm.convert --config config/eval/xsum.yaml \
        --checkpoint-dir artifacts/xsum --out artifacts-mlx/xsum
    python -m mlx_ncm.evaluate --config config/eval/xsum.yaml \
        --checkpoint-dir artifacts-mlx/xsum --data-dir data/xsum

See ``docs/MLX_MIGRATION.md`` for the full migration notes.
"""

__version__ = "0.1.0"

__all__ = [
    "evaluation",
    "models",
    "ops",
    "registry",
    "tokenization",
    "utils",
    "weights",
]
