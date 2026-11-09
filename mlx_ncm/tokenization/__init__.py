"""MLX tokenizers (port of the ``tokenization/`` package).

Kept import-light on purpose: the concrete tokenizers are imported explicitly
(``from mlx_ncm.tokenization.vqvae_tokenizer import VQVAETokenizer``) so that a
missing optional dependency (e.g. ``transformers`` for the HF tokenizer) does not
break evaluation of tasks that do not need it.
"""

