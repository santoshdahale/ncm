"""Generative perplexity under a reference LM (MLX port of ``evaluation/gen_ppl.py``).

The scoring maths is identical to the PyTorch version. The *reference model* is
abstracted behind ``ref_fn`` -- a callable mapping integer token ids to logits::

    ref_fn(input_ids: np.ndarray[B, T]) -> np.ndarray[B, T, V]

Two adapters are provided:

* ``mlx_ref_fn(model)`` wraps any ``mlx_lm`` model (pure MLX, recommended).
* ``torch_ref_fn(model)`` wraps a ``transformers`` causal LM (needs torch).

Because MLX has no float64, all reductions run in float32; corpus-level
perplexity is accumulated in Python floats so long documents do not lose
precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import numpy as np


@dataclass
class GenPPLResult:
    nll_sum: np.ndarray
    token_count: np.ndarray
    ppl: np.ndarray
    mean_nll: float
    mean_ppl: float


def mlx_ref_fn(model):
    """Wrap an ``mlx_lm`` model so it can be scored by ``generation_perplexity``."""

    def _fn(input_ids: np.ndarray) -> np.ndarray:
        logits = model(mx.array(input_ids, dtype=mx.int32))
        return np.array(logits.astype(mx.float32))

    return _fn


def torch_ref_fn(model):
    """Wrap a ``transformers`` causal LM (requires torch)."""
    import torch  # noqa: F401

    device = None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None

    def _fn(input_ids: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(input_ids.astype(np.int64))
        if device is not None:
            ids = ids.to(device)
        with torch.no_grad():
            out = model(input_ids=ids)
        return out.logits.float().cpu().numpy()

    return _fn




def generation_perplexity(
    ref_fn,
    tokenizer,
    prompts: List[str],
    generated_texts: List[str],
    *,
    score_only_generated: bool = True,
    batch_size: int = 8,
    max_length: Optional[int] = None,
) -> GenPPLResult:

    assert len(prompts) == len(generated_texts), "prompts and generated_texts must align"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_offsets = getattr(tokenizer, "is_fast", False)

    n = len(prompts)
    nll_sums = []
    tok_counts = []

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_prompts = prompts[batch_start:batch_end]
        batch_gens = generated_texts[batch_start:batch_end]

        full_texts = [p + g for p, g in zip(batch_prompts, batch_gens)]

        _full_max = max_length if max_length is not None else 1024
        enc_full = tokenizer(
            full_texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=_full_max,
            return_offsets_mapping=use_offsets,
        )

        input_ids = np.asarray(enc_full["input_ids"], dtype=np.int32)
        attn_mask = np.asarray(enc_full["attention_mask"], dtype=np.int32)
        offset_mapping = enc_full.get("offset_mapping", None)
        if offset_mapping is not None:
            offset_mapping = np.asarray(offset_mapping, dtype=np.int64)

        logits = np.asarray(ref_fn(input_ids), dtype=np.float32)

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attn_mask[:, 1:]

        include = shift_mask.astype(bool).copy()

        if score_only_generated:
            B = input_ids.shape[0]
            T = input_ids.shape[1]

            if offset_mapping is not None:
                real_token = (
                    (attn_mask == 1)
                    & ~((offset_mapping[:, :, 0] == 0) & (offset_mapping[:, :, 1] == 0))
                )

                prompt_char_lens = np.array([len(p) for p in batch_prompts])[:, None]
                is_gen = real_token & (offset_mapping[:, :, 0] >= prompt_char_lens)

                # argmax returns 0 for all-False rows, so fix those up explicitly.
                no_gen = ~is_gen.any(axis=1)
                first_gen = is_gen.argmax(axis=1)
                first_gen[no_gen] = T

                # In shift_labels (= input_ids[:, 1:]) the first generated token sits at first_gen - 1.
                start_pos_vec = np.clip(first_gen - 1, 0, None)
                positions = np.arange(T - 1)[None, :]
                include = include & (positions >= start_pos_vec[:, None])
            else:
                enc_prompt = tokenizer(
                    batch_prompts,
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                prompt_lens = np.asarray(enc_prompt["attention_mask"], dtype=np.int64).sum(axis=1)
                # argmax on attn_mask gives the index of the first real token (N for left-padding with N pads).
                padding_offset = attn_mask.astype(np.int64).argmax(axis=1)
                for i in range(B):
                    start_pos = max(int(padding_offset[i] + prompt_lens[i]) - 1, 0)
                    include[i, :start_pos] = False

        # log_softmax (numpy, float32) gathered at the labels.
        shifted = shift_logits - shift_logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        nll_tok = -np.take_along_axis(log_probs, shift_labels[..., None], axis=-1)[..., 0]

        nll_sum = (nll_tok * include.astype(np.float32)).sum(axis=1)
        tok_count = include.sum(axis=1).astype(np.int64)

        nll_sums.append(nll_sum)
        tok_counts.append(tok_count)

    nll_sum = np.concatenate(nll_sums, axis=0)
    tok_count = np.concatenate(tok_counts, axis=0)

    tok_count_safe = np.maximum(tok_count, 1)
    mean_nll_per_ex = nll_sum / tok_count_safe.astype(np.float32)
    ppl_per_ex = np.exp(mean_nll_per_ex)

    total_nll = float(nll_sum.sum())
    total_tok = int(tok_count.sum())
    mean_nll = total_nll / max(total_tok, 1)
    mean_ppl = math.exp(mean_nll)

    return GenPPLResult(
        nll_sum=nll_sum,
        token_count=tok_count,
        ppl=ppl_per_ex,
        mean_nll=mean_nll,
        mean_ppl=mean_ppl,
    )
