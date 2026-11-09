"""MLX port of ``models/next_concept.py`` (the Next-Concept generator)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..utils import count_params
from .common import Block, LayerNorm, Transformer, precompute_rope_cos_sin


def _trunc_normal(shape, std, dtype=mx.float32):
    """``torch.nn.init.trunc_normal_(..., std=std)`` (mean 0, truncated at +/-2 std)."""
    if hasattr(mx.random, "truncated_normal"):
        return mx.random.truncated_normal(-2.0, 2.0, shape, dtype=dtype) * std
    return mx.random.normal(shape, dtype=dtype) * std



class _TiedCodebookHead(nn.Module):
    def __init__(self, n_embd, codebook, learn_scale=True, trainable=False, use_bias=False):
        super().__init__()
        V, d = codebook.shape
        self.proj = nn.Linear(n_embd, d, bias=False)
        self.bias = mx.zeros((V,)) if use_bias else None
        self.trainable = bool(trainable)
        if self.trainable:
            self.E = mx.array(codebook)
        else:
            # torch keeps this as a non-persistent buffer: it is derived from the
            # VQ codebook, so it never round-trips through the checkpoint.
            self.E = mx.array(codebook)
        self.out_features = V
        # Distance form, the inverse of z_to_idx's argmin ||z - E_i||^2: ||z||^2 cancels
        # in the softmax, leaving 2 z.E_i - ||E_i||^2.
        E_sq = mx.sum(mx.array(codebook) ** 2, axis=-1)
        if not self.trainable:
            self.E_sq = E_sq
        self.d_scale = mx.maximum(mx.mean(E_sq), mx.array(1e-12))
        self.log_inv_T = mx.array(0.0) if learn_scale else 0.0

    def __call__(self, x):
        z = self.proj(x)
        E_sq = mx.sum(self.E ** 2, axis=-1) if self.trainable else self.E_sq
        neg_d2 = (2.0 * (z @ self.E.T) - E_sq) / self.d_scale
        out = neg_d2 * mx.exp(self.log_inv_T)
        return out if self.bias is None else out + self.bias


class _SharedHeadView(nn.Module):
    """One level's view of a shared output projection (optionally with a per-level bias)."""

    def __init__(self, owner, index):
        super().__init__()
        # Underscore attributes are not traversed by ``parameters()``, so the
        # shared projection and the per-level biases stay registered exactly once.
        self._owner = owner
        self._index = int(index)

    def __call__(self, x):
        out = mx.matmul(x, self._owner.head.weight.T)
        bias = self._owner.biases[self._index] if self._owner.biases is not None else None
        return out if bias is None else out + bias


class _SharedHeads(nn.Module):
    # one projection shared by every level; with per_level_bias each level owns its bias
    def __init__(self, head, n, per_level_bias=False):
        super().__init__()
        self.head = head
        self.n = int(n)
        self.per_level_bias = bool(per_level_bias)
        if self.per_level_bias:
            self.biases = [mx.zeros((head.weight.shape[0],)) for _ in range(self.n)]
            self._views = [_SharedHeadView(self, i) for i in range(self.n)]
        else:
            self.biases = None
            self._views = None

    def __getitem__(self, i):
        return self._views[i] if self.per_level_bias else self.head

    def __len__(self):
        return self.n


class NextConceptTransformer(Transformer):

    def __init__(self,
                 vqvae_local,
                 config
                 ):
        super().__init__()
        n_embd = config.n_embd
        n_layer = config.n_layer
        dropout = config.dropout
        bias = config.bias
        vocab_size = config.vocab_size
        attn_pattern = config.attn_pattern if hasattr(config, 'attn_pattern') else 'block_diagonal'
        self.config = config

        # A tuple keeps the frozen VQ-VAE out of the module tree and the state dict
        # (MLX does not traverse tuples when collecting parameters, same as torch).
        self.vqvae_local = (vqvae_local,)

        self.levels = list(self.vqvae_local[0].vq.levels)
        self.L = sum(self.levels)

        init_std = math.sqrt(1 / n_embd)

        self.query_level0 = bool(getattr(config, 'query_level0', False))
        self.l0_ar = bool(getattr(config, 'l0_autoregressive', False)) and self.query_level0
        self._lvl_first = 0 if self.query_level0 else 1
        T = self.L if self.query_level0 else self.L - self.levels[0]
        P = config.prefix_len

        # Per-level RoPE: prefix at [0..P-1], each target chunk reset to [0..l_k-1].
        self.use_rope = bool(getattr(config, 'use_rope', False))
        self.rope_base = int(getattr(config, 'rope_base', 10000))

        if self.use_rope:
            head_dim = n_embd // config.n_head
            levels_target = self.levels[self._lvl_first:]

            pos_full = mx.concatenate(
                [mx.arange(P)] + [mx.arange(l) for l in levels_target])
            cos_full, sin_full = precompute_rope_cos_sin(pos_full, head_dim, self.rope_base)
            self.rope_cos = cos_full
            self.rope_sin = sin_full

            for k, l_k in enumerate(levels_target):
                pos_k = mx.concatenate([mx.arange(P), mx.arange(l_k)])
                cos_k, sin_k = precompute_rope_cos_sin(pos_k, head_dim, self.rope_base)
                setattr(self, f'rope_cos_gen_{k}', cos_k)
                setattr(self, f'rope_sin_gen_{k}', sin_k)
        else:
            T_max = max(P, max(self.levels[self._lvl_first:]))
            self.wpe = nn.Embedding(T_max, n_embd)
            self.wp_prefix = mx.arange(P).reshape(1, -1)
            self.wp_1L = mx.concatenate(
                [mx.arange(l) for l in self.levels[self._lvl_first:]]).reshape(1, -1)
            self.wpe.weight = _trunc_normal(self.wpe.weight.shape, init_std)

        self.transformer = nn.Module()
        self.transformer.wte = nn.Linear(self.vqvae_local[0].embed_dim, n_embd, bias=bias)

        ctx_cfg = getattr(config, 'ctx_encoder', None)
        self.use_ctx_encoder = bool(getattr(ctx_cfg, 'enabled', False)) if ctx_cfg is not None else False
        self.ctx_pretrained = str(getattr(ctx_cfg, 'pretrained', '') or '') if ctx_cfg is not None else ''
        if self.use_ctx_encoder and self.ctx_pretrained:
            raise NotImplementedError(
                "ctx_encoder.pretrained loads a PyTorch T5EncoderModel; the MLX port does "
                "not support it. Use ctx_encoder.enabled: false (all released checkpoints).")
        if self.use_ctx_encoder:
            n_ctx_layer = int(getattr(ctx_cfg, 'n_layer', 4))
            ctx_vocab = int(getattr(ctx_cfg, 'vocab_size', self.vqvae_local[0].in_vocab_size))
            self.ctx_wte = nn.Embedding(ctx_vocab, n_embd)
            if bool(getattr(ctx_cfg, 'init_from_gpt2', False)):
                raise NotImplementedError(
                    "ctx_encoder.init_from_gpt2 reads PyTorch GPT-2 weights; use "
                    "init_from_vqvae or train the context encoder in MLX.")
            elif bool(getattr(ctx_cfg, 'init_from_vqvae', False)):
                src = self.vqvae_local[0].encoder.wte.weight
                proj = self.transformer.wte.weight
                w = mx.matmul(src.astype(proj.dtype), proj.T)
                if self.transformer.wte.bias is not None:
                    w = w + self.transformer.wte.bias
                V = min(w.shape[0], ctx_vocab)
                cur = self.ctx_wte.weight
                self.ctx_wte.weight = mx.concatenate(
                    [w[:V].astype(cur.dtype), cur[V:]], axis=0)
            else:
                self.ctx_wte.weight = _trunc_normal(self.ctx_wte.weight.shape, init_std)
            self.ctx_blocks = [Block(config) for _ in range(n_ctx_layer)]
            self.ctx_ln = LayerNorm(n_embd, bias=bias) if n_ctx_layer > 0 else nn.Identity()

        puc_cfg = getattr(config, 'post_upsample_conv', None)
        self.use_post_upsample_conv = bool(getattr(puc_cfg, 'enabled', False)) if puc_cfg is not None else False
        if self.use_post_upsample_conv:
            k = int(getattr(puc_cfg, 'kernel_size', 3))
            assert k % 2 == 1, "post_upsample_conv.kernel_size must be odd for same-length padding"
            vq_dim = self.vqvae_local[0].embed_dim
            self.post_upsample_convs = [
                nn.Conv1d(vq_dim, vq_dim, kernel_size=k, padding=k // 2)
                for _ in range(len(self.levels) - (0 if self.query_level0 else 1))
            ]
            for conv in self.post_upsample_convs:
                conv.weight = mx.zeros(conv.weight.shape)
                conv.bias = mx.zeros(conv.bias.shape)

        if config.use_level_encoding:
            self.transformer.wle = nn.Embedding(
                len(self.levels) - (0 if self.query_level0 else 1), n_embd)

        self.transformer.drop = nn.Dropout(dropout)
        self.transformer.blocks = [Block(config) for _ in range(n_layer)]
        self.transformer.ln_f = LayerNorm(n_embd, bias=bias)

        _n_heads = len(self.levels) - (0 if self.query_level0 else 1)
        assert len(vocab_size) == _n_heads, \
            f"vocab_size length {len(vocab_size)} must equal {_n_heads} " \
            f"(query_level0={self.query_level0}, levels={len(self.levels)})"
        self.vocab_sizes = list(vocab_size)

        if bool(getattr(config, 'shared_head_mlp', False)):
            raise ValueError("shared_head_mlp is not supported")
        self.shared_output_head = bool(getattr(config, 'shared_output_head', False))
        if self.shared_output_head:
            assert len(set(self.vocab_sizes)) == 1, "shared_output_head requires equal vocab sizes"
            _plb = bool(getattr(config, 'shared_head_per_level_bias', False))
            _hb = bool(getattr(config, 'untied_head_bias', True)) and not _plb
            self.lm_heads = _SharedHeads(nn.Linear(n_embd, self.vocab_sizes[0], bias=_hb),
                                         len(self.vocab_sizes), per_level_bias=_plb)
        elif bool(getattr(config, 'tie_heads_to_codebook', False)):
            _vq = self.vqvae_local[0].vq
            _ls = bool(getattr(config, 'tie_heads_learn_scale', True))
            _tr = bool(getattr(config, 'tie_heads_trainable', False))
            _bi = bool(getattr(config, 'tie_heads_bias', False))
            self.lm_heads = [
                _TiedCodebookHead(n_embd, _vq.embeddings[i + self._lvl_first].weight,
                                  learn_scale=_ls, trainable=_tr, use_bias=_bi)
                for i in range(len(self.vocab_sizes))
            ]
        else:
            _hb = bool(getattr(config, 'untied_head_bias', True))
            self.lm_heads = [
                nn.Linear(n_embd, vs, bias=_hb) for vs in self.vocab_sizes
            ]

        if self.query_level0:
            self.l0_queries = _trunc_normal(
                (1, self.levels[0], self.vqvae_local[0].embed_dim), init_std)

        self.prefix_len = P

        _ppid = getattr(config, 'prefix_pad_token_id', None)
        self.prefix_pad_id = int(_ppid) if _ppid is not None else None

        self.predict_level0 = bool(getattr(config, 'predict_level0', False))
        self.max_vocab = max(self.vocab_sizes + [self.vqvae_local[0].vq.vqvae_vocab_sizes[0]])
        if self.predict_level0:
            v0 = self.vqvae_local[0].vq.vqvae_vocab_sizes[0]
            assert P >= self.levels[0], (
                f"predict_level0 reads the last {self.levels[0]} prefix positions; "
                f"prefix_len={P} is too short"
            )
            if bool(getattr(config, 'tie_heads_to_codebook', False)):
                self.lm_head_l0 = _TiedCodebookHead(
                    n_embd, self.vqvae_local[0].vq.embeddings[0].weight,
                    learn_scale=bool(getattr(config, 'tie_heads_learn_scale', True)),
                    trainable=bool(getattr(config, 'tie_heads_trainable', False)),
                    use_bias=bool(getattr(config, 'tie_heads_bias', False)))
            else:
                self.lm_head_l0 = nn.Linear(n_embd, v0, bias=False)

        assert attn_pattern in ['block_diagonal', 'block_causal'], \
            "attn_pattern must be either 'block_diagonal' or 'block_causal'"

        d = mx.concatenate(
            [mx.full((l,), i, dtype=mx.int32)
             for i, l in enumerate(self.levels[self._lvl_first:])]).reshape(1, T, 1)
        dT = d.transpose(0, 2, 1)
        self.lvl_1L = dT[:, 0]

        neg_inf = mx.array(float('-inf'))
        if attn_pattern == 'block_diagonal':
            target_mask = mx.where(d == dT, mx.array(0.0), neg_inf).reshape(T, T)
        else:
            target_mask = mx.where(d >= dT, mx.array(0.0), neg_inf).reshape(T, T)
        if self.l0_ar:
            _l0 = self.levels[0]
            ii = mx.arange(T)[:, None]
            jj = mx.arange(T)[None, :]
            in_l0 = (ii < _l0) & (jj < _l0)
            ar_mask = mx.where(jj > ii, neg_inf, mx.array(0.0))
            target_mask = mx.where(in_l0, ar_mask, target_mask)

        top_left = mx.zeros((P, P))
        top_right = mx.full((P, T), float('-inf'))
        bottom_left = mx.zeros((T, P))
        bottom_right = target_mask

        self.attn_mask_for_masking = mx.concatenate([
            mx.concatenate([top_left, top_right], axis=1),
            mx.concatenate([bottom_left, bottom_right], axis=1),
        ], axis=0).reshape(1, 1, P + T, P + T)

        self.use_flex_attention = bool(getattr(config, 'use_flex_attention', False))
        if self.use_flex_attention:
            raise NotImplementedError(
                "use_flex_attention requires torch's flex_attention, which has no MLX "
                "equivalent; set use_flex_attention: false (all released checkpoints).")

        # GPT-2 style scaled init for residual projections.
        for name, mod in self.named_modules():
            if name.endswith('c_proj') and hasattr(mod, 'weight'):
                mod.weight = mx.random.normal(mod.weight.shape) * (0.02 / math.sqrt(2 * n_layer))

        total_params, non_embd_params = count_params(self, exclude_params=['wte', 'wle', 'wpe'])

        print("NCP Transformer, Total parameter count: %.2fM" % (total_params / 1e6,))
        print("NCP Transformer, Non embedding parameter count: %.2fM" % (non_embd_params / 1e6,))



    def _get_flex_block_mask(self, device=None):
        raise NotImplementedError(
            "flex_attention block masks are not available in MLX; "
            "set use_flex_attention: false in the config")

    def _apply_post_upsample_convs_multi(self, x_target):
        refined_chunks = []
        offset = 0
        for i, level in enumerate(self.levels[self._lvl_first:]):
            chunk = x_target[:, offset:offset + level, :]
            delta = self.post_upsample_convs[i](chunk)
            refined_chunks.append(chunk + delta)
            offset += level
        return mx.concatenate(refined_chunks, axis=1)

    def _apply_post_upsample_conv_single(self, chunk, level_idx_in_target):
        return chunk + self.post_upsample_convs[level_idx_in_target](chunk)

    def _encode_context(self, prefix_ids):
        if self.ctx_pretrained:
            raise NotImplementedError("pretrained T5 context encoders are not ported to MLX")
        h = self.ctx_wte(prefix_ids)
        rope_p = (self.rope_cos[:self.prefix_len],
                  self.rope_sin[:self.prefix_len]) if self.use_rope else None
        P = self.prefix_len
        ctx_mask = mx.zeros((1, 1, P, P))
        for block in self.ctx_blocks:
            h = block(h, attn_mask=ctx_mask, rope_cos_sin=rope_p)
        return self.ctx_ln(h)

    def __call__(self, x, prefix_ids=None, prefix_pad_mask=None):
        if self.use_ctx_encoder:
            assert prefix_ids is not None, "ctx_encoder requires prefix_ids"
            prefix_emb = self._encode_context(prefix_ids)
            x_target = x
            if self.use_post_upsample_conv:
                x_target = self._apply_post_upsample_convs_multi(x_target)
            x = self.transformer.wte(x_target)
            if not self.use_rope:
                x = x + self.wpe(self.wp_1L)
            if self.config.use_level_encoding:
                x = x + self.transformer.wle(self.lvl_1L)
            return self._run_stack(prefix_emb, x, prefix_pad_mask)

        z_prefix = x[:, :self.prefix_len, :]
        x_target = x[:, self.prefix_len:, :]

        if self.use_post_upsample_conv:
            x_target = self._apply_post_upsample_convs_multi(x_target)

        prefix_emb = self.transformer.wte(z_prefix)
        x = self.transformer.wte(x_target)

        if not self.use_rope:
            prefix_emb = prefix_emb + self.wpe(self.wp_prefix)
            x = x + self.wpe(self.wp_1L)

        if self.config.use_level_encoding:
            x = x + self.transformer.wle(self.lvl_1L)

        return self._run_stack(prefix_emb, x, prefix_pad_mask)

    def build_l0_query_input(self, batch_size, dtype=None):
        q = mx.broadcast_to(self.l0_queries, (batch_size,) + self.l0_queries.shape[1:])
        return q if dtype is None else q.astype(dtype)

    def build_l0_ar_input(self, l0_idx, dtype=None):
        B = l0_idx.shape[0]
        prev = self.vqvae_local[0].vq.embeddings[0](l0_idx[:, :self.levels[0] - 1])
        head = mx.broadcast_to(self.l0_queries[:, :1],
                               (B,) + self.l0_queries[:, :1].shape[1:]).astype(prev.dtype)
        x = mx.concatenate([head, prev], axis=1)
        return x if dtype is None else x.astype(dtype)

    def _prefix_pad_bias(self, pad_mask, S, dtype):
        B, P = pad_mask.shape
        pm = mx.where(pad_mask[:, None, None, :],
                      mx.array(float('-inf'), dtype=dtype),
                      mx.array(0.0, dtype=dtype))
        tail = mx.zeros((B, 1, 1, S - P), dtype=dtype)
        return mx.concatenate([pm, tail], axis=-1)

    def _level0_slice(self, x, prefix_pad_mask=None):
        P, l0 = self.prefix_len, self.levels[0]
        if prefix_pad_mask is None:
            return x[:, P - l0:P, :]
        last = mx.sum(prefix_pad_mask.astype(mx.int32) == 0, axis=1) - 1
        last = mx.maximum(last, l0 - 1)
        offs = mx.arange(l0 - 1, -1, -1)
        pos = mx.clip(last[:, None] - offs[None, :], 0, P - 1)
        return mx.take_along_axis(x, pos[:, :, None], axis=1)

    def _run_stack(self, prefix_emb, x, prefix_pad_mask=None):
        x = mx.concatenate([prefix_emb, x], axis=1)

        attn_mask = self.attn_mask_for_masking
        if prefix_pad_mask is not None:
            attn_mask = attn_mask + self._prefix_pad_bias(
                prefix_pad_mask, x.shape[1], attn_mask.dtype)

        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        for block in self.transformer.blocks:
            x = block(x, attn_mask=attn_mask, rope_cos_sin=rope)
        x = self.transformer.ln_f(x)

        x_target = x[:, self.prefix_len:, :]
        logits_list = []
        if self.predict_level0:
            l0_h = self._level0_slice(x, prefix_pad_mask)
            logits_list.append(self._pad_vocab(self.lm_head_l0(l0_h)))
        offset = 0
        for i, level in enumerate(self.levels[self._lvl_first:]):
            logits_list.append(
                self._pad_vocab(self.lm_heads[i](x_target[:, offset:offset + level, :])))
            offset += level
        return mx.concatenate(logits_list, axis=1)

    def _pad_vocab(self, logits):
        pad = self.max_vocab - logits.shape[-1]
        if pad == 0:
            return logits
        return mx.pad(logits, [(0, 0)] * (logits.ndim - 1) + [(0, pad)],
                      constant_values=mx.finfo(logits.dtype).min)



    def _set_kv_caching(self, enable: bool):
        for block in self.transformer.blocks:
            block.attn.kv_caching(enable)

    def _sample_level(self, logits, temperature, top_k, top_p):
        _c = getattr(self, "_cfg", None)
        if _c is not None and _c["scale"] > 0.0:
            B = _c["B"]
            t = _c["scale"] * (_c["level"] / max(_c["nlev"] - 1, 1))
            logits = (1.0 + t) * logits[:B] - t * logits[B:]
            pred = self._sample_level_raw(logits, temperature, top_k, top_p)
            return mx.concatenate([pred, pred], axis=0)
        return self._sample_level_raw(logits, temperature, top_k, top_p)

    def _sample_level_raw(self, logits, temperature, top_k, top_p):
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        logits_flat = logits.reshape(-1, logits.shape[-1]) / temperature
        if top_k > 0:
            kth = mx.topk(logits_flat, top_k, axis=-1)[:, -1:]
            logits_flat = mx.where(logits_flat < kth,
                                   mx.full(logits_flat.shape, float('-inf')), logits_flat)
        order = mx.argsort(-logits_flat, axis=-1)
        sorted_logits = mx.take_along_axis(logits_flat, order, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        cumprobs = mx.cumsum(probs, axis=-1)
        remove = (cumprobs - probs) > top_p
        sorted_logits = mx.where(remove, mx.full(sorted_logits.shape, float('-inf')),
                                 sorted_logits)
        rows = mx.arange(logits_flat.shape[0])[:, None]
        # ``order`` is a full permutation, so every slot is written; sampling from the
        # (masked) logits is equivalent to torch.multinomial(softmax(...)).
        masked = mx.zeros(logits_flat.shape).at[rows, order].add(sorted_logits)
        out = mx.random.categorical(masked.astype(mx.float32))
        return out.reshape(logits.shape[:-1])

    def _decode_level_logits(self, x, prefix_emb, level_idx, pad_bias=None, hidx=None):
        P = self.prefix_len
        hidx = level_idx if hidx is None else hidx
        L_cur = self.levels[level_idx + 1]
        x = mx.concatenate([prefix_emb, x], axis=1)

        S = P + L_cur
        ii = mx.arange(S)[:, None]
        jj = mx.arange(S)[None, :]
        gen_mask = mx.where((ii < P) & (jj >= P),
                            mx.array(float('-inf')), mx.array(0.0)
                            ).reshape(1, 1, S, S)
        if pad_bias is not None:
            gen_mask = gen_mask + mx.pad(
                pad_bias, [(0, 0)] * (pad_bias.ndim - 1) + [(0, L_cur)])
        rope_k = (
            getattr(self, f'rope_cos_gen_{hidx}'),
            getattr(self, f'rope_sin_gen_{hidx}'),
        ) if self.use_rope else None
        for block in self.transformer.blocks:
            x = block(x, attn_mask=gen_mask, rope_cos_sin=rope_k)
        x = self.transformer.ln_f(x)
        return self.lm_heads[hidx](x[:, P:, :])

    def generate(self, idx, prefix_tokens, temperature=1.0, top_k=50, top_p=0.9,
                 use_kv_cache=False, cfg_scale=0.0, cfg_null_eos=50256):
        assert not (self.query_level0 and use_kv_cache), \
            "query_level0 generation supports the eager path only"

        neg_inf = mx.array(float('-inf'))
        self._cfg = None
        # CFG: double the batch to [document; null] before the prefix is encoded;
        # cfg_scale=0 builds no null branch.
        if cfg_scale > 0.0:
            if self.prefix_pad_id is None:
                raise ValueError("cfg_scale needs prefix_pad_id to build the null prefix")
            _cB = prefix_tokens.shape[0]
            _null = mx.full(prefix_tokens.shape, self.prefix_pad_id,
                            dtype=prefix_tokens.dtype)
            _null = _null.at[:, 0].add(
                mx.array(cfg_null_eos, dtype=prefix_tokens.dtype) - _null[:, 0])
            prefix_tokens = mx.concatenate([prefix_tokens, _null], axis=0)
            if idx is not None:
                idx = mx.concatenate([idx, idx], axis=0)
            self._cfg = {"scale": float(cfg_scale), "B": _cB,
                         "nlev": len(self.levels), "level": 0}

        B = prefix_tokens.shape[0]
        vq = self.vqvae_local[0].vq
        z_hat = mx.zeros((B, self.levels[-1], self.vqvae_local[0].embed_dim),
                         dtype=mx.float32)

        if self.use_ctx_encoder:
            prefix_emb = self._encode_context(prefix_tokens[:, :self.prefix_len])
        else:
            if bool(getattr(self.config, 'raw_token_prefix', False)):
                _enc = self.vqvae_local[0].encoder
                if bool(getattr(self.config, 'contextual_prefix', False)) and \
                        len(getattr(_enc, 'blocks', [])) > 0:
                    z_prefix = _enc.forward_nopos(prefix_tokens)[:, :self.prefix_len, :]
                else:
                    z_prefix = _enc.wte(prefix_tokens)[:, :self.prefix_len, :]
            else:
                z_prefix = self.vqvae_local[0].encode_z(prefix_tokens)[:, :self.prefix_len, :]
            prefix_emb = self.transformer.wte(z_prefix)
        if not self.use_rope:
            prefix_emb = prefix_emb + self.wpe(self.wp_prefix)

        if self.config.use_level_encoding:
            level_pos_embed = self.transformer.wle(self.lvl_1L)
        else:
            level_pos_embed = mx.zeros(
                (1, self.L - self.levels[0], self.config.n_embd), dtype=prefix_emb.dtype)

        if not self.use_rope:
            level_pos_embed = level_pos_embed + self.wpe(self.wp_1L)

        P = self.prefix_len

        _pbias = None
        _pm = None
        if self.prefix_pad_id is not None:
            _pm = (prefix_tokens[:, :P] == self.prefix_pad_id)
            _pbias = mx.where(_pm[:, None, None, :], neg_inf, mx.array(0.0))

        if idx is None and self.query_level0:
            l0 = self.levels[0]
            _q0 = self.build_l0_query_input(B, prefix_emb.dtype)
            if self.use_post_upsample_conv:
                _q0 = self._apply_post_upsample_conv_single(_q0, 0)
            q = self.transformer.wte(_q0)
            if not self.use_rope:
                q = q + self.wpe(self.wp_1L[:, :l0])
            if self.config.use_level_encoding:
                q = q + self.transformer.wle(self.lvl_1L[:, :l0])
            h = mx.concatenate([prefix_emb, q], axis=1)
            S0 = P + l0
            a = mx.arange(S0)[:, None]
            b = mx.arange(S0)[None, :]
            base0 = mx.where((a < P) & (b >= P), neg_inf, mx.array(0.0))
            if self.l0_ar:
                in_blk = (a >= P) & (b >= P)
                ar = mx.where(b > a, neg_inf, mx.array(0.0))
                base0 = mx.where(in_blk, ar, base0)
            m0 = base0.reshape(1, 1, S0, S0)
            if _pbias is not None:
                m0 = m0 + mx.pad(_pbias, [(0, 0), (0, 0), (0, 0), (0, l0)])
            rope_0 = (self.rope_cos_gen_0, self.rope_sin_gen_0) if self.use_rope else None
            if self.l0_ar:
                _E0 = vq.embeddings[0]
                base_q = self.build_l0_query_input(B, prefix_emb.dtype)
                x_in, codes = base_q, []
                for t in range(l0):
                    q = self.transformer.wte(x_in)
                    if not self.use_rope:
                        q = q + self.wpe(self.wp_1L[:, :l0])
                    if self.config.use_level_encoding:
                        q = q + self.transformer.wle(self.lvl_1L[:, :l0])
                    h = mx.concatenate([prefix_emb, q], axis=1)
                    for block in self.transformer.blocks:
                        h = block(h, attn_mask=m0, rope_cos_sin=rope_0)
                    h = self.transformer.ln_f(h)
                    codes.append(self._sample_level(
                        self.lm_heads[0](h[:, P + t:P + t + 1, :]),
                        temperature, top_k, top_p))
                    if t + 1 < l0:
                        known = mx.concatenate(codes, axis=1)
                        x_in = mx.concatenate([base_q[:, :1],
                                               _E0(known).astype(base_q.dtype),
                                               base_q[:, t + 2:]], axis=1)
                idx = mx.concatenate(codes, axis=1)
            else:
                for block in self.transformer.blocks:
                    h = block(h, attn_mask=m0, rope_cos_sin=rope_0)
                h = self.transformer.ln_f(h)
                if self._cfg is not None:
                    self._cfg["level"] = 0
                idx = self._sample_level(self.lm_heads[0](h[:, P:, :]),
                                         temperature, top_k, top_p)

        if idx is None:
            assert self.predict_level0, \
                "generate(idx=None) requires model.predict_level0 or query_level0"
            rope_p = (self.rope_cos_gen_0, self.rope_sin_gen_0) if self.use_rope else None
            h = prefix_emb
            pfx_mask = mx.zeros((1, 1, P, P))
            if _pbias is not None:
                pfx_mask = pfx_mask + _pbias
            for block in self.transformer.blocks:
                h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
            h = self.transformer.ln_f(h)
            l0_logits = self.lm_head_l0(
                self._level0_slice(h, _pm if self.prefix_pad_id is not None else None))
            if self._cfg is not None:
                self._cfg["level"] = 0
            idx = self._sample_level(l0_logits, temperature, top_k, top_p)
        generated_idx = [idx]
        idx_next = idx
        did_cache = False
        if use_kv_cache:
            self._set_kv_caching(True)
            did_cache = True

        try:
            prefix_kv = None
            if use_kv_cache:
                rope_p = (self.rope_cos_gen_0, self.rope_sin_gen_0) if self.use_rope else None
                h = prefix_emb
                pfx_mask = mx.zeros((1, 1, P, P))
                for block in self.transformer.blocks:
                    h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
                prefix_kv = [(b.attn.cached_k, b.attn.cached_v)
                             for b in self.transformer.blocks]

            level_acum = self.levels[0] if self.query_level0 else 0
            for level_idx in range(len(self.levels) - 1):
                if self._cfg is not None:
                    self._cfg["level"] = level_idx + 1

                z_q_level = vq.embeddings[level_idx](idx_next)
                z_hat, x = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)

                if self.use_post_upsample_conv:
                    x = self._apply_post_upsample_conv_single(
                        x, level_idx + (1 - self._lvl_first))

                x = self.transformer.wte(x) + level_pos_embed[
                    :, level_acum:level_acum + self.levels[level_idx + 1], :]

                L_cur = self.levels[level_idx + 1]
                # Heads and RoPE buffers are indexed by absolute level when level 0 occupies slot 0.
                _hi = level_idx + (1 - self._lvl_first)
                rope_k = (
                    getattr(self, f'rope_cos_gen_{_hi}'),
                    getattr(self, f'rope_sin_gen_{_hi}'),
                ) if self.use_rope else None

                if use_kv_cache:
                    for blk, (pk, pv) in zip(self.transformer.blocks, prefix_kv):
                        blk.attn.cached_k, blk.attn.cached_v = pk, pv
                    lvl_mask = mx.zeros((1, 1, L_cur, P + L_cur))
                    for block in self.transformer.blocks:
                        x = block(x, attn_mask=lvl_mask, rope_cos_sin=rope_k)
                    x = self.transformer.ln_f(x)
                    logits = self.lm_heads[_hi](x)
                    idx_next = self._sample_level(logits, temperature, top_k, top_p)
                else:
                    logits = self._decode_level_logits(
                        x, prefix_emb, level_idx, _pbias, hidx=_hi)
                    idx_next = self._sample_level(logits, temperature, top_k, top_p)

                generated_idx.append(idx_next)
                level_acum += self.levels[level_idx + 1]
        finally:
            if did_cache:
                self._set_kv_caching(False)

        out = mx.concatenate(generated_idx, axis=1)
        if self._cfg is not None:
            out = out[:self._cfg["B"]]
        self._cfg = None
        return out



