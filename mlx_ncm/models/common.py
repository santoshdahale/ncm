"""MLX port of ``models/common.py`` (RoPE, LayerNorm, RMSNorm, attention, MLP, blocks)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..ops import additive_causal_mask, causal_additive_mask

# ``torch.nn.attention.flex_attention`` has no MLX equivalent. Every released
# config sets ``use_flex_attention: false``, so the flag is rejected loudly
# instead of being silently ignored (see ``SelfAttention.__call__``).
HAS_FLEX_ATTENTION = False


def precompute_rope_cos_sin(positions, head_dim, base=10000):
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    half = head_dim // 2
    inv_freq = 1.0 / mx.power(float(base), mx.arange(0, half, dtype=mx.float32) / half)
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1)
    sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


class LayerNorm(nn.Module):
    """LayerNorm with the reference implementation's ``eps=1e-6`` and optional bias."""

    def __init__(self, ndim, bias=False, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((ndim,))
        self.bias = mx.zeros((ndim,)) if bias else None
        self.eps = eps

    def __call__(self, x):
        return mx.fast.layer_norm(x, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """RMSNorm computed in float32 then cast back, as in ``models/common.py``."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        orig_dtype = x.dtype
        x_f = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(x_f * x_f, axis=-1, keepdims=True) + self.eps)
        return (x_f * rms).astype(orig_dtype) * self.weight


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # GQA: K/V use n_kv_heads <= n_head; the default reproduces full MHA.
        self.n_kv_head = int(getattr(config, 'n_kv_heads', config.n_head))
        assert config.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_heads"
        _hd = config.n_embd // config.n_head
        self._kv_dim = self.n_kv_head * _hd
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self._kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # MLX's fused SDPA has no dropout argument, so attention dropout is applied
        # to the attention output while training; eval-time behaviour is identical.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.caching, self.cached_k, self.cached_v = False, None, None

        # QK norm: RMSNorm over head_dim of Q and K, applied before RoPE as in Llama 3.
        self.use_qk_norm = bool(getattr(config, 'use_qk_norm', False))
        if self.use_qk_norm:
            head_dim = config.n_embd // config.n_head
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    def __call__(self, x, attn_mask=None, rope_cos_sin=None, block_mask=None):
        B, T, C = x.shape
        q, k, v = mx.split(self.c_attn(x), [self.n_embd, self.n_embd + self._kv_dim], axis=2)
        hd = C // self.n_head
        q = q.reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_head, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_head, hd).transpose(0, 2, 1, 3)
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # With KV caching the cache already holds rotated K, so rotate only [past_len, past_len+T).
        if rope_cos_sin is not None:
            cos_full, sin_full = rope_cos_sin
            past_len = self.cached_k.shape[2] if (self.caching and self.cached_k is not None) else 0
            cos = cos_full[past_len:past_len + T]
            sin = sin_full[past_len:past_len + T]
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = mx.concatenate([self.cached_k, k], axis=2)
                v = self.cached_v = mx.concatenate([self.cached_v, v], axis=2)

        if block_mask is not None:
            raise NotImplementedError(
                "flex_attention block masks are not supported by the MLX port; "
                "set use_flex_attention: false in the config")
        # Causal only for multiple tokens with no explicit mask; a cached single token sees all.
        use_causal = True if attn_mask is None and T > 1 else False
        mask = attn_mask
        if use_causal:
            mask = (additive_causal_mask(T, q.dtype) if k.shape[2] == T
                    else causal_additive_mask(T, k.shape[2], q.dtype))
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(hd), mask=mask)
        if mask is not None:
            # torch's SDPA uses a NaN-safe softmax: fully masked query rows produce
            # zeros. MLX returns NaN for them, which would poison later layers
            # through the value matmul (0 * NaN), so reproduce torch's behaviour.
            # Occurs when every key of a query is masked, e.g. the CFG null prefix.
            all_masked = mx.all(mask <= -float("inf"), axis=-1, keepdims=True)
            y = mx.where(all_masked, mx.array(0.0, dtype=y.dtype), y)


        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(self.attn_dropout(y) if self.training else y))
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_swiglu = bool(getattr(config, "use_swiglu", False))
        self.use_geglu = bool(getattr(config, "use_geglu", False))
        self.use_relu2 = bool(getattr(config, "use_relu2", False))
        self.gated_ffn = self.use_swiglu or self.use_geglu
        if self.gated_ffn:
            h = int(round(8 * config.n_embd / 3 / 8) * 8)
            self.w_gate = nn.Linear(config.n_embd, h, bias=config.bias)
            self.w_up = nn.Linear(config.n_embd, h, bias=config.bias)
            self.c_proj = nn.Linear(h, config.n_embd, bias=config.bias)
            self.act = nn.SiLU() if self.use_swiglu else nn.GELU()
        else:
            _ratio = float(getattr(config, 'mlp_ratio', 4.0))
            _h = int(round(_ratio * config.n_embd))
            self.c_fc = nn.Linear(config.n_embd, _h, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(_h, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        if self.gated_ffn:
            x = self.c_proj(self.act(self.w_gate(x)) * self.w_up(x))
        elif self.use_relu2:
            x = self.c_proj(mx.maximum(self.c_fc(x), 0) ** 2)
        else:
            x = self.c_proj(self.gelu(self.c_fc(x)))
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.parallel_block = bool(getattr(config, "parallel_block", False))
        self.normformer = bool(getattr(config, "normformer", False))
        if self.normformer:
            self.ln_post_attn = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_post_mlp = LayerNorm(config.n_embd, bias=config.bias)
        self.use_layerscale = bool(getattr(config, "use_layerscale", False))
        if self.use_layerscale:
            _ls = float(getattr(config, "layerscale_init", 1e-4))
            self.ls1 = mx.full((config.n_embd,), _ls)
            self.ls2 = mx.full((config.n_embd,), _ls)
        self.use_rezero = bool(getattr(config, "use_rezero", False))
        if self.use_rezero:
            self.rz1 = mx.zeros((1,))
            self.rz2 = mx.zeros((1,))

    def _ls_scale(self, h, which):
        if self.use_layerscale:
            return (self.ls1 if which == 1 else self.ls2) * h
        if self.use_rezero:
            return (self.rz1 if which == 1 else self.rz2) * h
        return h

    def __call__(self, x, attn_mask=None, rope_cos_sin=None, block_mask=None):
        if self.parallel_block:
            n = self.ln_1(x)
            a = self.attn(n, attn_mask, rope_cos_sin=rope_cos_sin, block_mask=block_mask)
            m = self.mlp(n)
            if self.normformer:
                a = self.ln_post_attn(a)
                m = self.ln_post_mlp(m)
            return x + self._ls_scale(a, 1) + self._ls_scale(m, 2)
        a = self.attn(self.ln_1(x), attn_mask, rope_cos_sin=rope_cos_sin, block_mask=block_mask)
        if self.normformer:
            a = self.ln_post_attn(a)
        x = x + self._ls_scale(a, 1)
        m = self.mlp(self.ln_2(x))
        if self.normformer:
            m = self.ln_post_mlp(m)
        x = x + self._ls_scale(m, 2)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight = mx.random.normal(module.weight.shape) * 0.02
            if module.bias is not None:
                module.bias = mx.zeros(module.bias.shape)
        elif isinstance(module, nn.Embedding):
            module.weight = mx.random.normal(module.weight.shape) * 0.02

    def get_num_params(self, non_embedding=True):
        _ = non_embedding
        raise NotImplementedError("This method should be implemented in subclasses.")

    def __call__(self, x):
        _ = x
        raise NotImplementedError("This method should be implemented in subclasses.")

    def generate(self, idx):
        _ = idx
        raise NotImplementedError("This method should be implemented in subclasses.")

