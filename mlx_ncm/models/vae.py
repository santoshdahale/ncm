"""MLX port of ``models/vae.py`` (VQ-VAE encoder / decoder, non-causal blocks)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..ops import causal_window_mask
from .common import LayerNorm, RMSNorm  # identical definitions; imported for key-path parity


class SelfAttention(nn.Module):
    """Full or windowed attention without a KV cache (encoder/decoder local attention)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_qk_norm = bool(getattr(config, "use_qk_norm", False))
        if self.use_qk_norm:
            head_dim = config.n_embd // config.n_head
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        # attn_window=w restricts attention to |i-j| <= w; 0 means full.
        self.attn_window = int(getattr(config, "attn_window", 0) or 0)

    def __call__(self, x):
        B, T, C = x.shape
        q, k, v = mx.split(self.c_attn(x), 3, axis=2)
        hd = C // self.n_head
        q = q.reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, hd).transpose(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_mask = causal_window_mask(T, self.attn_window, q.dtype) if self.attn_window > 0 else None
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(hd), mask=attn_mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(self.attn_dropout(y) if self.training else y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def _diag_conv1d(dim, kernel_size, stride, scale):
    """Conv1d initialised to a stride-``kernel_size`` average over channels (MLX layout)."""
    conv = nn.Conv1d(dim, dim, kernel_size, stride=stride, bias=True)
    w = mx.zeros(conv.weight.shape)  # (out, k, in)
    idx = mx.arange(dim)
    w[idx, :, idx] = scale
    conv.weight = w
    conv.bias = mx.zeros(conv.bias.shape)
    return conv


def _diag_conv_transpose1d(dim, kernel_size, stride):
    """ConvTranspose1d initialised to a stride-``kernel_size`` upsample (MLX layout)."""
    conv = nn.ConvTranspose1d(dim, dim, kernel_size, stride=stride, bias=True)
    w = mx.zeros(conv.weight.shape)  # (out, k, in)
    idx = mx.arange(dim)
    w[idx, :, idx] = 1.0
    conv.weight = w
    conv.bias = mx.zeros(conv.bias.shape)
    return conv


class Encoder(nn.Module):
    def __init__(
        self,
        in_vocab_size,
        context_length,
        embed_dim,
        n_layers,
        compression_factor=4,
        kernel_size=8,
        stride=2,
        bias=False,
        padding=3,
        dropout=0.1,
        enc_layers=0,
        use_qk_norm=False,
        enc_attn_window=0,
        enc_conv_layers=0,
        enc_conv_kernel=3,
        use_pos_emb=True,
    ):
        super().__init__()

        assert compression_factor == 1 or compression_factor % 2 == 0, \
            "Compression factor must be a power of 2"
        self.compression_factor = compression_factor
        self.wte = nn.Embedding(in_vocab_size, embed_dim)
        self.wpe = nn.Embedding(context_length, embed_dim) if use_pos_emb else None
        self.drop = nn.Dropout(dropout)

        # n_layers is the head count here, not a depth.
        self.blocks = [
            Block(config=type("Config", (object,), {
                "n_embd": embed_dim, "n_head": n_layers, "bias": True,
                "dropout": dropout, "use_qk_norm": use_qk_norm,
                "attn_window": int(enc_attn_window)}))
            for _ in range(int(enc_layers))
        ]
        k = int(enc_conv_kernel)
        self.conv_blocks = []
        for _ in range(int(enc_conv_layers)):
            blk = nn.Module()
            blk.ln = LayerNorm(embed_dim)
            blk.c1 = nn.Conv1d(embed_dim, embed_dim, k, padding=k // 2)
            blk.c2 = nn.Conv1d(embed_dim, embed_dim, k, padding=k // 2)
            blk.c2.weight = mx.zeros(blk.c2.weight.shape)
            blk.c2.bias = mx.zeros(blk.c2.bias.shape)
            self.conv_blocks.append(blk)
        self.ln_e = LayerNorm(embed_dim) if (int(enc_layers) > 0 or int(enc_conv_layers) > 0) else None

        # Sequence compression: f>1 shortens the sequence before quantization; init is an average pool.
        self.down = None
        if int(compression_factor) > 1:
            f = int(compression_factor)
            self.down = _diag_conv1d(embed_dim, f, f, 1.0 / f)

    def _conv_stack(self, x):
        # x is (B, T, C) in both frameworks; MLX convolutions are channel-last, which
        # is what the reference code's transpose(1, 2) ... transpose(1, 2) pair reaches.
        for blk in self.conv_blocks:
            h = blk.ln(x)
            h = blk.c2(nn.gelu(blk.c1(h)))
            x = x + h
        return x

    def __call__(self, idx):
        tok_emb = self.wte(idx)
        if self.wpe is not None:
            pos = mx.arange(0, idx.shape[1], dtype=mx.int32)
            tok_emb = tok_emb + self.wpe(pos)
        x = self.drop(tok_emb)
        x = self._conv_stack(x)
        for blk in self.blocks:
            x = blk(x)
        if self.ln_e is not None:
            x = self.ln_e(x)
        if self.down is not None:
            x = self.down(x)
        return x

    def forward_nopos(self, idx):
        B, T = idx.shape[0], idx.shape[1]
        C = self.wte.weight.shape[1]
        pos = mx.arange(T, dtype=mx.float32).reshape(T, 1)
        i = mx.arange(0, C, 2, dtype=mx.float32)
        div = mx.exp(-math.log(10000.0) * i / C)
        pe = mx.zeros((T, C), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(pos * div)
        pe[:, 1::2] = mx.cos(pos * div)
        x = self.drop(self.wte(idx) + pe.astype(self.wte.weight.dtype)[None])
        x = self._conv_stack(x)
        for blk in self.blocks:
            x = blk(x)
        if self.ln_e is not None:
            x = self.ln_e(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_vocab_size,
        embed_dim,
        n_layers,
        compression_factor=4,
        kernel_size=8,
        stride=2,
        bias=False,
        padding=3,
        dropout=0.1,
        use_qk_norm=False,
        dec_layers=1,
        use_pos_emb=False,
        context_length=None,
    ):
        super().__init__()

        assert compression_factor == 1 or compression_factor % 2 == 0, \
            "Compression factor must be a power of 2"

        self.wpe = nn.Embedding(int(context_length), embed_dim) if use_pos_emb else None
        self.drop = nn.Dropout(dropout)
        self.compression_factor = compression_factor

        _cfg = type("Config", (object,), {
            "n_embd": embed_dim, "n_head": n_layers, "bias": True,
            "dropout": dropout, "use_qk_norm": use_qk_norm})
        self.middle_block = Block(config=_cfg)
        self.extra_blocks = [Block(config=_cfg) for _ in range(max(0, int(dec_layers) - 1))]

        self.output_layer = nn.Linear(embed_dim, in_vocab_size, bias=False)
        self.ln_f = LayerNorm(embed_dim)

        self.up = None
        if int(compression_factor) > 1:
            f = int(compression_factor)
            self.up = _diag_conv_transpose1d(embed_dim, f, f)

    def __call__(self, z):
        if self.up is not None:
            z = self.up(z)
        if self.wpe is not None:
            z = z + self.wpe(mx.arange(z.shape[1], dtype=mx.int32))
        x = self.middle_block(z)
        for blk in self.extra_blocks:
            x = blk(x)
        x = self.ln_f(x)
        x = self.output_layer(x)
        return x

