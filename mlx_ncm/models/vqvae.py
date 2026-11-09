"""MLX port of ``models/vqvae.py``."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..registry import vector_quantizers
from ..utils import count_params, import_class
from .vae import Decoder, Encoder


class VQVAE(nn.Module):
    def __init__(
        self,
        in_vocab_size,
        context_length,
        embed_dim,
        vqvae_vocab_size,
        n_layers,
        vector_quantizer_config,
        compression_factor=4,
        dropout=0.1,
        pre_quant_groupnorm=None,
        pre_quant_dropout=None,
        use_qk_norm=False,
        enc_layers=0,
        enc_attn_window=0,
        enc_conv_layers=0,
        enc_conv_kernel=3,
        dec_layers=1,
        pre_quant_norm_type='group',
        enc_positional=True,
        dec_positional=False,
        **kwargs
    ):
        super().__init__()
        _ = kwargs
        self.embed_dim = embed_dim
        self.vqvae_vocab_size = vqvae_vocab_size
        self.in_vocab_size = in_vocab_size
        self.context_length = context_length

        # enc_layers=0 / dec_layers=1 reproduce the original architecture: encoder = wte+wpe lookup, decoder = one Block.
        self.encoder = Encoder(
            in_vocab_size, context_length, embed_dim, n_layers, compression_factor=compression_factor,
            dropout=dropout, enc_layers=enc_layers, use_qk_norm=use_qk_norm,
            enc_attn_window=enc_attn_window,
            enc_conv_layers=enc_conv_layers, enc_conv_kernel=enc_conv_kernel,
            use_pos_emb=enc_positional,
        )

        vq = import_class(vector_quantizers[vector_quantizer_config.clss])
        self.vq = vq.from_config({
            **{'vqvae_vocab_size': vqvae_vocab_size, 'embed_dim': embed_dim},
            **dict(vector_quantizer_config)})

        self.decoder = Decoder(
            in_vocab_size, embed_dim, n_layers, compression_factor=compression_factor,
            dropout=dropout, use_qk_norm=use_qk_norm, dec_layers=dec_layers,
            use_pos_emb=dec_positional, context_length=context_length,
        )

        # affine=False: ||z-e||^2 is not scale-invariant, so a learnable gamma would inflate without bound.
        if not pre_quant_groupnorm:
            self.pre_quant_norm = None
        elif str(pre_quant_norm_type) == 'layer':
            self.pre_quant_norm = nn.LayerNorm(embed_dim, affine=False)
            self._pre_quant_is_layer = True
        else:
            # ``pytorch_compatible=True`` keeps PyTorch's (B, C, L) convention.
            self.pre_quant_norm = nn.GroupNorm(
                int(pre_quant_groupnorm), embed_dim, affine=False, pytorch_compatible=True)
            self._pre_quant_is_layer = False

        self.pre_quant_drop = (
            nn.Dropout(float(pre_quant_dropout)) if pre_quant_dropout else None
        )

        self.context_length = context_length

        total_params, non_embd_params = count_params(self, exclude_params=['wte', 'wpe'])

        print("VQVAE, Total parameter count: %.2fM" % (total_params / 1e6,))
        print("VQVAE, Non embedding parameter count: %.2fM" % (non_embd_params / 1e6,))

    def _encode_pre_quant(self, x):
        z = self.encoder(x)
        if self.pre_quant_norm is not None:
            z = self.pre_quant_norm(z)
        if self.pre_quant_drop is not None:
            z = self.pre_quant_drop(z)
        return z

    def __call__(self, x):
        z = self._encode_pre_quant(x)
        z_q, vq_loss, per_level_codebook_util, global_codebook_util, ppl = self.vq(z)
        x = self.decoder(z_q)
        return x, vq_loss, per_level_codebook_util, global_codebook_util, ppl

    def encode(self, x):
        z = self._encode_pre_quant(x)
        idx = self.vq.z_to_idx(z)
        return idx

    def encode_z(self, x):
        return self._encode_pre_quant(x)

    def decode(self, idx):
        z_q = self.vq.idx_to_z_q(idx)
        x = self.decoder(z_q)
        return x

    def reconstruct(self, x):
        z = self._encode_pre_quant(x)
        z_q, _, _, _, _ = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon

    def get_ncm_inputs(self, pre_tokenized_ids):
        idx = self.encode(pre_tokenized_ids)
        x = self.vq.idx_to_ncm_input(idx)
        return x, idx[:, self.vq.levels[0]:]

    def reconstruct_multiscales(self, idx_pre_tokenized):
        B = idx_pre_tokenized.shape[0]
        z = self._encode_pre_quant(idx_pre_tokenized)
        _, multiscale_zqs = self.vq.z_to_idx(z, return_zlevels=True)
        reconstructions = []
        for z_q in multiscale_zqs:
            logits = self.decoder(z_q)
            reconstructions.append(mx.argmax(logits, axis=-1))

        reconstructions_reordered = [[] for _ in range(B)]
        for i in range(len(reconstructions)):
            for j in range(B):
                reconstructions_reordered[j].append(reconstructions[i][j].tolist())
        return reconstructions_reordered
