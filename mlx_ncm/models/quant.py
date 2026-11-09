"""MLX port of ``models/quant.py`` (multiscale residual vector quantizer).

Inference paths (``z_to_idx``, ``idx_to_zhats``, ``idx_to_ncm_input``,
``idx_to_z_q``, ``get_next_autoregressive_input``) are used by evaluation and
are a direct translation. The training path (``__call__``) is also ported,
including EMA codebook updates, but three training-only facilities are
simplified:

* ``torch.distributed`` EMA rescaling is a no-op (single device).
* ``torch.rand``/``torch.randint`` use MLX's global random state.
* gradients: MLX does not record them unless the caller asks for them, so the
  ``no_grad`` decorators of the reference are simply unnecessary here.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..ops import conv1d_bcl, conv_transpose1d_bcl, interpolate1d
from ..registry import level_schedulers
from .containers import Sequential, TorchIndexed


# Phi (quant_resi): per-level residual conv after upsample, blending (1-r)*h + r*Conv1d(h).
class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi, learnable_ratio=False):
        ks = 3
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=ks,
            stride=1,
            padding=ks // 2,
        )
        init_ratio = abs(quant_resi)
        self._learnable_ratio = bool(learnable_ratio)
        if self._learnable_ratio:
            r_clamped = min(max(init_ratio, 1e-4), 1 - 1e-4)
            init_logit = math.log(r_clamped / (1 - r_clamped))
            self.ratio_logit = mx.array(init_logit, dtype=mx.float32)
        else:
            self._ratio_const = init_ratio

    @property
    def resi_ratio(self):
        if self._learnable_ratio:
            return mx.sigmoid(self.ratio_logit)
        return self._ratio_const

    def __call__(self, h_BCL):
        r = self.resi_ratio
        # ``conv1d_bcl(self, ...)`` would re-enter this method; call the layer directly.
        h_last = h_BCL.transpose(0, 2, 1)
        y = nn.Conv1d.__call__(self, h_last).transpose(0, 2, 1)
        return h_BCL * (1 - r) + y * r


class PhiConvStack(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, hidden_mult=1):
        super().__init__()
        ks = kernel_size
        pad = ks // 2
        h = embed_dim * hidden_mult
        self.c1 = nn.Conv1d(embed_dim, h, kernel_size=ks, padding=pad)
        self.act1 = nn.GELU()
        self.c2 = nn.Conv1d(h, h, kernel_size=ks, padding=pad)
        self.act2 = nn.GELU()
        self.c3 = nn.Conv1d(h, embed_dim, kernel_size=ks, padding=pad)
        self.c3.weight = mx.zeros(self.c3.weight.shape)
        self.c3.bias = mx.zeros(self.c3.bias.shape)

    def __call__(self, h_BCL):
        h = h_BCL.transpose(0, 2, 1)
        y = self.act1(self.c1(h))
        y = self.act2(self.c2(y))
        y = self.c3(y)
        return h_BCL + y.transpose(0, 2, 1)


class PhiShared(nn.Module):
    def __init__(self, qresi):
        super().__init__()
        self.qresi = qresi

    def select(self, _at_from_0_to_1):
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls):
        super().__init__()
        self.qresi_ls = list(qresi_ls)
        K = len(self.qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def select(self, at_from_0_to_1):
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self):
        return f'ticks={self.ticks}'


class PhiNonShared(TorchIndexed):
    def __init__(self, qresi):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def select(self, at_from_0_to_1):
        # NOTE: the reference implementation selects through ``__getitem__`` with a
        # float key. MLX's ``Module.update`` walks children with ``dst[i]``, so a
        # float-keyed ``__getitem__`` would silently mis-route weight loading;
        # the tick lookup therefore lives in ``select`` and ``__getitem__`` keeps
        # its integer (torch ModuleList) meaning.
        return self._mods[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self):
        return f'ticks={self.ticks}'


def _select_quant_resi(quant_resi, at_from_0_to_1):
    """``quant_resi[at]`` in the reference implementation (``nn.Identity`` passes through)."""
    sel = getattr(quant_resi, "select", None)
    return sel(at_from_0_to_1) if sel is not None else quant_resi



def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key, default)
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return default


def _build_quant_resi(embed_dim, num_levels, quant_resi_config):
    enabled = bool(_cfg_get(quant_resi_config, 'enabled', True))
    if not enabled:
        return None
    raw_ratio = _cfg_get(quant_resi_config, 'ratio', 0.5)
    share_mode = int(_cfg_get(quant_resi_config, 'share_mode', 0))
    learnable_ratio = bool(_cfg_get(quant_resi_config, 'learnable_ratio', False))
    arch = str(_cfg_get(quant_resi_config, 'arch', 'blend'))
    hidden_mult = int(_cfg_get(quant_resi_config, 'hidden_mult', 1))
    kernel_size = int(_cfg_get(quant_resi_config, 'kernel_size', 3))

    if isinstance(raw_ratio, (list, tuple)) or (hasattr(raw_ratio, '__len__') and not isinstance(raw_ratio, str)):
        ratios_per_level = [float(r) for r in raw_ratio]
        if share_mode != 0:
            raise ValueError(
                f"per-level ratio list requires share_mode=0 (non-shared); got share_mode={share_mode}"
            )
        if len(ratios_per_level) != num_levels:
            raise ValueError(
                f"ratio list must have length {num_levels} (one per level); got {len(ratios_per_level)}"
            )
    else:
        ratios_per_level = [float(raw_ratio)] * num_levels

    def make_phi(r):
        if arch == 'convstack':
            return PhiConvStack(embed_dim, kernel_size=kernel_size, hidden_mult=hidden_mult)
        if learnable_ratio:
            return Phi(embed_dim, r, learnable_ratio=True)
        return Phi(embed_dim, r) if abs(r) > 1e-6 else nn.Identity()

    if share_mode == 0:
        return PhiNonShared([make_phi(r) for r in ratios_per_level])
    if share_mode == 1:
        return PhiShared(make_phi(ratios_per_level[0]))
    return PhiPartiallyShared([make_phi(ratios_per_level[0]) for _ in range(share_mode)])


def _make_learned_down_l1(D, stride=4):
    conv = nn.Conv1d(D, D, kernel_size=stride, stride=stride, bias=True)
    w = mx.zeros(conv.weight.shape)  # (out, k, in)
    idx = mx.arange(D)
    w[idx, :, idx] = 1.0 / stride
    conv.weight = w
    conv.bias = mx.zeros(conv.bias.shape)
    return conv


def _make_learned_up_l1(D, stride=4):
    convT = nn.ConvTranspose1d(D, D, kernel_size=stride, stride=stride, bias=True)
    w = mx.zeros(convT.weight.shape)  # (out, k, in)
    idx = mx.arange(D)
    w[idx, :, idx] = 1.0
    convT.weight = w
    convT.bias = mx.zeros(convT.bias.shape)
    return convT


def _plan_cascade_strides(total_stride, base=4):
    S = int(total_stride)
    strides = []
    while S > 1:
        f = min(base, S)
        while f > 1 and S % f != 0:
            f -= 1
        if f <= 1:
            strides.append(S)
            S = 1
        else:
            strides.append(f)
            S //= f
    return strides or [1]


def _cascade_norm(D, groups=8):
    return nn.GroupNorm(groups, D, affine=False, pytorch_compatible=True)


def _make_cascaded_down_l1(D, total_stride, base=4, inter_norm=False, final_norm=False):
    strides = _plan_cascade_strides(total_stride, base)
    mods = []
    for i, s in enumerate(strides):
        if inter_norm and i > 0:
            mods.append(_cascade_norm(D))
        mods.append(_make_learned_down_l1(D, stride=s))
    if final_norm:
        mods.append(_cascade_norm(D))
    return Sequential(*mods)


def _make_cascaded_up_l1(D, total_stride, base=4, inter_norm=False):
    strides = list(reversed(_plan_cascade_strides(total_stride, base)))
    mods = []
    for i, s in enumerate(strides):
        if inter_norm and i > 0:
            mods.append(_cascade_norm(D))
        mods.append(_make_learned_up_l1(D, stride=s))
    return Sequential(*mods)


def _make_zeroinit_residual_conv(D, kernel_size=3):
    pad = kernel_size // 2
    conv = nn.Conv1d(D, D, kernel_size=kernel_size, stride=1, padding=pad, bias=True)
    conv.weight = mx.zeros(conv.weight.shape)
    conv.bias = mx.zeros(conv.bias.shape)
    return conv


class VectorQuantizerBaseClass(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        _ = inputs
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**{k: v for k, v in config.items() if k != "clss"})


class MultiscaleResidualVectorQuantizer(VectorQuantizerBaseClass):
    def __init__(
        self,
        vqvae_vocab_size,
        embed_dim,
        levels,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        quant_resi=None,
        learned_l1_sampling=False,
        learned_all_sampling=False,
        learned_l1_cascade=False,
        learned_l1_cascade_base=4,
        learned_l1_cascade_norm=False,
        learned_l1_cascade_norm_final=False,
        restart_threshold=0.0,
        ema_scale_by_world_size=False,
        kmeans_init=False,
        kmeans_iters=10,
        pre_quant_norm_type='group',
        aux=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.ema_scale_by_world_size = bool(ema_scale_by_world_size)
        if isinstance(decay, (list, tuple)):
            self.base_decay = float(decay[0])
            self.base_decay_per_level = [float(x) for x in decay]
        else:
            self.base_decay = float(decay)
            self.base_decay_per_level = None
        # Fine-level dropout: hand the decoder a z_hat truncated at a random level >= min_keep. Train-only.
        _fd = _cfg_get(aux, 'fine_drop', None)
        self.fine_drop_prob = float(_cfg_get(_fd, 'prob', 0.0))
        self.fine_drop_min_keep = int(_cfg_get(_fd, 'min_keep', 2))
        _ps = _cfg_get(_fd, 'prob_start', None)
        self.fine_drop_prob_start = float(_ps) if _ps is not None else None
        self.fine_drop_anneal_steps = int(_cfg_get(_fd, 'anneal_steps', 0) or 0)
        self._fd_step = mx.array(0, dtype=mx.int32)
        self.learned_all_sampling = bool(learned_all_sampling)
        self.learned_l1_sampling = bool(learned_l1_sampling) or self.learned_all_sampling
        self.learned_l1_cascade = bool(learned_l1_cascade)
        self.learned_l1_cascade_base = int(learned_l1_cascade_base)
        self.learned_l1_cascade_norm = bool(learned_l1_cascade_norm)
        self.learned_l1_cascade_norm_final = bool(learned_l1_cascade_norm_final)
        self.restart_threshold = float(restart_threshold)
        self.kmeans_init = bool(kmeans_init)
        self.kmeans_iters = int(kmeans_iters)
        self._km_buf = [[] for _ in range(len(vqvae_vocab_size))] if self.kmeans_init else None
        if self.kmeans_init:
            self._km_done = [False] * len(vqvae_vocab_size)

        if levels.use_manual_levels:
            self.levels = list(levels.manual_levels)
        else:
            self.levels = level_schedulers[levels.scheduler](levels.start, levels.end, levels.num_levels)

        assert len(vqvae_vocab_size) == len(self.levels), \
            f"vqvae_vocab_size length {len(vqvae_vocab_size)} must match number of levels {len(self.levels)}"
        self.vqvae_vocab_sizes = list(vqvae_vocab_size)

        self.embeddings = [nn.Embedding(vs, embed_dim) for vs in self.vqvae_vocab_sizes]
        # Codebooks are updated by EMA, not gradients.
        for emb in self.embeddings:
            emb.weight = mx.random.normal(emb.weight.shape)
            emb.freeze()
        for i, vs in enumerate(self.vqvae_vocab_sizes):
            setattr(self, f'ema_cluster_size_{i}', mx.zeros((vs,), dtype=mx.float32))
            setattr(self, f'ema_w_{i}', mx.array(self.embeddings[i].weight))
            setattr(self, f'codebook_hit_{i}', mx.zeros((vs,), dtype=mx.int32))

        self.quant_resi = _build_quant_resi(embed_dim, len(self.levels), quant_resi)
        self.quant_resi_enabled = self.quant_resi is not None
        self._qr_num_levels = len(self.levels)

        # Learned down/upsampler at level 0 only, replacing F.interpolate; init matches area-pool / nearest-repeat.
        if self.learned_l1_sampling:
            L = self.levels[-1]
            ell_1 = self.levels[0]
            stride = L // ell_1
            assert L % ell_1 == 0 and stride * ell_1 == L, \
                f"learned_l1_sampling requires L={L} divisible by ell_1={ell_1}"
            if self.learned_l1_cascade:
                b = self.learned_l1_cascade_base
                nrm = self.learned_l1_cascade_norm
                self.learned_down_l1 = _make_cascaded_down_l1(
                    embed_dim, stride, base=b, inter_norm=nrm,
                    final_norm=self.learned_l1_cascade_norm_final)
                self.learned_up_l1 = _make_cascaded_up_l1(embed_dim, stride, base=b, inter_norm=nrm)
            else:
                self.learned_down_l1 = _make_learned_down_l1(embed_dim, stride=stride)
                self.learned_up_l1 = _make_learned_up_l1(embed_dim, stride=stride)
        else:
            self.learned_down_l1 = None
            self.learned_up_l1 = None

        if self.learned_all_sampling:
            num_levels = len(self.levels)
            self.learned_down_refine = [
                _make_zeroinit_residual_conv(embed_dim, kernel_size=3) if 1 <= i <= num_levels - 2 else nn.Identity()
                for i in range(num_levels - 1)
            ]
            self.learned_up_refine = [
                _make_zeroinit_residual_conv(embed_dim, kernel_size=3) if 1 <= i <= num_levels - 2 else nn.Identity()
                for i in range(num_levels - 1)
            ]
        else:
            self.learned_down_refine = None
            self.learned_up_refine = None





    def kmeans_init_level(self, level_idx, samples, n_iters=20):
        V = self.vqvae_vocab_sizes[level_idx]
        N = samples.shape[0]
        if N >= V:
            perm = mx.random.permutation(N)[:V]
            centers = samples.take(perm, axis=0).astype(mx.float32)
        else:
            idx = mx.random.randint(0, N, (V,))
            centers = samples.take(idx, axis=0).astype(mx.float32)
        chunk = max(1, min(N, 8192))
        for _ in range(n_iters):
            assigns = []
            for i in range(0, N, chunk):
                xs = samples[i:i + chunk].astype(mx.float32)
                d2 = ((xs ** 2).sum(1, keepdims=True)
                      + (centers ** 2).sum(1)
                      - 2 * (xs @ centers.T))
                assigns.append(mx.argmin(d2, axis=1))
            assigns = mx.concatenate(assigns)
            counts = mx.zeros((V,), dtype=mx.float32).at[assigns].add(
                mx.ones(assigns.shape, dtype=mx.float32))
            new_centers = mx.zeros(centers.shape, dtype=mx.float32).at[assigns].add(
                samples.astype(mx.float32))
            mask = counts > 0
            new_centers = mx.where(mask[:, None],
                                   new_centers / mx.maximum(counts, 1.0)[:, None],
                                   new_centers)
            empty = mx.nonzero(~mask)[0]
            if empty.size > 0:
                rand_idx = mx.random.randint(0, N, (empty.size,))
                new_centers[empty] = samples.take(rand_idx, axis=0).astype(mx.float32)
            centers = new_centers
        self.embeddings[level_idx].weight = centers.astype(self.embeddings[level_idx].weight.dtype)
        setattr(self, f'ema_cluster_size_{level_idx}', mx.ones((V,), dtype=mx.float32))
        setattr(self, f'ema_w_{level_idx}', centers)

    def _apply_quant_resi(self, h_BLC, level_idx):
        if not self.quant_resi_enabled:
            return h_BLC
        h_BCL = h_BLC.transpose(0, 2, 1)
        at = level_idx / max(self._qr_num_levels - 1, 1)
        h_BCL = _select_quant_resi(self.quant_resi, at)(h_BCL)
        return h_BCL.transpose(0, 2, 1)

    def _distances(self, flat_input, emb_w):
        return (
            mx.sum(flat_input ** 2, axis=1, keepdims=True)
            + mx.sum(emb_w ** 2, axis=1)
            - 2 * (flat_input @ emb_w.T)
        )

    def _down_for_quant(self, z_BLC, level_idx, target_len):
        # The learned cascade works in the channel-last (B, L, C) layout that
        # mlx.nn convolutions use; the interpolate fallback works on (B, C, L).
        if level_idx == 0 and self.learned_l1_sampling:
            return self.learned_down_l1(z_BLC)
        zd_BCL = interpolate1d(z_BLC.transpose(0, 2, 1), target_len, "area")
        if self.learned_all_sampling and 1 <= level_idx <= len(self.levels) - 2:
            zd_BCL = zd_BCL + conv1d_bcl(self.learned_down_refine[level_idx], zd_BCL)
        return zd_BCL.transpose(0, 2, 1)

    def _up_to_full(self, q_BLC, level_idx, target_len):
        if level_idx == 0 and self.learned_l1_sampling:
            return self.learned_up_l1(q_BLC)
        qu_BCL = interpolate1d(q_BLC.transpose(0, 2, 1), target_len, "linear")
        if self.learned_all_sampling and 1 <= level_idx <= len(self.levels) - 2:
            qu_BCL = qu_BCL + conv1d_bcl(self.learned_up_refine[level_idx], qu_BCL)
        return qu_BCL.transpose(0, 2, 1)

    def _get_ema_decay(self):
        return self._scale_decay(self.base_decay)

    def _scale_decay(self, d):
        # torch.distributed is not used by the MLX port; keep the hook for parity.
        return d

    def _get_ema_decay_level(self, i):
        if getattr(self, "base_decay_per_level", None) is None:
            return self._get_ema_decay()
        return self._scale_decay(
            self.base_decay_per_level[min(i, len(self.base_decay_per_level) - 1)])

    def _current_fine_drop_prob(self):
        if self.fine_drop_prob_start is None or self.fine_drop_anneal_steps <= 0:
            return self.fine_drop_prob
        t = float(self._fd_step.item()) / float(self.fine_drop_anneal_steps)
        t = min(max(t, 0.0), 1.0)
        return self.fine_drop_prob_start + t * (self.fine_drop_prob - self.fine_drop_prob_start)

    def _restart_dead_codes(self, flat_input, ema_cs, ema_w, emb_i):
        dead = ema_cs < self.restart_threshold
        n_dead = int(mx.sum(dead).item())
        if n_dead == 0 or flat_input.shape[0] == 0:
            return None
        src = mx.random.randint(0, flat_input.shape[0], (n_dead,))
        new_vecs = flat_input.take(src, axis=0).astype(emb_i.weight.dtype)
        positions = mx.nonzero(dead)[0]
        w = mx.array(emb_i.weight)
        e = mx.array(ema_w)
        w[positions] = new_vecs
        e[positions] = new_vecs
        cs = mx.where(dead, mx.full(ema_cs.shape, self.restart_threshold), ema_cs)
        return w, e, cs


    # ------------------------------------------------------------------
    # Training path (EMA codebook updates, straight-through, fine-drop)
    # ------------------------------------------------------------------
    def __call__(self, z):
        z = z.astype(mx.float32)
        B, L, C = z.shape
        z_no_grad = mx.stop_gradient(z)

        z_rest = mx.array(z_no_grad)
        z_hat = mx.zeros(z_rest.shape, dtype=z_rest.dtype)

        z_hat_keep = []
        _fd_p = self._current_fine_drop_prob()
        if self.training:
            self._fd_step = self._fd_step + 1
        do_fine_drop = (self.training and _fd_p > 0.0
                        and float(mx.random.uniform().item()) < _fd_p)
        track_cum = do_fine_drop

        vq_loss = mx.array(0.0)
        num_levels = len(self.levels)
        decay = self._get_ema_decay()

        for i, level in enumerate(self.levels):
            emb_i = self.embeddings[i]
            vs_i = self.vqvae_vocab_sizes[i]

            if i != num_levels - 1:
                flat_input_BLC = self._down_for_quant(z_rest, i, level)
            else:
                flat_input_BLC = z_rest
            flat_input = flat_input_BLC.reshape(-1, C)

            distances = self._distances(flat_input, emb_i.weight)

            indices = mx.argmin(distances, axis=1).reshape(B, level)

            encodings = mx.zeros((B * level, vs_i), dtype=z.dtype)
            encodings = encodings.at[mx.arange(B * level), indices.reshape(-1)].add(1.0)

            if self.training:
                decay = self._get_ema_decay_level(i)
                ema_cs = getattr(self, f'ema_cluster_size_{i}')
                ema_w = getattr(self, f'ema_w_{i}')
                encodings_sum = mx.sum(encodings, axis=0)
                dw = encodings.T @ mx.stop_gradient(flat_input)
                ema_cs = ema_cs * decay + encodings_sum * (1 - decay)
                n = mx.sum(ema_cs)
                smoothed = (ema_cs + self.epsilon) / (n + vs_i * self.epsilon) * n
                ema_w = ema_w * decay + dw * (1 - decay)
                emb_i.weight = ema_w / smoothed[:, None]
                setattr(self, f'ema_cluster_size_{i}', ema_cs)
                setattr(self, f'ema_w_{i}', ema_w)
                if self.restart_threshold > 0:
                    res = self._restart_dead_codes(
                        mx.stop_gradient(flat_input), ema_cs, ema_w, emb_i)
                    if res is not None:
                        w, e, cs = res
                        emb_i.weight = w
                        setattr(self, f'ema_w_{i}', e)
                        setattr(self, f'ema_cluster_size_{i}', cs)

            q_BLC = emb_i(indices)
            # Straight-through estimator at level 0 only; at higher levels it diverges within ~100 steps.
            if i == 0 and self.learned_l1_sampling:
                q_BLC = flat_input_BLC + mx.stop_gradient(q_BLC - flat_input_BLC)

            if i != num_levels - 1:
                quantized = self._up_to_full(q_BLC, i, L)
            else:
                quantized = q_BLC

            quantized = self._apply_quant_resi(quantized, i)

            z_hat = z_hat + quantized
            if track_cum:
                z_hat_keep.append(z_hat)
            z_rest = z_rest - quantized

            _z_hat_const = mx.stop_gradient(z_hat)
            vq_loss = vq_loss + self.commitment_cost * mx.mean((_z_hat_const - z) ** 2)
            vq_loss = vq_loss + mx.mean((z_hat - z_no_grad) ** 2)

            hit = mx.array(getattr(self, f'codebook_hit_{i}'))
            hit[indices.reshape(-1)] = 1
            setattr(self, f'codebook_hit_{i}', hit)

        vq_loss = vq_loss / num_levels
        if do_fine_drop and len(z_hat_keep) >= 2:
            kmin = max(self.fine_drop_min_keep - 1, 0)
            kmax = len(z_hat_keep) - 2
            if kmax >= kmin:
                K = int(mx.random.randint(kmin, kmax + 1, (1,)).item())
                z_hat = z_hat_keep[K]
        z_hat = z + mx.stop_gradient(z_hat - z)

        per_level_codebook_utilization = [
            mx.sum(getattr(self, f'codebook_hit_{i}')).item() / self.vqvae_vocab_sizes[i]
            for i in range(num_levels)
        ]
        total_slots = sum(self.vqvae_vocab_sizes)
        used_slots = sum(
            mx.sum(getattr(self, f'codebook_hit_{i}')).item() for i in range(num_levels)
        )
        global_codebook_utilization = used_slots / total_slots

        return (
            z_hat,
            vq_loss,
            per_level_codebook_utilization,
            global_codebook_utilization,
            0.1
        )

    def set_stochastic_encode(self, enabled, temperature=0.5, top_k=10, top_p=0.95):
        self._stoch_encode = bool(enabled)
        self._stoch_temp = float(temperature)
        self._stoch_top_k = int(top_k)
        self._stoch_top_p = float(top_p)

    def _sample_codes(self, distances):
        logits = -distances / self._stoch_temp
        k = self._stoch_top_k
        if k and k > 0:
            kth = mx.topk(logits, min(k, logits.shape[-1]), axis=-1)[:, -1:]
            logits = mx.where(logits < kth, mx.full(logits.shape, -float('inf')), logits)
        p = self._stoch_top_p
        if p and p < 1.0:
            order = mx.argsort(-logits, axis=-1)
            sl = mx.take_along_axis(logits, order, axis=-1)
            probs = mx.softmax(sl, axis=-1)
            remove = (mx.cumsum(probs, axis=-1) - probs) > p
            sl = mx.where(remove, mx.full(sl.shape, -float('inf')), sl)
            scattered = mx.zeros(logits.shape).at[
                mx.arange(logits.shape[0])[:, None], order].add(sl)
            logits = scattered
        return mx.random.categorical(logits.astype(mx.float32))

    # ------------------------------------------------------------------
    # Inference / evaluation paths
    # ------------------------------------------------------------------
    def z_to_idx(self, z, return_zhats=False, return_zlevels=False):
        z = z.astype(mx.float32)
        B, L, C = z.shape
        z_no_grad = mx.stop_gradient(z)

        z_rest = mx.array(z_no_grad)
        z_hat = mx.zeros(z_rest.shape, dtype=z_rest.dtype)

        z_hats = []
        z_levels = []
        idx = []

        num_levels = len(self.levels)

        for i, level in enumerate(self.levels):
            emb_i = self.embeddings[i]

            if i != num_levels - 1:
                flat_input = self._down_for_quant(z_rest, i, level).reshape(-1, C)
            else:
                flat_input = z_rest.reshape(-1, C)

            distances = self._distances(flat_input, emb_i.weight)

            if getattr(self, '_stoch_encode', False):
                indices = self._sample_codes(distances).reshape(B, level)
            else:
                indices = mx.argmin(distances, axis=1).reshape(B, level)
            idx.append(indices)

            if i != num_levels - 1:
                quantized = self._up_to_full(emb_i(indices), i, L)
            else:
                quantized = emb_i(indices)
            quantized = self._apply_quant_resi(quantized, i)

            z_hat = z_hat + quantized
            z_rest = z_rest - quantized
            z_hats.append(z_hat)
            z_levels.append(
                interpolate1d(z_hat.transpose(0, 2, 1), level, "area").transpose(0, 2, 1)
                if i != num_levels - 1
                else z_hat
            )

        idx = mx.concatenate(idx, axis=-1)

        response = (idx,)

        if return_zhats:
            response = response + (z_hats,)
        if return_zlevels:
            response = response + (z_levels,)

        if not (return_zhats or return_zlevels):
            return idx
        return response

    def idx_to_zhats(self, idx, return_zlevels=True):
        z_hats = []
        z_levels = []
        z_hat = mx.zeros((idx.shape[0], self.levels[-1], self.embed_dim),
                         dtype=mx.float32)

        level_accum = 0
        for level_idx in range(len(self.levels)):
            emb_i = self.embeddings[level_idx]
            emb_at_level = emb_i(idx[:, level_accum:level_accum + self.levels[level_idx]])
            if level_idx != len(self.levels) - 1:
                z_q_level = self._up_to_full(emb_at_level, level_idx, self.levels[-1])
            else:
                z_q_level = emb_at_level
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat = z_hat + z_q_level
            z_hats.append(z_hat)
            if return_zlevels:
                z_levels.append(
                    interpolate1d(z_hat.transpose(0, 2, 1), self.levels[level_idx], "area").transpose(0, 2, 1)
                    if level_idx != len(self.levels) - 1
                    else z_hat
                )
            level_accum += self.levels[level_idx]

        results = (z_hats,)
        if return_zlevels:
            results = results + (z_levels,)
        return results

    def idx_to_ncm_input(self, idx):
        next_concepts = []
        z_hat = mx.zeros((idx.shape[0], self.levels[-1], self.embed_dim),
                         dtype=mx.float32)
        n_levels = len(self.levels)

        level_accum = 0
        for level_idx in range(n_levels - 1):
            emb_i = self.embeddings[level_idx]
            emb_at_level = emb_i(idx[:, level_accum:level_accum + self.levels[level_idx]])
            z_q_level = self._up_to_full(emb_at_level, level_idx, self.levels[-1])
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat = z_hat + z_q_level
            next_concepts.append(
                interpolate1d(z_hat.transpose(0, 2, 1), self.levels[level_idx + 1], "area").transpose(0, 2, 1)
            )
            level_accum += self.levels[level_idx]

        return mx.concatenate(next_concepts, axis=1)

    def get_next_autoregressive_input(self, level_idx, z_hat, z_q_level):
        n_levels = len(self.levels)
        if level_idx != n_levels - 1:
            upsampled = self._up_to_full(z_q_level, level_idx, self.levels[-1])
            upsampled = self._apply_quant_resi(upsampled, level_idx)
            z_hat = z_hat + upsampled
            return z_hat, interpolate1d(
                z_hat.transpose(0, 2, 1), self.levels[level_idx + 1], "area"
            ).transpose(0, 2, 1)
        z_q_level = self._apply_quant_resi(z_q_level, level_idx)
        z_hat = z_hat + z_q_level
        return z_hat, z_hat

    def idx_to_z_q(self, indices):
        z_hat = mx.zeros((indices.shape[0], self.levels[-1], self.embed_dim),
                         dtype=mx.float32)

        level_accum = 0
        for level_idx in range(len(self.levels)):
            emb_i = self.embeddings[level_idx]
            emb_at_level = emb_i(indices[:, level_accum:level_accum + self.levels[level_idx]])
            if level_idx != len(self.levels) - 1:
                z_q_level = self._up_to_full(emb_at_level, level_idx, self.levels[-1])
            else:
                z_q_level = emb_at_level
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat = z_hat + z_q_level
            level_accum += self.levels[level_idx]
        return z_hat

