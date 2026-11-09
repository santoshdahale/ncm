"""Numerical parity between the PyTorch reference repo and the MLX port.

Requires the *reference* implementation, i.e. run it from a checkout that still
has ``models/`` plus a torch install::

    python -m mlx_ncm.tests.parity_torch

It builds a small VQ-VAE + generator in PyTorch, loads that ``state_dict`` into
the MLX model (through ``mlx_ncm.weights``), and compares encoder/decoder
outputs, quantizer indices, logits, greedy/sampled generation, the KV-cache path
and classifier-free guidance. Every check asserts, so a non-zero exit means the
port drifted from the reference.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from omegaconf import OmegaConf

LEVELS = [4, 6, 8]
EMBED = 16
CTX = 8
PREFIX = 6
ATOL = 2e-5  # float32 accumulation differences only

VQ_KW = dict(in_vocab_size=16, context_length=CTX, embed_dim=EMBED,
             vqvae_vocab_size=[8, 8, 8], n_layers=2, compression_factor=1,
             pre_quant_groupnorm=4, dropout=0.0, dec_layers=1, use_qk_norm=True,
             pre_quant_norm_type="group", pre_quant_dropout=0.0,
             vector_quantizer_config=OmegaConf.create({
                 "clss": "multiscale_residual_vector_quantizer",
                 "decay": 0.99, "epsilon": 1e-5, "commitment_cost": 0.25,
                 "learned_l1_sampling": True, "learned_l1_cascade": True,
                 "learned_l1_cascade_base": 2, "learned_l1_cascade_norm": True,
                 "learned_l1_cascade_norm_final": True, "learned_all_sampling": False,
                 "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0,
                                "learnable_ratio": False},
                 "aux": {"fine_drop": {"prob": 0.0, "min_keep": 1}},
                 "levels": {"use_manual_levels": True, "manual_levels": LEVELS},
             }))


def model_cfg(query_level0):
    return OmegaConf.create({
        "n_embd": 32, "n_head": 4, "n_layer": 2, "bias": True, "dropout": 0.0,
        "vocab_size": [8] * (len(LEVELS) if query_level0 else len(LEVELS) - 1),
        "attn_pattern": "block_diagonal", "use_positional_encoding": True,
        "use_level_encoding": True, "prefix_len": PREFIX, "use_rope": True,
        "rope_base": 10000, "use_qk_norm": True, "raw_token_prefix": True,
        "prefix_pad_token_id": 5, "predict_level0": False,
        "query_level0": query_level0, "tie_heads_to_codebook": True,
        "tie_heads_learn_scale": True, "use_flex_attention": False,
        "ctx_encoder": {"enabled": False},
    })


def _npc(t):
    return np.array(t.detach().cpu().numpy())


def _mx(a):
    return np.array(a.tolist())


def check(name, a, b, atol=ATOL):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    diff = float(np.max(np.abs(a - b)))
    assert diff <= atol, f"{name}: max|diff| {diff:.3e} > {atol:.1e}"
    print(f"PASS {name}  (max|diff| {diff:.3e})")


def check_exact(name, a, b):
    assert np.array_equal(np.asarray(a), np.asarray(b)), f"{name}: values differ"
    print(f"PASS {name} (exact)")


def main():
    import mlx.core as mx

    from mlx_ncm.models.next_concept import NextConceptTransformer as MlxNCP
    from mlx_ncm.models.vqvae import VQVAE as MlxVQVAE
    from mlx_ncm.weights import load_torch_state_dict

    try:
        from models.next_concept import NextConceptTransformer as TorchNCP
        from models.vqvae import VQVAE as TorchVQVAE
    except ImportError as exc:
        print(f"reference implementation not importable ({exc}); "
              "run this from a checkout that contains models/")
        return 2

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    tvq = TorchVQVAE(**VQ_KW).eval()
    mx.random.seed(0)
    mvq = MlxVQVAE(**VQ_KW)
    load_torch_state_dict(mvq, tvq.state_dict(), strict=True)
    print("PASS VQVAE state_dict -> MLX (strict)")

    ids_np = np.random.RandomState(3).randint(0, 16, (2, CTX)).astype(np.int64)
    ids_t = torch.from_numpy(ids_np)
    ids_m = mx.array(ids_np.astype(np.int32))

    check("encode_z", _npc(tvq.encode_z(ids_t)), _mx(mvq.encode_z(ids_m)))

    idx_t = _npc(tvq.encode(ids_t)).astype(np.int64)
    idx_m = np.array(mvq.encode(ids_m).tolist()).astype(np.int64)
    check_exact("encode indices", idx_t, idx_m)
    idx_t_mx = torch.from_numpy(idx_t)
    idx_m_mx = mx.array(idx_m.astype(np.int32))

    check("decode logits", _npc(tvq.decode(idx_t_mx)), _mx(mvq.decode(idx_m_mx)))
    check("idx_to_z_q", _npc(tvq.vq.idx_to_z_q(idx_t_mx)), _mx(mvq.vq.idx_to_z_q(idx_m_mx)))
    check("idx_to_ncm_input", _npc(tvq.vq.idx_to_ncm_input(idx_t_mx)),
          _mx(mvq.vq.idx_to_ncm_input(idx_m_mx)))

    _, zl_t = tvq.vq.idx_to_zhats(idx_t_mx, return_zlevels=True)
    _, zl_m = mvq.vq.idx_to_zhats(idx_m_mx, return_zlevels=True)
    for k, (a, b) in enumerate(zip(zl_t, zl_m)):
        check(f"idx_to_zhats[{k}]", _npc(a), _mx(b))

    ncm_t = _npc(tvq.vq.idx_to_ncm_input(idx_t_mx))
    prefix_np = np.random.RandomState(7).randint(0, 16, (2, PREFIX)).astype(np.int64)
    zpre_t = _npc(tvq.encode_z(torch.from_numpy(prefix_np)))[:, :PREFIX, :]
    l0_np = np.random.RandomState(8).randint(0, 8, (2, LEVELS[0])).astype(np.int64)
    padded = np.random.RandomState(9).randint(0, 16, (2, PREFIX)).astype(np.int64)
    padded[1, :2] = 5

    for query_level0 in (False, True):
        tag = f"query_level0={query_level0}"
        torch.manual_seed(1)
        tnet = TorchNCP(vqvae_local=tvq, config=model_cfg(query_level0)).eval()
        mx.random.seed(1)
        mnet = MlxNCP(vqvae_local=mvq, config=model_cfg(query_level0))
        load_torch_state_dict(mnet, tnet.state_dict(), strict=True)
        print(f"PASS generator state_dict -> MLX ({tag}, strict)")

        target = (np.random.RandomState(11).randn(2, sum(LEVELS), EMBED).astype(np.float32) * 0.5
                  if query_level0 else ncm_t)
        x_np = np.concatenate([zpre_t, target], axis=1)
        check(f"forward logits [{tag}]", _npc(tnet(torch.from_numpy(x_np))),
              _mx(mnet(mx.array(x_np.astype(np.float32)))))

        idx_arg_t = None if query_level0 else torch.from_numpy(l0_np)
        idx_arg_m = None if query_level0 else mx.array(l0_np.astype(np.int32))
        gen_t = _npc(tnet.generate(idx_arg_t, torch.from_numpy(prefix_np),
                                   temperature=0.0)).astype(np.int64)
        gen_m = np.array(mnet.generate(idx_arg_m, mx.array(prefix_np.astype(np.int32)),
                                       temperature=0.0).tolist()).astype(np.int64)
        check_exact(f"greedy generation [{tag}]", gen_t, gen_m)

        if not query_level0:
            gen_kv = np.array(mnet.generate(
                idx_arg_m, mx.array(prefix_np.astype(np.int32)),
                temperature=0.0, use_kv_cache=True).tolist()).astype(np.int64)
            check_exact("kv-cache generation == eager", gen_t, gen_kv)

        cfg_t = _npc(tnet.generate(idx_arg_t, torch.from_numpy(padded), temperature=0.0,
                                   cfg_scale=1.0, cfg_null_eos=5)).astype(np.int64)
        cfg_m = np.array(mnet.generate(idx_arg_m, mx.array(padded.astype(np.int32)),
                                       temperature=0.0, cfg_scale=1.0,
                                       cfg_null_eos=5).tolist()).astype(np.int64)
        check_exact(f"cfg generation [{tag}]", cfg_t, cfg_m)

    print("\nALL PARITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

