"""Timing: MLX (GPU) vs PyTorch (CPU) for an xsum-sized generator, random weights.

    python -m mlx_ncm.tests.bench_vs_torch

The model is built from the xsum eval config's shapes (9 levels, 768 dims, 12
layers, ~102M parameters) with *random* weights, so no download is needed and the
measurement isolates the port's runtime. PyTorch runs on CPU here (no CUDA on
Apple silicon); the paper's timings come from CUDA GPUs, so use this only as a
same-machine baseline for the MLX port.
"""

import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

LEVELS = [4, 6, 10, 14, 20, 26, 32, 40, 48]   # xsum levels
EMBED = 384
CTX = 48
PREFIX = 128          # shortened from 512 to keep the torch CPU run short
N_EMBD, N_HEAD, N_LAYER = 768, 12, 12
B = 4


vq_cfg = OmegaConf.create({
    "clss": "multiscale_residual_vector_quantizer",
    "decay": 0.99, "epsilon": 1e-5, "commitment_cost": 0.25,
    "learned_l1_sampling": True, "learned_l1_cascade": True,
    "learned_l1_cascade_base": 4, "learned_l1_cascade_norm": True,
    "learned_l1_cascade_norm_final": True, "learned_all_sampling": False,
    "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
    "aux": {"fine_drop": {"prob": 0.0, "min_keep": 1}},
    "levels": {"use_manual_levels": True, "manual_levels": LEVELS},
})
VQ_KW = dict(in_vocab_size=50304, context_length=CTX, embed_dim=EMBED,
             vqvae_vocab_size=[4096] * len(LEVELS), n_layers=6,
             vector_quantizer_config=vq_cfg, compression_factor=1,
             pre_quant_groupnorm=8, dropout=0.0, dec_layers=1, use_qk_norm=True,
             pre_quant_norm_type="group", pre_quant_dropout=0.0)
MODEL_CFG = OmegaConf.create({
    "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER, "bias": True,
    "dropout": 0.0, "vocab_size": [4096] * len(LEVELS),
    "attn_pattern": "block_diagonal", "use_positional_encoding": True,
    "use_level_encoding": True, "prefix_len": PREFIX, "use_rope": True,
    "rope_base": 100000, "use_qk_norm": True, "raw_token_prefix": True,
    "prefix_pad_token_id": 50257, "predict_level0": False, "query_level0": True,
    "tie_heads_to_codebook": True, "tie_heads_learn_scale": True,
    "use_flex_attention": False, "ctx_encoder": {"enabled": False},
})

B = 4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.next_concept import NextConceptTransformer as TorchNCP  # noqa: E402
from models.vqvae import VQVAE as TorchVQVAE  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)
tvq = TorchVQVAE(**VQ_KW).eval()
tnet = TorchNCP(vqvae_local=tvq, config=MODEL_CFG).eval()
sd = tnet.state_dict()

import mlx.core as mx  # noqa: E402

from mlx_ncm.models.next_concept import NextConceptTransformer as MlxNCP  # noqa: E402
from mlx_ncm.models.vqvae import VQVAE as MlxVQVAE  # noqa: E402
from mlx_ncm.weights import load_torch_state_dict  # noqa: E402

mx.random.seed(0)
mvq = MlxVQVAE(**VQ_KW)
load_torch_state_dict(mvq, tvq.state_dict(), strict=True)
mnet = MlxNCP(vqvae_local=mvq, config=MODEL_CFG)
load_torch_state_dict(mnet, sd, strict=True)
n_params = sum(v.size for v in __import__("mlx_ncm.ops", fromlist=["x"]).flatten_params(mnet).values())
print(f"generator params: {n_params/1e6:.1f}M   prefix={PREFIX}  batch={B}  levels={len(LEVELS)}")

prefix_np = np.random.RandomState(0).randint(0, 50304, (B, PREFIX)).astype(np.int64)
prefix_t = torch.from_numpy(prefix_np)
prefix_m = mx.array(prefix_np.astype(np.int32))


def bench(fn, repeat=3):
    fn()  # warm-up (Metal kernel compilation / caches)
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, mx.array):
            mx.eval(out)
        best = min(best, time.perf_counter() - t0)
    return best, out


t_torch, gen_t = bench(lambda: tnet.generate(None, prefix_t, temperature=0.0))
print(f"torch  CPU fp32 : {t_torch:6.2f}s")
t_mlx, gen_m = bench(lambda: mnet.generate(None, prefix_m, temperature=0.0))
print(f"mlx    GPU fp32 : {t_mlx:6.2f}s  ({t_torch / t_mlx:5.1f}x vs torch CPU)")
print("same tokens:", np.array_equal(np.array(gen_t.numpy()), np.array(gen_m.tolist())))

mnet16 = MlxNCP(vqvae_local=mvq, config=MODEL_CFG)
load_torch_state_dict(mnet16, sd, strict=True)
mnet16.set_dtype(mx.float16)
t_16, gen16 = bench(lambda: mnet16.generate(None, prefix_m, temperature=0.0))
print(f"mlx    GPU fp16 : {t_16:6.2f}s  ({t_torch / t_16:5.1f}x vs torch CPU)")
print("fp16 tokens match fp32:",
      np.array_equal(np.array(gen16.tolist()), np.array(gen_m.tolist())))

