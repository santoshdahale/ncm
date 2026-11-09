"""Self-contained checks for the MLX port.

Run with either::

    python -m mlx_ncm.tests.test_port            # plain asserts
    python -m pytest mlx_ncm/tests -q

No PyTorch is required: the layout tests synthesise a *PyTorch-layout* state
dict from the MLX parameters and check that conversion reproduces them.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from omegaconf import OmegaConf

from ..models.containers import Sequential
from ..models.next_concept import NextConceptTransformer
from ..models.quant import MultiscaleResidualVectorQuantizer
from ..models.vqvae import VQVAE
from ..ops import area_pool1d, flatten_params, linear_upsample1d
from ..weights import apply_weights, convert_state_dict, is_derived, load_module_weights

LEVELS = (4, 6, 8)
EMBED = 16


def vq_config(levels=LEVELS):
    return OmegaConf.create({
        "clss": "multiscale_residual_vector_quantizer",
        "decay": 0.99,
        "epsilon": 1e-5,
        "commitment_cost": 0.25,
        "learned_l1_sampling": True,
        "learned_l1_cascade": True,
        "learned_l1_cascade_base": 2,
        "learned_l1_cascade_norm": True,
        "learned_l1_cascade_norm_final": True,
        "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0,
                       "learnable_ratio": False},
        "aux": {"fine_drop": {"prob": 0.0, "min_keep": 1}},
        "levels": {"use_manual_levels": True, "manual_levels": list(levels)},
    })


def build_vqvae(levels=LEVELS, context_length=8, vocab=16, embed_dim=EMBED):
    return VQVAE(
        in_vocab_size=vocab, context_length=context_length, embed_dim=embed_dim,
        vqvae_vocab_size=[8, 8, 8], n_layers=2, vector_quantizer_config=vq_config(levels),
        compression_factor=1, pre_quant_groupnorm=4, dropout=0.0, dec_layers=1,
    )


def model_config(levels=LEVELS, query_level0=False, **overrides):
    n_heads = len(levels) if query_level0 else len(levels) - 1
    cfg = {
        "n_embd": 32, "n_head": 4, "n_layer": 2, "bias": True, "dropout": 0.0,
        "vocab_size": [8] * n_heads,
        "attn_pattern": "block_diagonal",
        "use_positional_encoding": True, "use_level_encoding": True,
        "prefix_len": 6, "use_rope": True, "rope_base": 10000, "use_qk_norm": True,
        "raw_token_prefix": True, "prefix_pad_token_id": 5,
        "predict_level0": False, "query_level0": query_level0,
        "tie_heads_to_codebook": True, "tie_heads_learn_scale": True,
        "use_flex_attention": False,
        "ctx_encoder": {"enabled": False},
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _torch_adaptive_pool_ref(x, out_len):
    """Reference for torch's adaptive_avg_pool1d on a (1, 1, L) array."""
    in_len = x.shape[-1]
    out = np.empty(out_len, dtype=np.float64)
    for a in range(out_len):
        start = (a * in_len) // out_len
        end = -((-(a + 1) * in_len) // out_len)
        out[a] = x[0, 0, start:end].mean()
    return out


def _torch_linear_ref(x, out_len):
    """Reference for torch's interpolate(mode='linear', align_corners=False)."""
    in_len = x.shape[-1]
    scale = in_len / out_len
    out = np.empty(out_len, dtype=np.float64)
    for d in range(out_len):
        src = max(scale * (d + 0.5) - 0.5, 0.0)
        i0 = int(np.floor(src))
        w1 = src - i0
        i1 = i0 + 1 if i0 < in_len - 1 else i0
        out[d] = (1 - w1) * x[0, 0, i0] + w1 * x[0, 0, i1]
    return out


def test_area_pool_matches_pytorch():
    for in_len, out_len in ((8, 4), (48, 40), (89, 67), (64, 11), (7, 13)):
        data = (np.arange(in_len, dtype=np.float32) ** 1.5).reshape(1, 1, -1)
        got = np.array(area_pool1d(mx.array(data), out_len).tolist()).reshape(-1)
        np.testing.assert_allclose(got, _torch_adaptive_pool_ref(data, out_len), atol=1e-5)


def test_linear_upsample_matches_pytorch():
    for in_len, out_len in ((4, 8), (6, 8), (11, 64), (40, 48)):
        data = (np.arange(in_len, dtype=np.float32) ** 2).reshape(1, 1, -1)
        got = np.array(linear_upsample1d(mx.array(data), out_len).tolist()).reshape(-1)
        np.testing.assert_allclose(got, _torch_linear_ref(data, out_len),
                                   rtol=1e-5, atol=1e-3)


def test_roundtrip_sizes_are_identity():
    data = mx.array(np.random.RandomState(0).randn(1, 3, 9).astype(np.float32))
    assert np.array_equal(np.array(area_pool1d(data, 9).tolist()), np.array(data.tolist()))
    assert np.array_equal(np.array(linear_upsample1d(data, 9).tolist()), np.array(data.tolist()))

def test_container_naming_matches_torch():
    seq = Sequential(nn.Conv1d(4, 4, 3), nn.Conv1d(4, 4, 3))
    names = sorted(flatten_params(seq))
    assert names == ["0.bias", "0.weight", "1.bias", "1.weight"], names


def test_quant_resi_naming_matches_torch():
    cfg = {k: v for k, v in vq_config().items() if k != "clss"}
    vq = MultiscaleResidualVectorQuantizer(
        vqvae_vocab_size=[8, 8, 8], embed_dim=EMBED, **cfg)
    names = sorted(flatten_params(vq))
    assert any(n.startswith("quant_resi.0.") for n in names), names
    assert any(n.startswith("learned_down_l1.0.") for n in names), names
    assert any(n.startswith("learned_up_l1.0.") for n in names), names



def _torch_layout_state_dict(model):
    """Synthesise a PyTorch-layout state dict from an MLX model.

    Applies the *inverse* of the torch->mlx permutation that ``weights.py`` uses.
    """
    modules = dict(model.named_modules())
    out = {}
    for name, value in flatten_params(model).items():
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        mod = modules.get(owner)
        arr = np.array(value)
        perm = None
        if name.endswith("weight"):
            if isinstance(mod, nn.ConvTranspose1d):
                perm = (1, 2, 0)
            elif isinstance(mod, nn.Conv1d):
                perm = (0, 2, 1)
        if perm is not None:
            arr = arr.transpose(np.argsort(perm))
        out[name] = arr
    return out



def test_conv_weight_permutation_roundtrip():
    model = nn.Module()
    model.conv = nn.Conv1d(4, 6, 3, padding=1)
    model.up = nn.ConvTranspose1d(6, 6, 4, stride=4)
    model.tied = nn.Conv1d(4, 4, 3)  # square: permutation must not be guessed from shape

    reference = flatten_params(model)
    torch_sd = _torch_layout_state_dict(model)
    assert torch_sd["conv.weight"].shape == (6, 4, 3)
    assert torch_sd["up.weight"].shape == (6, 6, 4)

    converted, mismatches = convert_state_dict(torch_sd, model)
    assert not mismatches, mismatches
    for name, value in reference.items():
        np.testing.assert_array_equal(np.array(converted[name].tolist()),
                                      np.array(value.tolist()))


def test_model_state_dict_roundtrip_and_strict_load():
    mx.random.seed(0)
    vqvae = build_vqvae()
    net = NextConceptTransformer(vqvae_local=vqvae, config=model_config())

    for model in (vqvae, net):
        torch_sd = _torch_layout_state_dict(model)
        converted, mismatches = convert_state_dict(torch_sd, model)
        assert not mismatches, mismatches
        missing, unexpected = apply_weights(model, converted, strict=True)
        assert not unexpected, unexpected
        assert all(is_derived(n) for n in missing), missing


def test_vqvae_roundtrip_shapes():
    mx.random.seed(1)
    vqvae = build_vqvae()
    ids = mx.array(np.random.RandomState(2).randint(0, 16, (2, 8)), dtype=mx.int32)

    z = vqvae.encode_z(ids)
    assert z.shape == (2, 8, EMBED), z.shape

    idx = vqvae.encode(ids)
    assert idx.shape == (2, sum(LEVELS)), idx.shape

    logits = vqvae.decode(idx)
    assert logits.shape == (2, 8, 16), logits.shape

    z_q = vqvae.vq.idx_to_z_q(idx)
    assert z_q.shape == (2, LEVELS[-1], EMBED), z_q.shape

    z_hats, z_levels = vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
    assert len(z_hats) == len(LEVELS) == len(z_levels)
    assert [z_.shape[1] for z_ in z_levels] == list(LEVELS)

    ncm = vqvae.vq.idx_to_ncm_input(idx)
    assert ncm.shape[1] == sum(LEVELS[1:]), ncm.shape


def test_quantizer_codes_are_nearest_neighbours():
    """z_to_idx must reproduce a brute-force argmin over the first level's codebook."""
    mx.random.seed(3)
    vq = MultiscaleResidualVectorQuantizer(
        vqvae_vocab_size=[8, 8], embed_dim=8, levels=vq_config((4, 4))["levels"],
        quant_resi={"enabled": False})
    z = mx.array(np.random.RandomState(4).randn(1, 4, 8).astype(np.float32))
    idx = vq.z_to_idx(z)

    w = np.array(vq.embeddings[0].weight.tolist())
    flat = np.array(z.tolist())[0]
    brute = np.argmin(((flat[:, None, :] - w[None, :, :]) ** 2).sum(-1), axis=1)
    np.testing.assert_array_equal(np.array(idx.tolist())[0][:4], brute)


def test_generate_shapes_and_kv_cache_parity():
    mx.random.seed(5)
    vqvae = build_vqvae()
    for query_level0 in (False, True):
        net = NextConceptTransformer(
            vqvae_local=vqvae, config=model_config(query_level0=query_level0))
        B = 2
        l0 = mx.array(np.random.RandomState(6).randint(0, 8, (B, LEVELS[0])), dtype=mx.int32)
        prefix = mx.array(np.random.RandomState(7).randint(0, 16, (B, 6)), dtype=mx.int32)
        idx_arg = None if query_level0 else l0

        out = net.generate(idx_arg, prefix, temperature=0.0)
        assert out.shape == (B, sum(LEVELS)), out.shape
        assert int(out.min()) >= 0 and int(out.max()) < 8

        if not query_level0:
            out_kv = net.generate(idx_arg, prefix, temperature=0.0, use_kv_cache=True)
            np.testing.assert_array_equal(np.array(out_kv.tolist()), np.array(out.tolist()))


def test_generate_cfg_and_prefix_padding():
    mx.random.seed(8)
    vqvae = build_vqvae()
    net = NextConceptTransformer(vqvae_local=vqvae, config=model_config(query_level0=True))
    B = 3
    base = np.random.RandomState(9).randint(0, 16, (B, 6))
    base[1, :3] = 5   # padded prefix
    base[2, :] = 5    # fully padded prefix
    prefix = mx.array(base, dtype=mx.int32)

    out = net.generate(None, prefix, temperature=0.0, cfg_scale=1.0, cfg_null_eos=5)
    assert out.shape == (B, sum(LEVELS)), out.shape
    assert np.all(np.isfinite(np.array(out.tolist(), dtype=np.float64)))


def test_forward_logits_shape():
    mx.random.seed(10)
    vqvae = build_vqvae()
    net = NextConceptTransformer(vqvae_local=vqvae, config=model_config())
    ids = mx.array(np.random.RandomState(11).randint(0, 16, (2, 8)), dtype=mx.int32)
    ncm = vqvae.vq.idx_to_ncm_input(vqvae.encode(ids))
    z_prefix = vqvae.encode_z(ids)[:, :6, :]
    logits = net(mx.concatenate([z_prefix, ncm], axis=1))
    assert logits.shape == (2, sum(LEVELS[1:]), 8), logits.shape


def test_float16_generation_runs():
    mx.random.seed(12)
    vqvae = build_vqvae()
    net = NextConceptTransformer(vqvae_local=vqvae, config=model_config())
    net.set_dtype(mx.float16)
    B = 1
    l0 = mx.array(np.random.RandomState(13).randint(0, 8, (B, LEVELS[0])), dtype=mx.int32)
    prefix = mx.array(np.random.RandomState(14).randint(0, 16, (B, 6)), dtype=mx.int32)
    out = net.generate(l0, prefix, temperature=0.0)
    assert out.shape == (B, sum(LEVELS))


def test_save_and_load_safetensors(tmp_path=None):
    """``save_weights`` must be unambiguous (no duplicate shared-module paths)."""
    import tempfile
    import os

    mx.random.seed(20)
    vqvae = build_vqvae()
    net = NextConceptTransformer(vqvae_local=vqvae, config=model_config())

    tmpdir = tempfile.mkdtemp()
    for name, model, factory in (
        ("tokenizer.safetensors", vqvae, lambda: build_vqvae()),
        ("generator.safetensors", net,
         lambda: NextConceptTransformer(vqvae_local=vqvae, config=model_config())),
    ):
        path = os.path.join(tmpdir, name)
        model.save_weights(path)
        assert os.path.exists(path), path
        reloaded = factory()
        missing, unexpected = load_module_weights(reloaded, path, strict=True)
        assert not unexpected, unexpected
        assert all(is_derived(n) for n in missing), missing

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_")]



def main():
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


