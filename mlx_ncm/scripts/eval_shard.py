"""MLX generation shard (port of ``scripts/eval_shard.py``).

Unlike the PyTorch version this does not need one process per GPU: MLX exposes a
single unified-memory device, so ``evaluate.py`` runs the shards sequentially and
``--shard/--nshards`` only exists to keep the output files comparable.
"""

from __future__ import annotations

import argparse
import os
import pickle

import mlx.core as mx
import numpy as np
from omegaconf import OmegaConf

from ..models.next_concept import NextConceptTransformer
from ..tokenization.vqvae_tokenizer import VQVAETokenizer
from ..weights import load_any


def _resolve_ckpt(ck_dir, name):
    """Resolve a config checkpoint name, preferring the MLX sibling of a .pt file."""
    stem, ext = os.path.splitext(name)
    direct = os.path.join(ck_dir, name)
    if ext in (".pt", ".pth", ".bin"):
        for mlx_ext in (".safetensors", ".npz"):
            candidate = os.path.join(ck_dir, stem + mlx_ext)
            if os.path.exists(candidate):
                return candidate
    return direct



def _walk(obj, ref):
    for key in ref.split("."):
        obj = obj[key] if isinstance(obj, dict) else getattr(obj, key)
    return obj


def _load_allowance(ck_dir, spec):
    """Load a float vector referenced as ``file:key.path`` (or a bare file name)."""
    path, _, ref = spec.partition(":")
    ref = ref or "inference.token_unigram"

    # 1. the MLX inference dump written by ``mlx_ncm.convert``
    inf_path = os.path.join(ck_dir, "inference.safetensors")
    if os.path.exists(inf_path):
        blob = dict(mx.load(inf_path))
        for key in (ref, f"inference.{ref.split('.')[-1]}", ref.split(".")[-1]):
            if key in blob:
                return blob[key].astype(mx.float32)

    # 2. any MLX file named in the spec
    mlx_path = _resolve_ckpt(ck_dir, path) if path else ""
    if mlx_path.endswith((".safetensors", ".npz")) and os.path.exists(mlx_path):
        blob = dict(mx.load(mlx_path))
        for key in (ref, ref.split(".")[-1]):
            if key in blob:
                return blob[key].astype(mx.float32)

    # 3. fall back to the PyTorch checkpoint when torch is installed
    pt_path = os.path.join(ck_dir, path) if path else ""
    if os.path.exists(pt_path) and pt_path.endswith((".pt", ".pth", ".bin")):
        import torch

        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        value = _walk(blob, ref)
        return mx.array(value.double().numpy()).astype(mx.float32)

    raise FileNotFoundError(
        f"cannot resolve {spec!r} in {ck_dir!r}: run `python -m mlx_ncm.convert` first "
        "or keep torch installed for on-the-fly conversion")


def build_parser():
    ap = argparse.ArgumentParser(prog="mlx_ncm.scripts.eval_shard")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", default="")
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--freq", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--adj", default=None)
    ap.add_argument("--l0", default="",
                    help="unigram (default), active, uniform, or smooth:<eps> "
                         "(mix eps of uniform into the unigram)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    return ap


def run(argv=None):
    a = build_parser().parse_args(argv)

    mx.random.seed(a.seed)
    mx.set_default_device(mx.gpu)

    CFG = OmegaConf.load(a.config)
    G = CFG.generation
    ck = a.checkpoint_dir or os.path.join("artifacts", CFG.task)
    dd = a.data_dir or os.path.join(os.environ.get("NCM_DATA_ROOT", "data"), CFG.task)

    def pick(cli, key, cast, default):
        return cast(cli if cli is not None else G.get(key, default))

    temp = pick(a.temp, "temperature", float, 1.0)
    top_k = pick(a.top_k, "top_k", int, 1)
    top_p = pick(a.top_p, "top_p", float, 0.0)
    cfg = pick(a.cfg, "cfg_scale", float, 0.0)
    freq = pick(a.freq, "frequency_penalty", float, 0.0)
    mu = pick(a.mu, "frequency_deficit", float, 0.0)
    adj = str(a.adj if a.adj is not None else G.get("adjacency_penalty", "") or "")
    split = a.split or CFG.eval.get("split", "test")
    n = int(a.n or CFG.eval.get("n_docs", 0))
    bs = int(a.bs or CFG.eval.batch_size)

    CFG.tokenizer.vqvae.checkpoint_path = _resolve_ckpt(ck, CFG.checkpoints.tokenizer)
    for key, name in (('vocab_file_path', 'bpe_vocab'), ('merges_file_path', 'bpe_merges')):
        if CFG.checkpoints.get(name):
            CFG.tokenizer.pre_tokenizer[key] = _resolve_ckpt(ck, CFG.checkpoints[name])

    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae.eval()

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model)
    load_any(net, _resolve_ckpt(ck, CFG.checkpoints.generator), strict=True)
    net.eval()
    if a.dtype != "float32":
        net.set_dtype(getattr(mx, a.dtype))

    P = int(CFG.model.prefix_len)
    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id

    ALLOW = _load_allowance(ck, str(G.frequency_allowance)) if G.get("frequency_allowance") else None
    L0_MODE = a.l0 or "unigram"
    L0_P = _load_allowance(ck, str(G.level0_unigram)) if G.get("level0_unigram") else None
    if L0_P is not None and L0_MODE == "active":
        active = L0_P > 0
        L0_P = active.astype(mx.float32) / mx.sum(active)
    elif L0_P is not None and L0_MODE == "uniform":
        L0_P = mx.ones(L0_P.shape, dtype=mx.float32) / L0_P.size
    elif L0_P is not None and L0_MODE.startswith("smooth:"):
        eps = float(L0_MODE.split(":", 1)[1])
        n_l0 = L0_P.size
        L0_P = (1.0 - eps) * L0_P + eps / n_l0

    ADJ = [float(v) for v in str(adj).split(",")] if adj else None

    UNCOND = (not CFG.data) or (not CFG.data.get(split))
    src_tok = None
    if UNCOND:
        seeds = [int(x) for x in CFG.eval.seeds][a.shard::a.nshards]
        n = a.n or int(CFG.eval.n_samples)
        items, refs, idxs = [], [], []
    else:
        items = pickle.load(open(os.path.join(dd, CFG.data[split]), "rb"))[:n]
        refs = pickle.load(open(os.path.join(dd, CFG.data[f"{split}_refs"]), "rb"))[:n]
        assert len(items) == len(refs), f"{len(items)} inputs vs {len(refs)} references"
        idxs = list(range(a.shard, len(items), a.nshards))
        if isinstance(items[0], str):
            from transformers import AutoTokenizer
            src_tok = AutoTokenizer.from_pretrained(
                str(CFG.tokenizer.pre_tokenizer.model_id))


    def prefix(item):
        ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
        return (ids[:P] + [PAD] * P)[:P]

    def pick_tokens(logits):
        if freq <= 0 and not ADJ:
            return mx.argmax(logits, axis=-1)
        # MLX has no float64; the reference used float64 here, which only matters
        # for very long sequences (counts stay exact in float32 below 2**24).
        B, T, V = logits.shape
        lg = logits.astype(mx.float32)
        allow = (ALLOW[:V] * T)[None, :] if ALLOW is not None else None
        al = ADJ[0] if ADJ else 0.0
        aw = int(ADJ[1]) if ADJ and len(ADJ) > 1 else 1
        ad = ADJ[2] if ADJ and len(ADJ) > 2 else 1.0
        cnt = mx.zeros((B, V), dtype=mx.float32)
        out = []
        hist = []
        for t in range(T):
            step = lg[:, t, :]
            if freq > 0:
                excess = mx.maximum(cnt - allow, 0.0) if allow is not None else cnt
                step = step - freq * excess
                if mu > 0 and allow is not None:
                    step = step + mu * ((T - t) / T) * mx.maximum(allow - cnt, 0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                penalty = -al * ad ** j
                step = step + mx.zeros((B, V)).at[mx.arange(B), h].add(penalty)
            p = mx.argmax(step, axis=-1)
            out.append(p[:, None])
            hist.append(p)
            cnt = cnt.at[mx.arange(B), p].add(1.0)
        return mx.concatenate(out, axis=1)

    CUT_AT_EOS = bool(G.get("cut_at_eos", True))

    def dec(ids):
        ids = list(ids)
        if CUT_AT_EOS and EOS in ids:
            ids = ids[:ids.index(EOS)]
        return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)],
                                      skip_special_tokens=True)

    RAW = []

    def run(x, l0=None):
        g = net.generate(l0, x, temperature=temp, top_k=top_k, top_p=top_p,
                         cfg_scale=cfg, cfg_null_eos=EOS)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        ids = np.array(pick_tokens(tk.vqvae.decoder(zl[-1])).tolist()).tolist()
        RAW.extend(ids)
        return [dec(r) for r in ids]

    hyps, sample_seed = [], []
    if UNCOND:
        assert EOS is not None, \
            "unconditional generation needs an eos_token_id in tokenizer.pre_tokenizer"
        L0 = tk.vqvae.vq.levels[0]
        for seed in seeds:
            mx.random.seed(seed)
            for s in range(0, n, bs):
                B = min(bs, n - s)
                if L0_P is not None:
                    logp = mx.log(mx.maximum(L0_P, 1e-30))
                    sampled = mx.random.categorical(
                        mx.broadcast_to(logp[None, :], (B * L0, L0_P.size)))
                    l0 = sampled.reshape(B, L0)
                else:
                    l0 = None
                hyps += run(mx.full((B, 1), EOS, dtype=mx.int32), l0)
                sample_seed += [seed] * B
            print(f"  shard{a.shard} seed {seed}: {n} samples", flush=True)
        idxs = list(range(len(hyps)))
        toks = [tk.pre_tokenizer.encode(h) for h in hyps]
        out = {"idxs": idxs, "hyps": hyps, "refs": [""] * len(hyps),
               "seeds": sample_seed, "toks": toks, "raw_ids": RAW}
    else:
        for s in range(0, len(idxs), bs):
            batch = idxs[s:s + bs]
            x = mx.array([prefix(items[i]) for i in batch], dtype=mx.int32)
            hyps += run(x)
            if a.shard == 0 and (s // bs) % 10 == 0:
                print(f"  shard0 {s}/{len(idxs)}", flush=True)
        out = {"idxs": idxs, "hyps": hyps,
               "refs": [r if isinstance(r, str) else dec(r)
                        for r in (refs[i] for i in idxs)]}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(out, open(a.out, "wb"))
    print(f"shard {a.shard}: {len(hyps)} -> {a.out}")
    return out


def main(argv=None):
    run(argv)


if __name__ == "__main__":
    main()

