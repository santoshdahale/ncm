"""Convert PyTorch NCM checkpoints to self-contained MLX safetensors files.

Usage::

    python -m mlx_ncm.convert --config config/eval/xsum.yaml \
        --checkpoint-dir artifacts/xsum --out artifacts-mlx/xsum

The eval config is what tells us how to rebuild the models (and therefore how to
tell a ``Conv1d`` weight from a ``ConvTranspose1d`` weight, whose layouts would
otherwise be ambiguous for square channel counts). Converting needs torch once;
running the MLX model afterwards does not.
"""

from __future__ import annotations

import argparse
import os
import shutil

import mlx.core as mx
from omegaconf import OmegaConf

from .models.next_concept import NextConceptTransformer
from .tokenization.vqvae_tokenizer import VQVAETokenizer
from .weights import load_torch_checkpoint, save_module_weights, save_state_dict


def convert(config_path, ckpt_dir, out_dir, overwrite=False):
    """Convert ``tokenizer``/``generator`` checkpoints named by ``config_path``."""
    CFG = OmegaConf.load(config_path)
    os.makedirs(out_dir, exist_ok=True)

    tok_src = os.path.join(ckpt_dir, CFG.checkpoints.tokenizer)
    gen_src = os.path.join(ckpt_dir, CFG.checkpoints.generator)
    tok_out = os.path.join(out_dir, "tokenizer.safetensors")
    gen_out = os.path.join(out_dir, "generator.safetensors")

    for src in (tok_src, gen_src):
        if not os.path.exists(src):
            raise FileNotFoundError(f"missing checkpoint {src}")

    if overwrite or not os.path.exists(tok_out):
        # VQVAETokenizer.from_config loads the PyTorch file through weights.load_any.
        CFG.tokenizer.vqvae.checkpoint_path = tok_src
        tk = VQVAETokenizer.from_config(CFG.tokenizer)
        save_module_weights(tk.vqvae, tok_out)
        print(f"[convert] tokenizer -> {tok_out}")
    else:
        CFG.tokenizer.vqvae.checkpoint_path = tok_src
        tk = VQVAETokenizer.from_config(CFG.tokenizer)
        print(f"[convert] tokenizer exists, reusing architecture from {tok_out}")

    if overwrite or not os.path.exists(gen_out):
        net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model)
        load_torch_checkpoint(net, gen_src, key="model", strict=True)
        save_module_weights(net, gen_out)
        print(f"[convert] generator -> {gen_out}")
    else:
        print(f"[convert] generator exists, skipping ({gen_out})")

    # Extra inference-time tensors (frequency allowances, level-0 unigram, ...).
    import torch

    ckpt = torch.load(gen_src, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "inference" in ckpt:
        inf = {f"inference.{k}": mx.array(v.float().numpy())
               for k, v in ckpt["inference"].items()}
        save_state_dict(inf, os.path.join(out_dir, "inference.safetensors"))
        print(f"[convert] inference tensors -> {os.path.join(out_dir, 'inference.safetensors')}")

    shutil.copyfile(config_path, os.path.join(out_dir, "config.yaml"))
    print(f"[convert] config -> {os.path.join(out_dir, 'config.yaml')}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mlx_ncm.convert")
    ap.add_argument("--config", required=True,
                    help="eval config that describes the checkpoint, e.g. config/eval/xsum.yaml")
    ap.add_argument("--checkpoint-dir", required=True, help="directory with .pt checkpoints")
    ap.add_argument("--out", default="", help="output directory (default <checkpoint-dir>-mlx)")
    ap.add_argument("--overwrite", action="store_true", help="re-convert existing files")
    a = ap.parse_args(argv)

    out = a.out or (a.checkpoint_dir.rstrip("/") + "-mlx")
    convert(a.config, a.checkpoint_dir, out, overwrite=a.overwrite)
    print(f"\nDone. Evaluate with:\n"
          f"  python -m mlx_ncm.evaluate --config {a.config} "
          f"--checkpoint-dir {out} --data-dir <data dir>")


if __name__ == "__main__":
    main()
