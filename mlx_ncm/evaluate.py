"""MLX evaluation entry point (port of ``evaluate.py``).

The PyTorch version spawns one generation shard per visible CUDA GPU. MLX exposes
a single device with unified memory, so shards run sequentially in-process; the
``--shards`` flag exists to keep the shard/merge file layout (and therefore
``scripts/eval_merge.py``) unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx
from omegaconf import OmegaConf

from .scripts import eval_merge, eval_shard


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mlx_ncm.evaluate",
        description="generation shards on the MLX device, then merge and score")
    ap.add_argument("--config", required=True, help="eval config, e.g. config/eval/xsum.yaml")
    ap.add_argument("--checkpoint-dir", required=True,
                    help="directory with the files named under checkpoints: "
                         "(.safetensors from mlx_ncm.convert, or .pt with torch installed)")
    ap.add_argument("--data-dir", default="",
                    help="directory with the files named under data: "
                         "(not needed for unconditional tasks)")
    ap.add_argument("--out", default="", help="output directory (default eval/<task>)")
    ap.add_argument("--shards", type=int, default=0,
                    help="number of sequential shards (default: eval.n_shards, or 1)")
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first n inputs")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--skip-genppl", action="store_true",
                    help="do not compute generative perplexity (no reference LM)")
    ap.add_argument("--ref-model", default="",
                    help="override eval.ref_model, e.g. mlx-community/gpt2-large")
    args, passthrough = ap.parse_known_args(argv)

    cfg = OmegaConf.load(args.config)
    n_shards = args.shards or int(cfg.eval.get("n_shards", 1) or 1)
    out = args.out or os.path.join("eval", cfg.task)
    os.makedirs(out, exist_ok=True)

    mx.set_default_device(mx.gpu)

    common = ["--config", args.config, "--checkpoint-dir", args.checkpoint_dir]
    if args.data_dir:
        common += ["--data-dir", args.data_dir]
    common += ["--nshards", str(n_shards), "--dtype", args.dtype]
    if args.n:
        common += ["--n", str(args.n)]
    common += passthrough

    print(f"[evaluate] {n_shards} shard(s) on {mx.default_device()} -> {out}/", flush=True)
    for k in range(n_shards):
        eval_shard.run(common + ["--shard", str(k),
                                 "--out", os.path.join(out, f"shard_{k}.pkl")])

    merge_argv = ["--config", args.config,
                  "--glob", os.path.join(out, "shard_*.pkl")]
    if args.skip_genppl:
        merge_argv.append("--skip-genppl")
    if args.ref_model:
        merge_argv += ["--ref-model", args.ref_model]
    eval_merge.run(merge_argv)


if __name__ == "__main__":
    sys.exit(main())
