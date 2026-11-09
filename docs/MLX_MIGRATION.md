# Migrating NCM to MLX (Apple silicon)

This document describes the MLX port of the Next-Concept Modeling (NCM) release in
this repository: what `mlx_ncm/` contains, how it maps onto the PyTorch reference
implementation, how to convert the released checkpoints, and what was validated.

The short version:

```bash
uv pip install -r requirements-mlx.txt

# one-time: convert the .pt checkpoints to MLX safetensors (needs torch)
python -m mlx_ncm.convert --config config/eval/xsum.yaml \
    --checkpoint-dir artifacts/xsum --out artifacts-mlx/xsum

# evaluate exactly like the PyTorch entry point
python -m mlx_ncm.evaluate --config config/eval/xsum.yaml \
    --checkpoint-dir artifacts-mlx/xsum --data-dir data/xsum
```

Everything runs on the Apple GPU through MLX with no CUDA and no PyTorch at
inference time. On an M-series laptop the port reproduces the reference
implementation to float32 round-off (`~5e-7`) and produces **identical token
ids** for greedy / CFG decoding.

---

## 1. What was migrated

The upstream release is an *evaluation* codebase: a multiscale residual VQ-VAE
tokenizer, a Next-Concept generator over its codebooks, and a shard/merge
evaluation harness. All of it is ported.

| PyTorch source | MLX port | Contents |
| --- | --- | --- |
| `models/common.py` | `mlx_ncm/models/common.py` | RoPE, `LayerNorm`, `RMSNorm`, `SelfAttention` (GQA, QK-norm, KV cache), `MLP`, `Block`, `Transformer` |
| `models/vae.py` | `mlx_ncm/models/vae.py` | VQ-VAE `Encoder` (wte/wpe lookup + optional conv/attn blocks + stride compression) and `Decoder` |
| `models/quant.py` | `mlx_ncm/models/quant.py` | `Phi`/`PhiShared`/`PhiPartiallyShared`/`PhiNonShared` residual convs, learned L1 down/up sampling, `MultiscaleResidualVectorQuantizer` |
| `models/vqvae.py` | `mlx_ncm/models/vqvae.py` | `VQVAE` (encode/decode/reconstruct/multiscale decode) |
| `models/next_concept.py` | `mlx_ncm/models/next_concept.py` | `_TiedCodebookHead`, `_SharedHeads`, `NextConceptTransformer` incl. `generate` (eager + KV cache + CFG) |
| `tokenization/*.py` | `mlx_ncm/tokenization/*.py` | BPE, HF, Sudoku and VQ-VAE tokenizers |
| `evaluation/gen_ppl.py` | `mlx_ncm/evaluation/gen_ppl.py` | Generative perplexity scoring (backend-agnostic) |
| `scripts/eval_shard.py` | `mlx_ncm/scripts/eval_shard.py` | Per-shard generation + decoding + frequency/adjacency penalties |
| `scripts/eval_merge.py` | `mlx_ncm/scripts/eval_merge.py` | Merge + ROUGE / sacreBLEU / chrF / Sudoku rates / gen-PPL |
| `evaluate.py` | `mlx_ncm/evaluate.py` | Shard orchestration + merge |
| — | `mlx_ncm/weights.py` | PyTorch `state_dict` ⇄ MLX conversion and loading |
| — | `mlx_ncm/ops.py` | Layout/kernel shims: conv conventions, interpolation, masks, param flattening |
| — | `mlx_ncm/convert.py` | CLI: `.pt` checkpoints → self-contained MLX safetensors |
| — | `mlx_ncm/registry.py`, `mlx_ncm/utils.py` | Registry + helpers (replacing `hydra.utils.get_class`) |
| — | `mlx_ncm/tests/` | Unit checks, PyTorch parity harness, benchmark |

The original `models/`, `tokenization/`, `scripts/`, `evaluate.py` and
`requirements.txt` are untouched: the PyTorch code remains the reference and both
implementations can live in the same checkout.

---

## 2. Requirements and install

`requirements-mlx.txt`:

| Package | Needed for |
| --- | --- |
| `mlx` | everything (runtime) |
| `numpy`, `omegaconf` | array glue, YAML configs |
| `transformers`, `huggingface_hub[cli]`, `regex` | HF/BPE tokenizers, checkpoint download |
| `rouge_score`, `sacrebleu` | XSum / WMT14 metrics |
| `mlx-lm` *(optional)* | native reference LM for generative perplexity |
| `torch` *(optional)* | only `python -m mlx_ncm.convert`, or `--data-dir`-style fallbacks |

Notes:

* Python 3.10+; MLX ships arm64 macOS wheels (`mlx==0.32.4` was used here).
* PyTorch is **not** imported by `mlx_ncm` at runtime; `weights.py` imports it
  lazily and only on the `.pt` path.
* `hydra-core` is not needed: the reference only used
  `hydra.utils.get_class(dotted_path)`, replaced by `mlx_ncm.utils.import_class`.


---

## 3. Quickstart

```bash
# 0) environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mlx.txt

# 1) released weights (torch .pt inside)
hf download hazyresearch/ncm-checkpoints --local-dir artifacts
hf download hazyresearch/ncm-datasets --repo-type dataset --local-dir data

# 2) convert once: writes tokenizer.safetensors / generator.safetensors /
#    inference.safetensors / config.yaml into artifacts-mlx/<task>
python -m mlx_ncm.convert --config config/eval/xsum.yaml \
    --checkpoint-dir artifacts/xsum --out artifacts-mlx/xsum

# 3) evaluate (identical flags to the PyTorch entry point)
python -m mlx_ncm.evaluate --config config/eval/xsum.yaml \
    --checkpoint-dir artifacts-mlx/xsum --data-dir data/xsum

# conditional tasks with guidance, e.g. WMT14:
python -m mlx_ncm.evaluate --config config/eval/wmt14.yaml \
    --checkpoint-dir artifacts-mlx/wmt14 --data-dir data/wmt14 --cfg 1.0
```

Unconditional tasks (TinyStories / OpenWebText / LM1B) use the `genppl` metric,
which needs a reference LM:

```bash
# native MLX reference LM (recommended)
uv pip install mlx-lm
python -m mlx_ncm.evaluate --config config/eval/tinystories.yaml \
    --checkpoint-dir artifacts-mlx/tinystories --ref-model mlx-community/gpt2-large

# or skip the perplexity columns entirely
python -m mlx_ncm.evaluate --config config/eval/tinystories.yaml \
    --checkpoint-dir artifacts-mlx/tinystories --skip-genppl
```

Conversion is optional if you keep torch installed: `mlx_ncm.evaluate` accepts a
directory of `.pt` files directly and converts them in memory
(`weights.load_any`). Converting first is what removes the torch dependency and
makes startup faster.

---

## 4. Framework mapping

### 4.1 API equivalents

| PyTorch | MLX | Notes |
| --- | --- | --- |
| `nn.Module.forward` | `nn.Module.__call__` | MLX modules are invoked directly |
| `nn.Linear` / `nn.Embedding` | `nn.Linear` / `nn.Embedding` | same weight layout |
| `F.scaled_dot_product_attention` | `mx.fast.scaled_dot_product_attention` | additive mask; `scale=1/sqrt(d)` passed explicitly |
| `torch.nn.functional.layer_norm` | `mx.fast.layer_norm(x, w, b, eps)` | `bias=None` allowed |
| `torch.sigmoid/exp/rsqrt/...` | `mx.sigmoid/exp/rsqrt/...` | |
| `torch.cat(dim=k)` | `mx.concatenate(axis=k)` | |
| `torch.split(x, [a,b,c], dim)` | `mx.split(x, [a, a+b], axis)` | |
| `x.view(...).transpose(1,2)` | `x.reshape(...).transpose(0,2,1,3)` | |
| `repeat_interleave(rep, dim=1)` | `mx.repeat(k, rep, axis=1)` | GQA |
| `torch.multinomial(softmax(l))` | `mx.random.categorical(l)` | samples from logits |
| `torch.topk(x, k).values[:,-1]` | `mx.topk(x, k, axis=-1)[:, -1:]` | |
| `torch.sort(descending=True)` | `mx.argsort(-x)` + `mx.take_along_axis` | |
| `tensor.scatter_/index_put_` | `a.at[idx].add(v)` / `a[idx] = v` | |
| `torch.finfo(dt).min` | `mx.finfo(dt).min` | vocabulary padding |
| `F.pad(x, (0, n), value=v)` | `mx.pad(x, [(0,0)...,(0,n)], constant_values=v)` | |
| `nn.init.trunc_normal_` | `mx.random.truncated_normal(-2,2,shape) * std` | |
| `torch.no_grad()` | nothing | MLX only tracks grads on request |
| `torch.distributed` | nothing | single device |
| `register_buffer(..., persistent=False)` | plain array attribute | see §4.6 |
| `hydra.utils.get_class` | `mlx_ncm.utils.import_class` | |

### 4.2 Convolution layout (the one place layouts differ)

| | PyTorch | MLX |
| --- | --- | --- |
| `Conv1d` input | `(B, C, L)` | `(B, L, C)` |
| `Conv1d` weight | `(out, in, k)` | `(out, k, in)` |
| `ConvTranspose1d` weight | `(in, out, k)` | `(out, k, in)` |

Consequences:

* `mlx_ncm/ops.py` exposes `conv1d_bcl` / `conv_transpose1d_bcl`, which keep the
  **PyTorch calling convention** (`(B, C, L)` in and out) so the model code reads
  like the reference. Inside the VQ-VAE encoder/decoder the reference already
  transposes around its convolutions, which is exactly MLX's native `(B, L, C)`
  layout, so those call sites simply drop the transpose pair.
* `mlx_ncm/weights.py` permutes 3-D weights by module type:
  `Conv1d → (0, 2, 1)`, `ConvTranspose1d → (1, 2, 0)`. The permutation is chosen
  from the **module type**, never guessed from shapes — `Conv1d(d, d, k)` and
  `ConvTranspose1d(d, d, k)` have ambiguous shapes when `in == out`
  (`learned_down_l1` / `learned_up_l1` are exactly that case).

### 4.3 Interpolation

MLX has no `F.interpolate`. The quantizer moves between levels with
`F.interpolate(mode="area")` (down) and `mode="linear"` (up), so both are
reimplemented in `ops.py`:

* `area_pool1d` == `adaptive_avg_pool1d` with PyTorch's exact window bounds
  `[floor(a·in/out), ceil((a+1)·in/out))`, as a precomputed averaging matrix.
* `linear_upsample1d` == `mode="linear", align_corners=False`
  (`src = scale·(d+0.5) − 0.5`, clamped, with edge duplication).

Both are validated against the PyTorch formulas (`test_area_pool_matches_pytorch`,
`test_linear_upsample_matches_pytorch`, and the end-to-end parity run).

### 4.4 Attention: three behavioural differences

1. **`is_causal`.** `mx.fast.scaled_dot_product_attention` has no `is_causal`
   flag, so `ops.additive_causal_mask` builds the bottom-right-aligned mask that
   torch would apply (``q_len != kv_len`` included).
2. **Dropout.** There is no `dropout_p` argument, so `attn_dropout` is applied to
   the attention *output* while training (`self.training`) and is a no-op at
   inference. Evaluation is unaffected; training-time behaviour is approximate
   and documented as such.
3. **NaN-safe softmax (important).** PyTorch's SDPA uses a *safe* softmax: a query
   row whose keys are all masked yields zeros. MLX yields `NaN`, and once a value
   is `NaN` it poisons later layers (`0 * NaN = NaN`). This only happens when an
   entire query row is masked — which is exactly the classifier-free-guidance
   *null prefix* (a prefix made only of pad tokens). `SelfAttention` therefore
   reproduces torch's behaviour explicitly:

   ```python
   y = mx.fast.scaled_dot_product_attention(q, k, v, scale=..., mask=mask)
   all_masked = mx.all(mask <= -float("inf"), axis=-1, keepdims=True)
   y = mx.where(all_masked, mx.array(0.0, dtype=y.dtype), y)
   ```

   Without this, `--cfg > 0` produced all-zero tokens in the port while torch
   produced sensible ones (see §6).

`torch.nn.attention.flex_attention` has no MLX equivalent. Every released config
sets `use_flex_attention: false`; the port **raises** if a config enables it
instead of silently falling back.

### 4.5 No float64

MLX has `float16`/`bfloat16`/`float32` only. Two places in the reference used
`float64`:

* `scripts/eval_shard.py::pick_tokens` (frequency / adjacency / deficit
  penalties) — now `float32`. Counts and penalties stay far below `2**24`, so
  decoding decisions are unchanged in practice; only ties resolved by float64
  rounding could differ.
* the `frequency_allowance` / `level0_unigram` vectors — stored and used as
  `float32`.

Everything else was already `float32`.

### 4.6 Modules, containers, and checkpoint keys

Conversion is a pure rename, which only works if MLX child names match
`torch.nn` ones. Two subtleties:

**(a) `Sequential` / `ModuleList` naming.** `mlx.nn.Sequential` nests children
under `layers.<i>`, and `torch.nn` uses `"0"`, `"1"`, …. `mlx_ncm/models/containers.py`
provides `Sequential` / `ModuleList` / `TorchIndexed` that store children under
numeric attribute names, so `vq.learned_down_l1.0.weight` keeps its name.

**(b) Do not overload `__getitem__`.** The reference selects a residual-conv rail
with `quant_resi[at]` where `at` is a **float** (`PhiNonShared` overrides
`__getitem__`). That cannot be reused: MLX's `Module.update` walks children with
`dst[i]`, so `load_weights` called `__getitem__(1)` and got the rail whose *tick*
was closest to `1.0` — i.e. rail 2 — silently leaving rail 1 with its random
initial values. The tick lookup now lives in `select(at)` (and `__getitem__`
keeps its integer `ModuleList` meaning). This bug was caught by the parity
harness; see §6.

### 4.7 Buffers vs parameters

MLX has no buffer namespace: every `mx.array` attribute is a "parameter" as far as
`parameters()`/`save_weights()` are concerned. Torch buffers registered with
`persistent=False` therefore show up as MLX parameters that the checkpoint does
not contain. `weights.is_derived()` lists them and `apply_weights(strict=True)`
allows exactly those to be missing:

* `rope_cos`, `rope_sin`, `rope_cos_gen_*`, `rope_sin_gen_*` (RoPE tables)
* `lm_head_l0.E`, `.E_sq`, `.d_scale` (derived from the tied codebook)
* `_fd_step` (fine-drop anneal counter), `flex_lvl_full`

Persistent torch buffers (`lvl_1L`, `attn_mask_for_masking`, `wp_prefix`, `wp_1L`,
`ema_*`, `codebook_hit_*`) *are* in the checkpoint and keep their names.

This strictness is a feature: a config/architecture mismatch fails loudly instead
of producing a model with random weights. For example, forgetting
`use_qk_norm: true` when rebuilding the VQ-VAE reports
`missing: ['decoder.middle_block.attn.k_norm.weight', ...]`.

### 4.8 Random number generation

`mx.random` and `torch` use different streams, so **sampled** generations
(`temperature > 0`) differ token-for-token even though the distributions match.
All *deterministic* paths are bit-identical:

| path | parity |
| --- | --- |
| greedy (`temperature=0`, `top_k<=1`, `top_p<=0`) | identical tokens |
| KV-cache vs eager decoding | identical tokens |
| CFG (`cfg_scale > 0`) greedy | identical tokens |
| sampling with `temperature>0` | same distribution, different draw |

Reproducibility within MLX is preserved: `eval_shard` calls `mx.random.seed(...)`
per seed, so a given `--seed`/config reproduces its own output.

### 4.9 Training-only infrastructure

The release ships no training script, but `models/quant.py` still contains the
training path. It is ported (EMA codebook updates, straight-through at level 0,
fine-drop, dead-code restart, k-means init) with these simplifications:

* `torch.distributed` all-reduce / world-size EMA scaling → no-ops.
* `torch.no_grad()` decorators → unnecessary (MLX tracks nothing unless asked).
* in-place tensor mutation (`ema_w.data.copy_`, `emb.weight.data[dead] = ...`) →
  functional updates (`a.at[i].add(...)`, `.at[].set(...)`, rebinding attributes).
* MLX autograd would need `mx.value_and_grad`; the port never builds a graph, so
  it is not an optimizer-ready training loop — it is the reference forward pass.

---

## 5. Checkpoint conversion

```bash
python -m mlx_ncm.convert --config config/eval/xsum.yaml \
    --checkpoint-dir artifacts/xsum --out artifacts-mlx/xsum [--overwrite]
```

`convert` needs the eval config because the config is what rebuilds the modules;
module *types* are what decide the conv weight permutations (§4.2). It writes:

| file | contents |
| --- | --- |
| `tokenizer.safetensors` | full VQ-VAE parameter set (self-contained, RoPE-free) |
| `generator.safetensors` | full `NextConceptTransformer` parameter set |
| `inference.safetensors` | every tensor from the checkpoint's `inference` dict, e.g. `inference.token_unigram`, `inference.level0_unigram` |
| `config.yaml` | copy of the eval config used |

After conversion the directory has no PyTorch dependency: `mx.load` +
`module.load_weights` are all that is needed. `scripts/eval_shard._resolve_ckpt`
prefers `<stem>.safetensors` over `<stem>.pt`, so a directory containing both
uses the MLX weights.

Loading is strict on purpose:

```
missing:    parameters the MLX model expects that the checkpoint lacks
            (must all be `weights.is_derived`)
unexpected: checkpoint tensors the MLX model does not have (must be empty)
```

Either list being non-empty raises `ValueError` with the offending names, which is
how §4.7's config-mismatch example surfaces.

## 6. Numerical validation

Two harnesses are committed:

```bash
python -m mlx_ncm.tests.test_port       # 14 self-contained checks (no torch needed)
python -m mlx_ncm.tests.parity_torch    # 18 checks against the PyTorch reference
```

The parity harness builds a small VQ-VAE + generator in PyTorch, converts that
`state_dict` into the MLX model (`strict=True`), and compares every stage. Results
on this machine (float32, CPU torch vs MLX GPU):

| quantity | result |
| --- | --- |
| `encode_z` | max abs diff `4.8e-07` |
| quantizer indices (`encode`) | **identical** |
| `decode` logits | `3.6e-07` |
| `idx_to_z_q`, `idx_to_ncm_input` | `3.0e-07`, `2.4e-07` |
| `idx_to_zhats` (all levels) | ≤ `3.0e-07` |
| generator `forward` logits | `4.8e-07` (`query_level0=False`), `6.0e-07` (`True`) |
| greedy generation | **identical tokens** |
| KV-cache vs eager generation | **identical tokens** |
| CFG greedy generation | **identical tokens** |
| sampled generation (`T>0`) | same distribution, different draw (RNG streams) |

An xsum-sized generator (102.5M parameters, 9 levels, prefix 128, batch 4) also
produced identical tokens to PyTorch at float32.

The unit suite additionally covers: interpolation vs the PyTorch formulas, torch
child naming (`quant_resi.0.weight`, `learned_down_l1.0.weight`), the
`Conv1d`/`ConvTranspose1d` permutation round-trip (including the ambiguous
square case), strict state-dict loading, nearest-neighbour quantizer codes,
shard shapes, CFG with padded prefixes, and `float16` execution.

Found by these harnesses and fixed during the port:

1. **float-keyed `__getitem__` broke weight loading** (§4.6) — rail 1 of
   `quant_resi` silently kept random weights. Would have degraded every released
   checkpoint's quality without any error. *(parity harness)*
2. **`NaN` in fully masked attention rows** (§4.4) — CFG produced all-zero tokens
   while torch produced valid ones. *(parity harness)*
3. `Sequential(*mods)` vs `nn.ModuleList`: `tuple(module)` on an MLX module
   iterates its *dict keys*, so a helper taking a single sequence must not be
   reused as a varargs constructor (`containers.py` now separates the two).
   *(first end-to-end run)*


## 7. Performance

`python -m mlx_ncm.tests.bench_vs_torch` builds an xsum-shaped generator
(102.5M parameters, 9 levels, prefix 128, batch 4) with random weights and times a
full multi-level generation on the same machine:

| backend | time | |
| --- | --- | --- |
| PyTorch, CPU, float32 | 0.89 s | baseline (no CUDA on Apple silicon) |
| MLX, GPU, float32 | 0.44 s | **2.0×** the CPU baseline, identical tokens |
| MLX, GPU, float16 | 0.49 s | no gain at batch 4; tokens can differ at near-ties |

Caveats worth stating plainly:

* The paper's numbers come from CUDA GPUs; this is a same-machine comparison
  against PyTorch on CPU, not a claim about parity with the paper's speedups.
* MLX keeps weights and activations in **unified memory**, so the 384-d VQ-VAE
  plus generator (~150M parameters for xsum) fit without host/device copies.
* Use `--dtype float16` / `bfloat16` on larger batches or bigger models; at small
  batches the decode is latency-bound and half precision does not help.
* `use_flex_attention` is unsupported (§4.4), so the port always uses the
  explicit additive mask. All released configs already do.

---

## 8. CLI reference

### `python -m mlx_ncm.convert`

| flag | meaning |
| --- | --- |
| `--config` (required) | eval config describing the checkpoint (`config/eval/*.yaml`) |
| `--checkpoint-dir` (required) | directory containing `tokenizer.pt` / `generator.pt` |
| `--out` | output dir (default `<checkpoint-dir>-mlx`) |
| `--overwrite` | re-convert files that already exist |

### `python -m mlx_ncm.evaluate`

Same flags as the PyTorch entry point, minus `--gpus`:

| flag | meaning |
| --- | --- |
| `--config` (required) | eval config |
| `--checkpoint-dir` (required) | `.safetensors` (or `.pt`, converted on the fly) |
| `--data-dir` | inputs/references for conditional tasks |
| `--out` | output dir (default `eval/<task>`) |
| `--shards` | number of sequential shards (default `eval.n_shards`, else 1) |
| `--n` | evaluate only the first `n` inputs/samples |
| `--dtype` | `float32` (default), `float16`, `bfloat16` |
| `--skip-genppl` | skip generative-perplexity columns |
| `--ref-model` | override `eval.ref_model`, e.g. `mlx-community/gpt2-large` |
| *(passthrough)* | `--temp`, `--top_k`, `--top_p`, `--cfg`, `--freq`, `--mu`, `--adj`, `--l0`, `--seed`, `--bs`, `--split` |

`--shards` runs **sequentially in one process** (MLX has a single unified-memory
device), so it only changes how the work and the `shard_*.pkl` files are split;
the merged numbers are identical to a single shard with the same seeds.

### `python -m mlx_ncm.scripts.eval_merge`

Useful standalone when you already have shard pickles:

```bash
python -m mlx_ncm.scripts.eval_merge --config config/eval/xsum.yaml \
    --glob 'eval/xsum/shard_*.pkl' [--skip-genppl] [--ref-model mlx-community/gpt2]
```

### Tests / benchmark

```bash
python -m mlx_ncm.tests.test_port      # 14 checks, no torch needed
python -m mlx_ncm.tests.parity_torch   # 18 checks vs the PyTorch reference
python -m mlx_ncm.tests.bench_vs_torch # timing vs PyTorch CPU
```

---

## 9. Limitations

| Area | Status |
| --- | --- |
| `use_flex_attention: true` | **not supported** (no MLX flex attention); raises at construction. All released configs use `false`. |
| `ctx_encoder.pretrained` (T5) | **not supported** (would need PyTorch weights); raises. All released configs use `enabled: false`. |
| `ctx_encoder.init_from_gpt2` | **not supported** (reads PyTorch GPT-2 weights); raises. |
| `ctx_encoder.init_from_vqvae` / random init | supported (MLX-only paths) |
| `float64` | unavailable; the two float64 sites use `float32` (§4.5) |
| sampling RNG | draws differ from PyTorch for `temperature > 0` (§4.8) |
| attention dropout semantics | applied to the output while training (§4.4); evaluation unaffected |
| `torch.distributed` | single device only |
| training | the forward pass is ported; no optimizer/backward loop |
| multi-process sharding | not used; shards run sequentially |
| generative perplexity | needs an MLX (`mlx-lm`) or torch (`transformers`) reference LM, or `--skip-genppl` |

Everything in the released eval configs (`config/eval/*.yaml`) is covered.

---

## 10. Troubleshooting

**`ValueError: checkpoint does not match the MLX model ... missing: [...]`**
The config used to build the MLX model does not describe the checkpoint. The
missing names point at the switch (e.g. `use_qk_norm`, `learned_l1_cascade`,
`quant_resi.enabled`, `tie_heads_to_codebook`). Use the `config.yaml` that ships
with the checkpoint — `convert` copies it next to the weights.

**`ValueError: [load] Unknown file format pt`**
`mx.load` only reads `.safetensors`/`.npz`. Either run `mlx_ncm.convert`, or keep
torch installed so `weights.load_any` can convert in memory.

**`RuntimeError: could not load the reference LM ...`**
`--skip-genppl`, or install `mlx-lm` and point `--ref-model` at an MLX-packaged
repo (e.g. `mlx-community/gpt2-large`) — the plain HF `gpt2` repo is a torch
checkpoint.

**All generated tokens are `0` with `--cfg > 0`**
This was the NaN-mask bug (§4.4); make sure you are on the current code
(`mlx_ncm/models/common.py` applies the all-masked-row guard).

**`TypeError: 'str' object is not callable` from `containers.py`**
A `TorchIndexed`/`Sequential` was constructed with a single module instead of a
sequence; `tuple(module)` iterates the module's dict keys (§6, item 3). Use
`Sequential(*mods)` (varargs) or `ModuleList(mods)` (one sequence).

**Memory pressure on long prefixes**
Use `--dtype float16` or reduce `--bs`; MLX keeps the whole model resident in
unified memory, so a large prefix plus a large batch is the usual cause.

---

## 11. Porting checklist (for future upstream changes)

When the reference implementation changes, the MLX port needs the matching edits:

1. **New module or attribute** → add it to the MLX model with the *same name and
   parent path*; otherwise the checkpoint key will not match.
2. **New conv** → check the calling convention: PyTorch `(B, C, L)` +
   `(out, in, k)` versus MLX `(B, L, C)` + `(out, k, in)`. Use
   `ops.conv1d_bcl` / `ops.conv_transpose1d_bcl` to keep the reference shape, or
   write the MLX-native version and rely on `weights.py` for the permutation.
3. **New container** → use `containers.Sequential` / `ModuleList` /
   `TorchIndexed`; never overload `__getitem__` with non-integer keys.
4. **New interpolation mode** → implement it in `ops.interpolate1d` and add a
   case to `test_port`.
5. **New tensor op** → prefer the equivalents in §4.1; watch for float64, for
   in-place mutation (`x[i] = v` → `x.at[i].set/add`), and for anything that can
   produce an all-masked attention row.
6. **New buffer** → if the reference registers it with `persistent=False`, add its
   name pattern to `weights._DERIVED_SUFFIXES/_DERIVED_SUBSTRINGS` so strict
   loading keeps working.
7. **Run both harnesses**: `python -m mlx_ncm.tests.test_port` and
   `python -m mlx_ncm.tests.parity_torch` (the latter needs a torch checkout of
   the reference models).
