# Tanimoto Predictor

A transformer model that predicts **scaled Tanimoto similarity** between an unknown molecule's mass spectrum and a proposed candidate structure (SMILES).

## Problem

Given:
- an experimental mass spectrum (peak list: m/z + intensity, plus metadata such as precursor m/z, ionization mode, collision energy) for an unknown compound, and
- a candidate SMILES string (e.g. from a spectral library match, formula search, or in silico generation),

predict the **scaled Tanimoto similarity** between the candidate structure and the true (unknown) structure's fingerprint. This score acts as a learned proxy for "how structurally close is this candidate to the right answer," without requiring the true structure at inference time.

This is useful for re-ranking candidate structures in small molecule / metabolite identification workflows, where the ground truth molecule is unknown but a Tanimoto-based confidence score can guide which candidate is most plausible.

## Approach

**Primary plan:** a Llama-based transformer (decoder-only, HuggingFace `LlamaModel` — RMSNorm, SwiGLU, rotary embeddings), repurposed as a regressor: feed in a tokenized sequence covering the spectrum and the candidate SMILES, and read out a scalar in `[0, 1]` for scaled Tanimoto similarity instead of next-token logits.

Vocabularies live in `src/tani_predictor/vocab/`. Two reference SMILES vocabularies are kept there for comparison:
- `chemberta-vocab.json` — 591 tokens (identical to `../ms2struct/vocab.txt`).
- `molformer-vocab.json` — 2,362 tokens (`<bos>=0`, `<eos>=1`, `<pad>=2`, `<mask>=3`, `<unk>=2361`); many more bracket-atom tokens (fewer `<unk>`s, shorter sequences).

Coverage check (`scripts/check_vocab_coverage.py`, all 14,461 FOAM `*_all_candidates.tsv` files, 36.1M SMILES unique within each file, 1.18B tokens, 133 distinct tokens in the data):

| | ChemBERTa | MoLFormer |
|---|---|---|
| SMILES fully covered | 99.987% | 99.996% |
| Missing SMILES tokens | 12 distinct, 4,829 occurrences (`[SiH-]`, `[SiH+]`, `[si]`, `[BH+]`, `[SH3]`, `[oH+]`, ...) | 5 distinct, 1,389 occurrences (`[F+]`, `[BH+]`, `[PH3]`, `[SH4]`, `[BrH+]`) |
| Missing format tokens | `[MS]`, `[SMILES]`, `,`, space, `0` | `[MS]`, `[SMILES]`, `,`, space |
| New tokens to add | 17 (vocab 591 -> 608) | 9 (vocab 2,362 -> 2,371) |
| Vocab entries never used | 455 of 576 (79%) | 2,229 of 2,357 (94.6%) |

Both vocabs lose almost nothing, so coverage isn't a deciding factor; adding the missing tokens gives 100% either way. The practical difference is size: ChemBERTa's needs 8 more additions and has a ~600-row embedding, while MoLFormer's has an embedding about 4x larger that is 95% unused.

**Minimal vocab (`src/tani_predictor/vocab/tani-vocab.json`, 142 tokens):** built by `scripts/build_vocab.py` from the data itself, so it covers 100% of the FOAM SMILES with nothing unused. Layout, by id:
- 0-5: `<pad>`, `<bos>`, `<eos>`, `<unk>`, `[MS]`, `[SMILES]`
- 6-18: input-format tokens `,` `space` `.` and digits `0`-`9`
- 19-141: the 123 remaining SMILES tokens in the data, most frequent first (the data has 133 distinct tokens; 10 of them, `.` and `1`-`9`, are shared with the format tokens above, so peak digits and ring-closure digits use the same embeddings)

The ids differ from both reference vocabs (`pad=0`, `bos=1`, `eos=2`, `vocab_size=142`), and the model is initialized with this vocab. It was built from 14,462 files while FOAM runs were still adding files; rerun `scripts/build_vocab.py` if the data grows and a new token appears (a changed vocab means retraining).

TODO: decide between this and an extended ChemBERTa/MoLFormer vocab (this one is the smallest and is exact for the current data, but any SMILES token not seen so far maps to `<unk>`).

**Input format:** a single string per (spectrum, candidate) pair:

```
[MS]<x, y:.4f>,…,<x, y:.4f>[SMILES]<smiles>
```

i.e. a `[MS]` marker, then comma-separated peaks (m/z `x` as an integer, so binned/rounded, and intensity `y` formatted to 4 decimals), then a `[SMILES]` marker and the candidate SMILES.

Tokenization: peak numbers are written character by character (digits, `.`, `,` and a space), and SMILES are split with the Schwaller regex. `[MS]`, `[SMILES]`, `,` and the space are in neither reference vocab, and `chemberta-vocab.json` has no `0` token, which is why `tani-vocab.json` was built instead. A peak is about 12 tokens, so the longest spectra (about 349 peaks) need roughly 4,300 tokens, within the 8192 context.

**Implementation:** `src/tani_predictor/` (see Status). The model is a Llama backbone (`LlamaModel`) plus a linear head on the final `<eos>` hidden state with a sigmoid, trained with MSE on the scaled target (`model.py`), initialized randomly with the 142-token `tani-vocab.json`.

**Future idea to try:** a dual-encoder architecture — independent spectrum encoder and molecule encoder whose outputs are fused via cross-attention (or concatenation + MLP) rather than one joint causal-LM sequence. Worth exploring later, not the initial implementation target.

Target labels are the Tanimoto similarity between each candidate and the known ground-truth structure, rescaled via a fitted quantile transform (see Data below) for training stability.

## Status

Data pipeline, tokenizer, model, training and evaluation are implemented and smoke-tested; a full-scale training run has not been done yet. `mypy --strict` is clean and `pytest` passes (`tests/`).

```
tani_predictor/
├── tani-pred/                   # Python 3.14 venv, `uv pip install -e ".[cuda,dev]"` (gitignored)
├── pyproject.toml               # FOAM's deps + transformers/pydantic/pyarrow; mypy + pytest config
├── configs/
│   ├── base.yaml                # 6 layers, hidden 768, context 8192 (56.7M parameters)
│   └── tiny.yaml                # smoke-test config (2 layers, hidden 64)
├── scripts/
│   ├── build_dataset.py         # FOAM TSVs + ms-pred spectra -> data/processed/ (parquet), fits the quantile transform
│   ├── check_vocab_coverage.py  # per-vocab coverage of the FOAM SMILES / new tokens needed
│   └── build_vocab.py           # builds tani-vocab.json: special + format + every SMILES token seen in the data
├── src/tani_predictor/
│   ├── config.py                # pydantic Config (model/data/train) + Manifest; YAML + `--set a.b=c` overrides
│   ├── tokenizer.py             # [MS]...[SMILES]... encoding, drops weakest peaks only if over max_len
│   ├── spectra.py               # reads `.ms` spectra from the ms-pred HDF5 (external links, relative paths)
│   ├── data.py                  # CandidateDataset (stratified per-spec sampling), Collator (pad to batch max)
│   ├── model.py                 # TaniRegressor: Llama backbone + sigmoid regression head
│   ├── train.py                 # HF Trainer entry point (resamples candidates every epoch)
│   ├── evaluate.py              # regression + per-spec ranking metrics, checkpoint evaluation CLI
│   ├── vocab/                   # chemberta-vocab.json, molformer-vocab.json (reference), tani-vocab.json (used)
│   └── py.typed
└── tests/                       # tokenizer, spectra, sampling/collator, model, config
```

### Usage

```
source tani-pred/bin/activate
python scripts/build_dataset.py --workers 64                     # -> data/processed/ (add --overwrite to replace an existing one)
python -m tani_predictor.train --config configs/base.yaml        # -> runs/base/model.pt
python -m tani_predictor.evaluate --config configs/base.yaml --checkpoint runs/base --split test --all-candidates
# multi-GPU (DDP): effective batch = per_device_batch_size x gradient_accumulation_steps x n_gpus
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc_per_node=2 -m tani_predictor.train --config configs/base.yaml
# smoke test: python scripts/build_dataset.py --max-specs 200 --out-dir data/processed_smoke
#             python -m tani_predictor.train --config configs/tiny.yaml
```

Output safety and checkpoints: `build_dataset.py` builds into `<out-dir>.building` and swaps it in only on success; it refuses a non-empty `--out-dir` without `--overwrite`, and never accepts the working directory, its parents, the home directory or the results root. When val specs exist, `train.py` evaluates every `eval_steps` (`save_steps` must be a multiple of it), keeps the checkpoint with the best `eval_mean_spec_spearman`, and `model.pt` holds that best model. `evaluate.py` loads `model.pt`, or the newest `checkpoint-N/` if a run was interrupted before `model.pt` was written.

Numerical stability (important): a full-scale run produced NaN gradients around step 1600 and then NaN weights. The cause was PyTorch's **cuDNN scaled-dot-product-attention backend, which returns NaN gradients in bf16 for some padded batches** (torch 2.14, H200; the loss stayed finite, only the backward pass was affected, and fp32 or the flash/mem-efficient backends give correct gradients). Using the captured tensors, the NaN appears in exactly the query rows whose softmax log-sum-exp is below -88.72 (= -ln(FLT_MAX)), which here all have an exactly-zero incoming gradient, and it disappears when a constant is added to the logits so that every row's log-sum-exp is above that threshold; cuDNN's forward log-sum-exp is finite and correct, so the NaN is created inside its backward kernel). `TaniRegressor` therefore disables the cuDNN SDPA backend globally (about 35% slower on the longest batches). As a safety net, `NonFiniteGuardTrainer` (`guard.py`) skips any optimizer step whose gradient norm is non-finite instead of letting one NaN destroy the weights, saves the offending micro-batches (and, once, the still-finite weights) to `<output_dir>/nonfinite_step<N>.pt`, and aborts after 20 skipped steps in a row. `train.max_grad_norm` must stay above 0 because the guard hooks into gradient clipping. **TODO:** disabling the cuDNN SDPA backend globally is a stopgap, not a fix: it costs speed (about 35% on the longest batches) and does not address why attention logits grow to 200-590. Replace it with a proper fix and re-enable the backend. Options to evaluate: report the bug upstream to PyTorch (`tests/diagnostics/sdpa_repro.py` with `runs/nan_repro/sdpa_failing_layer.pt` is a ready torch-only repro) and re-enable once a PyTorch/cuDNN release fixes it; stop the logit growth with QK-norm, a logit soft-cap, or a lower learning rate / longer warmup, which would also keep the logits within bf16 precision; or at least scope the workaround with `torch.nn.attention.sdpa_kernel` around the training forward instead of a global flag. Resume a run with `--resume runs/<name>/checkpoint-N`. The tools used to find and characterize the bug are in `tests/diagnostics/` (run from the repo root as `python -m tests.diagnostics.<name> <file>`): `nan_dump_analysis`, `nan_fix_check`, `capture_sdpa_tensors` (saves the failing call to `runs/nan_repro/`), `sdpa_repro` (torch-only repro), `sdpa_nan_map`, `sdpa_lse_compare`, `sdpa_lse_threshold` and `attention_logit_growth` (how attention logits grow from init to the failing step).

Design choices: context 8192 with all peaks (the longest spectra need about 4,300 tokens, so no truncation in practice); batches are random and padded only to the longest sequence in the batch; candidates are sampled per spec stratified over the scaled-target range, a fresh sample each epoch.

Full processed dataset (`data/processed/`, built in about 2 minutes): 9,935 specs (9,850 train, 45 val, 40 test), 34.4M deduplicated candidates, median 2,565 candidates per train spec (max 25,817). Raw Tanimoto is low and clustered (mean 0.16, median 0.15, 99th percentile 0.43), and the scaled target is close to uniform. Encoded inputs average 1,186 tokens (99th percentile 2,905, max 4,215) with no `<unk>` and exact SMILES round-trips on 1,000 sampled pairs.

Base config on the real data (300 steps, effective batch 32, one shared H200): training loss 0.143 -> 0.075 (0.083 is a constant prediction of the mean), about 53 examples/s, so one epoch of the default 1.25M sampled examples (128 per spec) is roughly 6.6 hours.

Smoke-test results (tiny model, 200 specs, 300 steps, scored on its own training data, so this only shows the pipeline learns): Pearson 0.31, mean per-spec Spearman 0.33, top-1 candidate Tanimoto 0.186 vs 0.153 for a random candidate. For the base config on one H200, a batch of 2 at length 8192 peaks at 5.8 GiB without gradient checkpointing (2.0 GiB with `model.gradient_checkpointing: true`).

## Data

Training data is generated from the **`lj-candidate-logging`** branch of FOAM at **`/tier1/home/ljing2/backup/foam`** (not `chem/foam` — that copy is unmodified upstream). This fork logs every GA candidate generated during structure-elucidation search, along with its Tanimoto similarity to the known ground-truth molecule, to `<save_dir>/<oracle.name>_all_candidates.tsv`:

```
# ground_truth_smiles=<...>
smiles	tanimoto
<candidate smiles 1>	0.1111
<candidate smiles 2>	0.0909
...
```

Key pieces already in place in that FOAM checkout:
- **`run_multi.py`** (+ `run_multi_sbatch.sh`, `run_multi_collect_cands.sh`) drives many parallel formula-constrained GraphGA runs (one per `spec_id`) against each spec's experimental spectrum (candidates are scored with ICEBERG-simulated spectra), writing results under `results/<expt_name>/<spec_id>/`. Train/val/test spec_id splits come from `split_1_testing_2.tsv`.
- **`build_ground_truth_extra_seeds.py`** / `ground_truth_extra_seeds.h5` inject the real ground-truth molecule (and near neighbors) into the GA seed pool, so the explored candidate pool isn't only low-Tanimoto noise.
- **`get_tani.py`** combines all `*_all_candidates.tsv` files under a results directory and plots the raw Tanimoto distribution.
- **`get_mces.py`** computes an alternative MCES-based structural distance oracle, for comparison against Tanimoto.
- **`build_tanimoto_uniform_transform.py`** fits a `sklearn.QuantileTransformer` (uniformizing/probability-integral transform) on the real logged Tanimoto distribution — raw Tanimoto scores cluster tightly, so this spreads them out to produce the **scaled Tanimoto** regression target used here. `compare_tanimoto_transforms.py` compares candidate transforms before picking one.

### Rescaling model (scaled Tanimoto)

The model that rescales raw Tanimoto to the training target is a fitted `sklearn.preprocessing.QuantileTransformer(output_distribution="uniform")`, kept at **`data/processed/tani_quantile.joblib`** (`/tier1/home/ljing2/chem/tani_predictor/data/processed/tani_quantile.joblib`, gitignored; written by `scripts/build_dataset.py`). It is fit on **train-split Tanimoto values only** (up to 1,000 per spec) and applied to every candidate to produce the `target` column. Load it with `joblib.load(...)`; `transform` gives scaled targets and `inverse_transform` converts model predictions back to raw Tanimoto, so keep it together with any trained checkpoint. Rebuilding the dataset refits and overwrites it, which changes the target scale, so a model must be retrained after a rebuild. (FOAM has its own older fit at `/tier1/home/ljing2/backup/foam/tanimoto_uniform_transform.joblib`; it is not used here.)

### Stereo handling

All SMILES lose their stereo information (chirality `@` and E/Z `/` `\`) before reaching the model. `strip_stereo` in `src/tani_predictor/chem.py` returns RDKit's canonical non-isomeric SMILES (SMILES without stereo marks are returned unchanged). It is applied in three places: `TaniTokenizer.smiles_to_ids` (so training and any inference input are treated identically), `scripts/build_dataset.py` (stereoisomers that collapse to one SMILES are merged and counted in the manifest, together with the largest Tanimoto disagreement among merged rows), and the vocab scripts. Stripping cannot change the labels: FOAM's Tanimoto uses Morgan fingerprints (radius 2, no chirality), and `tests/test_chem.py` checks that the fingerprint is identical before and after stripping.

Current data is unaffected: none of the 34.2M FOAM candidates carries stereo marks (FOAM's generated candidates are stereo-free), so the vocab and dataset are the same as without stripping. It matters for candidates from other sources, such as library or PubChem SMILES.

### What `scripts/build_dataset.py` does

1. Scans `*/<spec_id>/*_all_candidates.tsv` under `/tier1/home/ljing2/backup/foam/results/`. Only 6 of the 10 experiment directories contain candidate logs (`cands_logging`, `_w_gt`, `_w_perma_gt`, `_w_perma_gt_val_test` (the val/test specs), `_mces_logging_w_perma_gt`, `_tani_logging_w_perma_gt`); the others are earlier runs without logging. Candidates are deduplicated per spec across experiments, and a bitmask records which experiments produced each one (`source_mask`).
2. Reads the real experimental spectra (NIST20 GC and OPCW, not simulated; ICEBERG is only FOAM's forward model for scoring candidates) from `ms-pred/data/spec_datasets/delta_1/comb_gc_delta_1/spec_files.hdf5`. Intensities are normalized to a per-spec max of 1 because some spectra are scaled to 999 and others to 9999.
3. Fits `QuantileTransformer(output_distribution="uniform")` on **train-split** Tanimoto values only, and stores the result as the `target` column (the scaled Tanimoto). The fitted transform, a before/after histogram and a `manifest.json` are written next to the data.

Findings that shaped the setup:
- **Splits:** the train fold (9,850 specs) is 99.7% NIST20 GC (33 OPCW specs). The val (45 specs) and test (40 specs) folds come from `..._w_perma_gt_val_test`, a FOAM run on the official `split_1` val/test specs, so far only about 7% of the 620 val and 627 test specs, and are **all OPCW**. They share no molecule with train (0 InChIKey overlaps, and none by connectivity block) or with each other. `build_dataset.py` uses the official split labels, so more val/test specs appear in the next rebuild as FOAM logs them, and `train.py` evaluates on val every `eval_steps`.
- **Reading val/test results:** (1) evaluation measures transfer from NIST to OPCW; (2) val/test pools always contain the ground truth (Tanimoto 1.0, as these are `w_perma_gt` runs; 55% of train pools do) and have higher raw Tanimoto (mean 0.20-0.23 vs 0.16), so the scaled target, fit on train, is not uniform there and MSE is not comparable with training loss; prefer the ranking metrics (`mean_spec_spearman`, `top1_tanimoto`, `top1_is_best`); (3) 40-45 specs make the numbers noisy; (4) score every candidate for the final numbers with `evaluate --all-candidates`, since the default 256 sampled candidates per spec rarely include the ground truth.
- **Ground truth:** the TSV ground-truth SMILES (the molecule the Tanimoto is measured against) is sometimes a tautomer or stereo variant of the `labels.tsv` SMILES (16 of 50 in a sample), always with the same molecular formula. The build checks the formula and stores both (`gt_smiles`, `label_smiles`).
- **Growing data:** FOAM runs were still adding files while this was built; rerun `build_dataset.py` (and `build_vocab.py`, since a new SMILES token means a new vocab and retraining) when the data changes.
