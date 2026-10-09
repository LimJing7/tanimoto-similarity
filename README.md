# Tani Predictor

A transformer model that predicts **scaled Tanimoto similarity** between an unknown molecule's mass spectrum and a proposed candidate structure (SMILES).

## Problem

Given:
- an experimental mass spectrum (peak list: m/z + intensity, plus metadata such as precursor m/z, ionization mode, collision energy) for an unknown compound, and
- a candidate SMILES string (e.g. from a spectral library match, formula search, or in silico generation),

predict the **scaled Tanimoto similarity** between the candidate structure and the true (unknown) structure's fingerprint. This score acts as a learned proxy for "how structurally close is this candidate to the right answer," without requiring the true structure at inference time.

This is useful for re-ranking candidate structures in small molecule / metabolite identification workflows, where the ground truth molecule is unknown but a Tanimoto-based confidence score can guide which candidate is most plausible.

## Approach

**Primary plan:** a Llama-based transformer (decoder-only, HuggingFace `LlamaForCausalLM` architecture — RMSNorm, SwiGLU, rotary embeddings), repurposed as a regressor: feed in a tokenized sequence covering the spectrum and the candidate SMILES, and read out a scalar in `[0, 1]` for scaled Tanimoto similarity instead of next-token logits.

Three untrained HF checkpoints with this config already exist at `/tier1/home/ljing2/chem/`, apparently prepared as size-variant starting points:
- `untrained_llama_based_model/` — `hidden_size=768`, `num_hidden_layers=6`, `intermediate_size=3072`
- `untrained_llama_based_model_2/` — `hidden_size=768`, `num_hidden_layers=20`, `intermediate_size=72` (deep/narrow MLP variant)
- `untrained_llama_based_model_hidden3/` — `hidden_size=768`, `num_hidden_layers=3`, `intermediate_size=3072`

All three share `vocab_size=598`, `bos_token_id=12`, `eos_token_id=13`, `pad_token_id=0`. These match `src/tani_predictor/chemberta-vocab.json` (591 tokens, `[PAD]=0`, `[CLS]=12`, `[SEP]=13`; token-for-token identical to `../ms2struct/vocab.txt`), with 7 extra embedding rows.

Two candidate SMILES vocabularies live in `src/tani_predictor/`:
- `chemberta-vocab.json` — 591 tokens; compatible with the existing checkpoints as-is.
- `molformer-vocab.json` — 2,362 tokens (`<bos>=0`, `<eos>=1`, `<pad>=2`, `<mask>=3`, `<unk>=2361`); many more bracket-atom tokens (fewer `<unk>`s, shorter sequences) but needs a new Llama config (`vocab_size>=2362`, new special-token IDs), so the checkpoints would be re-initialized.

Coverage check (`scripts/check_vocab_coverage.py`, all 14,461 FOAM `*_all_candidates.tsv` files, 36.1M SMILES unique within each file, 1.18B tokens, 133 distinct tokens in the data):

| | ChemBERTa | MoLFormer |
|---|---|---|
| SMILES fully covered | 99.987% | 99.996% |
| Missing SMILES tokens | 12 distinct, 4,829 occurrences (`[SiH-]`, `[SiH+]`, `[si]`, `[BH+]`, `[SH3]`, `[oH+]`, ...) | 5 distinct, 1,389 occurrences (`[F+]`, `[BH+]`, `[PH3]`, `[SH4]`, `[BrH+]`) |
| Missing format tokens | `[MS]`, `[SMILES]`, `,`, space, `0` | `[MS]`, `[SMILES]`, `,`, space |
| New tokens to add | 17 (vocab 591 -> 608) | 9 (vocab 2,362 -> 2,371) |
| Vocab entries never used | 455 of 576 (79%) | 2,229 of 2,357 (94.6%) |

Both vocabs lose almost nothing, so coverage isn't a deciding factor; adding the missing tokens gives 100% either way. The practical difference is size: ChemBERTa's needs 8 more additions but keeps the checkpoints' special-token IDs and a ~600-row embedding, while MoLFormer's needs a re-initialized config and an embedding about 4x larger that is 95% unused.

**Minimal vocab (`src/tani_predictor/tani-vocab.json`, 142 tokens):** built by `scripts/build_vocab.py` from the data itself, so it covers 100% of the FOAM SMILES with nothing unused. Layout, by id:
- 0-5: `<pad>`, `<bos>`, `<eos>`, `<unk>`, `[MS]`, `[SMILES]`
- 6-18: input-format tokens `,` `space` `.` and digits `0`-`9`
- 19-141: the 123 remaining SMILES tokens in the data, most frequent first (the data has 133 distinct tokens; 10 of them, `.` and `1`-`9`, are shared with the format tokens above, so peak digits and ring-closure digits use the same embeddings)

The ids differ from both reference vocabs and from the existing Llama checkpoints (`pad=0` is kept; `bos=1`, `eos=2`, `vocab_size=142`), so the model config is re-initialized with this vocab. It was built from 14,462 files while FOAM runs were still adding files; rerun `scripts/build_vocab.py` if the data grows and a new token appears (a changed vocab means retraining).

TODO: decide between this and an extended ChemBERTa/MoLFormer vocab (this one is the smallest and is exact for the current data, but any SMILES token not seen so far maps to `<unk>`).

**Input format:** a single string per (spectrum, candidate) pair:

```
[MS]<x, y:.4f>,…,<x, y:.4f>[SMILES]<smiles>
```

i.e. a `[MS]` marker, then comma-separated peaks (m/z `x` as an integer, so binned/rounded, and intensity `y` formatted to 4 decimals), then a `[SMILES]` marker and the candidate SMILES.

Gaps between this format and the two vocab files (neither covers it as-is):
- `[MS]`, `[SMILES]`, `,` and the peak-internal separator are not in either vocab, so they need to be added as new tokens (growing `vocab_size`) or handled by a separate tokenizer for the spectrum part.
- Peak numbers would be tokenized digit by digit. `chemberta-vocab.json` has no `0` token (`molformer-vocab.json` does), so ChemBERTa's vocab can't represent most numbers without modification. This is a point in favor of MoLFormer's vocab or an extended ChemBERTa one.
- Spectrum length matters: the checkpoints have `max_position_embeddings=512`. With integer `x` (about 3 digits) and `y` at 4 decimals (about 6 tokens), plus separators, a peak is roughly 10-12 tokens, so only a few dozen peaks fit alongside the SMILES. This is an estimate; the real count depends on the tokenizer.

**Open question — code not yet located:** no training script, tokenizer-assembly code, or model-wrapping code (i.e. how spectrum peaks get tokenized/concatenated with the SMILES sequence, and how the regression head attaches to the Llama backbone) was found in `chem/` or `backup/` during this search. The only artifacts found are the three initialized checkpoints above. If you find/recall where that code lives, point me at it and I'll fold the real details in here; otherwise this will need to be written from scratch.

**Future idea to try:** a dual-encoder architecture — independent spectrum encoder and molecule encoder whose outputs are fused via cross-attention (or concatenation + MLP) rather than one joint causal-LM sequence. Worth exploring later, not the initial implementation target.

Target labels are the Tanimoto similarity between each candidate and the known ground-truth structure, rescaled via a fitted quantile transform (see Data below) for training stability.

## Status

Environment is built; no model/data code written yet. Current layout:

```
tani_predictor/
├── tani-pred/                   # Python 3.14 venv, built via `uv pip install -e ".[cuda]"` (gitignored)
├── pyproject.toml               # FOAM's deps + transformers/tokenizers/safetensors/accelerate
├── README.md
├── .gitignore
├── scripts/
│   ├── check_vocab_coverage.py  # tokenizes all FOAM candidate SMILES, reports per-vocab coverage / new tokens needed
│   └── build_vocab.py           # builds tani-vocab.json: special + format + every SMILES token seen in the data
└── src/
    └── tani_predictor/
        ├── __init__.py          # empty — package stub only, so the editable install resolves
        ├── chemberta-vocab.json # 591-token SMILES vocab (matches the Llama checkpoints' special-token IDs)
        ├── molformer-vocab.json # 2,362-token SMILES vocab (alternative; needs a re-initialized config)
        └── tani-vocab.json      # 142-token minimal vocab: specials + format tokens + SMILES tokens seen in the data
```

Planned (not yet created):
```
src/tani_predictor/
├── data/            # dataset classes, FOAM-candidate parsing, fingerprint/label generation
├── models/          # Llama-backbone regressor, (future) dual-encoder variant
├── train.py
└── evaluate.py
configs/             # training/model configs
notebooks/           # exploration, EDA
```

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
- **`run_multi.py`** (+ `run_multi_sbatch.sh`, `run_multi_collect_cands.sh`) drives many parallel formula-constrained GraphGA runs (one per `spec_id`) against ICEBERG-simulated spectra, writing results under `results/<expt_name>/<spec_id>/`. Train/val/test spec_id splits come from `split_1_testing_2.tsv`.
- **`build_ground_truth_extra_seeds.py`** / `ground_truth_extra_seeds.h5` inject the real ground-truth molecule (and near neighbors) into the GA seed pool, so the explored candidate pool isn't only low-Tanimoto noise.
- **`get_tani.py`** combines all `*_all_candidates.tsv` files under a results directory and plots the raw Tanimoto distribution.
- **`get_mces.py`** computes an alternative MCES-based structural distance oracle, for comparison against Tanimoto.
- **`build_tanimoto_uniform_transform.py`** fits a `sklearn.QuantileTransformer` (uniformizing/probability-integral transform) on the real logged Tanimoto distribution — raw Tanimoto scores cluster tightly, so this spreads them out to produce the **scaled Tanimoto** regression target used here. `compare_tanimoto_transforms.py` compares candidate transforms before picking one.

This project's data pipeline will:
1. Collect `(spec_id, candidate SMILES, tanimoto)` rows from the `*_all_candidates.tsv` files produced by FOAM runs.
2. Join each `spec_id` to its source spectrum via FOAM's `spec-lib-dir` / `spec-lib-label` (ICEBERG-simulated, from `ms-pred` results).
3. Apply the fitted quantile transform to get the scaled-Tanimoto training label.

TODO: write the extraction script (`src/data/parse_foam_candidates.py`) that reads the FOAM `results/` tree directly.

## Setup

`pyproject.toml` here is adapted from FOAM's (`backup/foam/pyproject.toml`, the `ms-gen-foam` package) rather than built from scratch, since this project consumes FOAM's data/tooling directly (RDKit fingerprints, `h5py` spectrum files, MCES oracle, etc.). On top of FOAM's dependency set it adds `transformers`, `tokenizers`, `safetensors`, and `accelerate` for the Llama backbone.

Targets **Python 3.14** (vs. FOAM's 3.12), which required dropping `dgl` / `torch-scatter` / `torch-sparse` / `pygmtools` from the dependency set — none of them have Python 3.14 wheels, and all four are only used by FOAM's graph-GA candidate *search* (lazily imported, not pulled in by `foam.oracles`/`foam.evaluators` or `ms_pred.common`, which is all we need to read `*_all_candidates.tsv` and the HDF5 spectrum files). Confirmed by tracing FOAM's actual import chains rather than assuming. Everything else in the dependency set (torch, numpy, pandas, scikit-learn, rdkit, transformers, tokenizers, lightgbm, jpype1, pymoo, etc.) publishes Python 3.14 wheels as of this writing.

```
uv venv tani-pred --python 3.14
source tani-pred/bin/activate
uv pip install -e ".[cuda]"   # or ".[cpu]"
```
