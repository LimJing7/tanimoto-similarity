"""Build the processed training data from FOAM candidate logs.

Outputs in ``--out-dir``:
  candidates.parquet   one row group per spec: spec_id, stereo-stripped smiles, tanimoto, target, source_mask
  specs.parquet        one row per spec: split, inchikey, gt_smiles (what Tanimoto is measured against),
                       label_smiles, row_group, peaks (mz, intensity)
  tani_quantile.joblib QuantileTransformer fit on train-split Tanimoto values only
  quantile_hist.png    before/after histogram of the train sample
  manifest.json        provenance (counts, seeds, experiment order for source_mask bits)
"""

import argparse
import csv
import datetime
import random
import shutil
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from sklearn.preprocessing import QuantileTransformer

from tani_predictor.chem import has_stereo_marks, strip_stereo
from tani_predictor.config import Manifest
from tani_predictor.spectra import read_spectra

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

RESULTS_ROOT = Path("/tier1/home/ljing2/backup/foam/results")
MS_PRED_ROOT = Path("/tier1/home/ljing2/backup/ms-pred")
DATASET_DIR = MS_PRED_ROOT / "data/spec_datasets/delta_1/comb_gc_delta_1"
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
MAX_EXPERIMENTS = 16  # source_mask is a uint16 bitmask

CANDIDATE_SCHEMA = pa.schema(
    [
        ("spec_id", pa.dictionary(pa.int32(), pa.string())),
        ("smiles", pa.string()),
        ("tanimoto", pa.float32()),
        ("target", pa.float32()),
        ("source_mask", pa.uint16()),
    ]
)


class SpecJob(NamedTuple):
    spec_id: str
    files: list[tuple[int, str]]  # (experiment index, tsv path)
    tmp_dir: str
    sample_cap: int
    seed: int


class SpecResult(NamedTuple):
    spec_id: str
    n_rows_before: int
    n_rows: int
    gt_smiles: str
    sample: NDArray[np.float32]
    n_stereo_stripped: int
    max_merge_diff: float


def find_candidate_files(root: Path) -> tuple[list[str], dict[str, list[tuple[int, str]]]]:
    """Map spec_id -> [(experiment index, tsv path)]; the spec_id is the TSV's parent directory name."""
    by_expt: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.glob("*/*/*_all_candidates.tsv")):
        by_expt[p.parent.parent.name].append(p)
    experiments = sorted(by_expt)
    by_spec: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, name in enumerate(experiments):
        for p in by_expt[name]:
            by_spec[p.parent.name].append((idx, str(p)))
    return experiments, by_spec


def process_spec(job: SpecJob) -> SpecResult:
    """Merge one spec's TSVs across experiments, dedupe SMILES, write a temporary parquet."""
    merged: dict[str, tuple[float, int]] = {}
    n_before = 0
    n_stripped = 0
    max_diff = 0.0
    gt_smiles = ""
    for expt_idx, path in job.files:
        bit = 1 << expt_idx
        with open(path) as f:
            for line in f:
                if line.startswith("# ground_truth_smiles="):
                    gt_smiles = gt_smiles or line.rstrip("\n").split("=", 1)[1]
                    continue
                if line.startswith("#"):
                    continue
                smi, _, rest = line.rstrip("\n").partition("\t")
                try:
                    tani = float(rest)
                except ValueError:  # embedded duplicate header rows
                    continue
                if not smi or tani != tani:
                    continue
                n_before += 1
                if has_stereo_marks(smi):
                    n_stripped += 1
                    smi = strip_stereo(smi)
                prev = merged.get(smi)
                if prev is not None:
                    max_diff = max(max_diff, abs(prev[0] - tani))
                merged[smi] = (tani, bit) if prev is None else (prev[0], prev[1] | bit)

    smiles = list(merged)
    tani_arr = np.fromiter((merged[s][0] for s in smiles), dtype=np.float32, count=len(smiles))
    mask_arr = np.fromiter((merged[s][1] for s in smiles), dtype=np.uint16, count=len(smiles))
    table = pa.table(
        {
            "spec_id": pa.array([job.spec_id] * len(smiles), type=pa.string()),
            "smiles": pa.array(smiles, type=pa.string()),
            "tanimoto": pa.array(tani_arr),
            "source_mask": pa.array(mask_arr),
        }
    )
    pq.write_table(table, Path(job.tmp_dir) / f"{job.spec_id}.parquet", compression="zstd")

    rng = np.random.default_rng([job.seed, zlib.crc32(job.spec_id.encode())])
    k = min(job.sample_cap, len(tani_arr))
    sample = rng.choice(tani_arr, size=k, replace=False) if k else tani_arr[:0]
    return SpecResult(job.spec_id, n_before, len(smiles), gt_smiles, sample, n_stripped, max_diff)


def read_tsv(path: Path, key: str, value: str) -> dict[str, str]:
    with open(path, newline="") as f:
        return {row[key]: row[value] for row in csv.DictReader(f, delimiter="\t")}


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def formula(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return CalcMolFormula(mol) if mol is not None else ""


def check_out_dir(out_dir: Path, results_root: Path, overwrite: bool) -> Path:
    """Resolve the output directory and refuse to clobber anything unless ``--overwrite`` is given."""
    out_dir = out_dir.resolve()
    cwd = Path.cwd().resolve()
    protected = [cwd, *cwd.parents, results_root.resolve(), DATASET_DIR.resolve(), Path.home()]
    if out_dir in protected or any(p.is_relative_to(out_dir) for p in protected):
        raise SystemExit(f"refusing to use {out_dir}: it is, or contains, a directory that must not be replaced")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise SystemExit(f"{out_dir} is not empty; pass --overwrite to replace it once the new build succeeds")
    return out_dir


def select_specs(by_split: dict[str, list[str]], max_specs: int | None, seed: int) -> dict[str, list[str]]:
    """Keep every spec, or a seeded random subset of ``max_specs`` specs (official split labels kept)."""
    if max_specs is None:
        return {k: sorted(v) for k, v in by_split.items()}
    pool = sorted((s, split) for split, ids in by_split.items() for s in ids)
    keep = random.Random(seed).sample(pool, min(max_specs, len(pool)))
    out: dict[str, list[str]] = {split: [] for split in by_split}
    for spec_id, split in keep:
        out[split].append(spec_id)
    return {k: sorted(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-specs", type=int, default=None, help="Build a small subset (split-proportional).")
    ap.add_argument("--sample-cap", type=int, default=1000, help="Max Tanimoto values per spec for the quantile fit.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing --out-dir after a successful build.")
    args = ap.parse_args()

    final_dir = check_out_dir(args.out_dir, args.results_root, args.overwrite)
    out_dir = final_dir.with_name(final_dir.name + ".building")  # staging dir, swapped in on success
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True)

    experiments, by_spec = find_candidate_files(args.results_root)
    if len(experiments) > MAX_EXPERIMENTS:
        raise SystemExit(f"{len(experiments)} experiments found but source_mask holds {MAX_EXPERIMENTS} bits")
    n_files = sum(len(v) for v in by_spec.values())
    print(f"{n_files} TSV files, {len(by_spec)} specs, {len(experiments)} experiments: {experiments}")

    splits_raw = read_tsv(DATASET_DIR / "splits/split_1.tsv", "spec", "Fold_0")
    by_split: dict[str, list[str]] = {s: [] for s in SPLIT_FRACTIONS}
    n_without_split = 0
    for spec_id in by_spec:
        split = splits_raw.get(spec_id)
        if split not in by_split:
            n_without_split += 1  # absent from split_1.tsv, or an unknown fold label
            continue
        by_split[split].append(spec_id)
    if n_without_split:
        print(f"WARNING: skipping {n_without_split} specs that have no train/val/test label in split_1.tsv")
    chosen = select_specs(by_split, args.max_specs, args.seed)
    if not chosen["train"]:
        raise SystemExit("no train-split specs selected; the quantile transform is fit on train data only")
    split_of = {spec_id: split for split, ids in chosen.items() for spec_id in ids}
    spec_ids = sorted(split_of)
    print({s: len(v) for s, v in chosen.items()})

    # Pass 1: merge/dedupe per spec in parallel.
    jobs = [SpecJob(s, sorted(by_spec[s]), str(tmp_dir), args.sample_cap, args.seed) for s in spec_ids]
    results: dict[str, SpecResult] = {}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_spec, jobs, chunksize=4), 1):
            results[res.spec_id] = res
            if i % 1000 == 0:
                print(f"  merged {i}/{len(jobs)} specs")

    empty = [s for s in spec_ids if results[s].n_rows == 0]
    if empty:
        print(f"dropping {len(empty)} specs with no candidate rows")
        spec_ids = [s for s in spec_ids if results[s].n_rows > 0]
        split_of = {s: split_of[s] for s in spec_ids}
        chosen = {split: [s for s in ids if s in split_of] for split, ids in chosen.items()}
        for s in empty:
            (tmp_dir / f"{s}.parquet").unlink()

    # Quantile transform fit on train-split values only.
    train_samples = [results[s].sample for s in spec_ids if split_of[s] == "train"]
    train_vals = np.concatenate(train_samples) if train_samples else np.zeros(0, dtype=np.float32)
    if len(train_vals) == 0:
        raise SystemExit("every train-split spec had no candidate rows; cannot fit the quantile transform")
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(train_vals)), output_distribution="uniform", subsample=None, random_state=args.seed
    )
    qt.fit(train_vals.reshape(-1, 1))
    joblib.dump(qt, out_dir / "tani_quantile.joblib")
    scaled = qt.transform(train_vals.reshape(-1, 1)).ravel()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.hist(train_vals, bins=100)
    ax0.set_title("train Tanimoto (raw)")
    ax1.hist(scaled, bins=100)
    ax1.set_title("train Tanimoto (scaled)")
    fig.tight_layout()
    fig.savefig(out_dir / "quantile_hist.png")
    print(f"quantile transform fit on {len(train_vals):,} train values")

    # Pass 2: write candidates.parquet (one row group per spec) with the scaled target.
    rows_per_split: dict[str, int] = defaultdict(int)
    row_group_of: dict[str, int] = {}
    with pq.ParquetWriter(out_dir / "candidates.parquet", CANDIDATE_SCHEMA, compression="zstd") as writer:
        for rg, spec_id in enumerate(spec_ids):
            t = pq.read_table(tmp_dir / f"{spec_id}.parquet")
            tani = t.column("tanimoto").to_numpy()
            target = qt.transform(tani.reshape(-1, 1)).ravel().astype(np.float32)
            t = t.append_column("target", pa.array(target))
            t = t.set_column(0, "spec_id", t.column("spec_id").dictionary_encode())
            writer.write_table(t.select(CANDIDATE_SCHEMA.names).cast(CANDIDATE_SCHEMA))
            row_group_of[spec_id] = rg
            rows_per_split[split_of[spec_id]] += t.num_rows
    shutil.rmtree(tmp_dir)
    n_row_groups = pq.ParquetFile(out_dir / "candidates.parquet").num_row_groups
    assert n_row_groups == len(spec_ids), (n_row_groups, len(spec_ids))

    # specs.parquet: metadata + spectra.
    labels = read_tsv(DATASET_DIR / "labels.tsv", "spec", "inchikey")
    label_smiles = read_tsv(DATASET_DIR / "labels.tsv", "spec", "smiles")
    spectra = read_spectra(DATASET_DIR / "spec_files.hdf5", spec_ids, MS_PRED_ROOT)
    formula_mismatches = 0
    non_identical = 0
    cols: dict[str, list[object]] = defaultdict(list)
    for spec_id in spec_ids:
        gt = results[spec_id].gt_smiles
        if canonical(gt) != canonical(label_smiles[spec_id]):
            non_identical += 1  # tautomer / stereo standardization differences
            if formula(gt) != formula(label_smiles[spec_id]):
                formula_mismatches += 1
        sp = spectra[spec_id]
        cols["spec_id"].append(spec_id)
        cols["split"].append(split_of[spec_id])
        cols["inchikey"].append(labels[spec_id])
        cols["gt_smiles"].append(gt)
        cols["label_smiles"].append(label_smiles[spec_id])
        cols["n_candidates"].append(results[spec_id].n_rows)
        cols["row_group"].append(row_group_of[spec_id])
        cols["mz"].append(sp.mz.astype(np.int32).tolist())
        cols["intensity"].append(sp.intensity.astype(np.float32).tolist())
    specs_schema = pa.schema(
        [
            ("spec_id", pa.string()),
            ("split", pa.string()),
            ("inchikey", pa.string()),
            ("gt_smiles", pa.string()),
            ("label_smiles", pa.string()),
            ("n_candidates", pa.int64()),
            ("row_group", pa.int64()),
            ("mz", pa.list_(pa.int32())),
            ("intensity", pa.list_(pa.float32())),
        ]
    )
    pq.write_table(pa.table(cols, schema=specs_schema), out_dir / "specs.parquet", compression="zstd")

    manifest = Manifest(
        results_root=str(args.results_root),
        experiments=experiments,
        n_tsv_files=n_files,
        n_specs=len(spec_ids),
        n_rows=sum(r.n_rows for r in results.values()),
        n_rows_before_dedupe=sum(r.n_rows_before for r in results.values()),
        specs_per_split={s: len(v) for s, v in chosen.items()},
        rows_per_split=dict(rows_per_split),
        n_specs_without_split=n_without_split,
        gt_smiles_non_identical=non_identical,
        gt_smiles_formula_mismatches=formula_mismatches,
        n_smiles_stereo_stripped=sum(r.n_stereo_stripped for r in results.values()),
        max_tanimoto_diff_on_merge=max((r.max_merge_diff for r in results.values()), default=0.0),
        quantile_fit_values=len(train_vals),
        seed=args.seed,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    print(manifest.model_dump_json(indent=2))
    print(f"{non_identical} specs: FOAM ground truth is a tautomer/stereo variant of labels.tsv (expected)")
    if formula_mismatches:
        raise SystemExit(f"{formula_mismatches} specs have a ground-truth formula that differs from labels.tsv")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    out_dir.rename(final_dir)
    print(f"wrote {final_dir}")


if __name__ == "__main__":
    main()
