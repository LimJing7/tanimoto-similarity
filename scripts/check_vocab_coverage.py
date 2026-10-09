"""Compare candidate SMILES vocabularies against the SMILES collected by FOAM.

For each vocab reports: how much of the data it covers, which data tokens are missing,
how many vocab entries are never used, and how many new tokens must be added
(missing SMILES tokens + tokens required by the [MS]...[SMILES]... input format).
"""

import argparse
import json
import re
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from tani_predictor.chem import strip_stereo
from tani_predictor.tokenizer import tokenize_smiles

# Tokens the input format needs besides SMILES tokens (digits/'.' for peak numbers).
FORMAT_TOKENS: list[str] = ["[MS]", "[SMILES]", ",", " ", ".", *"0123456789"]

HERE: Path = Path(__file__).resolve().parent.parent / "src" / "tani_predictor" / "vocab"
DEFAULT_VOCABS: dict[str, Path] = {
    "chemberta": HERE / "chemberta-vocab.json",
    "molformer": HERE / "molformer-vocab.json",
}
DEFAULT_ROOT: Path = Path("/tier1/home/ljing2/backup/foam/results")


def scan_file(path: Path) -> tuple[Counter[str], list[frozenset[str]], int]:
    """Return (token counts, per-SMILES token sets, n_untokenizable) over unique stereo-stripped SMILES in one TSV."""
    smiles: set[str] = set()
    with open(path) as f:
        for line in f:
            if line.startswith("# ground_truth_smiles="):
                smiles.add(strip_stereo(line.rstrip("\n").split("=", 1)[1]))
            elif line.startswith("#") or line.startswith("smiles\t"):
                continue
            else:
                smiles.add(strip_stereo(line.split("\t", 1)[0].strip()))
    smiles.discard("")
    counts: Counter[str] = Counter()
    bad = 0
    per_smiles: list[frozenset[str]] = []
    for smi in smiles:
        toks = tokenize_smiles(smi)
        if "".join(toks) != smi:
            bad += 1
            continue
        counts.update(toks)
        per_smiles.append(frozenset(toks))
    return counts, per_smiles, bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path, default=[DEFAULT_ROOT])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-files", type=int, default=None, help="Debug: only scan this many files.")
    ap.add_argument("--top", type=int, default=25, help="Show this many most frequent missing tokens.")
    args = ap.parse_args()

    files = sorted(p for r in args.roots for p in r.rglob("*_all_candidates.tsv"))
    if args.max_files:
        files = files[: args.max_files]
    print(f"Scanning {len(files)} files with {args.workers} workers...")

    total: Counter[str] = Counter()
    bad = 0
    smiles_token_sets: Counter[frozenset[str]] = Counter()  # distinct token-set -> number of SMILES
    n_smiles = 0
    with Pool(args.workers) as pool:
        for i, (counts, sets, b) in enumerate(pool.imap_unordered(scan_file, files, chunksize=4), 1):
            total.update(counts)
            bad += b
            n_smiles += len(sets)
            smiles_token_sets.update(sets)
            if i % 2000 == 0:
                print(f"  {i}/{len(files)} files")

    n_occ = sum(total.values())
    print(f"\nSMILES (unique within each file): {n_smiles:,}   token occurrences: {n_occ:,}")
    print(f"Distinct tokens in data: {len(total):,}   SMILES the regex couldn't tokenize losslessly: {bad:,}")

    for name, vpath in DEFAULT_VOCABS.items():
        vocab: dict[str, int] = json.loads(vpath.read_text())
        specials = {t for t in vocab if re.fullmatch(r"\[unused\d+\]|<[a-z]+>|\[(PAD|UNK|CLS|SEP|MASK)\]", t)}
        missing = {t: c for t, c in total.items() if t not in vocab}
        covered_occ = n_occ - sum(missing.values())
        smiles_ok = sum(n for s, n in smiles_token_sets.items() if not any(t not in vocab for t in s))
        unused = [t for t in vocab if t not in total and t not in specials]
        fmt_missing = [t for t in FORMAT_TOKENS if t not in vocab]
        new_needed = set(missing) | set(fmt_missing)

        print(f"\n=== {name} ({len(vocab):,} entries, {len(specials)} special/unused-placeholder) ===")
        print(f"  token-occurrence coverage : {covered_occ / n_occ:.4%}")
        print(f"  SMILES fully covered      : {smiles_ok:,} / {n_smiles:,} ({smiles_ok / n_smiles:.4%})")
        print(f"  data tokens missing       : {len(missing):,} distinct ({sum(missing.values()):,} occurrences)")
        print(
            f"  unused vocab entries      : {len(unused):,} of {len(vocab) - len(specials):,} non-special "
            f"({len(unused) / (len(vocab) - len(specials)):.1%}) never appear in the data"
        )
        print(f"  format tokens missing     : {len(fmt_missing)} -> {fmt_missing}")
        new_size = len(vocab) + len(new_needed)
        print(f"  NEW tokens to add (total) : {len(new_needed)}  -> vocab_size would be {new_size:,}")
        if missing:
            top = sorted(missing.items(), key=lambda x: -x[1])[: args.top]
            print(f"  most frequent missing data tokens: {top}")


if __name__ == "__main__":
    main()
