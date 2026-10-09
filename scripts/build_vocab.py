"""Build a minimal vocab: special tokens + input-format tokens + every SMILES token seen in the FOAM data.

SMILES tokens are ordered by descending frequency (ties broken alphabetically) so the output is deterministic.
"""

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_vocab_coverage import DEFAULT_ROOT, scan_file

SPECIAL_TOKENS: list[str] = ["<pad>", "<bos>", "<eos>", "<unk>", "[MS]", "[SMILES]"]
# Peak numbers are written digit by digit with a '.', separated by ',' and ' '.
FORMAT_TOKENS: list[str] = [",", " ", ".", *"0123456789"]

DEFAULT_OUT: Path = Path(__file__).resolve().parent.parent / "src" / "tani_predictor" / "vocab" / "tani-vocab.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path, default=[DEFAULT_ROOT])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    files = sorted(p for r in args.roots for p in r.rglob("*_all_candidates.tsv"))
    print(f"Scanning {len(files)} files with {args.workers} workers...")
    total: Counter[str] = Counter()
    with Pool(args.workers) as pool:
        for counts, _, _ in pool.imap_unordered(scan_file, files, chunksize=4):
            total.update(counts)

    tokens = list(SPECIAL_TOKENS)
    for t in FORMAT_TOKENS:
        if t not in tokens:
            tokens.append(t)
    for t, _ in sorted(total.items(), key=lambda x: (-x[1], x[0])):
        if t not in tokens:
            tokens.append(t)

    vocab = {t: i for i, t in enumerate(tokens)}
    args.out.write_text(json.dumps(vocab, ensure_ascii=False, separators=(",", ":")))
    shared = [t for t in FORMAT_TOKENS if t in total]
    print(f"Wrote {len(vocab)} tokens to {args.out}")
    print(
        f"  special: {len(SPECIAL_TOKENS)}  format-only: {len(FORMAT_TOKENS) - len(shared)}  "
        f"SMILES tokens in data: {len(total)} (of which shared with format tokens: {len(shared)})"
    )


if __name__ == "__main__":
    main()
