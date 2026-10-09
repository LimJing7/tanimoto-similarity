"""Tokenizer for ``[MS]<x, y:.4f>,...,<x, y:.4f>[SMILES]<smiles>`` inputs."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from tani_predictor.chem import strip_stereo

# Schwaller et al. SMILES regex (same as ms2struct/smiles_tokenizers.py).
SMI_REGEX: re.Pattern[str] = re.compile(
    r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
MS_MARKER, SMILES_MARKER = "[MS]", "[SMILES]"
PEAK_SEP = ","

# bos + [MS] + [SMILES] + eos
_N_FIXED_TOKENS = 4


def tokenize_smiles(smiles: str) -> list[str]:
    """Split a SMILES string into regex tokens."""
    return SMI_REGEX.findall(smiles)


def format_peak(mz: int, intensity: float) -> str:
    return f"{mz}, {intensity:.4f}"


class Encoded(NamedTuple):
    input_ids: list[int]
    n_peaks_kept: int
    n_peaks_dropped: int


class TaniTokenizer:
    def __init__(self, vocab_path: Path) -> None:
        self.vocab: dict[str, int] = json.loads(vocab_path.read_text())
        self.inverse: dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.pad_id: int = self.vocab[PAD]
        self.bos_id: int = self.vocab[BOS]
        self.eos_id: int = self.vocab[EOS]
        self.unk_id: int = self.vocab[UNK]
        self.ms_id: int = self.vocab[MS_MARKER]
        self.smiles_id: int = self.vocab[SMILES_MARKER]

    def __len__(self) -> int:
        return len(self.vocab)

    def smiles_to_ids(self, smiles: str) -> list[int]:
        """Token ids for a SMILES; stereo information is always stripped first."""
        return [self.vocab.get(tok, self.unk_id) for tok in tokenize_smiles(strip_stereo(smiles))]

    def ids_to_smiles(self, ids: Sequence[int]) -> str:
        """Decode the SMILES part of an encoded sequence (everything after ``[SMILES]``)."""
        ids = list(ids)
        start = ids.index(self.smiles_id) + 1 if self.smiles_id in ids else 0
        out: list[str] = []
        for i in ids[start:]:
            if i == self.eos_id:
                break
            out.append(self.inverse[i])
        return "".join(out)

    def peaks_to_ids(self, mz: NDArray[np.int64], intensity: NDArray[np.float64]) -> list[int]:
        """Ids for ``x, y,x, y,...`` with peaks in the given order (no markers)."""
        text = PEAK_SEP.join(format_peak(int(m), float(y)) for m, y in zip(mz, intensity, strict=True))
        return [self.vocab.get(ch, self.unk_id) for ch in text]

    def encode(
        self,
        mz: NDArray[np.int64],
        intensity: NDArray[np.float64],
        smiles: str,
        max_len: int = 8192,
    ) -> Encoded:
        """Build the full input; if too long, drop the lowest-intensity peaks first."""
        smiles_ids = self.smiles_to_ids(smiles)
        budget = max_len - _N_FIXED_TOKENS - len(smiles_ids)
        if budget < 0:
            raise ValueError(f"SMILES needs {len(smiles_ids) + _N_FIXED_TOKENS} tokens, over max_len={max_len}")

        order = np.argsort(mz, kind="stable")
        mz, intensity = mz[order], intensity[order]
        lengths = np.array(
            [len(format_peak(int(m), float(y))) + len(PEAK_SEP) for m, y in zip(mz, intensity, strict=True)],
            dtype=np.int64,
        )
        keep = np.ones(len(mz), dtype=bool)
        excess = int(lengths.sum()) - len(PEAK_SEP) - budget
        if excess > 0:
            weakest_first = np.argsort(intensity, kind="stable")
            removed = np.cumsum(lengths[weakest_first])
            n_drop = int(np.searchsorted(removed, excess, side="left")) + 1
            keep[weakest_first[: min(n_drop, len(mz))]] = False

        peak_ids = self.peaks_to_ids(mz[keep], intensity[keep])
        input_ids = [self.bos_id, self.ms_id, *peak_ids, self.smiles_id, *smiles_ids, self.eos_id]
        n_kept = int(keep.sum())
        return Encoded(input_ids, n_kept, len(mz) - n_kept)
