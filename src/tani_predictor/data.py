"""Candidate dataset (stratified per-spec sampling) and dynamic-padding collator."""

from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from tani_predictor.tokenizer import TaniTokenizer


class Item(TypedDict):
    input_ids: list[int]
    labels: float


Batch = dict[str, Tensor]  # keys: input_ids, attention_mask, labels


def stratified_sample(
    targets: NDArray[np.float32], k: int, n_bins: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Pick up to ``k`` indices, spread evenly over ``n_bins`` equal-width target bins.

    Each bin contributes ``k // n_bins`` candidates (or all it has); the shortfall is filled
    uniformly from the candidates not yet chosen.
    """
    n = len(targets)
    if n <= k:
        return np.arange(n, dtype=np.int64)
    bins = np.minimum((targets * n_bins).astype(np.int64), n_bins - 1)
    quota = k // n_bins
    chosen: list[NDArray[np.int64]] = []
    for b in range(n_bins):
        members = np.flatnonzero(bins == b)
        take = min(quota, len(members))
        if take:
            chosen.append(rng.choice(members, size=take, replace=False))
    picked = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
    short = k - len(picked)
    if short > 0:
        rest = np.setdiff1d(np.arange(n, dtype=np.int64), picked, assume_unique=True)
        picked = np.concatenate([picked, rng.choice(rest, size=short, replace=False)])
    return np.sort(picked)


class CandidateDataset(Dataset[Item]):
    """(spectrum, candidate SMILES) -> scaled Tanimoto, resampled per epoch via :meth:`resample`."""

    def __init__(
        self,
        processed_dir: Path,
        split: str,
        tokenizer: TaniTokenizer,
        max_len: int,
        candidates_per_spec: int,
        num_target_bins: int,
        seed: int,
        max_specs: int | None = None,
        fixed: bool = False,
    ) -> None:
        self.processed_dir = processed_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.k = candidates_per_spec
        self.n_bins = num_target_bins
        self.seed = seed
        self.fixed = fixed  # evaluation: same sample every epoch

        specs = pq.read_table(processed_dir / "specs.parquet").to_pydict()
        keep = [i for i, s in enumerate(specs["split"]) if s == split]
        if max_specs is not None:
            keep = keep[:max_specs]
        self.spec_ids: list[str] = [specs["spec_id"][i] for i in keep]
        self._row_groups: list[int] = [specs["row_group"][i] for i in keep]
        self._mz: list[NDArray[np.int64]] = [np.asarray(specs["mz"][i], dtype=np.int64) for i in keep]
        self._intensity: list[NDArray[np.float64]] = [
            np.asarray(specs["intensity"][i], dtype=np.float64) for i in keep
        ]
        self._peak_cache: dict[int, list[int]] = {}

        self._smiles: list[str] = []
        self.spec_index: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self.targets: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self.tanimoto: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self.resample(0)

    def resample(self, epoch: int) -> None:
        """Draw a fresh stratified sample of candidates for every spec."""
        pf = pq.ParquetFile(self.processed_dir / "candidates.parquet")
        smiles: list[str] = []
        spec_idx: list[NDArray[np.int64]] = []
        targets: list[NDArray[np.float32]] = []
        tanimoto: list[NDArray[np.float32]] = []
        for i, rg in enumerate(self._row_groups):
            table = pf.read_row_group(rg, columns=["smiles", "tanimoto", "target"])
            target_all = table.column("target").to_numpy()
            tanimoto_all = table.column("tanimoto").to_numpy()
            rng = np.random.default_rng([self.seed, 0 if self.fixed else epoch, rg])
            idx = stratified_sample(target_all, self.k, self.n_bins, rng)
            smiles.extend(table.column("smiles").take(idx).to_pylist())
            targets.append(target_all[idx])
            tanimoto.append(tanimoto_all[idx])
            spec_idx.append(np.full(len(idx), i, dtype=np.int64))
        self._smiles = smiles
        self.spec_index = np.concatenate(spec_idx) if spec_idx else np.zeros(0, dtype=np.int64)
        self.targets = np.concatenate(targets) if targets else np.zeros(0, dtype=np.float32)
        self.tanimoto = np.concatenate(tanimoto) if tanimoto else np.zeros(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._smiles)

    def _peak_ids(self, spec: int) -> list[int]:
        cached = self._peak_cache.get(spec)
        if cached is None:
            cached = self.tokenizer.peaks_to_ids(self._mz[spec], self._intensity[spec])
            self._peak_cache[spec] = cached
        return cached

    def __getitem__(self, index: int) -> Item:
        spec = int(self.spec_index[index])
        smiles = self._smiles[index]
        peak_ids = self._peak_ids(spec)
        smiles_ids = self.tokenizer.smiles_to_ids(smiles)
        tok = self.tokenizer
        if len(peak_ids) + len(smiles_ids) + 4 <= self.max_len:
            ids = [tok.bos_id, tok.ms_id, *peak_ids, tok.smiles_id, *smiles_ids, tok.eos_id]
        else:
            ids = tok.encode(self._mz[spec], self._intensity[spec], smiles, self.max_len).input_ids
        return {"input_ids": ids, "labels": float(self.targets[index])}


class Collator:
    """Right-pad each batch to the longest sequence *in that batch* (no length bucketing)."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, items: list[Item]) -> Batch:
        longest = max(len(it["input_ids"]) for it in items)
        input_ids = torch.full((len(items), longest), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(items), longest), dtype=torch.long)
        for row, it in enumerate(items):
            n = len(it["input_ids"])
            input_ids[row, :n] = torch.tensor(it["input_ids"], dtype=torch.long)
            attention_mask[row, :n] = 1
        labels = torch.tensor([it["labels"] for it in items], dtype=torch.float32)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
