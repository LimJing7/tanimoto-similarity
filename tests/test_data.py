import numpy as np
import torch

from tani_predictor.data import Collator, Item, stratified_sample


def test_stratified_sample_covers_all_bins() -> None:
    rng = np.random.default_rng(0)
    targets = np.concatenate([rng.random(5000) * 0.1, rng.random(50)]).astype(np.float32)  # heavily skewed low
    idx = stratified_sample(targets, k=100, n_bins=10, rng=rng)
    assert len(idx) == 100 and len(set(idx.tolist())) == 100
    high = (targets[idx] >= 0.1).sum()
    assert high >= 45  # 50 candidates exist above 0.1 and all bins 1..9 are sampled


def test_stratified_sample_small_pool_returns_everything() -> None:
    idx = stratified_sample(np.array([0.1, 0.9, 0.5], dtype=np.float32), k=10, n_bins=10, rng=np.random.default_rng(0))
    assert idx.tolist() == [0, 1, 2]


def test_collator_pads_to_longest_in_batch() -> None:
    items: list[Item] = [
        {"input_ids": [1, 4, 2], "labels": 0.2},
        {"input_ids": [1, 4, 5, 6, 7, 2], "labels": 0.9},
    ]
    batch = Collator(pad_id=0)(items)
    assert batch["input_ids"].shape == (2, 6)
    assert batch["attention_mask"].sum(dim=1).tolist() == [3, 6]
    assert batch["input_ids"][0, 3:].tolist() == [0, 0, 0]
    assert batch["labels"].dtype == torch.float32
