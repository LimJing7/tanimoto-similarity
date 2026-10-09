from pathlib import Path

import numpy as np
import pytest

from tani_predictor.evaluate import compute_metrics, find_weights


def test_find_weights_prefers_final_model(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"x")
    (tmp_path / "checkpoint-5").mkdir()
    (tmp_path / "checkpoint-5" / "model.safetensors").write_bytes(b"x")
    assert find_weights(tmp_path) == tmp_path / "model.pt"


def test_find_weights_falls_back_to_newest_checkpoint(tmp_path: Path) -> None:
    for step in (9, 100, 25):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "model.safetensors").write_bytes(b"x")
    assert find_weights(tmp_path) == tmp_path / "checkpoint-100" / "model.safetensors"


def test_find_weights_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_weights(tmp_path)


def test_perfect_ranking_metrics() -> None:
    target = np.array([0.1, 0.5, 0.9, 0.2, 0.6, 0.8], dtype=np.float32)
    spec = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    tani = np.array([0.05, 0.2, 0.7, 0.1, 0.3, 0.6], dtype=np.float32)
    m = compute_metrics(target.copy(), target, spec, tani)
    assert m["mse"] == 0.0 and m["mean_spec_spearman"] == pytest.approx(1.0)
    assert m["top1_tanimoto"] == pytest.approx(m["best_tanimoto"])


def test_top1_is_best_fraction() -> None:
    spec = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    tani = np.array([0.1, 0.9, 0.2, 0.3, 0.8, 0.4], dtype=np.float32)
    target = tani.copy()
    pred = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # spec 0 picks best, spec 1 picks 0.3
    assert compute_metrics(pred, target, spec, tani)["top1_is_best"] == 0.5
