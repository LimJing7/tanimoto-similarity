from pathlib import Path

import pytest
from build_dataset import check_out_dir

RESULTS = Path("/nonexistent/results")


def test_new_and_empty_dirs_are_allowed(tmp_path: Path) -> None:
    assert check_out_dir(tmp_path / "out", RESULTS, overwrite=False) == (tmp_path / "out").resolve()
    (tmp_path / "empty").mkdir()
    check_out_dir(tmp_path / "empty", RESULTS, overwrite=False)


def test_non_empty_dir_needs_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(SystemExit, match="--overwrite"):
        check_out_dir(out, RESULTS, overwrite=False)
    check_out_dir(out, RESULTS, overwrite=True)


@pytest.mark.parametrize("path", [".", "..", "~"])
def test_protected_dirs_are_refused_even_with_overwrite(path: str) -> None:
    with pytest.raises(SystemExit, match="refusing"):
        check_out_dir(Path(path).expanduser(), RESULTS, overwrite=True)


def test_results_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="refusing"):
        check_out_dir(tmp_path, tmp_path, overwrite=True)
