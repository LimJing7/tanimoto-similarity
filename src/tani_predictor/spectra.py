"""Reading SIRIUS-format ``.ms`` spectra stored in the ms-pred HDF5 files."""

import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
from numpy.typing import NDArray


class Spectrum(NamedTuple):
    """Integer m/z values (ascending) with intensities normalized to a max of 1."""

    mz: NDArray[np.int64]
    intensity: NDArray[np.float64]


EMPTY_SPECTRUM = Spectrum(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))


def parse_ms(text: str) -> Spectrum:
    """Parse the peak blocks (``>collision ...``) of a ``.ms`` file.

    Peaks from several collision blocks are merged by taking the max intensity per integer m/z.
    """
    best: dict[int, float] = {}
    in_peaks = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(">collision"):
            in_peaks = True
        elif not line or line.startswith((">", "#")):
            in_peaks = False
        elif in_peaks:
            mz_str, inten_str, *_ = line.split()
            mz = round(float(mz_str))
            best[mz] = max(best.get(mz, 0.0), float(inten_str))
    if not best:
        return EMPTY_SPECTRUM
    mz_arr = np.array(sorted(best), dtype=np.int64)
    inten = np.array([best[int(m)] for m in mz_arr], dtype=np.float64)
    top = float(inten.max())
    if top <= 0.0:
        return EMPTY_SPECTRUM
    return Spectrum(mz_arr, inten / top)


def read_spectra(h5_path: Path, spec_ids: Iterable[str], link_root: Path) -> dict[str, Spectrum]:
    """Read spectra by id.

    The HDF5 file holds external links with paths relative to the working directory of the
    ms-pred checkout, so reading happens with ``link_root`` as the working directory.
    """
    out: dict[str, Spectrum] = {}
    h5_abs = h5_path.resolve()
    with contextlib.chdir(link_root), h5py.File(h5_abs, "r") as f:
        for spec_id in spec_ids:
            blob = f[f"{spec_id}.ms"][0]
            text = blob.decode("utf-8") if isinstance(blob, bytes) else str(blob)
            out[spec_id] = parse_ms(text)
    return out
