import numpy as np

from tani_predictor.spectra import parse_ms

MS = """>compound X
>formula H2
#NAME X
#NUM PEAKS 3

>collision 70
1 21.0
5 999.0
5 500.0
3 10.5
"""


def test_parse_ms_merges_and_normalizes() -> None:
    sp = parse_ms(MS)
    assert sp.mz.tolist() == [1, 3, 5]
    assert np.isclose(sp.intensity.max(), 1.0)
    assert np.isclose(sp.intensity[0], 21.0 / 999.0)


def test_parse_ms_without_peaks_is_empty() -> None:
    sp = parse_ms(">compound X\n#NAME X\n")
    assert len(sp.mz) == 0 and len(sp.intensity) == 0
