import numpy as np

from tani_predictor.tokenizer import TaniTokenizer, format_peak, tokenize_smiles


def test_smiles_regex_is_lossless() -> None:
    for smi in ["CC(=O)Oc1ccccc1C(=O)O", "C[C@H](N)C(=O)O", "[Na+].[Cl-]", "C%12CC%12", "N#CCCc1ccc(O)cc1"]:
        assert "".join(tokenize_smiles(smi)) == smi


def test_vocab_special_ids(tokenizer: TaniTokenizer) -> None:
    assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id) == (0, 1, 2, 3)
    assert len(tokenizer) == 142


def test_encode_layout_and_roundtrip(tokenizer: TaniTokenizer) -> None:
    mz = np.array([50, 120, 77], dtype=np.int64)
    inten = np.array([0.25, 1.0, 0.5])
    enc = tokenizer.encode(mz, inten, "CCO")
    ids = enc.input_ids
    assert ids[0] == tokenizer.bos_id and ids[1] == tokenizer.ms_id and ids[-1] == tokenizer.eos_id
    assert ids.count(tokenizer.smiles_id) == 1
    assert tokenizer.ids_to_smiles(ids) == "CCO"
    assert tokenizer.unk_id not in ids
    # peaks are written in ascending m/z, intensity with 4 decimals
    body = "".join(tokenizer.inverse[i] for i in ids[2 : ids.index(tokenizer.smiles_id)])
    assert body == ",".join([format_peak(50, 0.25), format_peak(77, 0.5), format_peak(120, 1.0)])
    assert body == "50, 0.2500,77, 0.5000,120, 1.0000"
    assert (enc.n_peaks_kept, enc.n_peaks_dropped) == (3, 0)


def test_unknown_smiles_token_maps_to_unk(tokenizer: TaniTokenizer) -> None:
    assert tokenizer.smiles_to_ids("[Xe]")[0] == tokenizer.unk_id


def test_overlong_spectrum_drops_weakest_peaks(tokenizer: TaniTokenizer) -> None:
    rng = np.random.default_rng(0)
    mz = np.arange(1, 301, dtype=np.int64)
    inten = rng.random(300)
    inten[17] = 1.0
    enc = tokenizer.encode(mz, inten, "CCO", max_len=500)
    assert len(enc.input_ids) <= 500
    assert enc.n_peaks_dropped > 0 and enc.n_peaks_kept + enc.n_peaks_dropped == 300
    body = "".join(tokenizer.inverse[i] for i in enc.input_ids[2 : enc.input_ids.index(tokenizer.smiles_id)])
    assert "18, " in body  # the strongest peak (mz 18) survives
    assert tokenizer.ids_to_smiles(enc.input_ids) == "CCO"
    peaks = body.replace(", ", ";").replace(",", "|").replace(";", ", ").split("|")
    kept = {int(peak.split(", ")[0]) for peak in peaks}
    assert len(kept) == enc.n_peaks_kept
