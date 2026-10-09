import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from tani_predictor.chem import has_stereo_marks, strip_stereo
from tani_predictor.tokenizer import TaniTokenizer

STEREO_CASES = [
    ("C[C@H](N)C(=O)O", "CC(N)C(=O)O"),
    ("C[C@@H](N)C(=O)O", "CC(N)C(=O)O"),  # enantiomers collapse to the same string
    ("C/C=C/C", "CC=CC"),
    ("C/C=C\\C", "CC=CC"),  # E and Z collapse
    ("O=C(N/N=C/c1ccco1)c1ccncc1", "O=C(NN=Cc1ccco1)c1ccncc1"),
]


@pytest.mark.parametrize(("smiles", "expected"), STEREO_CASES)
def test_strip_stereo(smiles: str, expected: str) -> None:
    assert has_stereo_marks(smiles)
    assert strip_stereo(smiles) == expected
    assert not has_stereo_marks(strip_stereo(smiles))


def test_stereo_free_smiles_unchanged_and_idempotent() -> None:
    for smi in ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]:
        assert strip_stereo(smi) == smi
    once = strip_stereo("C[C@H](N)C(=O)O")
    assert strip_stereo(once) == once


@pytest.mark.parametrize(("smiles", "_"), STEREO_CASES)
def test_morgan_fingerprint_unchanged_by_stripping(smiles: str, _: str) -> None:
    """FOAM's Tanimoto uses Morgan r=2 without chirality, so stripping cannot change labels."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def bits(s: str) -> list[int]:
        mol = Chem.MolFromSmiles(s)
        assert mol is not None
        return list(gen.GetFingerprint(mol).GetOnBits())

    assert bits(smiles) == bits(strip_stereo(smiles))


def test_unparsable_smiles_falls_back_to_removing_marks() -> None:
    assert strip_stereo("C/C=C/[Xx]") == "CC=C[Xx]"


def test_tokenizer_strips_stereo(tokenizer: TaniTokenizer) -> None:
    assert tokenizer.smiles_to_ids("C[C@H](N)C(=O)O") == tokenizer.smiles_to_ids("CC(N)C(=O)O")
    assert tokenizer.unk_id not in tokenizer.smiles_to_ids("C/C=C/C")
