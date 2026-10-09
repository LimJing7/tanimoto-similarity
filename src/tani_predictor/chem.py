"""SMILES preprocessing shared by dataset building, vocab building and the tokenizer."""

import re

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

_STEREO_MARKS = ("@", "/", "\\")
_FALLBACK = re.compile(r"[/\\@]")


def has_stereo_marks(smiles: str) -> bool:
    """True if the SMILES string carries chirality (``@``) or double-bond direction (``/``, ``\\``)."""
    return any(c in smiles for c in _STEREO_MARKS)


def strip_stereo(smiles: str) -> str:
    """Remove all stereo information (chirality and E/Z) and return a canonical non-isomeric SMILES.

    SMILES without stereo marks are returned unchanged (FOAM emits RDKit-canonical SMILES, for which
    this equals full canonicalization). If RDKit cannot parse a stereo-bearing SMILES, the marks are
    removed textually as a last resort.
    """
    if not has_stereo_marks(smiles):
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _FALLBACK.sub("", smiles)
    return str(Chem.MolToSmiles(mol, isomericSmiles=False))


def is_parseable(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None
