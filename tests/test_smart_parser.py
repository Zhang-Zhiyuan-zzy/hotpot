"""Syntax regressions for the active SMARTS parser."""

import pytest

from hotpot.cheminfo.search.smarts import substructure_from_smarts, tokenize


VALID_SMARTS = (
    "C1CCCCC1",
    "[CH3]C[CH2]C",
    "c1cc(ccc1)C[CH]2CC[C](C(=O)O)C2C(=O)O",
    "C%12C[CH](O)CC[CH](N)C(=O)C%12C[CH](C)O",
    "[#6,#7][CH](N)C(=O)O[C]1CC[C](C)CC1C(=O)O",
    "[*:1]C[CH](N)C(=O)O[C]1CC[C](C)C1C(=O)O",
    "[O;H1]C(=O)C[CH](N)C(=O)O[C]1CC[C](C)C1",
    "[C;R;H1]1CC[C](C(=O)O)OCC[C]2CC[C]12N",
)

INVALID_SMARTS = (
    "CC]",
    "[C@H]1CC[C@H](C(=O)O)C1[",
    "[C[CH]]C",
    "[CH]C[CH]]C",
    "C1CC",
    "C%12CC",
    "C1CCC1C1",
    "C-",
    "C()",
    "C(=)N",
)

UNIMPLEMENTED_SMARTS = (
    "C[C@H](N)C(=O)O",
    "C/C=C\\C",
    "[13CH4]",
)


@pytest.mark.parametrize("smarts", VALID_SMARTS)
def test_valid_smarts_compile(smarts):
    assert tokenize(smarts)
    assert substructure_from_smarts(smarts).query_atoms


@pytest.mark.parametrize("smarts", INVALID_SMARTS)
def test_invalid_brackets_and_ring_labels_raise(smarts):
    with pytest.raises(ValueError):
        substructure_from_smarts(smarts)


@pytest.mark.parametrize("smarts", UNIMPLEMENTED_SMARTS)
def test_unimplemented_smarts_features_raise(smarts):
    with pytest.raises(NotImplementedError):
        substructure_from_smarts(smarts)
