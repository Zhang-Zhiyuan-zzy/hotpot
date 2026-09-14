"""Syntax regressions for the active SMARTS parser."""

import pytest

from hotpot.cheminfo.search.smarts import substructure_from_smarts, tokenize


VALID_SMARTS = (
    "C1CCCCC1",
    "[CH3]C[CH2]C",
    "C[C@H](N)C(=O)O",
    "C[C@H](N)C(=O)O[C@@H]1CC[C@H](C(=O)O)CC1",
    "c1cc(ccc1)C[C@H]2CC[C@H](C(=O)O)C2C(=O)O",
    "C%12C[C@H](O)CC[C@@H](N)C(=O)C%12C[C@H](C)O",
    "[#6,#7][C@H](N)C(=O)O[C@H]1CC[C@H](C)CC1C(=O)O",
    "[*:1]C[C@H](N)C(=O)O[C@H]1CC[C@H](C)C1C(=O)O",
    "[O;H1]C(=O)C[C@H](N)C(=O)O[C@@H]1CC[C@H](C)C1",
    "[C;R;H1]1CC[C@H](C(=O)O)OCC[C@H]2CC[C@@H]12N",
)

INVALID_SMARTS = (
    "CC]",
    "[C@H]1CC[C@H](C(=O)O)C1[",
    "[C[CH]]C",
    "[CH]C[CH]]C",
    "C1CC",
    "C%12CC",
    "C1CCC1C1",
)


@pytest.mark.parametrize("smarts", VALID_SMARTS)
def test_valid_smarts_compile(smarts):
    assert tokenize(smarts)
    assert substructure_from_smarts(smarts).query_atoms


@pytest.mark.parametrize("smarts", INVALID_SMARTS)
def test_invalid_brackets_and_ring_labels_raise(smarts):
    with pytest.raises(ValueError):
        substructure_from_smarts(smarts)
