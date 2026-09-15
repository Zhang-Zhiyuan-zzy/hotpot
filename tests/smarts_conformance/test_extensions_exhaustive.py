import pytest
from openbabel import openbabel as ob

import hotpot as hp


pytestmark = pytest.mark.smarts_core


PERIOD_BOUNDS = {
    1: range(1, 3),
    2: range(3, 11),
    3: range(11, 19),
    4: range(19, 37),
    5: range(37, 55),
    6: range(55, 87),
    7: range(87, 119),
}


@pytest.fixture(scope="module")
def elemental_targets():
    return {
        atomic_number: hp.read_mol(f"[{ob.GetSymbol(atomic_number)}]", "smi")
        for atomic_number in range(1, 119)
    }


def matching_atomic_numbers(query_text, elemental_targets):
    searcher = hp.Searcher(hp.Substructure.from_smarts(query_text))
    return {
        atomic_number
        for atomic_number, molecule in elemental_targets.items()
        if searcher.search(molecule)
    }


def test_lanthanide_and_actinide_extensions_have_exact_frozen_ranges(elemental_targets):
    assert matching_atomic_numbers("[Ln]", elemental_targets) == set(range(57, 72))
    assert matching_atomic_numbers("[An]", elemental_targets) == set(range(89, 104))


def test_period_extensions_cover_exact_periods_and_inclusive_ranges(elemental_targets):
    for period, expected_range in PERIOD_BOUNDS.items():
        assert matching_atomic_numbers(f"[NP{period}]", elemental_targets) == set(
            expected_range
        )

    assert matching_atomic_numbers("[NP3-5]", elemental_targets) == set(range(11, 55))


def test_group_extensions_follow_hotpot_atom_group_metadata(elemental_targets):
    for group in range(1, 19):
        expected = {
            atomic_number
            for atomic_number, molecule in elemental_targets.items()
            if molecule.atoms[0].group == group
        }
        assert matching_atomic_numbers(f"[NG{group}]", elemental_targets) == expected

    expected_range = {
        atomic_number
        for atomic_number, molecule in elemental_targets.items()
        if 3 <= molecule.atoms[0].group <= 8
    }
    assert matching_atomic_numbers("[NG3-8]", elemental_targets) == expected_range


def test_metal_extension_matches_hotpot_metal_classification(elemental_targets):
    expected = {
        atomic_number
        for atomic_number, molecule in elemental_targets.items()
        if molecule.atoms[0].is_metal
    }

    assert matching_atomic_numbers("[M]", elemental_targets) == expected
    assert (
        matching_atomic_numbers("[!M]", elemental_targets)
        == set(elemental_targets) - expected
    )
