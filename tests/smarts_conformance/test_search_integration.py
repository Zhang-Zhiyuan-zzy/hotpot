import pytest

import hotpot as hp


pytestmark = pytest.mark.smarts_core


def target_sets(hits):
    return tuple(hit.atom_indices for hit in hits)


@pytest.mark.smarts_smoke
def test_public_search_entry_points_agree():
    molecule = hp.read_mol("CC(=O)N", "smi")
    query = hp.Substructure.from_smarts("C(=O)N")

    assert target_sets(molecule.search_substructure("C(=O)N")) == target_sets(
        hp.Searcher(query).search(molecule)
    )


def test_reused_searcher_matches_fresh_searchers_for_multiple_targets():
    query = hp.Substructure.from_smarts("[O;H1]")
    reused = hp.Searcher(query)
    targets = [hp.read_mol(smiles, "smi") for smiles in ("CCO", "CC=O", "O", "CC(O)C")]

    reused_results = [target_sets(reused.search(target)) for target in targets]
    fresh_results = [
        target_sets(hp.Searcher(query).search(target)) for target in targets
    ]

    assert reused_results == fresh_results


def test_linear_batch_baseline_has_stable_ids_and_is_repeatable():
    records = [
        ("ethanol", hp.read_mol("CCO", "smi")),
        ("acetaldehyde", hp.read_mol("CC=O", "smi")),
        ("ethylamine", hp.read_mol("CCN", "smi")),
        ("sodium", hp.read_mol("[Na+]", "smi")),
    ]
    searcher = hp.Searcher(hp.Substructure.from_smarts("[O]"))

    first = [record_id for record_id, molecule in records if searcher.search(molecule)]
    second = [record_id for record_id, molecule in records if searcher.search(molecule)]

    assert first == second == ["ethanol", "acetaldehyde"]


def test_local_query_survives_unrelated_disconnected_target_component():
    base = hp.read_mol("CCO", "smi")
    extended = hp.read_mol("CCO.[Na+]", "smi")

    assert target_sets(base.search_substructure("CO")) == target_sets(
        extended.search_substructure("CO")
    )


def test_equivalent_smiles_order_preserves_match_existence():
    for smarts in ("CO", "[O;H1]", "[C;D1]", "C~O"):
        assert bool(hp.read_mol("CCO", "smi").search_substructure(smarts)) == bool(
            hp.read_mol("OCC", "smi").search_substructure(smarts)
        )


def test_current_capability_boundary_has_no_native_search_options_or_index():
    searcher = hp.Searcher(hp.Substructure.from_smarts("C"))

    assert not hasattr(searcher, "search_many")
    assert not hasattr(searcher, "max_matches")
    assert not hasattr(searcher, "use_chirality")
