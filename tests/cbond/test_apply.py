import inspect

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.AImodels.cbond import apply
from hotpot.cheminfo.AImodels.cbond.constants import DEFAULT_CBOND_THRESHOLD
from hotpot.cheminfo.core import Atom, Molecule


def _scores_by_element(context, values):
    return {
        atom_index: values[context.molecule.atoms[atom_index].symbol]
        for atom_index in context.candidate_indices
    }


def test_all_structures_branches_after_first_bond_and_merges_paths(monkeypatch):
    visited_states = []

    def predict_state(context, donor_indices, runtime):
        selected = frozenset(
            context.molecule.atoms[index].symbol for index in donor_indices
        )
        visited_states.append(selected)
        if not selected:
            values = {"N": 0.0, "O": 0.0, "S": -1.0}
        elif selected == {"N"}:
            values = {"N": 0.0, "O": 0.0, "S": 0.0}
        elif selected == {"O"}:
            values = {"N": 0.0, "O": 0.0, "S": -1.0}
        else:
            values = {"N": -1.0, "O": -1.0, "S": -1.0}
        return _scores_by_element(context, values)

    monkeypatch.setattr(apply, "_predict_state", predict_state)
    results = apply.build_all_possible_cbond(
        read_mol("NCCOCCS"),
        "Eu",
        return_details=True,
    )

    assert [result.path_count for result in results] == [2, 1]
    assert [result.probability for result in results] == pytest.approx([2 / 3, 1 / 3])
    assert [{step.element for step in result.steps} for result in results] == [
        {"N", "O"},
        {"N", "S"},
    ]
    assert len(visited_states) == len(set(visited_states)) == 5


def test_state_graph_has_no_metal_self_loop_or_duplicate_edges(monkeypatch):
    runtime = object()
    inspected_states = []

    def predict(model_data, observed_runtime):
        assert observed_runtime is runtime
        cbond_index = np.asarray(model_data["cbond_index"]).reshape(2, -1)
        metal_index = int(cbond_index[0, 0])
        candidates = tuple(int(index) for index in cbond_index[1])
        edges = [tuple(edge) for edge in np.asarray(model_data["edge_index"]).T]
        assert all(source != target for source, target in edges)
        assert len(edges) == len(set(edges))
        selected = frozenset(
            index
            for index in candidates
            if (metal_index, index) in edges and (index, metal_index) in edges
        )
        inspected_states.append(selected)
        scores = np.asarray(
            [2.0 if model_data["x"][index] == 7 else 1.0 for index in candidates]
        )
        return scores, cbond_index, model_data["is_cbond"]

    monkeypatch.setattr(apply, "cbond_prediction", predict)
    result = apply.auto_build_cbond(
        read_mol("NCCO"),
        "Eu",
        threshold=0.0,
        runtime=runtime,
        return_details=True,
    )

    assert [step.element for step in result.steps] == ["N", "O"]
    assert result.path_probability == pytest.approx(
        apply.signmod_with_offset(2.0) * apply.signmod_with_offset(1.0)
    )
    assert len(inspected_states) == len(set(inspected_states)) == 3
    metal = result.molecule.metals[0]
    assert {atom.symbol for atom in metal.neighbours} == {"N", "O"}


def test_default_threshold_is_strict(monkeypatch):
    def predict_state(context, donor_indices, runtime):
        return {
            atom_index: DEFAULT_CBOND_THRESHOLD
            for atom_index in context.candidate_indices
        }

    monkeypatch.setattr(apply, "_predict_state", predict_state)
    molecules, probabilities = apply.build_all_possible_cbond(
        read_mol("CN"),
        "Eu",
    )

    assert molecules == []
    assert probabilities == []


def test_single_structure_raises_when_no_bond_clears_threshold(monkeypatch):
    monkeypatch.setattr(
        apply,
        "_predict_state",
        lambda context, donor_indices, runtime: {
            atom_index: DEFAULT_CBOND_THRESHOLD
            for atom_index in context.candidate_indices
        },
    )

    with pytest.raises(ValueError, match="No coordination bond exceeded"):
        apply.auto_build_cbond(read_mol("CN"), "Eu")


def test_negative_logit_above_threshold_is_selected(monkeypatch):
    score = -0.1

    def predict_state(context, donor_indices, runtime):
        return {atom_index: score for atom_index in context.candidate_indices}

    monkeypatch.setattr(apply, "_predict_state", predict_state)
    result = apply.auto_build_cbond(
        read_mol("CN"),
        "Eu",
        return_details=True,
    )

    assert len(result.steps) == 1
    assert result.steps[0].score == score
    assert result.steps[0].probability == pytest.approx(
        apply.signmod_with_offset(score)
    )


def test_all_structures_enforces_state_limit(monkeypatch):
    def predict_state(context, donor_indices, runtime):
        return {atom_index: 0.0 for atom_index in context.candidate_indices}

    monkeypatch.setattr(apply, "_predict_state", predict_state)
    with pytest.raises(RuntimeError, match="max_states=2"):
        apply.build_all_possible_cbond(
            read_mol("NCCO"),
            "Eu",
            max_states=2,
        )


def test_structure_probability_and_unnormalized_path_weight_are_distinct(
    monkeypatch,
):
    def predict_state(context, donor_indices, runtime):
        return {atom_index: 0.0 for atom_index in context.candidate_indices}

    monkeypatch.setattr(apply, "_predict_state", predict_state)
    molecule = read_mol("NCCO")
    results = apply.build_all_possible_cbond(
        molecule,
        "Eu",
        return_details=True,
    )
    _, weights = apply.build_all_possible_cbond(
        molecule,
        "Eu",
        normalize_prob=False,
    )

    assert len(results) == 1
    assert results[0].probability == 1.0
    assert results[0].path_weight == pytest.approx(0.5)
    assert weights == pytest.approx([0.5])


def test_equal_scores_preserve_legacy_high_index_tie_break(monkeypatch):
    def predict_state(context, donor_indices, runtime):
        score = -1.0 if donor_indices else 0.0
        return {atom_index: score for atom_index in context.candidate_indices}

    monkeypatch.setattr(apply, "_predict_state", predict_state)
    result = apply.auto_build_cbond(
        read_mol("NCCO"),
        "Eu",
        return_details=True,
    )

    assert len(result.steps) == 1
    assert result.steps[0].element == "O"


def test_all_structures_does_not_mutate_input(monkeypatch):
    molecule = read_mol("CN")
    original_smiles = molecule.smiles
    original_atom_count = len(molecule.atoms)

    monkeypatch.setattr(
        apply,
        "_predict_state",
        lambda context, donor_indices, runtime: {
            atom_index: -1.0 for atom_index in context.candidate_indices
        },
    )
    apply.build_all_possible_cbond(molecule, "Eu")

    assert molecule.smiles == original_smiles
    assert len(molecule.atoms) == original_atom_count
    assert not molecule.metals


def test_existing_coordination_bond_is_reused_without_duplicate_edges(monkeypatch):
    molecule = read_mol("NCCO")
    metal = molecule.add_atom(Atom(symbol="Eu"))
    nitrogen = next(atom for atom in molecule.atoms if atom.symbol == "N")
    molecule.add_bond(metal, nitrogen)
    observed = {}

    def predict_state(context, donor_indices, runtime):
        observed["donors"] = donor_indices
        edges = [
            tuple(edge)
            for edge in apply._edge_index_for_state(context, donor_indices).T
        ]
        assert len(edges) == len(set(edges))
        assert all(source != target for source, target in edges)
        return {atom_index: -1.0 for atom_index in context.candidate_indices}

    monkeypatch.setattr(apply, "_predict_state", predict_state)
    results = apply.build_all_possible_cbond(
        molecule,
        metal,
        return_details=True,
    )

    assert observed["donors"] == {nitrogen.idx}
    assert len(results) == 1
    assert results[0].probability == 1.0
    assert results[0].steps == ()
    result_metal = results[0].molecule.metals[0]
    assert [atom.symbol for atom in result_metal.neighbours] == ["N"]


def test_multiple_metals_are_rejected_before_inference():
    molecule = read_mol("NCCO")
    europium = molecule.add_atom(Atom(symbol="Eu"))
    molecule.add_atom(Atom(symbol="Gd"))

    with pytest.raises(ValueError, match="exactly one metal centre"):
        apply.auto_build_cbond(molecule, europium)


def test_all_structures_with_no_candidate_returns_empty():
    molecules, probabilities = apply.build_all_possible_cbond(
        read_mol("CC"),
        "Eu",
    )

    assert molecules == []
    assert probabilities == []


def test_public_backend_defaults_are_synchronized():
    assert (
        inspect.signature(apply.auto_build_cbond).parameters["threshold"].default
        == DEFAULT_CBOND_THRESHOLD
    )
    assert (
        inspect.signature(Molecule.build_all_pair_links).parameters["threshold"].default
        == DEFAULT_CBOND_THRESHOLD
    )
    assert (
        inspect.signature(Molecule.auto_pair_metal).parameters["threshold"].default
        == DEFAULT_CBOND_THRESHOLD
    )
    assert (
        inspect.signature(apply.build_one_cbond).parameters["threshold"].default
        == DEFAULT_CBOND_THRESHOLD
    )
    assert (
        inspect.signature(apply.build_all_possible_cbond)
        .parameters["threshold"]
        .default
        == DEFAULT_CBOND_THRESHOLD
    )


def test_auto_build_cbond_real_runtime_smoke(runtime):
    result, probability = apply.auto_build_cbond(
        read_mol("CN"),
        "Eu",
        runtime=runtime,
    )

    metal = result.metals[0]
    assert metal.neighbours
    assert all(bond.atom1 is not bond.atom2 for bond in result.bonds)
    assert 0.0 < probability <= 1.0


def test_documented_aminodiol_reference_matches_packaged_model(runtime):
    results = apply.build_all_possible_cbond(
        read_mol("NCC(O)CO"),
        "Eu",
        runtime=runtime,
        return_details=True,
    )

    assert [result.molecule.smiles for result in results] == [
        "C1O[Eu]2O[C@@H]1C[NH2+]2",
        "NC[C@@H]1CO[Eu]O1",
    ]
    assert [result.probability for result in results] == pytest.approx(
        [0.5787371244916796, 0.4212628755083204]
    )
    assert [
        [(step.atom_index, step.element) for step in result.steps] for result in results
    ] == [[(0, "N"), (3, "O"), (5, "O")], [(3, "O"), (5, "O")]]
    assert [step.score for result in results for step in result.steps] == pytest.approx(
        [
            0.8958554863929749,
            3.6056129932403564,
            4.415280342102051,
            2.3229620456695557,
            3.092538833618164,
        ]
    )
