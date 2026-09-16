"""Coordination-bond inference and structure assembly."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import os
from typing import Any, Union

import numpy as np

from ...core import Atom, Molecule
from .. import data_extract as de
from .constants import DEFAULT_CBOND_THRESHOLD, DEFAULT_MAX_STATES
from .runtime import CBondRuntime, padding_rings


@dataclass(frozen=True)
class CBondStep:
    """One coordination bond selected along an inference path."""

    atom_index: int
    element: str
    score: float
    probability: float


@dataclass(frozen=True)
class CBondPathResult:
    """A single greedily assembled coordination structure."""

    molecule: Molecule
    steps: tuple[CBondStep, ...]
    path_probability: float
    donor_indices: tuple[int, ...]


@dataclass(frozen=True)
class CBondStructureResult:
    """A terminal structure produced by exact path enumeration."""

    molecule: Molecule
    probability: float
    steps: tuple[CBondStep, ...]
    donor_indices: tuple[int, ...]
    path_count: int
    log_path_weight: float


@dataclass
class _SearchState:
    log_weight: float
    map_log_weight: float
    map_steps: tuple[CBondStep, ...]
    path_count: int


@dataclass(frozen=True)
class _SearchContext:
    molecule: Molecule
    metal_index: int
    existing_donors: frozenset[int]
    candidate_indices: tuple[int, ...]
    base_data: dict[str, Any]
    ligand_edge_index: np.ndarray


@lru_cache(maxsize=None)
def get_cbond_runtime(
    device: str = None,
    model_dir: str = None,
) -> CBondRuntime:
    return CBondRuntime(
        model_dir=model_dir,
        device=device or os.environ.get("HOTPOT_CBOND_DEVICE", "auto"),
    )


def extract_cbond_inputs(mol: Molecule) -> dict[str, Any]:
    data = de.extract_atom_attrs(mol, {}, atomic_number_only=True)
    data = de.extract_bond_attrs(mol, data=data)
    data = de.extract_ring_attrs(mol, data=data)
    return de.extract_potentials_cbonds(mol, data=data)


def get_graph_cbond_inputs(data: dict[str, Any]):
    return {
        "x": data["x"],
        "edge_index": data["edge_index"],
    }


def get_cbond_inputs_model(data: dict[str, Any], xg):
    return padding_rings(
        xg,
        data["rings_node_index"],
        data["rings_node_nums"],
    )


def pred_xg(mol_data: dict[str, Any], runtime: CBondRuntime = None):
    inputs = get_graph_cbond_inputs(mol_data)
    runtime = runtime or get_cbond_runtime()
    return runtime.embed_graph(inputs["x"], inputs["edge_index"])


def pred_cb_value(
    xg,
    padded_xr,
    rings_mask,
    cbond_index,
    runtime: CBondRuntime = None,
):
    runtime = runtime or get_cbond_runtime()
    return runtime.predict(xg, padded_xr, rings_mask, cbond_index)


def cbond_prediction(mol_data: dict[str, Any], runtime: CBondRuntime = None):
    runtime = runtime or get_cbond_runtime()
    xg = pred_xg(mol_data, runtime)
    padded_xr, rings_mask = get_cbond_inputs_model(mol_data, xg)
    cbond_index = mol_data["cbond_index"]
    cbond = pred_cb_value(
        xg,
        padded_xr,
        rings_mask,
        cbond_index,
        runtime,
    )
    return cbond, cbond_index, mol_data["is_cbond"]


def signmod_with_offset(x, offset: float = 0.0):
    """Return a stable sigmoid while retaining the historical public helper."""
    return np.exp(-np.logaddexp(0.0, -(np.asarray(x) - offset)))


def _sigmoid(value: float) -> float:
    return float(np.exp(-np.logaddexp(0.0, -float(value))))


def _log_sigmoid(value: float) -> float:
    return float(-np.logaddexp(0.0, -float(value)))


def init_metal_ligand_pair(
    mol: Molecule,
    metal: Union[int, str, Atom],
):
    if isinstance(metal, str):
        metal = Atom(symbol=metal)
    elif isinstance(metal, int):
        metal = Atom(atomic_number=metal)
    elif not isinstance(metal, Atom):
        raise TypeError(
            "metal should be the atomic_number(int), atomic_symbol(str) "
            "or an Atom object"
        )

    assert metal.is_metal, f"{metal.symbol} is not a metal"
    mol.add_hydrogens()
    mol.force_remove_polar_hydrogens()
    if metal not in mol.atoms:
        existing_metals = mol.metals
        if (
            len(existing_metals) == 1
            and existing_metals[0].atomic_number == metal.atomic_number
        ):
            metal = existing_metals[0]
        else:
            assert len(existing_metals) == 0, (
                "Only support identification of coordination pattern between "
                "a single metal and a ligand"
            )
            metal = mol.add_atom(metal)

    return mol, metal


def _prepare_search(
    mol: Molecule,
    metal: Union[int, str, Atom],
) -> _SearchContext:
    molecule, metal_atom = init_metal_ligand_pair(mol, metal)
    data = extract_cbond_inputs(molecule)
    metal_index = metal_atom.idx
    cbond_index = np.asarray(data["cbond_index"], dtype=np.int64)
    candidate_indices = (
        tuple(dict.fromkeys(int(index) for index in cbond_index.reshape(2, -1)[1]))
        if cbond_index.size
        else ()
    )

    edge_index = np.asarray(data["edge_index"], dtype=np.int64).reshape(2, -1)
    ligand_mask = (edge_index[0] != metal_index) & (edge_index[1] != metal_index)
    ligand_edge_index = edge_index[:, ligand_mask]
    existing_donors = frozenset(
        atom.idx for atom in metal_atom.neighbours if atom.idx != metal_index
    )
    return _SearchContext(
        molecule=molecule,
        metal_index=metal_index,
        existing_donors=existing_donors,
        candidate_indices=candidate_indices,
        base_data=data,
        ligand_edge_index=ligand_edge_index,
    )


def _edge_index_for_state(
    context: _SearchContext,
    donor_indices: frozenset[int],
) -> np.ndarray:
    donors = tuple(sorted(donor_indices))
    if not donors:
        return context.ligand_edge_index
    coordination_edges = np.asarray(
        [
            (context.metal_index,) * len(donors) + donors,
            donors + (context.metal_index,) * len(donors),
        ],
        dtype=np.int64,
    )
    return np.concatenate((context.ligand_edge_index, coordination_edges), axis=1)


def _predict_state(
    context: _SearchContext,
    donor_indices: frozenset[int],
    runtime: CBondRuntime = None,
) -> dict[int, float]:
    if not context.candidate_indices:
        return {}
    model_data = dict(context.base_data)
    model_data["edge_index"] = _edge_index_for_state(context, donor_indices)
    prediction, cbond_index, _ = cbond_prediction(model_data, runtime)
    candidate_indices = np.asarray(cbond_index, dtype=np.int64).reshape(2, -1)[1]
    scores = np.asarray(prediction, dtype=float).reshape(-1)
    return {
        int(atom_index): float(score)
        for atom_index, score in zip(candidate_indices, scores)
    }


def _eligible_candidates(
    scores: dict[int, float],
    donor_indices: frozenset[int],
    threshold: float,
    greedy: bool,
) -> list[tuple[int, float]]:
    if not scores:
        return []
    if not greedy:
        highest_index, highest_score = max(
            scores.items(),
            key=lambda item: (item[1], -item[0]),
        )
        if highest_index in donor_indices and highest_score > threshold:
            return []
    return [
        (atom_index, score)
        for atom_index, score in scores.items()
        if atom_index not in donor_indices and score > threshold
    ]


def _step(context: _SearchContext, atom_index: int, score: float) -> CBondStep:
    return CBondStep(
        atom_index=atom_index,
        element=context.molecule.atoms[atom_index].symbol,
        score=score,
        probability=_sigmoid(score),
    )


def _materialize_state(
    context: _SearchContext,
    donor_indices: frozenset[int],
) -> Molecule:
    molecule = context.molecule.copy()
    for atom_index in sorted(donor_indices - context.existing_donors):
        molecule.add_bond(context.metal_index, atom_index)
    return molecule


def auto_build_cbond(
    mol: Molecule,
    metal: Union[int, str, Atom],
    threshold: float = DEFAULT_CBOND_THRESHOLD,
    greedy: bool = True,
    sum_prob: bool = True,
    runtime: CBondRuntime = None,
    *,
    return_details: bool = False,
):
    """Greedily add the highest-scoring eligible bond until convergence."""
    context = _prepare_search(mol, metal)
    donor_indices = context.existing_donors
    steps = []

    while True:
        scores = _predict_state(context, donor_indices, runtime)
        eligible = _eligible_candidates(scores, donor_indices, threshold, greedy)
        if not eligible:
            break
        atom_index, score = max(
            eligible,
            key=lambda item: (item[1], -item[0]),
        )
        logging.debug(
            "Selected coordination atom %s with raw score %.6f",
            atom_index,
            score,
        )
        steps.append(_step(context, atom_index, score))
        donor_indices = donor_indices | {atom_index}

    for atom_index in sorted(donor_indices - context.existing_donors):
        context.molecule.add_bond(context.metal_index, atom_index)

    path_probability = float(np.exp(sum(_log_sigmoid(step.score) for step in steps)))
    result = CBondPathResult(
        molecule=context.molecule,
        steps=tuple(steps),
        path_probability=path_probability,
        donor_indices=tuple(sorted(donor_indices)),
    )
    if return_details:
        return result
    if sum_prob:
        return result.molecule, result.path_probability
    return result.molecule, [step.probability for step in result.steps]


def build_one_cbond(
    mol: Molecule,
    metal: Union[int, str, Atom],
    threshold: float = DEFAULT_CBOND_THRESHOLD,
    get_all: bool = False,
    runtime: CBondRuntime = None,
):
    """Build one-bond seed structures using raw-logit thresholding."""
    metal_spec = metal.atomic_number if isinstance(metal, Atom) else metal
    context = _prepare_search(mol.copy(), metal_spec)
    scores = _predict_state(context, context.existing_donors, runtime)
    eligible = sorted(
        _eligible_candidates(
            scores,
            context.existing_donors,
            threshold,
            greedy=True,
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not eligible:
        logging.info("Not found any suitable coordination bond")
        return None, None

    if not get_all:
        atom_index, score = eligible[0]
        context.molecule.add_bond(context.metal_index, atom_index)
        return context.molecule, [score]

    molecules = []
    scores_by_molecule = []
    for atom_index, score in eligible:
        molecule = context.molecule.copy()
        molecule.add_bond(context.metal_index, atom_index)
        molecules.append(molecule)
        scores_by_molecule.append([score])
    return molecules, scores_by_molecule


def _merge_child_state(
    next_states: dict[frozenset[int], _SearchState],
    child_indices: frozenset[int],
    parent: _SearchState,
    step: CBondStep,
) -> bool:
    edge_log_weight = _log_sigmoid(step.score)
    log_weight = parent.log_weight + edge_log_weight
    map_log_weight = parent.map_log_weight + edge_log_weight
    map_steps = parent.map_steps + (step,)
    child = next_states.get(child_indices)
    if child is None:
        next_states[child_indices] = _SearchState(
            log_weight=log_weight,
            map_log_weight=map_log_weight,
            map_steps=map_steps,
            path_count=parent.path_count,
        )
        return True

    child.log_weight = float(np.logaddexp(child.log_weight, log_weight))
    child.path_count += parent.path_count
    if map_log_weight > child.map_log_weight:
        child.map_log_weight = map_log_weight
        child.map_steps = map_steps
    return False


def build_all_possible_cbond(
    mol: Molecule,
    m: Union[int, str, Atom],
    threshold: float = DEFAULT_CBOND_THRESHOLD,
    greedy: bool = True,
    normalize_prob: bool = True,
    runtime: CBondRuntime = None,
    *,
    max_states: int = DEFAULT_MAX_STATES,
    return_details: bool = False,
):
    """Enumerate every terminal coordination state above ``threshold``.

    Each unique donor-index set is evaluated once. Different bond-order paths
    reaching the same state are merged in log space. Terminal path weights are
    optionally normalized over the returned structures.
    """
    metal_spec = m.atomic_number if isinstance(m, Atom) else m
    context = _prepare_search(mol.copy(), metal_spec)
    initial_indices = context.existing_donors
    frontier = {
        initial_indices: _SearchState(
            log_weight=0.0,
            map_log_weight=0.0,
            map_steps=(),
            path_count=1,
        )
    }
    terminal_states = {}
    discovered_states = 1

    while frontier:
        next_states = {}
        for donor_indices in sorted(frontier, key=lambda state: tuple(sorted(state))):
            state = frontier[donor_indices]
            scores = _predict_state(context, donor_indices, runtime)
            eligible = _eligible_candidates(
                scores,
                donor_indices,
                threshold,
                greedy,
            )
            if not eligible:
                if donor_indices != initial_indices or initial_indices:
                    terminal_states[donor_indices] = state
                continue

            for atom_index, score in sorted(
                eligible,
                key=lambda item: (-item[1], item[0]),
            ):
                child_indices = donor_indices | {atom_index}
                is_new = _merge_child_state(
                    next_states,
                    child_indices,
                    state,
                    _step(context, atom_index, score),
                )
                if is_new:
                    discovered_states += 1
                    if discovered_states > max_states:
                        raise RuntimeError(
                            "CBond enumeration exceeded max_states="
                            f"{max_states}; raise --threshold or --max-states"
                        )
        frontier = next_states

    if not terminal_states:
        return [] if return_details else ([], [])

    terminal_items = list(terminal_states.items())
    log_weights = np.asarray(
        [state.log_weight for _, state in terminal_items],
        dtype=float,
    )
    if normalize_prob:
        max_log_weight = float(np.max(log_weights))
        weights = np.exp(log_weights - max_log_weight)
        probabilities = weights / np.sum(weights)
    else:
        probabilities = np.exp(log_weights)

    results = [
        CBondStructureResult(
            molecule=_materialize_state(context, donor_indices),
            probability=float(probability),
            steps=state.map_steps,
            donor_indices=tuple(sorted(donor_indices)),
            path_count=state.path_count,
            log_path_weight=state.log_weight,
        )
        for (donor_indices, state), probability in zip(
            terminal_items,
            probabilities,
        )
    ]
    results.sort(key=lambda result: (-result.probability, result.donor_indices))
    if return_details:
        return results
    return (
        [result.molecule for result in results],
        [result.probability for result in results],
    )
