import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot.cheminfo.rules.mathematics import geometry as geo
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.structure._selection import _capture_topology_selection, _selected_bonds
from hotpot.cheminfo.operations.topology_queries import _project_molecule
from hotpot.cheminfo.forcefields import backend as ob_backend
from hotpot.cheminfo.forcefields import ff, repair
from hotpot.cheminfo.forcefields import ff as forcefield_utils
from hotpot.cheminfo.forcefields.trajectory import ForceFieldTrajectory
from hotpot.cheminfo.structure.forcefields import (
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)


def _UntanglingMolecule():
    molecule = _ring_molecule()
    molecule.coordinates = np.zeros((3, 3))
    molecule.events = []
    return molecule


class _TraceMolecule:
    def __init__(self, frames):
        self.coordinates = np.asarray(frames[-1], dtype=float).copy()
        self._conformers = [
            {"coordinates": np.asarray(frame, dtype=float).copy(), "energy": float(index)}
            for index, frame in enumerate(frames)
        ]
        self._conformers_index = len(self._conformers) - 1

    @property
    def conformers_number(self):
        return len(self._conformers)

    def conformer_get(self, index):
        return self._conformers[index]

    def conformer_clear(self):
        self._conformers.clear()
        self._conformers_index = 0

    def conformer_add(self, coordinates, energies):
        coordinate_frames = np.asarray(coordinates, dtype=float)
        energy_values = np.asarray(energies, dtype=float).reshape(-1)
        self._conformers.extend(
            {
                "coordinates": frame.copy(),
                "energy": float(energy),
            }
            for frame, energy in zip(coordinate_frames, energy_values)
        )

    def conformer_load(self, index):
        self._conformers_index = index
        self.coordinates = self._conformers[index]["coordinates"].copy()


def _key(bond):
    return tuple(sorted((bond.atom1.idx, bond.atom2.idx)))


def _report(count):
    findings = tuple(
        SimpleNamespace(
            target=SimpleNamespace(
                ring=SimpleNamespace(ring=f"ring-{index}"),
                bond=SimpleNamespace(bond=f"bond-{index}"),
            ),
            relation=SimpleNamespace(
                state=repair.geo.PiercingState.PIERCES,
            ),
        )
        for index in range(count)
    )
    return SimpleNamespace(
        findings=findings,
        piercings=findings,
        undetermined=(),
        ring_scope="ligand_skeleton",
        max_ring_size=16,
        selected_ring_count=1 if count else 0,
        excluded_ring_count=0,
        candidate_pair_count=count,
        aabb_separated_pair_count=0,
        exact_pair_count=count,
        piercing_pair_count=count,
        does_not_pierce_pair_count=0,
        undetermined_pair_count=0,
        scan_complete=True,
        state=(
            repair.geo.PiercingState.PIERCES
            if count
            else repair.geo.PiercingState.DOES_NOT_PIERCE
        ),
    )


def _mock_ring_watch(monkeypatch, states, *, events=None):
    watch = (
        repair._WatchedRingPiercing(
            repair._BondRingPairKey((0, 1, 2), (3, 4)),
            ((0, 1),),
        ),
    )
    state_iterator = iter(states)

    monkeypatch.setattr(
        repair,
        "_ring_piercing_watch",
        lambda *args, **kwargs: watch,
    )

    def scan(*args, **kwargs):
        if events is not None:
            events.append(("targeted_scan",))
        state = next(state_iterator)
        return repair._RingPiercingWatchResult(
            state,
            watch if state is repair.geo.PiercingState.PIERCES else (),
        )

    monkeypatch.setattr(repair, "_scan_ring_piercing_watch", scan)
    return watch


def _optimization(energy):
    return ob_backend._CandidateOptimizationResult(
        energy=float(energy),
        energy_unit="kJ/mol",
        exploded=False,
    )


def _ring_molecule():
    molecule = Molecule()
    for coordinate in ((0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.75, 1.3, 0.0)):
        molecule.create_atom(atomic_number=6, coordinates=coordinate)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        molecule.add_bond(first, second, bond_order=1.0)
    return molecule


def _pierced_ring_molecule():
    molecule = Molecule()
    coordinates = (
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    )
    for coordinate in coordinates:
        molecule.create_atom(atomic_number=6, coordinates=coordinate)
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)):
        molecule.add_bond(first, second, bond_order=1.0)
    return molecule


def _untangling_result(
    *,
    piercing_count=0,
):
    report = forcefield_utils.RingUntanglingReport(
        attempt_limit=3,
        attempts_completed=1,
        initial_piercing_count=piercing_count,
        final_piercing_count=0,
        minimum_piercing_count=0,
        resolved=True,
    )
    return repair._RingUntanglingResult(
        report=report,
        energy=1.0,
        checkpoint_report=_report(0),
    )


def test_ring_untangling_opens_perturbs_optimizes_closes_and_rechecks(monkeypatch):
    molecule = _UntanglingMolecule()
    opening_edge = molecule.bond(0, 1)
    initial_checkpoint = _report(1)
    scans = iter(
        (
            _report(0),
            _report(0),
        )
    )

    def scan(*args, **kwargs):
        molecule.events.append(("scan",))
        return next(scans)

    def perturb(coordinates, **kwargs):
        molecule.events.append(("perturb",))
        return np.asarray(coordinates) + 1.0

    def optimize(current_molecule, forcefield, steps, *, selection=None):
        molecule.events.append(("optimize", steps))
        assert len(current_molecule.bonds) == 3
        if selection is not None:
            assert _selected_bonds(current_molecule, selection) == tuple(
                bond for bond in current_molecule.bonds if bond is not opening_edge)
        return _optimization(steps)

    monkeypatch.setattr(repair, "_scan_ring_checkpoint", scan)
    _mock_ring_watch(
        monkeypatch,
        (repair.geo.PiercingState.DOES_NOT_PIERCE,),
        events=molecule.events,
    )
    monkeypatch.setattr(
        repair,
        "_select_ring_opening_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(repair, "_perturbed_coordinates", perturb)
    monkeypatch.setattr(repair, "_single_ob_optimization", optimize)

    result = repair._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=2,
        short_steps=4,
        settling_steps=9,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        checkpoint_report=initial_checkpoint,
    )

    assert molecule.events == [
        ("perturb",),
        ("optimize", 4),
        ("targeted_scan",),
        ("scan",),
        ("optimize", 9),
        ("scan",),
    ]
    assert result.report.resolved
    assert result.report.attempts_completed == 1
    assert result.checkpoint_report.state is geo.PiercingState.DOES_NOT_PIERCE


def test_single_watched_repair_only_observes_the_fixed_watch_batch(monkeypatch):
    molecule = _UntanglingMolecule()
    opening_edge = molecule.bond(0, 1)
    watch = _mock_ring_watch(
        monkeypatch,
        (repair.geo.PiercingState.DOES_NOT_PIERCE,),
        events=molecule.events,
    )
    monkeypatch.setattr(
        repair,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: pytest.fail(
            "a single watched repair must not run a full checkpoint"
        ),
    )
    monkeypatch.setattr(
        repair,
        "_select_ring_opening_edge",
        lambda *args, **kwargs: opening_edge,
    )

    def perturb(coordinates, **kwargs):
        molecule.events.append(("perturb",))
        return np.asarray(coordinates) + 1.0

    def optimize(current_molecule, forcefield, steps, *, selection=None):
        molecule.events.append(("optimize", steps))
        assert len(current_molecule.bonds) == 3
        if selection is not None:
            assert _selected_bonds(current_molecule, selection) == tuple(
                bond for bond in current_molecule.bonds if bond is not opening_edge)
        return _optimization(steps)

    monkeypatch.setattr(repair, "_perturbed_coordinates", perturb)
    monkeypatch.setattr(repair, "_single_ob_optimization", optimize)
    trajectory_recorder = repair._RingTrajectoryRecorder(
        mol=molecule,
        trajectory=None,
        stage=TrajectoryStage.COMPLEX_UNTANGLING,
        enabled=False,
    )

    observation = repair._repair_watched_ring_piercings_once(
        molecule,
        "UFF",
        current_piercings=watch,
        watch_batch=watch,
        attempt=1,
        short_steps=4,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        trajectory_recorder=trajectory_recorder,
    )

    assert observation == repair._WatchedRingRepairObservation(
        repair._RingPiercingWatchResult(
            repair.geo.PiercingState.DOES_NOT_PIERCE,
            (),
        )
    )
    assert molecule.events == [
        ("perturb",),
        ("optimize", 4),
        ("targeted_scan",),
    ]


def test_ring_piercing_watch_tracks_only_the_confirmed_pair():
    molecule = _pierced_ring_molecule()
    report = repair._scan_ring_checkpoint(
        molecule,
        ring_scope="ligand_skeleton",
    )

    assert report.state is geo.PiercingState.PIERCES
    watch = repair._ring_piercing_watch(molecule, report)
    assert len(watch) == 1
    assert repair._scan_ring_piercing_watch(molecule, watch) == (
        repair._RingPiercingWatchResult(
            geo.PiercingState.PIERCES,
            watch,
        )
    )

    molecule.atoms[4].coordinates = (3.0, 0.0, -1.0)
    molecule.atoms[5].coordinates = (3.0, 0.0, 1.0)

    assert repair._scan_ring_piercing_watch(molecule, watch) == (
        repair._RingPiercingWatchResult(
            geo.PiercingState.DOES_NOT_PIERCE,
            (),
        )
    )


def test_chelate_ring_watch_only_exposes_metal_ligand_opening_edges():
    molecule = Molecule()
    for atomic_number, coordinate in zip(
        (30, 7, 6, 6, 6),
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.5, 0.3, -1.0),
            (0.5, 0.3, 1.0),
        ),
    ):
        molecule.create_atom(
            atomic_number=atomic_number,
            coordinates=coordinate,
        )
    metal_nitrogen = molecule.add_bond(0, 1, bond_order=1.0)
    organic_edge = molecule.add_bond(1, 2, bond_order=1.0)
    metal_carbon = molecule.add_bond(2, 0, bond_order=1.0)
    target_bond = molecule.add_bond(3, 4, bond_order=1.0)
    ring = molecule.rings_for_scope("full_graph")[0]
    finding = SimpleNamespace(
        target=SimpleNamespace(
            ring=SimpleNamespace(
                key=tuple(atom.idx for atom in ring.atoms),
                ring=ring,
            ),
            bond=SimpleNamespace(key=_key(target_bond), bond=target_bond),
        ),
    )
    report = SimpleNamespace(
        ring_scope="full_graph",
        piercings=(finding,),
    )

    watch = repair._ring_piercing_watch(molecule, report)

    assert watch[0].opening_edge_keys == tuple(sorted((
        _key(metal_nitrogen),
        _key(metal_carbon),
    )))
    assert _key(organic_edge) not in watch[0].opening_edge_keys


def test_unrepairable_chelate_watch_warns_without_opening_a_bond(monkeypatch):
    molecule = _UntanglingMolecule()
    molecule.remove_bonds(molecule.bonds)
    checkpoint = _report(1)
    watch = (
        repair._WatchedRingPiercing(
            repair._BondRingPairKey((0, 1, 2), (3, 4)),
            (),
        ),
    )
    monkeypatch.setattr(
        repair,
        "_ring_piercing_watch",
        lambda *args, **kwargs: watch,
    )
    monkeypatch.setattr(
        repair,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: checkpoint,
    )

    result = repair._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=3,
        short_steps=1,
        settling_steps=0,
        perturb_sigma=0.0,
        rng=np.random.default_rng(7),
        checkpoint_report=checkpoint,
    )

    assert result.report.attempts_completed == 0
    assert result.report.resolved is False
    assert "no eligible ring-opening edge" in result.report.warning_messages[0]
    assert not any(event[0] == "open" for event in molecule.events)

def test_ring_checkpoint_preserves_a_non_piercing_report(monkeypatch):
    molecule = _UntanglingMolecule()
    expected = _report(0)
    calls = []

    def screen(current_molecule, *, ring_scope, max_ring_size):
        calls.append((current_molecule, ring_scope, max_ring_size))
        return expected

    monkeypatch.setattr(repair.geometry_adapter, "screen_bond_ring_relations", screen)

    result = repair._scan_ring_checkpoint(
        molecule,
        ring_scope="full_graph",
    )

    assert result is expected
    assert calls == [(molecule, "full_graph", repair._BOND_RING_MAX_SIZE)]


def test_ring_untangling_consumes_the_entry_checkpoint_without_rescanning(
    monkeypatch,
):
    molecule = _UntanglingMolecule()
    checkpoint = _report(0)

    monkeypatch.setattr(
        repair,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the repair entry must not rescan")
        ),
    )

    result = repair._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=2,
        short_steps=4,
        settling_steps=0,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        checkpoint_report=checkpoint,
    )

    assert result.report.resolved
    assert result.checkpoint_report is checkpoint


def test_ring_untangling_trajectory_preserves_open_and_closed_topologies(
    monkeypatch,
):
    molecule = _ring_molecule()
    opening_edge = molecule.bond(0, 1)
    initial_checkpoint = _report(1)
    scans = iter(
        (
            _report(0),
            _report(0),
        )
    )

    monkeypatch.setattr(
        repair,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: next(scans),
    )
    _mock_ring_watch(
        monkeypatch,
        (repair.geo.PiercingState.DOES_NOT_PIERCE,),
    )
    monkeypatch.setattr(
        repair,
        "_select_ring_opening_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(
        repair,
        "_perturbed_coordinates",
        lambda coordinates, **kwargs: np.asarray(coordinates) + 1.0,
    )
    monkeypatch.setattr(
        repair,
        "_single_ob_optimization",
        lambda current_molecule, forcefield, steps, **kwargs: _optimization(steps),
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.COMPLEX_UNTANGLING,
    )
    repair._record_ring_checkpoint(
        molecule,
        initial_checkpoint,
        trajectory=trajectory,
        stage=TrajectoryStage.COMPLEX_UNTANGLING,
    )

    repair._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=2,
        short_steps=4,
        settling_steps=9,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        checkpoint_report=initial_checkpoint,
        trajectory=trajectory,
    )

    assert tuple(frame.event for frame in trajectory) == (
        TrajectoryEvent.TOPOLOGY_CHECKPOINT,
        TrajectoryEvent.RING_OPENED,
        TrajectoryEvent.PERTURBED,
        TrajectoryEvent.OPTIMIZED,
        TrajectoryEvent.TOPOLOGY_CHECKPOINT,
        TrajectoryEvent.TOPOLOGY_CHECKPOINT,
        TrajectoryEvent.TERMINAL,
    )
    assert tuple(len(trajectory.topology(frame.index).bonds) for frame in trajectory) == (
        3,
        2,
        2,
        2,
        3,
        3,
        3,
    )
    assert tuple(frame.energy_kj_mol for frame in trajectory) == (
        None,
        None,
        None,
        4.0,
        None,
        9.0,
        9.0,
    )
    assert trajectory[0].evidence == repair._ring_frame_evidence(
        geo.PiercingState.PIERCES,
        initial_checkpoint,
    )
    assert trajectory[4].evidence == repair._ring_frame_evidence(
        geo.PiercingState.DOES_NOT_PIERCE,
        _report(0),
    )
    assert trajectory.selected_index == 6


def test_ring_untangling_does_not_assign_open_topology_energy_to_closed_frame(
    monkeypatch,
):
    molecule = _ring_molecule()
    opening_edge = molecule.bond(0, 1)
    initial_checkpoint = _report(1)
    scans = iter(
        (
            _report(1),
            _report(2),
        )
    )
    monkeypatch.setattr(
        repair,
        "_scan_ring_checkpoint",
        lambda *args, **kwargs: next(scans),
    )
    _mock_ring_watch(
        monkeypatch,
        (repair.geo.PiercingState.PIERCES,),
    )
    monkeypatch.setattr(
        repair,
        "_select_ring_opening_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(
        repair,
        "_perturbed_coordinates",
        lambda coordinates, **kwargs: np.asarray(coordinates) + 1.0,
    )
    monkeypatch.setattr(
        repair,
        "_single_ob_optimization",
        lambda current_molecule, forcefield, steps, **kwargs: _optimization(steps),
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.COMPLEX_UNTANGLING,
    )

    result = repair._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=1,
        short_steps=4,
        settling_steps=9,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        checkpoint_report=initial_checkpoint,
        trajectory=trajectory,
    )

    assert np.isnan(result.energy)
    assert trajectory.selected_frame is not None
    assert trajectory.selected_frame.event is TrajectoryEvent.TERMINAL
    assert trajectory.selected_frame.energy_kj_mol is None


def test_ring_repair_operation_selection_is_independent_of_coexisting_projection(monkeypatch):
    molecule = _ring_molecule()
    originals = tuple(molecule.bonds)
    other_selection = _capture_topology_selection(molecule, excluded_bonds=(originals[1],))
    projected = _project_molecule(molecule, Molecule, selection=other_selection)
    opening_edge = projected.bond(0, 1)
    projected_bonds = tuple(projected.bonds)
    watch = _mock_ring_watch(monkeypatch, (geo.PiercingState.DOES_NOT_PIERCE,))
    monkeypatch.setattr(repair, '_select_ring_opening_edge', lambda *args: opening_edge)
    monkeypatch.setattr(repair, '_perturbed_coordinates', lambda coordinates, **kwargs: coordinates)
    seen = []
    def optimize(current, forcefield, steps, *, selection=None):
        seen.append(_selected_bonds(current, selection))
        assert tuple(current.bonds) == projected_bonds
        return _optimization(1.)
    monkeypatch.setattr(repair, '_single_ob_optimization', optimize)
    recorder = repair._RingTrajectoryRecorder(projected, None, TrajectoryStage.COMPLEX_UNTANGLING, False)
    repair._repair_watched_ring_piercings_once(
        projected, 'UFF', current_piercings=watch, watch_batch=watch, attempt=1,
        short_steps=4, perturb_sigma=.5, rng=np.random.default_rng(3), trajectory_recorder=recorder)
    assert seen == [tuple(bond for bond in projected_bonds if bond is not opening_edge)]
    assert tuple(projected.bonds) == projected_bonds
    assert tuple(molecule.bonds) == originals
    assert _selected_bonds(molecule, other_selection) == (originals[0], originals[2])


def test_ring_untangling_budget_scans_only_selected_watch_frame(monkeypatch):
    molecule = _UntanglingMolecule()
    opening_edge = molecule.bond(0, 1)
    initial_checkpoint = _report(3)
    selected_checkpoint = _report(1)
    full_scan_count = 0
    optimization_count = 0

    def scan(*args, **kwargs):
        nonlocal full_scan_count
        full_scan_count += 1
        return selected_checkpoint

    def optimize(current_molecule, forcefield, steps, *, selection=None):
        nonlocal optimization_count
        optimization_count += 1
        current_molecule.coordinates = np.full_like(current_molecule.coordinates, optimization_count)
        return _optimization(optimization_count)

    monkeypatch.setattr(
        repair,
        "_scan_ring_checkpoint",
        scan,
    )
    _mock_ring_watch(
        monkeypatch,
        (
            repair.geo.PiercingState.PIERCES,
            repair.geo.PiercingState.PIERCES,
        ),
    )
    monkeypatch.setattr(
        repair,
        "_select_ring_opening_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(
        repair,
        "_perturbed_coordinates",
        lambda coordinates, **kwargs: coordinates,
    )
    monkeypatch.setattr(repair, "_single_ob_optimization", optimize)

    result = repair._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=2,
        short_steps=4,
        settling_steps=9,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        checkpoint_report=initial_checkpoint,
    )

    assert result.report.initial_piercing_count == 3
    assert result.report.minimum_piercing_count == 1
    assert result.report.final_piercing_count == 1
    assert not result.report.resolved
    assert full_scan_count == 1
    assert result.checkpoint_report is selected_checkpoint
    np.testing.assert_array_equal(molecule.coordinates, np.full((3, 3), 2.0))


def test_budget_exhaustion_returns_best_frame_without_settling_or_rescanning(
    monkeypatch,
):
    molecule = _UntanglingMolecule()
    opening_edge = molecule.bond(0, 1)
    initial_checkpoint = _report(3)
    selected_checkpoint = _report(1)
    full_scan_count = 0
    optimization_steps = []

    def scan(*args, **kwargs):
        nonlocal full_scan_count
        full_scan_count += 1
        return selected_checkpoint

    def optimize(current_molecule, forcefield, steps, *, selection=None):
        optimization_steps.append(steps)
        marker = 1.0 if steps == 4 else 9.0
        current_molecule.coordinates = np.full_like(current_molecule.coordinates, marker)
        return _optimization(marker)

    monkeypatch.setattr(
        repair,
        "_scan_ring_checkpoint",
        scan,
    )
    _mock_ring_watch(
        monkeypatch,
        (repair.geo.PiercingState.PIERCES,),
    )
    monkeypatch.setattr(
        repair,
        "_select_ring_opening_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(
        repair,
        "_perturbed_coordinates",
        lambda coordinates, **kwargs: coordinates,
    )
    monkeypatch.setattr(repair, "_single_ob_optimization", optimize)

    result = repair._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=1,
        short_steps=4,
        settling_steps=9,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        checkpoint_report=initial_checkpoint,
    )

    assert optimization_steps == [4]
    assert full_scan_count == 1
    assert result.report.final_piercing_count == 1
    assert result.checkpoint_report is selected_checkpoint
    assert result.report.warning_messages == (
        (
            "Confirmed bond-ring piercing remains after 1 untangling attempts; "
            "retaining the closed-topology frame with the lowest piercing count"
        ),
    )
    np.testing.assert_array_equal(molecule.coordinates, np.ones((3, 3)))


def test_public_staged_workflow_attempt_defaults():
    expected_defaults = {
        "ligand_untangling_attempts": 20,
        "coordination_restoration_attempts": 20,
        "coordination_relaxation_steps": 100,
        "complex_untangling_attempts": 30,
    }
    build_parameters = inspect.signature(ff.build_complex3d).parameters
    optimize_parameters = inspect.signature(ff.optimize_complex).parameters
    workflow_parameters = inspect.signature(ff.complexes_build).parameters

    assert build_parameters["ligand_untangling_attempts"].default == 20
    assert build_parameters["coordination_restoration_attempts"].default == 20
    assert optimize_parameters["complex_untangling_attempts"].default == 30
    assert {
        name: workflow_parameters[name].default
        for name in expected_defaults
    } == expected_defaults


def test_selected_backend_optimizes_independent_graph_and_publishes_only_coordinates(monkeypatch):
    molecule = _ring_molecule()
    originals = tuple(molecule.bonds)
    _ = molecule.rings
    selection = _capture_topology_selection(molecule, excluded_bonds=(originals[0],))
    coordinates = molecule.coordinates.copy()
    calls = []
    def optimize(target, forcefield, steps):
        calls.append((target, forcefield, steps))
        assert target is not molecule and target._owner_token is not molecule._owner_token
        assert len(target.bonds) == 2 and target.rings == []
        assert len(molecule.bonds) == 3 and len(molecule.rings) == 1
        target.coordinates = target.coordinates + 2.
        return SimpleNamespace(energy=4., energy_unit='kJ/mol', exploded=False)
    monkeypatch.setattr(ob_backend, '_native_single_optimize', optimize)
    result = ob_backend._single_ob_optimization(molecule, 'UFF', 4, selection=selection)
    assert len(calls) == 1 and calls[0][1:] == ('UFF', 4)
    assert result.energy == 4.
    assert tuple(molecule.bonds) == originals
    np.testing.assert_array_equal(molecule.coordinates, coordinates + 2.)


@pytest.mark.parametrize('failure', ['native', 'source_geometry', 'source_topology'])
def test_selected_backend_failure_does_not_publish_partial_projected_coordinates(monkeypatch, failure):
    molecule = _ring_molecule()
    selection = _capture_topology_selection(molecule, excluded_bonds=(molecule.bonds[0],))
    coordinates = molecule.coordinates.copy()
    def optimize(target, forcefield, steps):
        target.coordinates = target.coordinates + 99.
        if failure == 'native':
            raise MemoryError('projected optimizer failed')
        if failure == 'source_geometry':
            molecule.coordinates = coordinates + 1.
        else:
            molecule._bonds.reverse()
        return SimpleNamespace(energy=4., energy_unit='kJ/mol', exploded=False)
    monkeypatch.setattr(ob_backend, '_native_single_optimize', optimize)
    with pytest.raises(MemoryError if failure == 'native' else RuntimeError):
        ob_backend._single_ob_optimization(molecule, 'UFF', 4, selection=selection)
    np.testing.assert_array_equal(molecule.coordinates, coordinates + (1. if failure == 'source_geometry' else 0.))


@pytest.mark.parametrize('failure', ['optimization', 'recording'])
def test_ring_repair_failure_keeps_complete_graph_and_records_closed_frame(monkeypatch, failure):
    molecule = _ring_molecule()
    bonds = tuple(molecule.bonds)
    original_coordinates = molecule.coordinates.copy()
    watch = _mock_ring_watch(monkeypatch, (geo.PiercingState.DOES_NOT_PIERCE,))
    monkeypatch.setattr(repair, '_select_ring_opening_edge', lambda *args: bonds[0])
    monkeypatch.setattr(repair, '_perturbed_coordinates', lambda coordinates, **kwargs: coordinates + 1.)
    trajectory = ForceFieldTrajectory.from_molecule(molecule, start=TrajectoryStart.COMPLEX_UNTANGLING)
    recorder = repair._RingTrajectoryRecorder(molecule, trajectory, TrajectoryStage.COMPLEX_UNTANGLING, True)
    error = MemoryError('failed during open operation')
    def optimize(*args, **kwargs):
        raise error
    monkeypatch.setattr(repair, '_single_ob_optimization', optimize)
    if failure == 'recording':
        record = ForceFieldTrajectory.record
        def fail_record(self, coordinates, bonds, **kwargs):
            if kwargs['event'] is TrajectoryEvent.PERTURBED:
                raise error
            return record(self, coordinates, bonds, **kwargs)
        monkeypatch.setattr(ForceFieldTrajectory, 'record', fail_record)
    with pytest.raises(MemoryError) as caught:
        repair._repair_watched_ring_piercings_once(
            molecule, 'UFF', current_piercings=watch, watch_batch=watch, attempt=1,
            short_steps=4, perturb_sigma=.5, rng=np.random.default_rng(3), trajectory_recorder=recorder)
    assert caught.value is error
    assert tuple(molecule.bonds) == bonds
    assert trajectory[-1].event is TrajectoryEvent.RING_CLOSED
    assert len(trajectory.topology(trajectory[-1].index).bonds) == 3
    assert trajectory[-1].energy_kj_mol is None
    # The helper retains the perturbation; outer closed-frame selection owns rollback.
    np.testing.assert_array_equal(molecule.coordinates, original_coordinates + 1.)
