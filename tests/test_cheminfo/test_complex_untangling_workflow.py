import inspect
from types import SimpleNamespace

import numpy as np

from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.forcefields import ff, ff39
from hotpot.cheminfo.forcefields import backend as ob_backend
from hotpot.cheminfo.forcefields import repair
from hotpot.cheminfo.forcefields import utils as forcefield_utils
from hotpot.cheminfo.forcefields import workflows
from hotpot.cheminfo.forcefields.trajectory import (
    CoordinationFrameEvidence,
    ForceFieldTrajectory,
    RingFrameEvidence,
    TrajectoryEvent,
    TrajectoryStart,
)
from hotpot.cheminfo.core import Molecule


class _Atom:
    def __init__(self, index):
        self.idx = index


class _Bond:
    def __init__(self, first, second, *, coordination=False):
        self.atom1 = _Atom(first)
        self.atom2 = _Atom(second)
        self.is_metal_ligand_bond = coordination


class _UntanglingMolecule:
    def __init__(self):
        self.coordinates = np.zeros((3, 3), dtype=float)
        self.events = []
        self.hidden_bonds = []

    def hide_bonds(self, *bonds, clear_conformers=False):
        self.hidden_bonds.extend(bonds)
        self.events.append(("open", bonds))

    def recover_hided_covalent_bonds(self, clear_conformers=False):
        self.hidden_bonds.clear()
        self.events.append(("close",))

    def restore_bonds(self, *bonds, clear_conformers=False):
        for bond in bonds:
            self.hidden_bonds.remove(bond)
        self.events.append(("restore", bonds))


class _CoordinationMolecule:
    def __init__(self, bonds):
        self.coordinates = np.zeros((4, 3), dtype=float)
        self._visible_bonds = list(bonds)
        self._hidden_bonds = []
        self.events = []

    @property
    def bonds(self):
        return tuple(self._visible_bonds)

    def hide_bonds(self, *bonds, clear_conformers=False):
        for bond in bonds:
            self._visible_bonds.remove(bond)
            self._hidden_bonds.append(bond)
        self.events.append(("hide", tuple(_key(bond) for bond in bonds)))

    def restore_bonds(self, *bonds, clear_conformers=False):
        for bond in bonds:
            self._hidden_bonds.remove(bond)
            self._visible_bonds.append(bond)
        self.events.append(("restore", tuple(_key(bond) for bond in bonds)))


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


def _coordination_counts(piercing=0, undetermined=0, excluded_rings=0):
    return repair._CoordinationRelationCounts(
        piercing=piercing,
        undetermined=undetermined,
        excluded_rings=excluded_rings,
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


def _coordination_molecule():
    molecule = Molecule()
    for atomic_number, coordinate in zip(
        (30, 7, 7),
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)),
    ):
        molecule.create_atom(
            atomic_number=atomic_number,
            coordinates=coordinate,
        )
    bonds = tuple(
        molecule.add_bond(0, donor_index, bond_order=1.0)
        for donor_index in (1, 2)
    )
    return molecule, bonds


def _mock_coordination_scans(monkeypatch, candidate_counts):
    monkeypatch.setattr(
        repair,
        "_screen_coordination_bond_relations",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        repair,
        "_candidate_coordination_relation_counts",
        candidate_counts,
    )


def _run_report(*, untangling=None):
    return forcefield_utils.ForceFieldRunReport(
        requested_forcefield=None,
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=True,
        epochs_completed=1,
        steps_submitted=1,
        initialization_steps=1,
        steps_completed=None,
        final_energy=1.0,
        best_energy=1.0,
        energy_unit="kJ/mol",
        rms_gradient=0.0,
        max_gradient=0.0,
        exploded=False,
        untangling=untangling,
    )


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
    opening_edge = _Bond(0, 1)
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

    def optimize(current_molecule, forcefield, steps):
        molecule.events.append(("optimize", steps))
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
        ("open", (opening_edge,)),
        ("perturb",),
        ("optimize", 4),
        ("restore", (opening_edge,)),
        ("targeted_scan",),
        ("scan",),
        ("optimize", 9),
        ("scan",),
    ]
    assert result.report.resolved
    assert result.report.attempts_completed == 1
    assert result.checkpoint_report.state is geo.PiercingState.DOES_NOT_PIERCE


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


def test_ring_checkpoint_preserves_a_non_piercing_report(monkeypatch):
    molecule = _UntanglingMolecule()
    expected = _report(0)
    calls = []

    def screen(current_molecule, *, ring_scope, max_ring_size):
        calls.append((current_molecule, ring_scope, max_ring_size))
        return expected

    monkeypatch.setattr(repair.geo, "screen_bond_ring_relations", screen)

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
        lambda current_molecule, forcefield, steps: _optimization(steps),
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.COMPLEX_UNTANGLING,
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
        TrajectoryEvent.INITIAL,
        TrajectoryEvent.RING_OPENED,
        TrajectoryEvent.PERTURBED,
        TrajectoryEvent.OPTIMIZED,
        TrajectoryEvent.RING_CLOSED,
        TrajectoryEvent.SETTLED,
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
    assert trajectory[0].evidence == RingFrameEvidence(1, 0)
    assert trajectory[4].evidence == RingFrameEvidence(0, 0)
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
        lambda current_molecule, forcefield, steps: _optimization(steps),
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


def test_ring_untangling_restores_only_the_edge_opened_by_this_attempt(
    monkeypatch,
):
    molecule = _UntanglingMolecule()
    preexisting_hidden_bond = _Bond(1, 2)
    opening_edge = _Bond(0, 1)
    molecule.hidden_bonds.append(preexisting_hidden_bond)
    initial_checkpoint = _report(1)
    scans = iter(
        (
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
        lambda coordinates, **kwargs: coordinates,
    )
    monkeypatch.setattr(
        repair,
        "_single_ob_optimization",
        lambda *args, **kwargs: _optimization(1.0),
    )

    repair._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=1,
        short_steps=4,
        settling_steps=0,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        checkpoint_report=initial_checkpoint,
    )

    assert molecule.hidden_bonds == [preexisting_hidden_bond]
    assert ("restore", (opening_edge,)) in molecule.events
    assert ("close",) not in molecule.events


def test_ring_untangling_budget_scans_only_selected_watch_frame(monkeypatch):
    molecule = _UntanglingMolecule()
    opening_edge = _Bond(0, 1)
    initial_checkpoint = _report(3)
    selected_checkpoint = _report(1)
    rejected_checkpoint = _report(2)
    scans = iter(
        (
            selected_checkpoint,
            rejected_checkpoint,
        )
    )
    full_scan_count = 0
    optimization_count = 0

    def scan(*args, **kwargs):
        nonlocal full_scan_count
        full_scan_count += 1
        return next(scans)

    def optimize(current_molecule, forcefield, steps):
        nonlocal optimization_count
        optimization_count += 1
        current_molecule.coordinates[:] = optimization_count
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
    assert full_scan_count == 2
    assert result.checkpoint_report is selected_checkpoint
    np.testing.assert_array_equal(molecule.coordinates, np.full((3, 3), 2.0))


def test_budget_exhaustion_settles_best_frame_and_rolls_back_if_it_worsens(
    monkeypatch,
):
    molecule = _UntanglingMolecule()
    opening_edge = _Bond(0, 1)
    initial_checkpoint = _report(3)
    scans = iter(
        (
            _report(1),
            _report(2),
        )
    )
    optimization_steps = []

    def optimize(current_molecule, forcefield, steps):
        optimization_steps.append(steps)
        marker = 1.0 if steps == 4 else 9.0
        current_molecule.coordinates[:] = marker
        return _optimization(marker)

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

    assert optimization_steps == [4, 9]
    assert result.report.final_piercing_count == 1
    np.testing.assert_array_equal(molecule.coordinates, np.ones((3, 3)))


def test_coordination_bonds_are_restored_one_by_one_after_safe_checks(monkeypatch):
    first = _Bond(0, 2, coordination=True)
    second = _Bond(0, 3, coordination=True)
    molecule = _CoordinationMolecule((first, second))
    second_checks = 0

    def relation_counts(report, bond):
        nonlocal second_checks
        if bond is first:
            return _coordination_counts()
        second_checks += 1
        return (
            _coordination_counts(piercing=1)
            if second_checks == 1
            else _coordination_counts()
        )

    _mock_coordination_scans(monkeypatch, relation_counts)
    monkeypatch.setattr(
        repair,
        "_single_ob_optimization",
        lambda *args, **kwargs: _optimization(1.0),
    )

    result = repair._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=3,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert molecule.events == [
        ("hide", ((0, 2), (0, 3))),
        ("restore", ((0, 2),)),
        ("restore", ((0, 3),)),
    ]
    assert result.report.attempts_completed == 1
    assert result.report.rejected_piercing_trial_count == 1
    assert result.report.forced_bond_keys == ()


def test_coordination_trajectory_records_hidden_trial_without_rollback_topology(
    monkeypatch,
):
    molecule, (first, second) = _coordination_molecule()
    second_checks = 0

    def relation_counts(report, bond):
        nonlocal second_checks
        if bond is first:
            return _coordination_counts()
        second_checks += 1
        return (
            _coordination_counts(piercing=1)
            if second_checks == 1
            else _coordination_counts()
        )

    _mock_coordination_scans(monkeypatch, relation_counts)
    monkeypatch.setattr(
        repair,
        "_single_ob_optimization",
        lambda current_molecule, forcefield, steps: _optimization(second_checks + 1),
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.COORDINATION_RESTORATION,
    )

    repair._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=3,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        trajectory=trajectory,
    )

    assert tuple(frame.event for frame in trajectory) == (
        TrajectoryEvent.COORDINATION_READY,
        TrajectoryEvent.BOND_TRIAL,
        TrajectoryEvent.BOND_ACCEPTED,
        TrajectoryEvent.OPTIMIZED,
        TrajectoryEvent.BOND_TRIAL,
        TrajectoryEvent.BOND_REJECTED,
        TrajectoryEvent.OPTIMIZED,
        TrajectoryEvent.BOND_TRIAL,
        TrajectoryEvent.BOND_ACCEPTED,
        TrajectoryEvent.OPTIMIZED,
        TrajectoryEvent.TERMINAL,
    )
    assert tuple(len(trajectory.topology(frame.index).bonds) for frame in trajectory) == (
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
    )
    assert trajectory[4].evidence == CoordinationFrameEvidence(
        bond_atom_indices=_key(second),
        accepted=None,
        pending_bond_count=1,
    )
    assert trajectory[5].evidence == CoordinationFrameEvidence(
        bond_atom_indices=_key(second),
        accepted=False,
        pending_bond_count=1,
        piercing_relation_count=1,
    )
    assert trajectory[5].topology_revision == trajectory[3].topology_revision
    assert trajectory[9].energy_kj_mol == 3.0
    assert trajectory[10].energy_kj_mol == 3.0
    assert trajectory.selected_index == 10


def test_safe_coordination_bonds_do_not_consume_the_stalled_attempt_budget(
    monkeypatch,
):
    bonds = tuple(
        _Bond(0, donor, coordination=True)
        for donor in (1, 2, 3)
    )
    molecule = _CoordinationMolecule(bonds)
    _mock_coordination_scans(
        monkeypatch,
        lambda *args, **kwargs: _coordination_counts(),
    )
    optimization_calls = 0

    def optimize(*args, **kwargs):
        nonlocal optimization_calls
        optimization_calls += 1
        return _optimization(optimization_calls)

    monkeypatch.setattr(repair, "_single_ob_optimization", optimize)

    result = repair._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=1,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert optimization_calls == 3
    assert result.report.attempts_completed == 0
    assert result.report.forced_bond_keys == ()


def test_coordination_restoration_perturbs_without_progress_then_forces_all(
    monkeypatch,
):
    bond = _Bond(0, 2, coordination=True)
    molecule = _CoordinationMolecule((bond,))
    perturb_calls = 0
    optimize_calls = 0

    def perturb(coordinates, **kwargs):
        nonlocal perturb_calls
        perturb_calls += 1
        return coordinates

    def optimize(*args, **kwargs):
        nonlocal optimize_calls
        optimize_calls += 1
        return _optimization(optimize_calls)

    _mock_coordination_scans(
        monkeypatch,
        lambda *args, **kwargs: _coordination_counts(piercing=1),
    )
    monkeypatch.setattr(repair, "_perturbed_coordinates", perturb)
    monkeypatch.setattr(repair, "_single_ob_optimization", optimize)

    result = repair._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=2,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert perturb_calls == 1
    assert optimize_calls == 2
    assert molecule.events[-1] == ("restore", ((0, 2),))
    assert result.report.forced_bond_keys == ((0, 2),)
    assert result.report.rejected_piercing_trial_count == 3


def test_coordination_trajectory_records_each_forced_bond_after_hidden_trials(
    monkeypatch,
):
    molecule, bonds = _coordination_molecule()
    second = bonds[1]
    molecule.remove_bond(bonds[0])
    _mock_coordination_scans(
        monkeypatch,
        lambda *args, **kwargs: _coordination_counts(piercing=1),
    )
    monkeypatch.setattr(
        repair,
        "_single_ob_optimization",
        lambda *args, **kwargs: _optimization(3.0),
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.COORDINATION_RESTORATION,
    )

    result = repair._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=1,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        trajectory=trajectory,
    )

    assert result.report.forced_bond_keys == (_key(second),)
    assert tuple(frame.event for frame in trajectory) == (
        TrajectoryEvent.COORDINATION_READY,
        TrajectoryEvent.BOND_TRIAL,
        TrajectoryEvent.BOND_REJECTED,
        TrajectoryEvent.OPTIMIZED,
        TrajectoryEvent.BOND_TRIAL,
        TrajectoryEvent.BOND_REJECTED,
        TrajectoryEvent.BOND_FORCED,
        TrajectoryEvent.TERMINAL,
    )
    assert tuple(len(trajectory.topology(frame.index).bonds) for frame in trajectory) == (
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
    )
    assert trajectory[6].evidence == CoordinationFrameEvidence(
        bond_atom_indices=_key(second),
        accepted=False,
        pending_bond_count=0,
        forced=True,
    )
    assert trajectory[7].energy_kj_mol is None


def test_coordination_restoration_forces_pending_bonds_on_last_relaxed_frame(
    monkeypatch,
):
    first = _Bond(0, 2, coordination=True)
    second = _Bond(0, 3, coordination=True)
    molecule = _CoordinationMolecule((first, second))
    attempt = 0

    def relation_counts(report, bond):
        if bond is first:
            return _coordination_counts()
        return _coordination_counts(piercing=1)

    def optimize(current_molecule, forcefield, steps):
        nonlocal attempt
        attempt += 1
        current_molecule.coordinates[:] = float(attempt)
        return _optimization(attempt)

    def perturb(coordinates, **kwargs):
        return np.full_like(coordinates, 50.0)

    _mock_coordination_scans(monkeypatch, relation_counts)
    monkeypatch.setattr(repair, "_single_ob_optimization", optimize)
    monkeypatch.setattr(repair, "_perturbed_coordinates", perturb)

    result = repair._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=3,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert result.report.forced_bond_keys == ((0, 3),)
    np.testing.assert_array_equal(molecule.coordinates, np.full((4, 3), 4.0))


def test_coordination_bond_restored_in_a_round_is_relaxed(monkeypatch):
    bond = _Bond(0, 2, coordination=True)
    molecule = _CoordinationMolecule((bond,))
    relation_results = iter(
        (_coordination_counts(piercing=1), _coordination_counts())
    )
    optimization_calls = 0

    def optimize(current_molecule, forcefield, steps):
        nonlocal optimization_calls
        optimization_calls += 1
        return _optimization(optimization_calls)

    _mock_coordination_scans(
        monkeypatch,
        lambda *args, **kwargs: next(relation_results),
    )
    monkeypatch.setattr(repair, "_single_ob_optimization", optimize)

    result = repair._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=1,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert optimization_calls == 2
    assert result.report.attempts_completed == 1
    assert result.report.forced_bond_keys == ()


def test_coordination_bond_check_counts_candidate_through_ligand_ring():
    candidate = _Bond(0, 4, coordination=True)
    finding = SimpleNamespace(
        target=SimpleNamespace(
            bond=SimpleNamespace(key=(0, 4)),
            ring=SimpleNamespace(key=(1, 2, 3)),
        ),
        relation=SimpleNamespace(state=repair.geo.PiercingState.PIERCES),
    )
    report = SimpleNamespace(
        actionable_findings=(finding,),
        excluded_ring_count=0,
    )

    assert repair._candidate_coordination_relation_counts(
        report,
        candidate,
    ) == _coordination_counts(piercing=1)


def test_coordination_bond_check_ignores_non_candidate_bonds():
    candidate = _Bond(0, 4, coordination=True)
    finding = SimpleNamespace(
        target=SimpleNamespace(
            bond=SimpleNamespace(key=(1, 3)),
            ring=SimpleNamespace(key=(0, 1, 2, 3, 4)),
        ),
        relation=SimpleNamespace(state=repair.geo.PiercingState.PIERCES),
    )
    report = SimpleNamespace(
        actionable_findings=(finding,),
        excluded_ring_count=0,
    )

    assert repair._candidate_coordination_relation_counts(
        report,
        candidate,
    ) == _coordination_counts()


def test_coordination_bond_check_reports_excluded_large_rings():
    candidate = _Bond(0, 4, coordination=True)
    report = SimpleNamespace(actionable_findings=(), excluded_ring_count=3)

    assert repair._candidate_coordination_relation_counts(
        report,
        candidate,
    ) == _coordination_counts(excluded_rings=3)


def test_failed_hypothetical_check_never_restores_candidate_bond(monkeypatch):
    candidate = _Bond(0, 2, coordination=True)
    molecule = _CoordinationMolecule((candidate,))
    molecule.hide_bonds(candidate, clear_conformers=False)

    scan_states = []

    def scan(current_molecule, bond):
        assert bond is candidate
        scan_states.append(candidate in current_molecule.bonds)
        return object()

    def relation_counts(report, bond):
        assert bond is candidate
        return _coordination_counts(piercing=1)

    monkeypatch.setattr(
        repair,
        "_screen_coordination_bond_relations",
        scan,
    )
    monkeypatch.setattr(
        repair,
        "_candidate_coordination_relation_counts",
        relation_counts,
    )

    restored, warnings, observations = (
        repair._restore_next_nonpiercing_coordination_bond(
            molecule,
            [candidate],
        )
    )

    assert restored is None
    assert warnings == ()
    assert observations == repair._CoordinationTrialStatistics(1, 0, 0)
    assert scan_states == [False]
    assert candidate not in molecule.bonds


def test_real_post_addition_chelate_cycle_does_not_reject_its_closing_bond():
    molecule = Molecule()
    coordinates = (
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 1.5, 0.0),
        (1.0, 1.5, 0.0),
        (1.0, 0.0, 0.0),
    )
    for atomic_number, coordinate in zip((30, 7, 6, 6, 7), coordinates):
        molecule.create_atom(
            atomic_number=atomic_number,
            coordinates=coordinate,
        )
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 4)):
        molecule.add_bond(first, second, bond_order=1.0)
    candidate = molecule.add_bond(4, 0, bond_order=1.0)
    molecule.hide_bonds(candidate, clear_conformers=False)
    pending = [candidate]

    restored, warnings, observations = (
        repair._restore_next_nonpiercing_coordination_bond(
            molecule,
            pending,
        )
    )

    assert restored is candidate
    assert pending == []
    assert warnings == ()
    assert observations == repair._CoordinationTrialStatistics(0, 0, 0)
    assert {
        frozenset(atom.idx for atom in ring.atoms)
        for ring in molecule.rings_for_scope("full_graph")
    } == {frozenset(range(5))}


def test_real_post_addition_bond_through_ligand_ring_is_rejected():
    molecule = Molecule()
    ring_coordinates = tuple(
        (
            2.0 * np.cos(index * np.pi / 3.0),
            2.0 * np.sin(index * np.pi / 3.0),
            0.0,
        )
        for index in range(6)
    )
    for atomic_number, coordinate in zip(
        (6, 6, 6, 6, 6, 6, 30, 7),
        ring_coordinates + ((0.0, 0.0, -2.0), (0.0, 0.0, 2.0)),
    ):
        molecule.create_atom(
            atomic_number=atomic_number,
            coordinates=coordinate,
        )
    for index in range(6):
        molecule.add_bond(index, (index + 1) % 6, bond_order=1.0)
    candidate = molecule.add_bond(6, 7, bond_order=1.0)
    molecule.hide_bonds(candidate, clear_conformers=False)
    pending = [candidate]

    restored, _, observations = repair._restore_next_nonpiercing_coordination_bond(
        molecule,
        pending,
    )

    assert restored is None
    assert pending == [candidate]
    assert candidate not in molecule.bonds
    assert observations == repair._CoordinationTrialStatistics(1, 0, 0)


def test_public_staged_workflow_attempt_defaults_match_between_versions():
    expected_defaults = {
        "ligand_untangling_attempts": 20,
        "coordination_restoration_attempts": 20,
        "coordination_relaxation_steps": 100,
        "complex_untangling_attempts": 30,
    }
    for module in (ff, ff39):
        build_parameters = inspect.signature(module.build_complex3d).parameters
        optimize_parameters = inspect.signature(module.optimize_complex).parameters
        workflow_parameters = inspect.signature(module.complexes_build).parameters

        assert build_parameters["ligand_untangling_attempts"].default == 20
        assert build_parameters["coordination_restoration_attempts"].default == 20
        assert optimize_parameters["complex_untangling_attempts"].default == 30
        assert {
            name: workflow_parameters[name].default
            for name in expected_defaults
        } == expected_defaults


def test_final_relaxation_repiercing_reenters_repair_and_reports_final_state(
    monkeypatch,
):
    molecule = _TraceMolecule((np.zeros((2, 3)),))
    events = []
    untangling_calls = 0
    optimization_calls = []

    def untangle(current_molecule, *args, **kwargs):
        nonlocal untangling_calls
        untangling_calls += 1
        events.append("untangle")
        marker = 0.0 if untangling_calls == 1 else 2.0
        current_molecule.coordinates[:] = marker
        return _untangling_result(
            piercing_count=1 if untangling_calls == 2 else 0,
        )

    def optimize(current_molecule, **kwargs):
        optimization_calls.append(kwargs["epochs"])
        events.append("optimize")
        current_molecule.coordinates[:] = (
            1.0 if len(optimization_calls) == 1 else 3.0
        )
        return _run_report()

    def scan(current_molecule, **kwargs):
        events.append("scan")
        if float(current_molecule.coordinates[0, 0]) == 1.0:
            return _report(1)
        return _report(0)

    def accept(*args, **kwargs):
        events.append("accept")
        return forcefield_utils.ForceFieldValidationReport(
            level="standard",
            passed=True,
            checks=(),
        )

    monkeypatch.setattr(workflows, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(workflows, "_optimize_working_mol", optimize)
    monkeypatch.setattr(workflows, "_scan_ring_checkpoint", scan)
    monkeypatch.setattr(workflows, "evaluate_structure_acceptance", accept)

    report = workflows._optimize_complex_working_mol(
        molecule,
        requested_forcefield=None,
        effective_forcefield="UFF",
        algorithm="conjugate",
        epochs=2,
        steps_per_epoch=5,
        complex_untangling_attempts=3,
        quality_level="standard",
        topology_reference=SimpleNamespace(),
        quality_thresholds=None,
        seed=3,
        perturb_interval=None,
        perturb_sigma=0.5,
        retain_epoch_history=False,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
        trajectory=ForceFieldTrajectory(
            (),
            start=TrajectoryStart.COMPLEX_UNTANGLING,
        ),
    )

    assert events == [
        "scan",
        "untangle",
        "optimize",
        "scan",
        "untangle",
        "optimize",
        "scan",
        "accept",
    ]
    assert report.untangling.initial_piercing_count == 0
    assert report.untangling.final_piercing_count == 0
    assert report.untangling.attempts_completed == 2
    assert optimization_calls == [2, 1]
    np.testing.assert_array_equal(molecule.coordinates, np.full((2, 3), 3.0))
