import inspect
from types import SimpleNamespace

import numpy as np

from hotpot.cheminfo.forcefields import ff, ff39
from hotpot.cheminfo.forcefields import utils as forcefield_utils
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
            )
        )
        for index in range(count)
    )
    return SimpleNamespace(piercings=findings, ring_scope="ligand_skeleton")


def _optimization(energy):
    return forcefield_utils._CandidateOptimizationResult(
        energy=float(energy),
        energy_unit="kJ/mol",
        exploded=False,
    )


def _coordination_counts(piercing=0, undetermined=0, excluded_rings=0):
    return forcefield_utils._CoordinationRelationCounts(
        piercing=piercing,
        undetermined=undetermined,
        excluded_rings=excluded_rings,
    )


def _mock_coordination_scans(monkeypatch, candidate_counts):
    monkeypatch.setattr(
        forcefield_utils,
        "_scan_full_graph_bond_ring_relations",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_candidate_coordination_relation_counts",
        candidate_counts,
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_coordination_topology_relation_counts",
        lambda *args, **kwargs: _coordination_counts(),
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
    frames=(),
    energies=(),
):
    report = forcefield_utils.RingUntanglingReport(
        attempt_limit=3,
        attempts_completed=1,
        initial_piercing_count=piercing_count,
        final_piercing_count=0,
        minimum_piercing_count=0,
        resolved=True,
    )
    return forcefield_utils._RingUntanglingResult(
        report=report,
        energy=1.0,
        frames=tuple(np.asarray(frame, dtype=float) for frame in frames),
        frame_energies=tuple(float(energy) for energy in energies),
    )


def test_ring_untangling_opens_perturbs_optimizes_closes_and_rechecks(monkeypatch):
    molecule = _UntanglingMolecule()
    opening_edge = _Bond(0, 1)
    scans = iter(
        (
            (forcefield_utils.geo.PiercingState.PIERCES, _report(1)),
            (forcefield_utils.geo.PiercingState.DOES_NOT_PIERCE, None),
            (forcefield_utils.geo.PiercingState.DOES_NOT_PIERCE, None),
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

    monkeypatch.setattr(forcefield_utils, "_scan_confirmed_ring_piercings", scan)
    monkeypatch.setattr(
        forcefield_utils,
        "_first_openable_ring_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(forcefield_utils, "_perturbed_coordinates", perturb)
    monkeypatch.setattr(forcefield_utils, "_single_ob_optimization", optimize)

    result = forcefield_utils._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=2,
        short_steps=4,
        settling_steps=9,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert molecule.events == [
        ("scan",),
        ("open", (opening_edge,)),
        ("perturb",),
        ("optimize", 4),
        ("restore", (opening_edge,)),
        ("scan",),
        ("optimize", 9),
        ("scan",),
    ]
    assert result.report.resolved
    assert result.report.attempts_completed == 1


def test_ring_untangling_restores_only_the_edge_opened_by_this_attempt(
    monkeypatch,
):
    molecule = _UntanglingMolecule()
    preexisting_hidden_bond = _Bond(1, 2)
    opening_edge = _Bond(0, 1)
    molecule.hidden_bonds.append(preexisting_hidden_bond)
    scans = iter(
        (
            (forcefield_utils.geo.PiercingState.PIERCES, _report(1)),
            (forcefield_utils.geo.PiercingState.DOES_NOT_PIERCE, None),
        )
    )

    monkeypatch.setattr(
        forcefield_utils,
        "_scan_confirmed_ring_piercings",
        lambda *args, **kwargs: next(scans),
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_first_openable_ring_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_perturbed_coordinates",
        lambda coordinates, **kwargs: coordinates,
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_single_ob_optimization",
        lambda *args, **kwargs: _optimization(1.0),
    )

    forcefield_utils._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=1,
        short_steps=4,
        settling_steps=0,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert molecule.hidden_bonds == [preexisting_hidden_bond]
    assert ("restore", (opening_edge,)) in molecule.events
    assert ("close",) not in molecule.events


def test_ring_untangling_budget_retains_lowest_piercing_frame(monkeypatch):
    molecule = _UntanglingMolecule()
    opening_edge = _Bond(0, 1)
    scans = iter(
        (
            (forcefield_utils.geo.PiercingState.PIERCES, _report(3)),
            (forcefield_utils.geo.PiercingState.PIERCES, _report(1)),
            (forcefield_utils.geo.PiercingState.PIERCES, _report(2)),
            (forcefield_utils.geo.PiercingState.PIERCES, _report(2)),
        )
    )
    optimization_count = 0

    def optimize(current_molecule, forcefield, steps):
        nonlocal optimization_count
        optimization_count += 1
        current_molecule.coordinates[:] = optimization_count
        return _optimization(optimization_count)

    monkeypatch.setattr(
        forcefield_utils,
        "_scan_confirmed_ring_piercings",
        lambda *args, **kwargs: next(scans),
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_first_openable_ring_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_perturbed_coordinates",
        lambda coordinates, **kwargs: coordinates,
    )
    monkeypatch.setattr(forcefield_utils, "_single_ob_optimization", optimize)

    result = forcefield_utils._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=2,
        short_steps=4,
        settling_steps=9,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert result.report.initial_piercing_count == 3
    assert result.report.minimum_piercing_count == 1
    assert result.report.final_piercing_count == 1
    assert not result.report.resolved
    np.testing.assert_array_equal(molecule.coordinates, np.ones((3, 3)))


def test_budget_exhaustion_settles_best_frame_and_rolls_back_if_it_worsens(
    monkeypatch,
):
    molecule = _UntanglingMolecule()
    opening_edge = _Bond(0, 1)
    scans = iter(
        (
            (forcefield_utils.geo.PiercingState.PIERCES, _report(3)),
            (forcefield_utils.geo.PiercingState.PIERCES, _report(1)),
            (forcefield_utils.geo.PiercingState.PIERCES, _report(2)),
        )
    )
    optimization_steps = []

    def optimize(current_molecule, forcefield, steps):
        optimization_steps.append(steps)
        marker = 1.0 if steps == 4 else 9.0
        current_molecule.coordinates[:] = marker
        return _optimization(marker)

    monkeypatch.setattr(
        forcefield_utils,
        "_scan_confirmed_ring_piercings",
        lambda *args, **kwargs: next(scans),
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_first_openable_ring_edge",
        lambda *args, **kwargs: opening_edge,
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_perturbed_coordinates",
        lambda coordinates, **kwargs: coordinates,
    )
    monkeypatch.setattr(forcefield_utils, "_single_ob_optimization", optimize)

    result = forcefield_utils._untangle_ring_piercings(
        molecule,
        "UFF",
        attempt_limit=1,
        short_steps=4,
        settling_steps=9,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
    )

    assert optimization_steps == [4, 9]
    assert result.report.final_piercing_count == 1
    np.testing.assert_array_equal(molecule.coordinates, np.ones((3, 3)))


def test_coordination_bonds_are_restored_one_by_one_after_safe_checks(monkeypatch):
    first = _Bond(0, 2, coordination=True)
    second = _Bond(0, 3, coordination=True)
    molecule = _CoordinationMolecule((first, second))
    second_checks = 0

    def relation_counts(before, after, bond):
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
        forcefield_utils,
        "_single_ob_optimization",
        lambda *args, **kwargs: _optimization(1.0),
    )

    result = forcefield_utils._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=3,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        save_movie=False,
    )

    assert molecule.events == [
        ("hide", ((0, 2), (0, 3))),
        ("restore", ((0, 2),)),
        ("restore", ((0, 3),)),
        ("hide", ((0, 3),)),
        ("restore", ((0, 3),)),
    ]
    assert result.report.attempts_completed == 1
    assert result.report.restored_without_forcing == 2
    assert result.report.forced_bond_keys == ()


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

    monkeypatch.setattr(forcefield_utils, "_single_ob_optimization", optimize)

    result = forcefield_utils._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=1,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        save_movie=False,
    )

    assert optimization_calls == 3
    assert result.report.attempts_completed == 0
    assert result.report.restored_without_forcing == 3
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
    monkeypatch.setattr(forcefield_utils, "_perturbed_coordinates", perturb)
    monkeypatch.setattr(forcefield_utils, "_single_ob_optimization", optimize)

    result = forcefield_utils._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=2,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        save_movie=False,
    )

    assert perturb_calls == 1
    assert optimize_calls == 2
    assert molecule.events[-1] == ("restore", ((0, 2),))
    assert result.report.forced_bond_keys == ((0, 2),)
    assert not result.report.resolved


def test_coordination_restoration_forces_pending_bonds_on_last_relaxed_frame(
    monkeypatch,
):
    first = _Bond(0, 2, coordination=True)
    second = _Bond(0, 3, coordination=True)
    molecule = _CoordinationMolecule((first, second))
    attempt = 0

    def relation_counts(before, after, bond):
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
    monkeypatch.setattr(forcefield_utils, "_single_ob_optimization", optimize)
    monkeypatch.setattr(forcefield_utils, "_perturbed_coordinates", perturb)

    result = forcefield_utils._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=3,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        save_movie=False,
    )

    assert result.report.restored_without_forcing == 1
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
    monkeypatch.setattr(forcefield_utils, "_single_ob_optimization", optimize)

    result = forcefield_utils._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=1,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        save_movie=False,
    )

    assert optimization_calls == 2
    assert result.report.attempts_completed == 1
    assert result.report.resolved


def test_coordination_restoration_reports_piercing_created_by_relaxation(
    monkeypatch,
):
    bond = _Bond(0, 2, coordination=True)
    molecule = _CoordinationMolecule((bond,))
    _mock_coordination_scans(
        monkeypatch,
        lambda *args, **kwargs: _coordination_counts(),
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_coordination_topology_relation_counts",
        lambda *args, **kwargs: _coordination_counts(piercing=1),
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_single_ob_optimization",
        lambda *args, **kwargs: _optimization(1.0),
    )

    result = forcefield_utils._restore_coordination_bonds_incrementally(
        molecule,
        "UFF",
        attempt_limit=1,
        relaxation_steps=5,
        perturb_sigma=0.5,
        rng=np.random.default_rng(3),
        save_movie=False,
    )

    assert result.report.forced_bond_keys == ()
    assert result.report.final_piercing_count == 1
    assert not result.report.resolved


def test_chelate_cycle_filter_only_ignores_the_candidate_bond_itself():
    candidate = _Bond(0, 4, coordination=True)
    same_cycle = SimpleNamespace(
        target=SimpleNamespace(
            bond=SimpleNamespace(key=(0, 4)),
            ring=SimpleNamespace(key=(0, 1, 2, 3, 4)),
        )
    )
    other_bond = SimpleNamespace(
        target=SimpleNamespace(
            bond=SimpleNamespace(key=(1, 3)),
            ring=SimpleNamespace(key=(0, 1, 2, 3, 4)),
        )
    )
    unrelated_cycle = SimpleNamespace(
        target=SimpleNamespace(
            bond=SimpleNamespace(key=(0, 4)),
            ring=SimpleNamespace(key=(0, 1, 2, 3)),
        )
    )

    assert forcefield_utils._is_coordination_cycle_closure(same_cycle, candidate)
    assert not forcefield_utils._is_coordination_cycle_closure(other_bond, candidate)
    assert not forcefield_utils._is_coordination_cycle_closure(
        unrelated_cycle,
        candidate,
    )


def test_coordination_bond_check_ignores_self_closure():
    candidate = _Bond(0, 4, coordination=True)
    finding = SimpleNamespace(
        target=SimpleNamespace(
            bond=SimpleNamespace(key=(0, 4)),
            ring=SimpleNamespace(key=(0, 1, 2, 3, 4)),
        ),
        relation=SimpleNamespace(state=forcefield_utils.geo.PiercingState.PIERCES),
    )
    before = SimpleNamespace(findings=(), excluded_ring_count=0)
    after = SimpleNamespace(findings=(finding,), excluded_ring_count=0)

    assert forcefield_utils._candidate_coordination_relation_counts(
        before,
        after,
        candidate,
    ) == _coordination_counts()


def test_coordination_bond_check_counts_other_bond_through_new_chelate_ring():
    candidate = _Bond(0, 4, coordination=True)
    finding = SimpleNamespace(
        target=SimpleNamespace(
            bond=SimpleNamespace(key=(1, 3)),
            ring=SimpleNamespace(key=(0, 1, 2, 3, 4)),
        ),
        relation=SimpleNamespace(state=forcefield_utils.geo.PiercingState.PIERCES),
    )
    before = SimpleNamespace(findings=(), excluded_ring_count=0)
    after = SimpleNamespace(findings=(finding,), excluded_ring_count=0)

    assert forcefield_utils._candidate_coordination_relation_counts(
        before,
        after,
        candidate,
    ) == _coordination_counts(piercing=1)


def test_coordination_bond_check_does_not_claim_preexisting_piercing():
    candidate = _Bond(0, 4, coordination=True)
    finding = SimpleNamespace(
        target=SimpleNamespace(
            bond=SimpleNamespace(key=(1, 3)),
            ring=SimpleNamespace(key=(0, 1, 2, 3, 4)),
        ),
        relation=SimpleNamespace(state=forcefield_utils.geo.PiercingState.PIERCES),
    )
    before = SimpleNamespace(findings=(finding,), excluded_ring_count=0)
    after = SimpleNamespace(findings=(finding,), excluded_ring_count=0)

    assert forcefield_utils._candidate_coordination_relation_counts(
        before,
        after,
        candidate,
    ) == _coordination_counts()


def test_coordination_bond_check_reports_excluded_large_rings():
    candidate = _Bond(0, 4, coordination=True)
    before = SimpleNamespace(findings=(), excluded_ring_count=2)
    after = SimpleNamespace(findings=(), excluded_ring_count=3)

    assert forcefield_utils._candidate_coordination_relation_counts(
        before,
        after,
        candidate,
    ) == _coordination_counts(excluded_rings=3)


def test_failed_post_addition_check_rehides_candidate_bond(monkeypatch):
    candidate = _Bond(0, 2, coordination=True)
    molecule = _CoordinationMolecule((candidate,))
    molecule.hide_bonds(candidate, clear_conformers=False)

    scan_states = []

    def scan(current_molecule):
        scan_states.append(candidate in current_molecule.bonds)
        return object()

    def relation_counts(before, after, bond):
        assert bond is candidate
        return _coordination_counts(piercing=1)

    monkeypatch.setattr(
        forcefield_utils,
        "_scan_full_graph_bond_ring_relations",
        scan,
    )
    monkeypatch.setattr(
        forcefield_utils,
        "_candidate_coordination_relation_counts",
        relation_counts,
    )

    restored, warnings = (
        forcefield_utils._restore_next_nonpiercing_coordination_bond(
            molecule,
            [candidate],
        )
    )

    assert restored is None
    assert warnings == ()
    assert scan_states == [False, True]
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

    restored, warnings = (
        forcefield_utils._restore_next_nonpiercing_coordination_bond(
            molecule,
            pending,
        )
    )

    assert restored is candidate
    assert pending == []
    assert warnings == ()
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

    restored, _ = forcefield_utils._restore_next_nonpiercing_coordination_bond(
        molecule,
        pending,
    )

    assert restored is None
    assert pending == [candidate]
    assert candidate not in molecule.bonds


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
        optimization_calls.append(
            (kwargs["epochs"], kwargs["stop_on_ring_piercing"])
        )
        events.append("optimize")
        current_molecule.coordinates[:] = (
            1.0 if len(optimization_calls) == 1 else 3.0
        )
        return _run_report()

    def scan(current_molecule, **kwargs):
        events.append("scan")
        if float(current_molecule.coordinates[0, 0]) == 1.0:
            return forcefield_utils.geo.PiercingState.PIERCES, _report(1)
        return forcefield_utils.geo.PiercingState.DOES_NOT_PIERCE, None

    monkeypatch.setattr(forcefield_utils, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(forcefield_utils, "_optimize_working_mol", optimize)
    monkeypatch.setattr(forcefield_utils, "_scan_confirmed_ring_piercings", scan)

    report = forcefield_utils._optimize_complex_working_mol(
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
        save_movie=False,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
    )

    assert events == [
        "untangle",
        "optimize",
        "scan",
        "untangle",
        "optimize",
        "scan",
    ]
    assert report.untangling.initial_piercing_count == 0
    assert report.untangling.final_piercing_count == 0
    assert report.untangling.attempts_completed == 2
    assert optimization_calls == [(2, True), (1, True)]
    np.testing.assert_array_equal(molecule.coordinates, np.full((2, 3), 3.0))


def test_movie_combines_coordination_restoration_untangling_and_final_frames(
    monkeypatch,
):
    stage_21 = np.zeros((2, 3))
    stage_22_open = np.ones((2, 3))
    stage_22_closed = np.full((2, 3), 2.0)
    final_frame = np.full((2, 3), 3.0)
    molecule = _TraceMolecule((stage_21,))

    def untangle(current_molecule, *args, **kwargs):
        current_molecule.coordinates = stage_22_closed.copy()
        return _untangling_result(
            frames=(stage_22_open, stage_22_closed),
            energies=(1.0, 2.0),
        )

    def optimize(current_molecule, **kwargs):
        current_molecule.conformer_clear()
        current_molecule.conformer_add(
            np.asarray((stage_22_closed, final_frame)),
            np.asarray((2.0, 3.0)),
        )
        current_molecule.conformer_load(1)
        return _run_report()

    monkeypatch.setattr(forcefield_utils, "_untangle_ring_piercings", untangle)
    monkeypatch.setattr(forcefield_utils, "_optimize_working_mol", optimize)
    monkeypatch.setattr(
        forcefield_utils,
        "_scan_confirmed_ring_piercings",
        lambda *args, **kwargs: (
            forcefield_utils.geo.PiercingState.DOES_NOT_PIERCE,
            None,
        ),
    )

    forcefield_utils._optimize_complex_working_mol(
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
        save_movie=True,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
    )

    frames = tuple(
        molecule.conformer_get(index)["coordinates"]
        for index in range(molecule.conformers_number)
    )
    assert len(frames) == 4
    for actual, expected in zip(
        frames,
        (stage_21, stage_22_open, stage_22_closed, final_frame),
    ):
        np.testing.assert_array_equal(actual, expected)
    assert molecule._conformers_index == 3


def test_movie_boundary_keeps_a_finite_energy_over_a_duplicate_nan_frame():
    initial_frame = np.zeros((2, 3))
    final_frame = np.ones((2, 3))
    molecule = _TraceMolecule((initial_frame, final_frame))
    molecule._conformers[0]["energy"] = float("nan")

    forcefield_utils._combine_conformer_traces(
        molecule,
        (initial_frame,),
        (10.0,),
    )

    assert molecule.conformers_number == 2
    assert molecule.conformer_get(0)["energy"] == 10.0
    assert molecule._conformers_index == 1
