"""D04/D05 canonical owners, native stopping and topology transport."""

from dataclasses import FrozenInstanceError, asdict, fields
import pickle
import subprocess
import sys

import pytest

from hotpot.cheminfo.adapters.structure.forcefields import capture_topology
from hotpot import Molecule
from hotpot.cheminfo.operations.forcefields import (
    ff,
    workflows,
)
from hotpot.cheminfo.adapters.native.forcefields import native as native_adapter
from hotpot.cheminfo.operations.forcefields.trajectory import ForceFieldTrajectory
from hotpot.cheminfo.structure.forcefields import (
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)
from hotpot.cheminfo.rules.physics.forcefields.contracts import OptimizationStoppingCriteria
from hotpot.cheminfo.structure.forcefields import (
    AtomTopologySignature,
    BondTopology,
    CoordinationFrameEvidence,
    OptimizationFrameEvidence,
    RingFrameEvidence,
    TopologyReference,
)


def test_canonical_classes_and_values_cold_pickle_at_their_actual_owners():
    classes = (
        OptimizationStoppingCriteria,
        AtomTopologySignature,
        BondTopology,
        TopologyReference,
    )
    atom = AtomTopologySignature(0, 91, 6, 0)
    bond = BondTopology((1, 0), 2.0, "double")
    values = (
        OptimizationStoppingCriteria(window=7),
        atom,
        bond,
        TopologyReference((atom, AtomTopologySignature(1, 17, 6, 0)), (bond,)),
    )
    for cls in classes:
        assert getattr(ff, cls.__name__) is cls
    assert ff.capture_topology is capture_topology
    assert not hasattr(ff, "BondTopologySignature")
    assert not hasattr(native_adapter, "OptimizationStoppingOptions")
    script = """
import pickle
import sys
classes, values = pickle.loads(sys.stdin.buffer.read())
assert tuple(type(value) for value in values) == classes
assert tuple(cls.__module__ for cls in classes) == (
    'hotpot.cheminfo.rules.physics.forcefields.contracts',
    'hotpot.cheminfo.structure.forcefields',
    'hotpot.cheminfo.structure.forcefields',
    'hotpot.cheminfo.structure.forcefields',
)
assert values[0].window == 7
assert values[2].atom_indices == (0, 1)
assert values[3].bonds == (values[2],)
"""

    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        input=pickle.dumps((classes, values)),
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode()


@pytest.mark.parametrize(
    ("legacy_class", "error_type"),
    (
        (
            b"chotpot.cheminfo.forcefields.native\nOptimizationStoppingOptions\n.",
            ModuleNotFoundError,
        ),
        (
            b"chotpot.cheminfo.forcefields.topology\nBondTopologySignature\n.",
            ModuleNotFoundError,
        ),
    ),
)
def test_retired_class_pickle_symbols_have_no_compatibility_alias(legacy_class, error_type):
    with pytest.raises(error_type):
        pickle.loads(legacy_class)


def test_capture_factory_and_trajectory_share_normalized_row_topology():
    mol = Molecule()
    mol.create_atom(atomic_number=6, id=91, coordinates=(0.0, 0.0, 0.0))
    mol.create_atom(atomic_number=6, id=17, coordinates=(1.3, 0.0, 0.0))
    bond = mol.add_bond(1, 0, 2.0)
    expected = BondTopology((1, 0), 2.0, "double")

    reference = capture_topology(mol, allow_added_hydrogens=False)
    trajectory = ForceFieldTrajectory.from_molecule(mol)
    frame = trajectory.record_molecule(
        mol,
        stage=TrajectoryStage.FINAL_OPTIMIZATION,
        event=TrajectoryEvent.INITIAL,
    )

    assert type(reference) is TopologyReference
    assert tuple(atom.atom_id for atom in reference.atoms) == (91, 17)
    assert reference.allow_added_hydrogens is False
    assert reference.bonds == trajectory.topology(frame.index).bonds == (expected,)
    assert type(reference.bonds[0]) is BondTopology
    assert type(trajectory.topology(frame.index).bonds[0]) is BondTopology
    assert BondTopology.from_bond(bond) == expected
    assert hash(expected) == hash(BondTopology((0, 1), 2.0, "double"))
    assert sorted((BondTopology((0, 2), 1.0, "single"), expected))[0] == expected
    assert asdict(expected) == {
        "atom_indices": (0, 1), "bond_order": 2.0, "bond_kind": "double",
    }
    with pytest.raises(FrozenInstanceError):
        expected.bond_order = 1.0


@pytest.mark.parametrize(
    "stopping",
    (
        None,
        OptimizationStoppingCriteria(
            window=7,
            maximum_energy_change_kj_mol=0.0,
            maximum_atom_displacement_angstrom=0.125,
            maximum_rms_gradient_kj_mol_angstrom=1.25,
            maximum_gradient_kj_mol_angstrom=2.5,
        ),
    ),
)
def test_workflow_passes_the_same_criteria_to_native_translation(monkeypatch, stopping):
    options = workflows._optimization_options(
        effective_forcefield="UFF",
        algorithm="conjugate",
        epochs=2,
        steps_per_epoch=3,
        untangling_attempt_limit=1,
        perturb_interval=None,
        perturb_sigma=0.5,
        trajectory_start=TrajectoryStart.FINAL_OPTIMIZATION,
        save_movie=False,
        stopping_criteria=stopping,
        convergence_level=ff.DEFAULT_CONVERGENCE_LEVEL,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
    )
    original = native_adapter._native_stopping_options
    observed = []

    def record_translation(criteria):
        observed.append(criteria)
        return original(criteria)

    monkeypatch.setattr(native_adapter, "_native_stopping_options", record_translation)

    translated = native_adapter._native_complex_optimization_options(options)

    assert options.stopping is stopping
    if stopping is None:
        assert observed == []
        assert translated.stopping is None
    else:
        assert len(observed) == 1 and observed[0] is stopping
        for field in fields(stopping):
            assert getattr(translated.stopping, field.name) == getattr(stopping, field.name)


def test_acceptance_classes_cold_pickle_at_the_true_rule_owner():
    from hotpot.cheminfo.operations.forcefields import contracts as mixed
    from hotpot.cheminfo.rules.mathematics import geometry as geo
    from hotpot.cheminfo.rules.physics.forcefields import contracts as canonical

    classes = tuple(getattr(canonical, name) for name in (
        'AcceptanceCheck', 'StructureAcceptanceThresholds', 'ForceFieldAcceptanceEvidence',
        '_AtomPairAcceptanceIssue', '_BondRingAcceptanceFinding', '_BondRingAcceptanceFacts',
    ))
    for cls in classes[:3]:
        assert getattr(mixed, cls.__name__) is cls
        assert getattr(ff, cls.__name__) is cls
    for name in ('AcceptanceLevel', 'ForceFieldStage', 'ForceFieldDiagnosticValue'):
        assert getattr(mixed, name) is getattr(canonical, name)
    finding = canonical._BondRingAcceptanceFinding((2, 3, 4), (0, 1), geo.PiercingState.PIERCES, ())
    values = (
        canonical.AcceptanceCheck('sample', True, atom_indices=(91,)),
        canonical.StructureAcceptanceThresholds(),
        canonical._AtomPairAcceptanceIssue('overlap', (0, 1), 0.0, 0.001),
        finding,
        canonical._BondRingAcceptanceFacts((finding,), 'ligand_skeleton', 16, 1, 0, 1, 0, True),
    )
    script = """
import pickle
import sys
classes, values = pickle.loads(sys.stdin.buffer.read())
assert all(cls.__module__ == 'hotpot.cheminfo.rules.physics.forcefields.contracts' for cls in classes)
assert tuple(type(value) for value in values) == classes[:2] + classes[3:]
assert values[0].atom_indices == (91,)
assert values[-1].piercings == (values[-2],)
assert values[-1].scan_complete is True
"""
    completed = subprocess.run(
        [sys.executable, '-B', '-c', script],
        input=pickle.dumps((classes, values)), capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()


def test_trajectory_values_cold_pickle_at_the_structure_owner():
    classes = (
        TrajectoryStart, TrajectoryStage, TrajectoryEvent,
        RingFrameEvidence, CoordinationFrameEvidence, OptimizationFrameEvidence,
    )
    values = (
        TrajectoryStart.COORDINATION_RESTORATION,
        TrajectoryStage.COMPLEX_UNTANGLING,
        TrajectoryEvent.BOND_ROLLBACK,
        RingFrameEvidence(2, ring_scope="full_graph", scan_complete=False),
        CoordinationFrameEvidence(
            (0, 1), False, pending_bond_count=2, safe_donor_atom_indices=(1, 3),
            minimum_normalized_clearance=0.75,
        ),
        OptimizationFrameEvidence(
            False, False, True, True, True, energy_change_kj_mol=0.125,
        ),
    )
    script = """
import pickle
import sys
from hotpot.cheminfo.structure import forcefields as owner
classes, values = pickle.loads(sys.stdin.buffer.read())
assert all(cls.__module__ == owner.__name__ for cls in classes)
assert all(getattr(owner, cls.__name__) is cls for cls in classes)
assert tuple(type(value) for value in values) == classes
assert values[0] is owner.TrajectoryStart.COORDINATION_RESTORATION
assert values[1] is owner.TrajectoryStage.COMPLEX_UNTANGLING
assert values[2] is owner.TrajectoryEvent.BOND_ROLLBACK
assert values[3].confirmed_piercing_count == 2
assert values[3].ring_scope == 'full_graph' and values[3].scan_complete is False
assert values[4].bond_atom_indices == (0, 1)
assert values[4].safe_donor_atom_indices == (1, 3)
assert values[4].accepted is False and values[4].pending_bond_count == 2
assert values[4].minimum_normalized_clearance == 0.75
assert values[5].converged is False and values[5].finite_energy is True
assert values[5].energy_change_kj_mol == 0.125
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        input=pickle.dumps((classes, values)), capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()


def test_b3d_moved_values_and_operation_classes_cold_pickle_at_actual_owners():
    from hotpot.cheminfo.structure.forcefields import (
        AtomIdentity, BondTopologyRevision, CoordinationEnvironment, ForceFieldFrame,
    )

    classes = (
        AtomIdentity, BondTopologyRevision, CoordinationEnvironment, ForceFieldFrame,
        ff.Build3DReport, ForceFieldTrajectory,
    )
    values = (
        AtomIdentity(0, 91, 6, 0, "C"),
        BondTopologyRevision(0, (BondTopology((0, 1), 1.0, "single"),)),
        CoordinationEnvironment(0, (1,), 1, 30, 2, (7,), ((1,),)),
        ForceFieldFrame(0, TrajectoryStage.LIGAND_BUILD, TrajectoryEvent.INITIAL, 0, 0),
        ff.Build3DReport(1, ff.ForceFieldValidationReport(True, ())),
    )
    for cls in classes:
        assert getattr(ff, cls.__name__) is cls
    script = """
import importlib
import pickle
import sys
classes, values = pickle.loads(sys.stdin.buffer.read())
assert tuple(type(value) for value in values) == classes[:5]
assert tuple(cls.__module__ for cls in classes) == (
    *('hotpot.cheminfo.structure.forcefields',) * 4,
    'hotpot.cheminfo.operations.forcefields.contracts',
    'hotpot.cheminfo.operations.forcefields.trajectory',
)
for cls in classes:
    assert getattr(importlib.import_module(cls.__module__), cls.__name__) is cls
assert values[0].symbol == 'C'
assert values[1].bonds[0].atom_indices == (0, 1)
assert values[2].chelate_groups == ((1,),)
assert values[3].stage.value == 'ligand_build'
assert not hasattr(values[4], 'atom_count')
assert values[4].added_hydrogen_count == 1
assert values[4].quality_report.passed
assert 'hotpot.cheminfo.forcefields' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        input=pickle.dumps((classes, values)), capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()


@pytest.mark.parametrize("legacy_module, name", (
    ("trajectory", "AtomIdentity"),
    ("trajectory", "BondTopologyRevision"),
    ("trajectory", "ForceFieldFrame"),
    ("contracts", "CoordinationEnvironment"),
))
def test_retired_b3d_pickle_paths_have_no_migration_loader(legacy_module, name):
    payload = f"chotpot.cheminfo.forcefields.{legacy_module}\n{name}\n.".encode()
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(payload)


@pytest.mark.parametrize("serializer", ("stage", "benchmark"))
def test_workflow_json_keeps_each_geometry_quality_at_its_result_owner(serializer):
    if serializer == "stage":
        from hotpot.cheminfo.operations.forcefields.stage import json_value
    else:
        from tests.benchmarks.coordination_complexes.io import json_value

    initial_quality = ff.ForceFieldValidationReport(
        passed=False,
        checks=(ff.AcceptanceCheck("initial_geometry", False),),
    )
    final_quality = ff.ForceFieldValidationReport(
        passed=True,
        checks=(ff.AcceptanceCheck("selected_geometry", True),),
    )
    build = ff.Build3DReport(added_hydrogen_count=2, quality_report=initial_quality)
    optimization = ff.ForceFieldRunReport(
        effective_forcefield="UFF", setup_succeeded=True, converged=False,
        epochs_completed=1, steps_submitted=7, initialization_steps=1,
        final_energy=12.0, best_energy=10.0, energy_unit="kJ/mol",
        rms_gradient=0.1, max_gradient=0.2, exploded=False,
        quality_report=final_quality,
    )
    workflow = ff.BuildAndOptimizeReport(
        effective_forcefield="UFF", build=build, optimization=optimization,
        quality_report=final_quality,
    )

    payload = json_value(workflow)

    assert workflow.quality_report is optimization.quality_report
    assert "quality_report" not in payload
    assert payload["build"]["quality_report"]["checks"][0]["name"] == "initial_geometry"
    assert payload["optimization"]["quality_report"]["checks"][0]["name"] == "selected_geometry"
    assert payload["optimization"]["best_energy"] == 10.0
    assert payload["optimization"]["final_energy"] == 12.0
    assert "trajectory" not in payload

    build_only = ff.ForceFieldWorkflowReport(
        effective_forcefield="UFF", build=build, optimization=None,
        quality_report=initial_quality,
    )
    assert json_value(build_only)["quality_report"]["checks"][0]["name"] == "initial_geometry"
