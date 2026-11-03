"""Known xTB local failures stay distinct from request and commit integrity."""

from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import StatePolicy
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.plugins.xtb import adapter, state_resolution, workflow
from hotpot.plugins.xtb.adapter import commit_xtb_coordinates
from hotpot.plugins.xtb.contracts import (
    XTBApplicabilityError,
    XTBError,
    XTBExecutableError,
    XTBExecutionError,
    XTBGeometry,
    XTBInputError,
    XTBMethod,
    XTBParsedResult,
    XTBResultError,
    XTBRunReport,
    XTBTask,
)


@pytest.mark.parametrize(
    "error_type, code",
    (
        (XTBError, None),
        (XTBExecutableError, None),
        (XTBInputError, None),
        (XTBApplicabilityError, "XTB-ELEMENT-UNSUPPORTED"),
    ),
)
def test_only_known_xtb_domain_errors_have_local_failure_codes(error_type, code):
    error = error_type("XTB-RESULT-UNAVAILABLE: same message across classes")
    assert error.sample_failure_code == code
    assert str(error) == "XTB-RESULT-UNAVAILABLE: same message across classes"


@pytest.mark.parametrize(
    "error_type, code",
    (
        (XTBExecutionError, "XTB-NATIVE-EXECUTION"),
        (XTBResultError, "XTB-RESULT-UNAVAILABLE"),
    ),
)
def test_process_and_result_errors_preserve_raw_report(error_type, code):
    report = object()
    error = error_type("raw backend failure", report)
    assert error.sample_failure_code == code
    assert error.report is report
    assert str(error) == "raw backend failure"


@pytest.fixture
def isolated_workflow(monkeypatch, tmp_path):
    """Use actual source checks/transactions around a stubbed external run."""
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1, bond_order=1.0)
    mol.coordinates = np.array(((0.0, 0.0, 0.0), (1.5, 0.0, 0.0)))
    state = SimpleNamespace(
        charge=0,
        uhf=0,
        scientific_state_identity=object(),
        facts_identity=object(),
        projection_convention="pure_spin_highest_ms",
    )
    backend = SimpleNamespace()
    report = XTBRunReport(
        backend_info=backend,
        requested_method=XTBMethod.GFN2_XTB,
        effective_method=XTBMethod.GFN2_XTB,
        task=XTBTask.OPTIMIZE,
        charge=0,
        unpaired_electrons=0,
        work_directory=tmp_path,
        argv=("stub-xtb",),
        return_code=0,
        stdout="raw stdout",
        stderr="raw stderr",
        elapsed_seconds=0.1,
        process_succeeded=True,
        converged=True,
        artifacts={},
        provenance=object(),
    )
    parsed = XTBParsedResult(
        energy_hartree=-1.0,
        gradient_norm=0.0,
        partial_charges=None,
        optimized_geometry=XTBGeometry(
            ("C", "C"), ((0.0, 0.0, 0.0), (1.6, 0.0, 0.0)),
        ),
        atom_order_verified=True,
    )
    monkeypatch.setattr(workflow, "prepare_xtb_input", lambda mol, path: parsed.optimized_geometry)
    monkeypatch.setattr(workflow, "probe_xtb_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(workflow, "validate_element_support", lambda *args: None)
    monkeypatch.setattr(workflow, "_validate_projection", lambda *args: None)
    monkeypatch.setattr(workflow, "run_xtb", lambda request: report)
    monkeypatch.setattr(workflow, "parse_xtb_artifacts", lambda *args: parsed)
    return mol, state, report, parsed


@pytest.mark.parametrize("mode", ("original_cause", "mapping"))
def test_coordinate_commit_integrity_error_preserves_identity_cause_and_rolls_back(
    isolated_workflow, monkeypatch, mode,
):
    mol, state, report, parsed = isolated_workflow
    original_coordinates = mol.coordinates.copy()
    cause = RuntimeError("commit identity damaged")
    original_error = XTBInputError("coordinate commit rejected")
    observed_errors = []
    if mode == "mapping":
        parsed = replace(
            parsed,
            optimized_geometry=XTBGeometry(
                ("N", "C"), parsed.optimized_geometry.coordinates,
            ),
        )

    def fail_commit(owner, result):
        owner.coordinates = owner.coordinates + 7.0
        if mode == "original_cause":
            observed_errors.append(original_error)
            raise original_error from cause
        try:
            commit_xtb_coordinates(owner, parsed)
        except XTBInputError as error:
            observed_errors.append(error)
            raise

    monkeypatch.setattr(workflow, "commit_xtb_coordinates", fail_commit)
    with pytest.raises(XTBInputError) as caught:
        workflow.run_gfn_xtb(mol, state=state, task=XTBTask.OPTIMIZE)
    assert caught.value is observed_errors[0]
    assert type(caught.value) is XTBInputError
    assert caught.value.sample_failure_code is None
    assert caught.value.__cause__ is (cause if mode == "original_cause" else None)
    np.testing.assert_array_equal(mol.coordinates, original_coordinates)
    assert report.coordinates_committed is False


def test_source_change_after_external_run_remains_unknown_integrity_error(isolated_workflow, monkeypatch):
    mol, state, report, parsed = isolated_workflow
    changed = mol.coordinates + 3.0

    def mutate_source(request):
        mol.coordinates = changed
        return report

    monkeypatch.setattr(workflow, "run_xtb", mutate_source)
    with pytest.raises(RuntimeError, match="geometry.*changed") as caught:
        workflow.run_gfn_xtb(mol, state=state, task=XTBTask.OPTIMIZE)
    assert getattr(caught.value, "sample_failure_code", None) is None
    np.testing.assert_array_equal(mol.coordinates, changed)


def test_parse_failure_retains_named_result_code_report_and_cause(isolated_workflow, monkeypatch):
    mol, state, report, parsed = isolated_workflow
    original = mol.coordinates.copy()
    cause = XTBResultError("required output absent", report)

    def fail_parse(*args):
        raise cause

    monkeypatch.setattr(workflow, "parse_xtb_artifacts", fail_parse)
    with pytest.raises(XTBResultError) as caught:
        workflow.run_gfn_xtb(mol, state=state, task=XTBTask.OPTIMIZE)
    assert caught.value.sample_failure_code == "XTB-RESULT-UNAVAILABLE"
    assert caught.value.__cause__ is cause
    assert str(caught.value) == str(cause)
    assert caught.value.report.stdout == "raw stdout"
    assert caught.value.report.stderr == "raw stderr"
    assert caught.value.report.scientific_state_identity is state.scientific_state_identity
    np.testing.assert_array_equal(mol.coordinates, original)


def _explicit_geometry_molecule(coordinates=((0.0, 0.0, 0.0), (1.5, 0.0, 0.0))):
    mol = Molecule()
    for _ in coordinates:
        mol.create_atom(atomic_number=2)
    if coordinates:
        mol.coordinates = np.asarray(coordinates)
    return mol


def test_actual_explicit_hydrogen_rejection_has_its_input_code_and_message():
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    atom.implicit_hydrogens = 4
    with pytest.raises(XTBInputError) as caught:
        adapter.molecule_to_xtb_geometry(mol)
    assert caught.value.sample_failure_code == "XTB-EXPLICIT-H-REQUIRED"
    assert "4 implicit hydrogens but has 0 explicit hydrogen atoms" in str(caught.value)


@pytest.mark.parametrize(
    "coordinates, message",
    (
        ((), "non-empty geometry"),
        (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "all atoms at the same position"),
        (((0.0, 0.0, 0.0), (np.nan, 0.0, 0.0)), "non-finite coordinates"),
    ),
)
def test_actual_input_geometry_rejections_have_call_site_code(coordinates, message):
    mol = _explicit_geometry_molecule(coordinates)
    before = mol.coordinates.copy()
    with pytest.raises(XTBInputError, match=message) as caught:
        adapter.molecule_to_xtb_geometry(mol)
    assert caught.value.sample_failure_code == "XTB-INPUT-GEOMETRY"
    np.testing.assert_array_equal(mol.coordinates, before)


@pytest.mark.parametrize("error_type, code", ((XTBInputError, "XTB-INPUT-GEOMETRY"), (ValueError, None)))
def test_input_validator_tags_only_its_known_error_and_preserves_original_cause(monkeypatch, error_type, code):
    mol = _explicit_geometry_molecule()
    error = error_type("unchanged validator failure")
    cause = RuntimeError("original validator cause")

    def fail(geometry):
        raise error from cause

    monkeypatch.setattr(adapter, "_validate_geometry", fail)
    with pytest.raises(error_type) as caught:
        adapter.molecule_to_xtb_geometry(mol)
    assert caught.value is error
    assert caught.value.__cause__ is cause
    assert getattr(caught.value, "sample_failure_code", None) == code


def test_actual_commit_geometry_validation_stays_uncoded_integrity_error():
    mol = _explicit_geometry_molecule()
    before = mol.coordinates.copy()
    result = XTBParsedResult(
        energy_hartree=-1.0, gradient_norm=0.0, partial_charges=None,
        optimized_geometry=XTBGeometry(("He", "He"), ((0.0, 0.0, 0.0), (np.nan, 0.0, 0.0))),
        atom_order_verified=True,
    )
    with pytest.raises(XTBInputError, match="non-finite coordinates") as caught:
        adapter.commit_xtb_coordinates(mol, result)
    assert caught.value.sample_failure_code is None
    np.testing.assert_array_equal(mol.coordinates, before)


def test_actual_illegal_backend_state_keeps_bundle_projection_and_local_code(fake_xtb_factory):
    mol = Molecule()
    mol.create_atom(atomic_number=1)
    with pytest.raises(state_resolution.XTBStateResolutionError) as caught:
        state_resolution.resolve_xtb_state(
            mol, XTBMethod.GFN2_XTB, executable=fake_xtb_factory(),
            environment=dict(os.environ), charge=0, uhf=0, policy=StatePolicy.STRICT,
        )
    error = caught.value
    assert error.sample_failure_code == "XTB-STATE-UNAVAILABLE"
    assert error.scientific_state.facts.identity == extract_molecular_facts(mol).identity
    assert error.projection_result.projection is None
    assert error.projection_result.request.uhf == 0
    assert any(finding.code == "HARD-013.UHF_PARITY" for finding in error.projection_result.findings)
    assert "HARD-013.UHF_PARITY" in str(error)


@pytest.mark.parametrize("mismatch", ("carried", "expected_facts", "expected_science"))
def test_state_identity_mismatch_remains_uncoded_input_integrity_failure(fake_xtb_factory, mismatch):
    mol = Molecule()
    mol.create_atom(atomic_number=2)
    executable = fake_xtb_factory()
    options = dict(
        executable=executable, environment=dict(os.environ), charge=0, uhf=0, policy=StatePolicy.STRICT,
    )
    resolved = state_resolution.resolve_xtb_state(mol, XTBMethod.GFN2_XTB, **options)
    if mismatch == "carried":
        mol.atoms[0].atomic_number = 1
        options.update(charge=None, carried_state=resolved.scientific_state)
    elif mismatch == "expected_facts":
        options["expected_facts_identity"] = object()
    else:
        options["expected_scientific_identity"] = object()
    with pytest.raises(XTBInputError) as caught:
        state_resolution.resolve_xtb_state(mol, XTBMethod.GFN2_XTB, **options)
    assert type(caught.value) is XTBInputError
    assert caught.value.sample_failure_code is None


def test_unknown_state_extraction_error_keeps_exact_object_and_no_code(monkeypatch):
    cause = ValueError("unknown molecular state extraction failure")
    monkeypatch.setattr(state_resolution, "probe_xtb_backend", lambda *args, **kwargs: object())

    def fail(mol):
        raise cause

    monkeypatch.setattr(state_resolution, "extract_molecular_facts", fail)
    with pytest.raises(ValueError) as caught:
        state_resolution.resolve_xtb_state(
            Molecule(), XTBMethod.GFN2_XTB, executable=None, environment={},
            charge=0, uhf=0, policy=StatePolicy.STRICT,
        )
    assert caught.value is cause
    assert getattr(caught.value, "sample_failure_code", None) is None
