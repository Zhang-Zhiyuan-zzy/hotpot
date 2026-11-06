"""Native optimization decisions remain distinct from finite result availability."""

from __future__ import annotations

import os

import numpy as np
import pytest

import hotpot
from hotpot.pipeline.contracts import MolecularPayload, MolecularRecord, StageSpec
from hotpot.pipeline.runner import run_pipeline
from hotpot.plugins.xtb.adapter import (
    commit_xtb_coordinates,
    parse_xtb_artifacts,
    prepare_xtb_input,
)
from hotpot.plugins.xtb.backend import probe_xtb_backend
from hotpot.plugins.xtb.contracts import (
    XTBExecutionError,
    XTBMethod,
    XTBRequest,
    XTBResultError,
    XTBTask,
)
from hotpot.plugins.xtb.runner import run_xtb


_CONVERGED = "*** GEOMETRY OPTIMIZATION CONVERGED AFTER 12 ITERATIONS ***"
_NOT_CONVERGED = "*** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 219 ITERATIONS ***"


def _water():
    mol = hotpot.read_mol("[H]O[H]", "smi")
    mol.coordinates = np.array(((-0.75, 0.0, 0.0), (0.0, 0.5, 0.0), (0.75, 0.0, 0.0)))
    return mol


@pytest.fixture
def native_output_factory(fake_xtb_factory):
    """Change only a temporary executable's native decision and optional artifact."""
    def create(stdout=_NOT_CONVERGED, scenario="success", missing=None, return_code=0):
        executable = fake_xtb_factory(scenario)
        text = executable.read_text()
        text = text.replace('print("GEOMETRY OPTIMIZATION CONVERGED")', f"print({stdout!r})")
        terminal = '    print("TOTAL ENERGY      -5.123456789 Eh")\n    return 0'
        removal = "" if missing is None else f"    Path({missing!r}).unlink()\n"
        text = text.replace(
            terminal,
            removal + '    print("TOTAL ENERGY      -5.123456789 Eh")\n'
            + ('' if return_code == 0 else '    print("SCC is not converged", file=sys.stderr)\n')
            + f"    return {return_code}",
        )
        executable.write_text(text)
        return executable
    return create


def _request(executable, work, mol, method, task=XTBTask.OPTIMIZE):
    work.mkdir()
    input_path = work / ("input.sdf" if method is XTBMethod.GFNFF else "input.xyz")
    geometry = prepare_xtb_input(mol, input_path)
    request = XTBRequest(
        backend_info=probe_xtb_backend(executable, environment=dict(os.environ)),
        method=method,
        task=task,
        input_path=input_path,
        work_directory=work,
        charge=0,
        unpaired_electrons=None if method is XTBMethod.GFNFF else 0,
        environment=dict(os.environ),
    )
    return request, geometry


@pytest.mark.parametrize("method", (XTBMethod.GFNFF, XTBMethod.GFN2_XTB))
@pytest.mark.parametrize(
    "stdout,converged",
    (
        (_CONVERGED, True),
        (_NOT_CONVERGED, False),
        (_CONVERGED + "\n" + _NOT_CONVERGED, False),
        (_NOT_CONVERGED + "\n" + _CONVERGED, True),
        ("", True),
    ),
)
def test_final_native_geometry_decision_takes_precedence_over_success_files(
    native_output_factory, tmp_path, method, stdout, converged,
):
    request, _ = _request(native_output_factory(stdout), tmp_path / "run", _water(), method)

    report = run_xtb(request)

    assert report.provenance.return_code == 0
    assert report.converged is converged
    assert {".xtboptok", "xtbopt.log"}.issubset(report.artifacts)
    assert report.provenance.stdout.startswith(stdout)


@pytest.mark.parametrize("method", (XTBMethod.GFNFF, XTBMethod.GFN2_XTB))
def test_finite_nonconverged_result_can_be_parsed_and_explicitly_committed(
    native_output_factory, tmp_path, method,
):
    mol = _water()
    before = mol.coordinates.copy()
    request, expected = _request(native_output_factory(), tmp_path / "run", mol, method)
    report = run_xtb(request)

    result = parse_xtb_artifacts(report, expected)

    assert report.converged is False
    assert result.energy_hartree == -5.123456789
    assert result.gradient_norm == 0.00001
    assert result.atom_order_verified is True
    np.testing.assert_array_equal(mol.coordinates, before)
    commit_xtb_coordinates(mol, result)
    np.testing.assert_allclose(mol.coordinates, before + 0.1, atol=1e-14)


def test_native_execution_failure_remains_unavailable(
    native_output_factory, tmp_path,
):
    mol = _water()
    before = mol.coordinates.copy()
    request, expected = _request(
        native_output_factory(return_code=17), tmp_path / "run", mol, XTBMethod.GFN2_XTB,
    )

    with pytest.raises(XTBExecutionError) as caught:
        run_xtb(request)

    report = caught.value.report
    assert report.provenance.return_code == 17
    assert report.converged is False
    assert "SCC is not converged" in report.provenance.stderr
    assert ".xtboptok" in report.artifacts
    with pytest.raises(XTBResultError):
        parse_xtb_artifacts(report, expected)
    np.testing.assert_array_equal(mol.coordinates, before)


@pytest.mark.parametrize("missing", (".xtboptok", "xtbopt.log", "xtbopt.xyz", "xtbout.json"))
def test_nonconverged_result_still_requires_every_native_artifact(
    native_output_factory, tmp_path, missing,
):
    mol = _water()
    before = mol.coordinates.copy()
    request, expected = _request(
        native_output_factory(missing=missing), tmp_path / "run", mol, XTBMethod.GFN2_XTB,
    )

    with pytest.raises(XTBResultError) as caught:
        run_xtb(request)

    report = caught.value.report
    assert report.converged is False
    assert report.provenance.return_code == 0
    assert missing not in report.artifacts
    with pytest.raises(XTBResultError):
        parse_xtb_artifacts(report, expected)
    np.testing.assert_array_equal(mol.coordinates, before)


@pytest.mark.parametrize(
    "scenario",
    ("nonfinite_energy", "nonfinite_gradient", "nonfinite_coordinates", "truncated_geometry", "reordered_elements"),
)
def test_nonconvergence_does_not_weaken_finite_metrics_or_atom_order(
    native_output_factory, tmp_path, scenario,
):
    mol = _water()
    # The fixture must not have a palindromic element sequence for reordered-elements.
    if scenario == "reordered_elements":
        mol = hotpot.read_mol("[H]F", "smi")
        mol.coordinates = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    before = mol.coordinates.copy()
    request, expected = _request(
        native_output_factory(scenario=scenario), tmp_path / "run", mol, XTBMethod.GFN2_XTB,
    )
    report = run_xtb(request)

    with pytest.raises(XTBResultError) as caught:
        parse_xtb_artifacts(report, expected)

    assert caught.value.report is report
    assert report.converged is False
    np.testing.assert_array_equal(mol.coordinates, before)


def test_nonconverged_corrupt_result_is_rejected_without_source_mutation(
    native_output_factory, tmp_path,
):
    mol = _water()
    before = mol.coordinates.copy()
    request, expected = _request(
        native_output_factory(), tmp_path / "run", mol, XTBMethod.GFN2_XTB,
    )
    report = run_xtb(request)
    report.artifacts["xtbout.json"].path.write_text("{malformed native output")

    with pytest.raises(XTBResultError) as caught:
        parse_xtb_artifacts(report, expected)

    assert caught.value.report is report
    assert report.converged is False
    np.testing.assert_array_equal(mol.coordinates, before)


@pytest.mark.parametrize(
    "flag,outcomes,output_count",
    (
        (None, ("returned", "returned"), 2),
        ("--sample-break-on", ("sample-break", "sample-break"), 0),
        ("--batch-break-on", ("returned", "not-attempted"), 1),
    ),
)
def test_convergence_finding_returns_finite_output_and_respects_per_code_breaks(
    native_output_factory, tmp_path, flag, outcomes, output_count,
):
    executable = native_output_factory()
    records = tuple(MolecularRecord(_water()) for _ in range(2))
    before = tuple(record.molecule.coordinates.copy() for record in records)
    argv = ("--method", "gfn2", "--task", "optimize", "--charge", "0", "--xtb-executable", str(executable))
    if flag is not None:
        argv += (flag, "XTB:convergence")

    run = run_pipeline(
        (StageSpec("xtb", argv),),
        results_directory=tmp_path / "pipeline",
        initial_payload=MolecularPayload(records),
    )

    result = run.stage_results[0]
    assert result.report["output_count"] == output_count
    assert tuple(row["outcome"] for row in result.report["sample_outcomes"]) == outcomes
    assert result.break_pipeline is (flag == "--batch-break-on")
    assert run.status.value == ("succeeded" if flag is None else "failed")
    for row in result.report["sample_outcomes"]:
        if row["outcome"] == "not-attempted":
            continue
        assert row["findings"] == ({"code": "XTB:convergence", "name": "convergence", "passed": False, "converged": False},)
        if flag == "--sample-break-on":
            observation = tmp_path / "pipeline" / "stages" / "00-xtb" / row["observation_path"]
            assert observation.is_file()
    for row, record, coordinates in zip(result.report["sample_outcomes"], records, before):
        if row["outcome"] == "not-attempted":
            np.testing.assert_array_equal(record.molecule.coordinates, coordinates)
        else:
            np.testing.assert_allclose(record.molecule.coordinates, coordinates + 0.1, atol=1e-14)
    for record in result.payload.records:
        np.testing.assert_allclose(record.molecule.coordinates, before[0] + 0.1, atol=1e-14)
    native_reports = result.report["xtb"]
    if isinstance(native_reports, dict):
        native_reports = (native_reports,)
    assert all(report["converged"] is False for report in native_reports)
    assert all(report["energy_hartree"] == -5.123456789 for report in native_reports)
