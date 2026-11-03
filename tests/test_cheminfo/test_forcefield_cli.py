from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from hotpot import __main__ as hotpot_main
from hotpot.commands import ff as cli
from hotpot.cheminfo.operations.forcefields import (
    ComplexBuildDiagnostics,
    ForceFieldError,
    ForceFieldSetupError,
    ForceFieldSetupReport,
    AcceptanceCheck,
    ForceFieldValidationReport,
    TrajectoryStart,
)
from hotpot.cheminfo.operations.forcefields.stage import ForceFieldRouteOptions


@dataclass(frozen=True)
class _Report:
    requested_forcefield: str | None
    effective_forcefield: str
    quality_report: ForceFieldValidationReport
    trajectory: object | None = None


class _Molecule:
    def __init__(
        self,
        name: str,
        *,
        has_3d: bool = False,
        has_metal: bool = False,
    ) -> None:
        self.name = name
        self.has_3d = has_3d
        self.has_metal = has_metal
        self.write_calls = []

    def write(self, *, fmt: str, write_single: bool) -> str:
        self.write_calls.append((fmt, write_single))
        return f"{fmt}:{self.name}\n"


def _report(*, passed: bool = True) -> _Report:
    checks = (AcceptanceCheck("finite_coordinates", True),)
    if not passed:
        checks += (AcceptanceCheck("atom_too_close", False, measured=0.1, threshold=0.5),)
    return _Report(None, "UFF", ForceFieldValidationReport("standard", passed, checks))


def _install_reader(monkeypatch, molecules):
    observed = []

    def reader(source, fmt=None):
        observed.append((source, fmt))
        return iter(molecules)

    monkeypatch.setattr(cli, "MolReader", reader)
    return observed


def _fail_if_called(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} should not have been called")

    return fail


def test_parser_defaults_and_documented_choices():
    args = cli.build_parser().parse_args(["CN"])

    assert args.inputs == ["CN"]
    assert args.output is None
    assert args.input_format is None
    assert args.output_format is None
    assert args.rebuild is False
    assert args.optimize_only is False
    assert args.route == "auto"
    assert args.forcefield == "auto"
    assert args.algorithm == "conjugate"
    assert args.epochs == 100
    assert args.steps_per_epoch == 100
    assert args.convergence_level == "fast"
    assert args.add_hydrogens is True
    assert args.quality == "standard"
    assert args.seed is None
    assert args.timeout == pytest.approx(1000.0)
    assert args.trajectory is None
    assert args.trajectory_start is None
    assert args.report is None
    assert args.jobs == 1
    assert args.overwrite is False

    explicit = cli.build_parser().parse_args(
        [
            "CN",
            "--route",
            "complex",
            "--forcefield",
            "uff",
            "--algorithm",
            "steepest",
            "--quality",
            "strict",
            "--convergence-level",
            "strict",
            "--trajectory-start",
            "coordination-restoration",
        ]
    )
    assert explicit.route == "complex"
    assert explicit.forcefield == "uff"
    assert explicit.algorithm == "steepest"
    assert explicit.quality == "strict"
    assert explicit.convergence_level == "strict"
    assert explicit.trajectory_start == "coordination-restoration"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--route", "unknown"),
        ("--forcefield", "unknown"),
        ("--algorithm", "unknown"),
        ("--convergence-level", "unknown"),
        ("--quality", "unknown"),
        ("--trajectory-start", "unknown"),
        ("--epochs", "0"),
        ("--steps-per-epoch", "0"),
        ("--timeout", "0"),
        ("--jobs", "0"),
    ],
)
def test_parser_rejects_invalid_choice_or_nonpositive_value(option, value):
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["CN", option, value])

    assert exit_info.value.code == 2


def test_rebuild_and_optimize_only_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["CN", "--rebuild", "--optimize-only"])

    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    (
        ("CN", "--output", "same", "--report", "same"),
        ("CN", "--output", "same", "--trajectory", "same"),
        ("CN", "--report", "same", "--trajectory", "same"),
        ("CN", "--trajectory", "-"),
        ("CN", "--trajectory-start", "final-optimization"),
    ),
)
def test_output_contract_rejects_ambiguous_paths(arguments):
    args = cli.build_parser().parse_args(arguments)

    with pytest.raises(ValueError):
        cli._check_output_paths(args)


def test_invalid_cross_option_request_is_concise(monkeypatch, capsys):
    _install_reader(monkeypatch, [_Molecule("invalid")])

    status = cli.main(
        ["CN", "--route", "complex", "--forcefield", "mmff94"]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "InputError" in captured.err
    assert "only UFF" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_output_contract_is_concise(capsys):
    status = cli.main(["CN", "--trajectory-start", "final-optimization"])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err == "hotpot ff: --trajectory-start requires --trajectory\n"


@pytest.mark.parametrize(
    ("output_option", "result_name", "trajectory_name"),
    (
        ("--output", "trajectory/result.mol2", "trajectory"),
        ("--report", "trajectory/report.json", "trajectory"),
        ("--output", "result.mol2", "result.mol2/trajectory"),
    ),
)
def test_trajectory_and_result_paths_must_not_overlap(
    tmp_path,
    output_option,
    result_name,
    trajectory_name,
):
    args = cli.build_parser().parse_args(
        [
            "CN",
            output_option,
            str(tmp_path / result_name),
            "--trajectory",
            str(tmp_path / trajectory_name),
        ]
    )

    with pytest.raises(ValueError, match="must not overlap"):
        cli._check_output_paths(args)


def test_invalid_smiles_is_a_concise_input_error(capsys):
    status = cli.main(["not-a-smiles??"])

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert "Could not read molecule input" in captured.err
    assert "Traceback" not in captured.err


def test_top_level_dispatches_ff(monkeypatch):
    observed = []

    def run(args):
        observed.append(args)
        return 7

    monkeypatch.setattr(cli, "run", run)

    assert hotpot_main.main(["ff", "CN"]) == 7
    assert len(observed) == 1
    assert observed[0].works == "ff"
    assert observed[0].inputs == ["CN"]


def test_direct_smiles_without_3d_builds_and_optimizes(monkeypatch, capsys):
    mol = _Molecule("CN")
    observed_reader = _install_reader(monkeypatch, [mol])
    observed_calls = []

    def build_and_optimize(molecule, forcefield, **options):
        observed_calls.append((molecule, forcefield, options))
        return _report()

    monkeypatch.setattr(cli, "build_and_optimize", build_and_optimize)
    monkeypatch.setattr(cli, "auto_optimize", _fail_if_called("auto_optimize"))

    assert cli.main(
        [
            "CN",
            "--seed",
            "19",
            "--epochs",
            "3",
            "--convergence-level",
            "strict",
        ]
    ) == 0

    captured = capsys.readouterr()
    assert captured.out == "mol2:CN\n"
    assert captured.err == ""
    assert observed_reader == [("CN", "smi")]
    assert mol.write_calls == [("mol2", True)]
    assert observed_calls[0][0] is mol
    assert observed_calls[0][1] is None
    assert observed_calls[0][2] == {
        "algorithm": "conjugate",
        "epochs": 3,
        "steps_per_epoch": 100,
        "convergence_level": cli.ConvergenceLevel.STRICT,
        "add_hydrogens": True,
        "quality_level": "standard",
        "seed": 19,
        "save_movie": False,
        "trajectory_path": None,
        "timeout": 1000.0,
    }


def test_existing_3d_uses_auto_optimize(monkeypatch, capsys):
    mol = _Molecule("existing", has_3d=True)
    _install_reader(monkeypatch, [mol])
    observed_calls = []

    def auto_optimize(molecule, forcefield, **options):
        observed_calls.append((molecule, forcefield, options))
        return _report()

    monkeypatch.setattr(cli, "auto_optimize", auto_optimize)
    monkeypatch.setattr(
        cli,
        "build_and_optimize",
        _fail_if_called("build_and_optimize"),
    )

    assert cli.main(["existing.mol2", "--forcefield", "uff"]) == 0

    assert capsys.readouterr().out == "mol2:existing\n"
    assert observed_calls[0][0] is mol
    assert observed_calls[0][1] == "UFF"
    assert "timeout" not in observed_calls[0][2]


@pytest.mark.parametrize(
    ("forcefield", "seed", "trajectory_start", "trajectory_path"),
    [
        (None, None, None, None),
        ("UFF", 0, TrajectoryStart.COORDINATION_RESTORATION, "trajectory"),
    ],
)
def test_route_options_are_forwarded_by_identity(
    monkeypatch,
    forcefield,
    seed,
    trajectory_start,
    trajectory_path,
):
    mol = _Molecule("forwarded")
    options = ForceFieldRouteOptions(
        rebuild=False,
        optimize_only=False,
        route="auto",
        forcefield=forcefield,
        algorithm="conjugate",
        epochs=2,
        steps_per_epoch=3,
        add_hydrogens=True,
        quality_level="standard",
        seed=seed,
        timeout=10.0,
        trajectory_start=trajectory_start,
    )
    report = _report()
    observed = []

    def execute(molecule, route_options, **kwargs):
        observed.append((molecule, route_options, kwargs))
        return report

    monkeypatch.setattr(cli, "execute_forcefield_route", execute)

    assert cli._run_forcefield(mol, options, trajectory_path) is report
    assert observed[0][0] is mol
    assert observed[0][1] is options
    assert observed[0][2]["trajectory_path"] == trajectory_path
    assert observed[0][2]["operations"].build3d is cli.build3d


@pytest.mark.parametrize("has_3d", [False, True])
def test_complex_uses_native_fast_auto_route(monkeypatch, has_3d):
    mol = _Molecule("complex", has_3d=has_3d, has_metal=True)
    options = ForceFieldRouteOptions(
        rebuild=False,
        optimize_only=False,
        route="auto",
        forcefield=None,
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        add_hydrogens=True,
        quality_level="standard",
        seed=None,
        timeout=10.0,
        trajectory_start=None,
    )
    observed = []
    monkeypatch.setattr(
        cli,
        "auto_optimize",
        lambda molecule, forcefield, **kwargs: (
            observed.append((molecule, forcefield, kwargs)) or _report()
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_and_optimize",
        _fail_if_called("build_and_optimize"),
    )

    cli._run_forcefield(mol, options, None)

    assert observed[0][0] is mol
    assert observed[0][2]["timeout"] == 10.0


@pytest.mark.parametrize(
    ("route", "has_3d", "expected_name"),
    [
        ("complex", False, "complexes_build"),
        ("complex", True, "optimize_complex"),
        ("organic", True, "optimize"),
    ],
)
def test_explicit_route_selects_the_corresponding_api(
    monkeypatch,
    route,
    has_3d,
    expected_name,
):
    mol = _Molecule("routed", has_3d=has_3d, has_metal=route == "complex")
    options = ForceFieldRouteOptions(
        rebuild=False,
        optimize_only=False,
        route=route,
        forcefield=None,
        algorithm="conjugate",
        epochs=2,
        steps_per_epoch=4,
        add_hydrogens=True,
        quality_level="standard",
        seed=None,
        timeout=10.0,
        trajectory_start=None,
    )
    observed = []

    for name in ("complexes_build", "optimize_complex", "optimize"):
        if name == expected_name:
            monkeypatch.setattr(
                cli,
                name,
                lambda molecule, forcefield, _name=name, **kwargs: (
                    observed.append((_name, molecule, forcefield, kwargs)) or _report()
                ),
            )
        else:
            monkeypatch.setattr(cli, name, _fail_if_called(name))

    assert cli._run_forcefield(mol, options, None) == _report()
    assert observed[0][0] == expected_name
    assert observed[0][1] is mol


def test_explicit_organic_rebuilds_then_optimizes(monkeypatch):
    mol = _Molecule("organic")
    options = ForceFieldRouteOptions(
        rebuild=False,
        optimize_only=False,
        route="organic",
        forcefield="UFF",
        algorithm="steepest",
        epochs=2,
        steps_per_epoch=3,
        add_hydrogens=True,
        quality_level="strict",
        seed=5,
        timeout=9.0,
        trajectory_start=None,
    )
    calls = []
    build_report = SimpleNamespace()

    def build3d(molecule, **kwargs):
        calls.append(("build3d", molecule, kwargs))
        return build_report

    optimization_report = _report()

    def optimize(molecule, forcefield, **kwargs):
        calls.append(("optimize", molecule, forcefield, kwargs))
        return optimization_report

    monkeypatch.setattr(cli, "build3d", build3d)
    monkeypatch.setattr(cli, "optimize", optimize)

    result = cli._run_forcefield(mol, options, None)

    assert [call[0] for call in calls] == ["build3d", "optimize"]
    assert calls[0][2] == {
        "add_hydrogens": True,
        "seed": 5,
        "timeout": 9.0,
    }
    assert calls[1][2] == "UFF"
    assert calls[1][3]["add_hydrogens"] is False
    assert result.build is build_report
    assert result.optimization is optimization_report


def test_rebuild_overrides_existing_coordinates(monkeypatch):
    mol = _Molecule("existing", has_3d=True)
    options = ForceFieldRouteOptions(
        rebuild=True,
        optimize_only=False,
        route="auto",
        forcefield=None,
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        add_hydrogens=True,
        quality_level="standard",
        seed=None,
        timeout=10.0,
        trajectory_start=None,
    )
    observed = []
    monkeypatch.setattr(
        cli,
        "build_and_optimize",
        lambda molecule, forcefield, **kwargs: (
            observed.append((molecule, forcefield, kwargs)) or _report()
        ),
    )
    monkeypatch.setattr(cli, "auto_optimize", _fail_if_called("auto_optimize"))

    cli._run_forcefield(mol, options, None)

    assert len(observed) == 1


def test_optimize_only_rejects_input_without_3d():
    mol = _Molecule("flat")
    options = ForceFieldRouteOptions(
        rebuild=False,
        optimize_only=True,
        route="auto",
        forcefield=None,
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        add_hydrogens=True,
        quality_level="standard",
        seed=None,
        timeout=10.0,
        trajectory_start=None,
    )

    with pytest.raises(ValueError, match="existing coordinates"):
        cli._run_forcefield(mol, options, None)


@pytest.mark.parametrize(
    ("has_metal", "optimizer_name"),
    [(False, "optimize"), (True, "optimize_complex")],
)
def test_optimize_only_preserves_existing_coordinates(
    monkeypatch,
    has_metal,
    optimizer_name,
):
    mol = _Molecule("existing", has_3d=True, has_metal=has_metal)
    options = ForceFieldRouteOptions(
        rebuild=False,
        optimize_only=True,
        route="auto",
        forcefield=None,
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        add_hydrogens=True,
        quality_level="standard",
        seed=None,
        timeout=10.0,
        trajectory_start=None,
    )
    observed = []

    for name in ("optimize", "optimize_complex"):
        if name == optimizer_name:
            monkeypatch.setattr(
                cli,
                name,
                lambda molecule, forcefield, _name=name, **kwargs: (
                    observed.append((_name, molecule, forcefield, kwargs))
                    or _report()
                ),
            )
        else:
            monkeypatch.setattr(cli, name, _fail_if_called(name))
    monkeypatch.setattr(cli, "auto_optimize", _fail_if_called("auto_optimize"))
    monkeypatch.setattr(
        cli,
        "build_and_optimize",
        _fail_if_called("build_and_optimize"),
    )

    assert cli._run_forcefield(mol, options, None) == _report()
    assert observed[0][0] == optimizer_name
    assert "timeout" not in observed[0][3]


def test_output_file_is_byte_equivalent_to_stdout(monkeypatch, tmp_path, capsys):
    mol_stdout = _Molecule("same")
    mol_file = _Molecule("same")
    read_count = 0

    def reader(source, fmt=None):
        nonlocal read_count
        molecule = (mol_stdout, mol_file)[read_count]
        read_count += 1
        return iter((molecule,))

    monkeypatch.setattr(cli, "MolReader", reader)
    monkeypatch.setattr(cli, "build_and_optimize", lambda *args, **kwargs: _report())

    assert cli.main(["CN"]) == 0
    stdout_payload = capsys.readouterr().out

    output_path = tmp_path / "optimized.mol2"
    assert cli.main(["CN", "-o", str(output_path)]) == 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert output_path.read_bytes() == stdout_payload.encode("utf-8")


def test_output_dash_explicitly_selects_stdout(monkeypatch, capsys):
    mol = _Molecule("stdout")
    _install_reader(monkeypatch, [mol])
    monkeypatch.setattr(cli, "build_and_optimize", lambda *args, **kwargs: _report())

    assert cli.main(["CN", "--output", "-"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "mol2:stdout\n"
    assert captured.err == ""


def test_stdin_dash_uses_requested_input_format(monkeypatch, capsys):
    mol = _Molecule("stdin")
    observed_reader = []

    def reader(source, fmt=None):
        source_path = Path(source)
        observed_reader.append(
            (source_path.read_text(encoding="utf-8"), fmt, source_path.suffix)
        )
        return iter((mol,))

    monkeypatch.setattr(cli, "MolReader", reader)
    monkeypatch.setattr(cli, "build_and_optimize", lambda *args, **kwargs: _report())
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("CN first\nCO second\n"))

    assert cli.main(["-", "--input-format", "smi"]) == 0

    assert observed_reader == [("CN first\nCO second\n", "smi", ".smi")]
    assert capsys.readouterr().out == "mol2:stdin\n"


def test_stdin_smiles_reads_every_record(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("CC first\nCO second\n"))

    records = cli._read_molecules(("-",), "smi")

    assert len(records) == 2
    assert tuple(record.source for record in records) == ("stdin", "stdin")
    assert tuple(record.source_record_index for record in records) == (0, 1)


def test_smiles_file_reads_every_record(tmp_path):
    input_path = tmp_path / "records.smi"
    input_path.write_text("CC first\nCO second\n", encoding="utf-8")

    records = cli._read_molecules((str(input_path),), None)

    assert len(records) == 2
    assert tuple(record.source_record_index for record in records) == (0, 1)


def test_multiple_records_preserve_input_order(monkeypatch, capsys):
    molecules = [_Molecule("first"), _Molecule("second"), _Molecule("third")]
    _install_reader(monkeypatch, molecules)
    calls = []

    def build(molecule, *args, **kwargs):
        calls.append(molecule.name)
        return _report()

    monkeypatch.setattr(cli, "build_and_optimize", build)

    assert cli.main(["records.smi"]) == 0

    assert calls == ["first", "second", "third"]
    assert capsys.readouterr().out == (
        "mol2:first\n"
        "mol2:second\n"
        "mol2:third\n"
    )


def test_trajectory_and_report_are_forwarded_and_persisted(
    monkeypatch,
    tmp_path,
    capsys,
):
    mol = _Molecule("tracked")
    _install_reader(monkeypatch, [mol])
    observed = []

    def build(molecule, forcefield, **options):
        observed.append(options)
        return _report()

    monkeypatch.setattr(cli, "build_and_optimize", build)
    trajectory_path = tmp_path / "trajectory"
    report_path = tmp_path / "report.json"

    assert (
        cli.main(
            [
                "CN",
                "--trajectory",
                str(trajectory_path),
                "--trajectory-start",
                "coordination-restoration",
                "--report",
                str(report_path),
            ]
        )
        == 0
    )

    assert observed[0]["save_movie"] is True
    assert observed[0]["trajectory_path"] == str(trajectory_path)
    assert observed[0]["trajectory_start"] is TrajectoryStart.COORDINATION_RESTORATION
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["results"][0]["status"] == "ok"
    assert report["results"][0]["forcefield_report"]["effective_forcefield"] == "UFF"
    assert capsys.readouterr().out == "mol2:tracked\n"


def test_known_forcefield_error_is_nonzero_without_fake_payload(
    monkeypatch,
    capsys,
):
    mol = _Molecule("failed")
    _install_reader(monkeypatch, [mol])

    def fail(*args, **kwargs):
        raise ForceFieldSetupError("backend did not produce usable coordinates")

    monkeypatch.setattr(cli, "build_and_optimize", fail)

    assert cli.main(["CN"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ForceFieldSetupError" in captured.err
    assert "backend did not produce usable coordinates" in captured.err
    assert mol.write_calls == []


def test_failed_overwrite_does_not_leave_stale_structure(
    monkeypatch,
    tmp_path,
    capsys,
):
    mol = _Molecule("failed")
    _install_reader(monkeypatch, [mol])
    output_path = tmp_path / "stale.mol2"
    output_path.write_text("old structure\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise ForceFieldSetupError("optimization failed")

    monkeypatch.setattr(cli, "build_and_optimize", fail)

    assert (
        cli.main(["CN", "--output", str(output_path), "--overwrite"])
        == 1
    )

    assert not output_path.exists()
    assert capsys.readouterr().out == ""


def test_forcefield_error_report_preserves_typed_evidence(
    monkeypatch,
    tmp_path,
):
    mol = _Molecule("failed")
    _install_reader(monkeypatch, [mol])
    setup_report = ForceFieldSetupReport(
        requested_forcefield="GAFF",
        effective_forcefield="GAFF",
        stage="setup",
        setup_succeeded=False,
        workflow_stage="complex_optimization",
    )
    diagnostics = ComplexBuildDiagnostics(1, 1, (), 0.25)

    def fail(*args, **kwargs):
        raise ForceFieldSetupError(
            "GAFF setup failed",
            setup_report,
            diagnostics,
        )

    monkeypatch.setattr(cli, "build_and_optimize", fail)
    report_path = tmp_path / "failure.json"

    assert cli.main(["CN", "--report", str(report_path)]) == 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    error = report["results"][0]["error"]
    assert error["type"] == "ForceFieldSetupError"
    assert error["evidence"]["code"] == "FF-SETUP-UNAVAILABLE"
    assert error["evidence"]["setup_report"] == {
        "effective_forcefield": "GAFF",
        "requested_forcefield": "GAFF",
        "setup_succeeded": False,
        "stage": "setup",
        "workflow_stage": "complex_optimization",
    }
    assert error["evidence"]["build_diagnostics"]["attempt_count"] == 1
    assert error["evidence"]["build_diagnostics"]["elapsed_seconds"] == 0.25


def test_nonhard_quality_failure_returns_payload_and_exit_zero(monkeypatch, capsys):
    mol = _Molecule("inspectable")
    _install_reader(monkeypatch, [mol])
    monkeypatch.setattr(
        cli,
        "build_and_optimize",
        lambda *args, **kwargs: _report(passed=False),
    )

    assert cli.main(["CN"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "mol2:inspectable\n"
    assert "quality profile 'standard' failed" in captured.err
    assert "inspectable structure was emitted" in captured.err


def test_tiny_real_smiles_forcefield_smoke(capsys):
    assert (
        cli.main(
            [
                "CC",
                "--epochs",
                "1",
                "--steps-per-epoch",
                "5",
                "--quality",
                "off",
                "--seed",
                "1",
                "--timeout",
                "30",
                "--output-format",
                "mol2",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("@<TRIPOS>MOLECULE\n")
    assert "@<TRIPOS>ATOM\n" in captured.out
    assert "@<TRIPOS>BOND\n" in captured.out
    assert captured.err == ""


def test_real_parallel_smiles_preserve_output_order(tmp_path, capsys):
    assert (
        cli.main(
            [
                "CC",
                "CO",
                "--jobs",
                "2",
                "--epochs",
                "1",
                "--steps-per-epoch",
                "2",
                "--quality",
                "off",
                "--seed",
                "1",
                "--output-format",
                "sdf",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    output_path = tmp_path / "parallel.sdf"
    output_path.write_text(captured.out, encoding="utf-8")
    molecules = tuple(cli.MolReader(output_path, fmt="sdf"))

    assert tuple(len(mol.atoms) for mol in molecules) == (8, 6)
    assert captured.err == ""


def test_doc_is_available_without_molecular_input(capsys):
    with pytest.raises(SystemExit) as exit_info:
        hotpot_main.main(["ff", "--doc"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert not output.startswith("# ")
    assert "hotpot ff" in output
    assert "Standard input" in output
    assert "coordination bonds" in output.lower()


def test_help_is_available_without_molecular_input(capsys):
    with pytest.raises(SystemExit) as exit_info:
        hotpot_main.main(["ff", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: hotpot ff" in output
    assert "--optimize-only" in output


@pytest.mark.parametrize("error_type", (ValueError, ForceFieldError, RuntimeError))
def test_unknown_execution_errors_propagate_same_object_without_next_sample(monkeypatch, error_type):
    molecules = [_Molecule("first"), _Molecule("middle"), _Molecule("last")]
    _install_reader(monkeypatch, molecules)
    error = error_type("backend did not produce usable coordinates")
    calls = []

    def run(molecule, *args, **kwargs):
        calls.append(molecule.name)
        if molecule.name == "middle":
            raise error
        return _report()

    monkeypatch.setattr(cli, "build_and_optimize", run)
    with pytest.raises(error_type) as caught:
        cli.main(["records.smi"])
    assert caught.value is error
    assert calls == ["first", "middle"]


def test_known_middle_setup_rejection_keeps_order_reports_and_trailing_output(monkeypatch, tmp_path, capsys):
    molecules = [_Molecule("first"), _Molecule("middle"), _Molecule("last")]
    _install_reader(monkeypatch, molecules)
    calls = []

    def run(molecule, *args, **kwargs):
        calls.append(molecule.name)
        if molecule.name == "middle":
            raise ForceFieldSetupError("backend did not produce usable coordinates")
        return _report()

    monkeypatch.setattr(cli, "build_and_optimize", run)
    report_path = tmp_path / "mixed.json"
    assert cli.main(["records.smi", "--report", str(report_path)]) == 1
    assert calls == ["first", "middle", "last"]
    assert capsys.readouterr().out == "mol2:first\nmol2:last\n"
    document = json.loads(report_path.read_text())
    assert document["schema_version"] == 1
    rows = document["results"]
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert [row["status"] for row in rows] == ["ok", "error", "ok"]
    assert rows[1]["error"]["evidence"]["code"] == "FF-SETUP-UNAVAILABLE"
    assert molecules[1].write_calls == []


def test_only_coded_route_request_is_input_error_with_exact_code(monkeypatch, tmp_path, capsys):
    _install_reader(monkeypatch, [_Molecule("flat")])
    report_path = tmp_path / "request.json"
    assert cli.main(["CN", "--optimize-only", "--report", str(report_path)]) == 1
    assert capsys.readouterr().out == ""
    error = json.loads(report_path.read_text())["results"][0]["error"]
    assert error["type"] == "InputError"
    assert error["evidence"]["code"] == "FF-REQUEST-COORDINATES"


@pytest.mark.parametrize("name", ("finite_final_energy", "finite_coordinates", "topology_atom_identity"))
def test_required_result_failure_preserves_report_code_and_emits_no_molecule(monkeypatch, tmp_path, capsys, name):
    mol = _Molecule("unavailable")
    _install_reader(monkeypatch, [mol])
    quality = ForceFieldValidationReport("strict", False, (
        AcceptanceCheck(name, False, measured=None, threshold="required", message="required result unavailable"),
    ))
    monkeypatch.setattr(cli, "build_and_optimize", lambda *args, **kwargs: _Report(None, "UFF", quality))
    report_path = tmp_path / "hard.json"
    assert cli.main(["CN", "--report", str(report_path)]) == 1
    assert capsys.readouterr().out == ""
    assert mol.write_calls == []
    row = json.loads(report_path.read_text())["results"][0]
    assert row["status"] == "error"
    assert row["error"]["evidence"]["code"] == f"FF:{name}"
    assert row["forcefield_report"]["quality_report"]["checks"][0]["name"] == name


def test_finite_findings_preserve_failed_checks_in_report_while_exit_zero(monkeypatch, tmp_path, capsys):
    mol = _Molecule("observation")
    _install_reader(monkeypatch, [mol])
    monkeypatch.setattr(cli, "build_and_optimize", lambda *args, **kwargs: _report(passed=False))
    report_path = tmp_path / "observation.json"
    assert cli.main(["CN", "--report", str(report_path)]) == 0
    assert capsys.readouterr().out == "mol2:observation\n"
    row = json.loads(report_path.read_text())["results"][0]
    assert row["status"] == "quality-failed"
    assert row["forcefield_report"]["quality_report"]["passed"] is False
    assert row["forcefield_report"]["quality_report"]["checks"][1]["name"] == "atom_too_close"
