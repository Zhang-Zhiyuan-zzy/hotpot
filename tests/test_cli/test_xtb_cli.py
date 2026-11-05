"""Black-box contracts for the standalone ``hotpot xtb`` command."""

from __future__ import annotations

import io
import json
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

import hotpot
from hotpot import __main__ as hotpot_main
from hotpot.cheminfo.operations.io import MolReader
from hotpot.plugins.xtb import cli


_FAKE_XTB_SOURCE = (
    Path(__file__).parents[1]
    / "test_plugin"
    / "test_xtb"
    / "fixtures"
    / "fake_xtb.py"
)


def _molecule(smiles: str, coordinates) -> hotpot.Molecule:
    mol = hotpot.read_mol(smiles, "smi")
    mol.coordinates = np.asarray(coordinates, dtype=float)
    return mol


def _water() -> hotpot.Molecule:
    return _molecule(
        "[H]O[H]",
        ((-0.7, 0.0, 0.0), (0.0, 0.0, 0.0), (0.7, 0.0, 0.0)),
    )


def _methane() -> hotpot.Molecule:
    return _molecule(
        "[H]C([H])([H])[H]",
        (
            (0.63, 0.63, 0.63),
            (0.0, 0.0, 0.0),
            (0.63, -0.63, -0.63),
            (-0.63, 0.63, -0.63),
            (-0.63, -0.63, 0.63),
        ),
    )


def _sdf_payload(*molecules: hotpot.Molecule) -> str:
    return "".join(
        molecule.write(fmt="sdf", write_single=True) for molecule in molecules
    )


def _write_sdf(path: Path, *molecules: hotpot.Molecule) -> Path:
    path.write_text(_sdf_payload(*molecules), encoding="utf-8")
    return path


def _fake_xtb(tmp_path: Path, scenario: str = "success") -> Path:
    executable_directory = tmp_path / f"fake-xtb-{scenario}"
    executable_directory.mkdir()
    executable = executable_directory / "xtb"
    shutil.copyfile(_FAKE_XTB_SOURCE, executable)
    executable.chmod(
        executable.stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )
    (executable_directory / "scenario.json").write_text(
        json.dumps({"scenario": scenario}),
        encoding="utf-8",
    )
    return executable


def _common_options(executable: Path) -> list[str]:
    return [
        "--xtb-executable",
        str(executable),
        "--charge",
        "0",
        "--unpaired-electrons",
        "0",
        "--post-check",
        "off",
    ]


def test_parser_defaults_and_documented_choices() -> None:
    args = cli.build_parser().parse_args(["input.sdf"])

    assert args.method == "gfn2"
    assert args.task == "optimize"
    assert args.charge is None
    assert args.unpaired_electrons is None
    assert args.state_policy is None
    assert not hasattr(args, "charge_model")
    assert args.input_format is None
    assert args.output_format == "sdf"
    assert args.output is None
    assert args.report is None
    assert args.native_log is None
    assert args.xtb_executable is None
    assert args.threads is None
    assert args.jobs == 1
    assert args.work_directory is None
    assert args.keep_work_directory is False
    assert args.post_check == "off"

    explicit = cli.build_parser().parse_args(
        [
            "input.sdf",
            "--method",
            "gfnff",
            "--task",
            "singlepoint",
            "--state-policy",
            "coordination-screening-v1",
            "--post-check",
            "strict",
            "--jobs",
            "2",
        ]
    )
    assert explicit.method == "gfnff"
    assert explicit.task == "singlepoint"
    assert explicit.state_policy == "coordination-screening-v1"
    assert explicit.post_check == "strict"
    assert explicit.jobs == 2


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--method", "unknown"),
        ("--task", "unknown"),
        ("--state-policy", "unknown"),
        ("--post-check", "unknown"),
        ("--jobs", "0"),
        ("--threads", "0"),
    ),
)
def test_parser_rejects_invalid_choice_or_nonpositive_count(
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.build_parser().parse_args(["input.sdf", option, value])

    assert exit_info.value.code == 2


def test_help_and_documentation_do_not_require_input(capsys) -> None:
    with pytest.raises(SystemExit) as help_exit:
        hotpot_main.main(["xtb", "--help"])
    assert help_exit.value.code == 0
    help_output = capsys.readouterr().out
    assert "usage: hotpot xtb" in help_output
    assert "--unpaired-electrons" in help_output

    with pytest.raises(SystemExit) as doc_exit:
        hotpot_main.main(["xtb", "--doc"])
    assert doc_exit.value.code == 0
    documentation = capsys.readouterr().out
    assert "hotpot xtb" in documentation
    assert "SDF" in documentation
    assert "standard input" in documentation.lower()


def test_stdin_stdout_is_a_pure_molecular_stream(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    executable = _fake_xtb(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(_sdf_payload(_water())))

    status = cli.main(
        [
            "-",
            "--input-format",
            "sdf",
            "--task",
            "singlepoint",
            *_common_options(executable),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "TOTAL ENERGY" not in captured.out
    assert "GEOMETRY OPTIMIZATION" not in captured.out
    output_path = tmp_path / "stdout.sdf"
    output_path.write_text(captured.out, encoding="utf-8")
    molecules = tuple(MolReader(output_path, fmt="sdf"))
    assert len(molecules) == 1
    assert tuple(atom.symbol for atom in molecules[0].atoms) == ("H", "O", "H")


def test_output_report_and_native_log_are_separate_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    executable = _fake_xtb(tmp_path)
    input_path = _write_sdf(tmp_path / "water.sdf", _water())
    output_path = tmp_path / "optimized.sdf"
    report_path = tmp_path / "report.json"
    native_log_path = tmp_path / "xtb.log"

    status = cli.main(
        [
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--native-log",
            str(native_log_path),
            *_common_options(executable),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.out == ""
    assert len(tuple(MolReader(output_path, fmt="sdf"))) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report
    assert "-5.123456789" in report_path.read_text(encoding="utf-8")
    native_log = native_log_path.read_text(encoding="utf-8")
    assert "TOTAL ENERGY" in native_log
    assert "TOTAL ENERGY" not in output_path.read_text(encoding="utf-8")


def test_multiple_records_keep_input_order_with_parallel_jobs(
    tmp_path: Path,
    capsys,
) -> None:
    executable = _fake_xtb(tmp_path)
    input_path = _write_sdf(tmp_path / "two.sdf", _water(), _methane())

    status = cli.main(
        [
            str(input_path),
            "--jobs",
            "2",
            "--task",
            "singlepoint",
            *_common_options(executable),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    output_path = tmp_path / "parallel.sdf"
    output_path.write_text(captured.out, encoding="utf-8")
    molecules = tuple(MolReader(output_path, fmt="sdf"))
    assert tuple(len(molecule.atoms) for molecule in molecules) == (3, 5)


def test_bare_smiles_is_rejected_instead_of_being_built(
    tmp_path: Path,
    capsys,
) -> None:
    executable = _fake_xtb(tmp_path)
    input_path = tmp_path / "two-dimensional.smi"
    input_path.write_text("CC\n", encoding="utf-8")

    status = cli.main(
        [
            str(input_path),
            "--input-format",
            "smi",
            *_common_options(executable),
        ]
    )

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert "3d" in captured.err.lower() or "coordinate" in captured.err.lower()


def test_backend_failure_is_nonzero_and_preserves_structured_report(
    tmp_path: Path,
    capsys,
) -> None:
    executable = _fake_xtb(tmp_path, "nonzero")
    input_path = _write_sdf(tmp_path / "water.sdf", _water())
    report_path = tmp_path / "failure.json"

    status = cli.main(
        [
            str(input_path),
            "--report",
            str(report_path),
            *_common_options(executable),
        ]
    )

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert "fake execution failure" in captured.err
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report
    assert "17" in report_path.read_text(encoding="utf-8")
    assert report["schema_version"] == 2
    assert report["results"][0]["error"]["code"] == "XTB-NATIVE-EXECUTION"


@pytest.mark.parametrize(
    "arguments",
    (
        ("input.sdf", "--output", "-", "--report", "-"),
        ("input.sdf", "--output", "-", "--native-log", "-"),
        ("input.sdf", "--report", "same", "--native-log", "same"),
    ),
)
def test_diagnostic_artifacts_cannot_share_the_molecular_stream(
    arguments,
    capsys,
) -> None:
    status = cli.main(list(arguments))

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""


def test_gfnff_rejects_an_unpaired_electron_override(
    tmp_path: Path,
    capsys,
) -> None:
    executable = _fake_xtb(tmp_path)
    input_path = _write_sdf(tmp_path / "water.sdf", _water())

    status = cli.main(
        [
            str(input_path),
            "--method",
            "gfnff",
            "--charge",
            "0",
            "--unpaired-electrons",
            "0",
            "--xtb-executable",
            str(executable),
        ]
    )

    captured = capsys.readouterr()
    assert status != 0
    assert captured.out == ""
    assert "unpaired" in captured.err.lower()


def _quality_report(name, *, passed=False, severity="error"):
    from hotpot.cheminfo.operations.forcefields.contracts import AcceptanceCheck, ForceFieldValidationReport

    check = AcceptanceCheck(name, passed, severity=severity, measured=0.1, threshold=0.5,
                            atom_indices=(0, 1), message="observed geometry finding")
    return ForceFieldValidationReport("strict", passed or severity == "warning", (check,))


@pytest.mark.parametrize("error_kind", ("input", "value", "stream", "commit", "unrecognized-code"))
def test_unknown_or_integrity_errors_propagate_same_object_without_next_sample(monkeypatch, tmp_path, error_kind):
    from hotpot.plugins.xtb.contracts import XTBInputError
    from hotpot.plugins.xtb.stream import XTBStreamError

    errors = {
        "input": XTBInputError("input unavailable"),
        "value": ValueError("input unavailable"),
        "stream": XTBStreamError("stream binding invalid"),
        "commit": XTBInputError("validated coordinate commit failed"),
        "unrecognized-code": XTBInputError("input unavailable"),
    }
    error = errors[error_kind]
    if error_kind == "unrecognized-code":
        error.sample_failure_code = "UNKNOWN"
    records = tuple(cli._InputRecord(index, _water(), None) for index in range(3))
    monkeypatch.setattr(cli, "_read_records", lambda *args: records)
    calls = []

    def run(record, options):
        calls.append(record.index)
        raise error

    monkeypatch.setattr(cli, "_run_xtb_record", run)
    with pytest.raises(type(error)) as caught:
        cli.main([str(tmp_path / "input.sdf")])
    assert caught.value is error
    assert calls == [0]


def test_input_stream_integrity_error_propagates(monkeypatch, tmp_path):
    from hotpot.plugins.xtb.stream import XTBStreamError

    input_path = _write_sdf(tmp_path / "water.sdf", _water())
    error = XTBStreamError("molecular provenance binding invalid")

    def fail(content):
        raise error

    monkeypatch.setattr(cli, "read_sdf_records", fail)
    with pytest.raises(XTBStreamError) as caught:
        cli.main([str(input_path)])
    assert caught.value is error


def test_known_middle_input_failure_keeps_trailing_output_and_exact_code(monkeypatch, tmp_path, capsys):
    from hotpot.plugins.xtb.contracts import XTBInputError

    executable = _fake_xtb(tmp_path)
    input_path = _write_sdf(tmp_path / "mixed.sdf", _water(), _methane(), _water())
    report_path = tmp_path / "mixed.json"
    native_log_path = tmp_path / "native.log"
    original = cli._run_xtb_record
    calls = []

    def run(record, options):
        calls.append(record.index)
        if record.index == 1:
            error = XTBInputError("input unavailable")
            error.sample_failure_code = "XTB-INPUT-GEOMETRY"
            raise error
        return original(record, options)

    monkeypatch.setattr(cli, "_run_xtb_record", run)
    assert cli.main([str(input_path), "--report", str(report_path), "--native-log", str(native_log_path), *_common_options(executable)]) == 1
    captured = capsys.readouterr()
    assert calls == [0, 1, 2]
    assert captured.out.count("$$$$") == 2
    rows = json.loads(report_path.read_text())["results"]
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert [row["status"] for row in rows] == ["ok", "error", "ok"]
    assert rows[1]["error"]["code"] == "XTB-INPUT-GEOMETRY"
    assert "record 2 stdout" in native_log_path.read_text()
    assert "TOTAL ENERGY" in native_log_path.read_text()


@pytest.mark.parametrize("name", ("atom_too_close", "bond_ring_piercing"))
def test_nonhard_post_check_retains_serialized_output_report_and_exit_zero(monkeypatch, tmp_path, capsys, name):
    executable = _fake_xtb(tmp_path)
    input_path = _write_sdf(tmp_path / "water.sdf", _water())
    report_path = tmp_path / "quality.json"
    monkeypatch.setattr(cli, "_post_check", lambda mol, level: _quality_report(name))
    assert cli.main([str(input_path), "--report", str(report_path), *_common_options(executable)]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("$$$$") == 1
    assert "post-check" in captured.err
    document = json.loads(report_path.read_text())
    assert document["schema_version"] == 2
    row = document["results"][0]
    assert row["status"] == "quality-failed"
    assert row["quality_report"]["passed"] is False
    assert row["quality_report"]["checks"][0]["name"] == name
    assert row["quality_report"]["checks"][0]["measured"] == 0.1
    assert "error" not in row


@pytest.mark.parametrize("name", ("coordinate_shape", "finite_coordinates"))
def test_required_post_check_omits_output_and_preserves_report_and_exact_code(monkeypatch, tmp_path, capsys, name):
    executable = _fake_xtb(tmp_path)
    input_path = _write_sdf(tmp_path / "water.sdf", _water())
    report_path = tmp_path / "unavailable.json"
    native_log_path = tmp_path / "unavailable.log"
    monkeypatch.setattr(cli, "_post_check", lambda mol, level: _quality_report(name))
    assert cli.main([str(input_path), "--report", str(report_path), "--native-log", str(native_log_path), *_common_options(executable)]) == 1
    assert capsys.readouterr().out == ""
    row = json.loads(report_path.read_text())["results"][0]
    assert row["status"] == "error"
    assert row["error"]["code"] == f"XTB:{name}"
    assert row["quality_report"]["checks"][0]["name"] == name
    assert row["xtb_report"]["process_succeeded"] is True
    assert "TOTAL ENERGY" in native_log_path.read_text()


def test_output_stream_integrity_error_is_not_a_local_failure(monkeypatch, tmp_path):
    from hotpot.plugins.xtb.stream import XTBStreamError

    executable = _fake_xtb(tmp_path)
    input_path = _write_sdf(tmp_path / "water.sdf", _water())
    error = XTBStreamError("output stream provenance mismatch")

    def fail(*args):
        raise error

    monkeypatch.setattr(cli, "_serialize_record", fail)
    with pytest.raises(XTBStreamError) as caught:
        cli.main([str(input_path), *_common_options(executable)])
    assert caught.value is error


def test_strict_post_check_uses_geometry_acceptance_without_ff_evidence(monkeypatch):
    from hotpot.cheminfo.operations.forcefields import acceptance

    report = _quality_report("finite_coordinates", passed=True)
    calls = []

    def evaluate(mol, *, level):
        calls.append(level)
        return report

    monkeypatch.setattr(acceptance, "_evaluate_geometry_acceptance", evaluate)
    assert cli._post_check(_water(), "strict") is report
    assert calls == ["strict"]


@pytest.mark.parametrize("error_type", (RuntimeError, ValueError))
def test_non_sdf_unknown_reader_error_propagates_exact_object(monkeypatch, tmp_path, error_type):
    input_path = tmp_path / "input.mol2"
    input_path.write_text("reader fixture")
    error = error_type("unknown reader failure")

    def fail(source, fmt):
        assert source == "reader fixture"
        assert fmt == "mol2"
        raise error

    monkeypatch.setattr(cli, "MolReader", fail)
    with pytest.raises(error_type) as caught:
        cli.main([str(input_path)])
    assert caught.value is error


def test_non_sdf_reader_oserror_retains_usage_cause_message_and_exit_two(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "input.mol2"
    input_path.write_text("reader fixture")
    error = OSError("input read unavailable")

    def fail(source, fmt):
        raise error

    monkeypatch.setattr(cli, "MolReader", fail)
    with pytest.raises(cli._CLIUsageError) as caught:
        cli._read_records(str(input_path), None)
    assert caught.value.__cause__ is error
    assert str(caught.value) == "could not read molecular input: input read unavailable"
    assert cli.main([str(input_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "hotpot xtb: could not read molecular input: input read unavailable\n"
