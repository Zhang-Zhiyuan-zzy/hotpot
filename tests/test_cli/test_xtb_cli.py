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
from hotpot.cheminfo._io import MolReader
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
    assert args.charge_model == "valence"
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
            "--charge-model",
            "preserve",
            "--post-check",
            "strict",
            "--jobs",
            "2",
        ]
    )
    assert explicit.method == "gfnff"
    assert explicit.task == "singlepoint"
    assert explicit.charge_model == "preserve"
    assert explicit.post_check == "strict"
    assert explicit.jobs == 2


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--method", "unknown"),
        ("--task", "unknown"),
        ("--charge-model", "unknown"),
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
