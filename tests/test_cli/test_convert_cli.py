"""Actual canonical-SMILES conversion through the public command dispatcher."""

from argparse import ArgumentError
import json
from pathlib import Path
import subprocess
import sys

import pytest

from hotpot import __main__ as entry


def test_convert_reads_every_smiles_record_and_overwrites_output(tmp_path):
    source = tmp_path / "molecules.smi"
    source.write_text("OCC ethanol\nCN methylamine\n[NH4+] ammonium\n")
    output = tmp_path / "result.smi"
    output.write_text("stale\n")

    assert entry.main(["convert", str(source), "-f", str(output), "-o", "smi"]) == 0

    assert output.read_text().splitlines() == ["CCO", "CN", "[NH4+]"]


def test_convert_infers_input_and_default_smiles_output_from_sdf(tmp_path):
    source = tmp_path / "water.sdf"
    source.write_text(
        "water\n  Hotpot fixture\n\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    output = tmp_path / "water.smi"

    assert entry.main(["convert", str(source), "--output_file", str(output)]) == 0

    assert output.read_text() == "O\n"


@pytest.mark.parametrize("directory_output", (False, True))
def test_convert_directory_uses_sorted_files_and_preserves_record_order(tmp_path, directory_output):
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "z.smi").write_text("CN\n[NH4+]\n")
    (source / "a.smi").write_text("OCC\nO\n")
    (source / "ignored.txt").write_text("bad molecule\n")
    output = tmp_path / ("results" if directory_output else "results.smi")
    if directory_output:
        output.mkdir()
    else:
        output.write_text("stale\n")

    assert entry.main([
        "convert", str(source), "-f", str(output), "-i", "smi", "-o", "smi"
    ]) == 0

    if directory_output:
        assert sorted(path.name for path in output.iterdir()) == ["0.smi", "1.smi"]
        assert (output / "0.smi").read_text().splitlines() == ["CCO", "O"]
        assert (output / "1.smi").read_text().splitlines() == ["CN", "[NH4+]"]
    else:
        assert output.read_text().splitlines() == ["CCO", "O", "CN", "[NH4+]"]


def test_convert_rejects_unsupported_output_without_writing_or_success_message(tmp_path, capsys):
    source = tmp_path / "input.smi"
    source.write_text("CN\n")
    output = tmp_path / "output.sdf"
    output.write_text("keep this file\n")

    with pytest.raises(NotImplementedError, match="supports only 'smi'"):
        entry.main(["convert", str(source), "-f", str(output), "-o", "sdf"])

    assert output.read_text() == "keep this file\n"
    assert "Done !!!" not in capsys.readouterr().out


def test_convert_requires_output_format_for_smiles_input(tmp_path):
    source = tmp_path / "input.smi"
    source.write_text("CN\n")

    with pytest.raises(ArgumentError, match="output format is not specified"):
        entry.main(["convert", str(source), "-f", str(tmp_path / "result")])


def test_convert_requires_explicit_output_path(tmp_path):
    source = tmp_path / "input.smi"
    source.write_text("CN\n")

    with pytest.raises(ArgumentError, match="output file or directory is not specified"):
        entry.main(["convert", str(source), "-o", "smi"])


@pytest.mark.parametrize("arguments", (["--help"], ["--version"], ["convert", "--help"]))
def test_cold_help_keeps_toolkits_and_optional_training_dependencies_unloaded(arguments):
    script = r"""
import json
import sys
from hotpot import __main__ as entry

try:
    status = entry.main(json.loads(sys.argv[1]))
except SystemExit as error:
    status = error.code
blocked = {"rdkit", "openbabel", "torch", "onnxruntime", "sklearn", "xgboost", "lightgbm"}
loaded = sorted(name for name in sys.modules if name.split(".")[0] in blocked)
print(json.dumps({"status": status, "loaded": loaded, "origin": entry.__file__}))
"""
    result = subprocess.run(
        [sys.executable, "-s", "-B", "-c", script, json.dumps(arguments)],
        check=True, capture_output=True, text=True,
    )
    observation = json.loads(result.stdout.splitlines()[-1])

    assert observation["status"] == 0
    assert observation["loaded"] == []
    assert Path(observation["origin"]).resolve() == Path(entry.__file__).resolve()
