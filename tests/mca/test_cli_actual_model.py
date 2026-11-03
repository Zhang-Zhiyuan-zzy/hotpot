"""Opt-in MCA CLI batch isolation checks with the installed model artifact."""

import math
import re
from pathlib import Path

import pytest

from hotpot.commands import mca as cli


pytestmark = pytest.mark.model_artifact


def test_actual_model_cli_keeps_charged_rejection_local(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "molecules.smi"
    input_text = "CN first\nC[NH3+] charged\nCO third\n"
    input_path.write_text(input_text, encoding="utf-8")
    output_path = tmp_path / "results.txt"

    returncode = cli.main(
        [
            str(input_path),
            "--backend", "cpu",
            "--variant", "fp16",
            "--cpu-count", "1",
            "--model-source", "local",
            "--output", str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert returncode == 1
    assert captured.out == ""
    assert "charg" in captured.err.lower()
    assert re.search(r"\b(?:molecule|record|input)\s*#?\s*2\b", captured.err, re.I)
    output = output_path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^Molecule (\d+):[^\n]*\n", output, re.M))
    assert [int(heading.group(1)) for heading in headings] == [1, 3]
    assert output[: headings[0].start()].strip() == ""
    assert output.count("MCA(kJ/mol)") == 2
    for position, expected_elements in enumerate((("C", "N"), ("C", "O"))):
        start = headings[position].end()
        end = headings[position + 1].start() if position == 0 else len(output)
        lines = output[start:end].strip().splitlines()
        assert lines[0].split() == ["No.", "Atom", "MCA(kJ/mol)", "is_Nuc_site"]
        rows = [line.split() for line in lines[1:]]
        assert len(rows) == 2
        assert all(len(row) == 4 for row in rows)
        assert [row[0] for row in rows] == ["1", "2"]
        assert tuple(row[1] for row in rows) == expected_elements
        assert all(math.isfinite(float(row[2])) for row in rows)
        assert all(row[3] in {"True", "False"} for row in rows)
    assert input_path.read_text(encoding="utf-8") == input_text
