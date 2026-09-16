from pathlib import Path

import numpy as np
import pytest

from hotpot import read_mol
from hotpot import __main__ as hotpot_main
from hotpot.cheminfo.AImodels.mca import cli
from hotpot.cheminfo.AImodels.mca.result_types import (
    AtomPrediction,
    MoleculePrediction,
    SitePrediction,
)


def _prediction(smiles="CN"):
    return MoleculePrediction(
        smiles=smiles,
        formal_charge=0,
        sites=(SitePrediction(1, "N", "Amine", 314.45),),
        model_variant="fp16",
        atom_predictions=(
            AtomPrediction(0, "C", 102.125),
            AtomPrediction(1, "N", 314.45),
        ),
    )


class _Predictor:
    def __init__(self, **kwargs):
        self.options = kwargs

    def predict(self, molecules):
        return [_prediction(molecule.smiles) for molecule in molecules]


def test_single_prediction_table_contract():
    assert cli.format_predictions([_prediction()]) == (
        "No.  Atom  MCA(kJ/mol)  is_Nuc_site\n"
        "1    C     102.12       False\n"
        "2    N     314.45       True"
    )


def test_top_level_mca_prints_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MCAPredictor", _Predictor)

    assert hotpot_main.main(["mca", "CN", "--device", "cpu"]) == 0

    output = capsys.readouterr().out
    assert "MCA(kJ/mol)" in output
    assert "2    N     314.45       True" in output
    assert "Done !!!" not in output


def test_top_level_mca_doc_prints_packaged_markdown_without_input(capsys):
    with pytest.raises(SystemExit) as exit_info:
        hotpot_main.main(["mca", "--doc"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert not output.startswith("# ")
    assert "hotpot mca" in output
    assert "Command synopsis" in output
    assert "$ hotpot mca inputs/*.mol2 -o results.txt" in output
    assert "Charged molecules and applicability" in output


def test_output_option_writes_table(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "MCAPredictor", _Predictor)
    output_path = tmp_path / "result.txt"

    assert hotpot_main.main(["mca", "CN", "-o", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    assert output_path.read_text(encoding="utf-8") == (
        f"{cli.format_predictions([_prediction()])}\n"
    )


def test_multimolecule_file_is_read_and_reported(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "MCAPredictor", _Predictor)
    input_path = tmp_path / "molecules.smi"
    input_path.write_text("CN first\nCN second\n", encoding="utf-8")

    assert hotpot_main.main(["mca", str(input_path)]) == 0

    output = capsys.readouterr().out
    assert "Molecule 1:" in output
    assert "Molecule 2:" in output
    assert output.count("MCA(kJ/mol)") == 2


def test_plot_data_marks_only_detected_sites_by_default():
    molecule = read_mol("CN")

    rd_molecules, values, highlighted = cli._plot_data(
        [molecule], [_prediction()], all_sites=False
    )

    assert np.isnan(values[0][0])
    assert values[0][1] == pytest.approx(314.45)
    assert highlighted == [314.45]
    assert not rd_molecules[0].GetAtomWithIdx(0).HasProp("atomNote")
    assert rd_molecules[0].GetAtomWithIdx(1).GetProp("atomNote") == "314.45"


def test_all_site_plot_data_marks_every_atom():
    molecule = read_mol("CN")

    rd_molecules, values, highlighted = cli._plot_data(
        [molecule], [_prediction()], all_sites=True
    )

    assert values[0].tolist() == pytest.approx([102.125, 314.45])
    assert highlighted == pytest.approx([102.125, 314.45])
    assert rd_molecules[0].GetAtomWithIdx(0).GetProp("atomNote") == "102.12"
    assert rd_molecules[0].GetAtomWithIdx(1).GetProp("atomNote") == "314.45"


def test_plot_option_delegates_to_plot_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "MCAPredictor", _Predictor)
    observed = []

    def record_plot(molecules, predictions, path, all_sites):
        observed.append((len(molecules), len(predictions), Path(path), all_sites))

    monkeypatch.setattr(cli, "save_plot", record_plot)
    plot_path = tmp_path / "mca.png"

    assert hotpot_main.main(
        ["mca", "CN", "--plot", str(plot_path), "--all-site"]
    ) == 0

    assert observed == [(1, 1, plot_path, True)]


def test_plot_backend_writes_png(tmp_path):
    plot_path = tmp_path / "mca.png"

    cli.save_plot([read_mol("CN")], [_prediction()], str(plot_path), False)

    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_range_expands_a_single_value():
    assert cli._plot_range([]) == (None, None)
    assert cli._plot_range([100.0]) == (95.0, 105.0)
