from types import SimpleNamespace

from hotpot import read_mol
from hotpot import __main__ as hotpot_main
from hotpot.cheminfo.AImodels.cbond import cli


def _install_backend(monkeypatch):
    observed = {}
    runtime = object()
    observed["runtime"] = runtime

    def get_runtime(device, model_dir):
        observed["runtime_options"] = (device, model_dir)
        return runtime

    def build(molecule, metal, threshold, greedy, runtime):
        observed["molecule"] = molecule
        observed["build_options"] = (metal, threshold, greedy, runtime)
        return SimpleNamespace(smiles="CN[Eu]"), 0.75

    monkeypatch.setattr(cli, "get_cbond_runtime", get_runtime)
    monkeypatch.setattr(cli, "auto_build_cbond", build)
    return observed


def test_top_level_cbond_prints_result_smiles(monkeypatch, capsys):
    observed = _install_backend(monkeypatch)

    assert hotpot_main.main(["cbond", "Eu", "CN", "--device", "cpu"]) == 0

    assert capsys.readouterr().out == "CN[Eu]\n"
    assert [atom.symbol for atom in observed["molecule"].atoms] == ["C", "N"]
    assert observed["runtime_options"] == ("cpu", None)
    assert observed["build_options"] == ("Eu", 0.0, True, observed["runtime"])


def test_cbond_reads_mol2_and_writes_output(monkeypatch, tmp_path, capsys):
    observed = _install_backend(monkeypatch)
    ligand_path = tmp_path / "ligand.mol2"
    ligand_path.write_text(read_mol("CN").write(fmt="mol2"), encoding="utf-8")
    output_path = tmp_path / "result.smi"
    model_dir = tmp_path / "models"

    assert hotpot_main.main(
        [
            "cbond",
            "63",
            str(ligand_path),
            "--input-format",
            "mol2",
            "--threshold",
            "1.25",
            "--no-greedy",
            "--device",
            "cuda",
            "--model-dir",
            str(model_dir),
            "-o",
            str(output_path),
        ]
    ) == 0

    assert capsys.readouterr().out == ""
    assert output_path.read_text(encoding="utf-8") == "CN[Eu]\n"
    assert [atom.symbol for atom in observed["molecule"].atoms] == ["C", "N"]
    assert observed["runtime_options"] == ("cuda", str(model_dir))
    assert observed["build_options"][:3] == (63, 1.25, False)
