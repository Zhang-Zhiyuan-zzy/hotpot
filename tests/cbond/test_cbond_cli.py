from types import SimpleNamespace

import pytest

from hotpot import __main__ as hotpot_main
from hotpot import read_mol
from hotpot.cheminfo.AImodels.cbond import cli
from hotpot.cheminfo.AImodels.cbond.constants import DEFAULT_CBOND_THRESHOLD


def _step(atom_index, element, score):
    return SimpleNamespace(
        atom_index=atom_index,
        element=element,
        score=score,
    )


def _result(smiles, probability=1.0, steps=()):
    return SimpleNamespace(
        molecule=SimpleNamespace(smiles=smiles),
        probability=probability,
        steps=tuple(steps),
    )


def _install_backend(monkeypatch):
    observed = {}
    runtime = object()
    observed["runtime"] = runtime

    def get_runtime(device, model_dir):
        observed["runtime_options"] = (device, model_dir)
        return runtime

    def build_single(
        molecule,
        metal,
        threshold,
        greedy,
        runtime,
        *,
        return_details,
    ):
        observed["molecule"] = molecule
        observed["single_options"] = (
            metal,
            threshold,
            greedy,
            runtime,
            return_details,
        )
        return _result(
            "CN[Eu]",
            steps=(_step(4, "O", 0.88682), _step(11, "N", 0.23455)),
        )

    def build_all(
        molecule,
        metal,
        threshold,
        greedy,
        runtime,
        *,
        max_states,
        return_details,
    ):
        observed["molecule"] = molecule
        observed["all_options"] = (
            metal,
            threshold,
            greedy,
            runtime,
            max_states,
            return_details,
        )
        return [
            _result("CN[Eu]", 0.749, (_step(4, "O", 0.88682),)),
            _result("C[NH2+][Eu]", 0.251, (_step(11, "N", 0.23455),)),
        ]

    monkeypatch.setattr(cli, "get_cbond_runtime", get_runtime)
    monkeypatch.setattr(cli, "auto_build_cbond", build_single)
    monkeypatch.setattr(cli, "build_all_possible_cbond", build_all)
    return observed


def test_top_level_cbond_prints_result_smiles(monkeypatch, capsys):
    observed = _install_backend(monkeypatch)

    assert hotpot_main.main(["cbond", "Eu", "CN", "--device", "cpu"]) == 0

    assert capsys.readouterr().out == "CN[Eu]\n"
    assert [atom.symbol for atom in observed["molecule"].atoms] == ["C", "N"]
    assert observed["runtime_options"] == ("cpu", None)
    assert observed["single_options"] == (
        "Eu",
        DEFAULT_CBOND_THRESHOLD,
        True,
        observed["runtime"],
        True,
    )


def test_parser_uses_shared_default_threshold():
    args = cli.build_parser().parse_args(["Eu", "CN"])

    assert args.threshold == DEFAULT_CBOND_THRESHOLD


def test_cbond_reads_mol2_and_writes_output(monkeypatch, tmp_path, capsys):
    observed = _install_backend(monkeypatch)
    ligand_path = tmp_path / "ligand.mol2"
    ligand_path.write_text(read_mol("CN").write(fmt="mol2"), encoding="utf-8")
    output_path = tmp_path / "result.smi"
    model_dir = tmp_path / "models"

    assert (
        hotpot_main.main(
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
        )
        == 0
    )

    assert capsys.readouterr().out == ""
    assert output_path.read_text(encoding="utf-8") == "CN[Eu]\n"
    assert [atom.symbol for atom in observed["molecule"].atoms] == ["C", "N"]
    assert observed["runtime_options"] == ("cuda", str(model_dir))
    assert observed["single_options"][:3] == (63, 1.25, False)


def test_bond_detail_omits_rank_and_probability(monkeypatch, capsys):
    _install_backend(monkeypatch)

    assert hotpot_main.main(["cbond", "Eu", "CN", "--bond-detail"]) == 0

    assert capsys.readouterr().out == (
        "CN[Eu]\n"
        "Cbond Detail:\n"
        "AtomIdx  Atom  Score\n"
        "4        O     0.88682\n"
        "11       N     0.23455\n"
        "-- End --\n"
    )


def test_all_structures_prints_ranked_probabilities(monkeypatch, capsys):
    observed = _install_backend(monkeypatch)

    assert (
        hotpot_main.main(
            ["cbond", "Eu", "CN", "--all-structures", "--max-states", "17"]
        )
        == 0
    )

    assert capsys.readouterr().out == (
        "CN[Eu]  --> Rank 1: Prob: 74.9%\n"
        "-----\n"
        "C[NH2+][Eu]  --> Rank 2: Prob: 25.1%\n"
        "-- End --\n"
    )
    assert observed["all_options"] == (
        "Eu",
        DEFAULT_CBOND_THRESHOLD,
        True,
        observed["runtime"],
        17,
        True,
    )


def test_all_structures_with_bond_details(monkeypatch, capsys):
    _install_backend(monkeypatch)

    assert (
        hotpot_main.main(["cbond", "Eu", "CN", "--all-structures", "--bond-detail"])
        == 0
    )

    assert capsys.readouterr().out == (
        "CN[Eu]  --> Rank 1: Prob: 74.9%\n"
        "Cbond Detail:\n"
        "AtomIdx  Atom  Score\n"
        "4        O     0.88682\n"
        "-----\n"
        "C[NH2+][Eu]  --> Rank 2: Prob: 25.1%\n"
        "Cbond Detail:\n"
        "AtomIdx  Atom  Score\n"
        "11       N     0.23455\n"
        "-- End --\n"
    )


def test_all_structures_writes_the_same_report_to_file(
    monkeypatch,
    tmp_path,
    capsys,
):
    _install_backend(monkeypatch)
    output_path = tmp_path / "ranked.txt"

    assert (
        hotpot_main.main(
            ["cbond", "Eu", "CN", "--all-structures", "-o", str(output_path)]
        )
        == 0
    )

    assert capsys.readouterr().out == ""
    assert output_path.read_text(encoding="utf-8") == (
        "CN[Eu]  --> Rank 1: Prob: 74.9%\n"
        "-----\n"
        "C[NH2+][Eu]  --> Rank 2: Prob: 25.1%\n"
        "-- End --\n"
    )


def test_empty_all_structures_has_explicit_output():
    assert cli.format_all_results([]) == (
        "No coordination structures exceeded the threshold.\n-- End --"
    )


def test_bond_detail_supports_an_empty_path():
    assert cli.format_single_result(_result("CN.[Eu]"), True) == (
        "CN.[Eu]\nCbond Detail:\nAtomIdx  Atom  Score\n-- End --"
    )


def test_top_level_cbond_doc_prints_packaged_markdown_without_input(capsys):
    with pytest.raises(SystemExit) as exit_info:
        hotpot_main.main(["cbond", "--doc"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert not output.startswith("# ")
    assert "hotpot cbond" in output
    assert "Enumerate candidate coordination structures" in output
    assert "$ hotpot cbond Eu ligand.mol2 --all-structures" in output
    assert "relative ranking probability" in output
