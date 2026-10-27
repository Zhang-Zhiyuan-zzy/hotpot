"""Smoke tests executed against an installed wheel outside the checkout."""

from importlib.resources import files
import subprocess
import sys


def _stdout(*arguments: str, input_text: str | None = None) -> str:
    return subprocess.run(
        arguments,
        input=input_text,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def core() -> None:
    import hotpot
    from hotpot.cheminfo.calculator import (
        ScientificOperationRequirements,
        StateRequest,
        extract_molecular_facts,
        resolve_scientific_state,
    )
    from hotpot.cheminfo.calculator.contracts import ScientificQuantity
    from hotpot.cheminfo.calculator.state_selection import StateDecisionAction
    from hotpot.cheminfo.graph import relevant_cycles
    from hotpot.cheminfo.search import _search_native

    assert _search_native.NativeProblem is not None
    assert relevant_cycles([(4, 7), (7, 9), (9, 4)]) == ((4, 7, 9),)
    search_hits = hotpot.read_mol("CCO", "smi").search_substructure("[C]-[O]")
    assert tuple(hit.atom_indices for hit in search_hits) == (frozenset({1, 2}),)

    facts = extract_molecular_facts(hotpot.read_mol("C", "smi"))
    state = resolve_scientific_state(
        facts,
        StateRequest(net_charge=0, two_s=0),
        requirements=ScientificOperationRequirements(
            "installed-wheel-smoke",
            frozenset(
                {
                    ScientificQuantity.NET_CHARGE,
                    ScientificQuantity.ELECTRONIC_STATE,
                }
            ),
            frozenset(),
        ),
    )
    assert state.execution_decision.action is StateDecisionAction.SELECT_STATE

    forcefield_package = files("hotpot.cheminfo.forcefields")
    assert forcefield_package.joinpath("ff.py").is_file()
    for obsolete_module in ("ff39.py", "utils.py", "utils39.py"):
        assert not forcefield_package.joinpath(obsolete_module).is_file()

    xtb_package = files("hotpot.plugins.xtb")
    assert xtb_package.joinpath("cli_doc.md").is_file()
    assert xtb_package.joinpath("README.md").read_text(
        encoding="utf-8"
    ).startswith("# Hotpot xTB plugin")
    for obsolete_module in ("core.py", "xtb_doc.md", ".cache.json"):
        assert not xtb_package.joinpath(obsolete_module).is_file()

    assert files("hotpot.pipeline").joinpath("cli_doc.md").read_text(
        encoding="utf-8"
    ).startswith("# `hotpot run`")
    assert files("hotpot.pipeline").joinpath("README.md").read_text(
        encoding="utf-8"
    ).startswith("# Hotpot controlled molecular pipeline")
    assert files("hotpot.cheminfo.geometry").joinpath("README.md").read_text(
        encoding="utf-8"
    ).startswith("# `hotpot.cheminfo.geometry` API Reference")
    assert files("hotpot.cheminfo.geometry").joinpath("README.zh.md").read_text(
        encoding="utf-8"
    ).startswith("# `hotpot.cheminfo.geometry` API 参考（中文版）")

    mca_documentation = _stdout("hotpot", "mca", "--doc")
    assert "hotpot mca" in mca_documentation
    assert "Command synopsis" in mca_documentation
    xtb_documentation = _stdout("hotpot", "xtb", "--doc")
    assert "hotpot xtb" in xtb_documentation
    assert "Independent methods" in xtb_documentation
    run_documentation = _stdout("hotpot", "run", "--doc")
    assert "hotpot run" in run_documentation
    assert "Results and failure semantics" in run_documentation


def mca() -> None:
    import hotpot
    from hotpot.cheminfo.AImodels.mca import MCAPredictor

    molecule = hotpot.to_hotpot_mol("C1CCCCN1")
    prediction = MCAPredictor(device="cpu").predict(molecule)
    actual_sites = [
        (site.atom_index, site.mca_kj_mol) for site in prediction.sites
    ]
    if abs(prediction.sites[0].mca_kj_mol - 503.25) >= 1e-6:
        print(
            "::error file=tests/test_packaging/installed_package_smoke.py,"
            f"line=101,title=MCA installed-wheel prediction::{actual_sites}",
            flush=True,
        )
    assert len(prediction.atom_predictions) == 6
    assert abs(prediction.sites[0].mca_kj_mol - 503.25) < 1e-6


def cbond() -> None:
    cbond_result = _stdout(
        "hotpot", "cbond", "Eu", "CN", "--device", "cpu"
    ).strip()
    assert "[Eu" in cbond_result
    assert len(cbond_result.splitlines()) == 1

    cbond_documentation = _stdout("hotpot", "cbond", "--doc")
    assert "hotpot cbond" in cbond_documentation
    assert "Enumerate terminal coordination structures" in cbond_documentation
    assert "default mode must not be interpreted as an alias for Rank 1" in (
        cbond_documentation
    )

    cbond_ranked = _stdout(
        "hotpot",
        "cbond",
        "Eu",
        "CN",
        "--device",
        "cpu",
        "--all-structures",
        "--bond-detail",
    )
    assert "Rank 1: Prob:" in cbond_ranked
    assert "Cbond Detail:" in cbond_ranked
    assert cbond_ranked.rstrip().endswith("-- End --")


def inference() -> None:
    mca()
    cbond()


def forcefield() -> None:
    cbond_result = _stdout(
        "hotpot", "cbond", "Eu", "CN", "--device", "cpu"
    ).strip()
    pipeline_result = _stdout(
        "hotpot",
        "ff",
        "-",
        "--input-format",
        "smi",
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
        input_text=f"{cbond_result}\n",
    )
    assert pipeline_result.startswith("@<TRIPOS>MOLECULE\n")
    assert "@<TRIPOS>ATOM\n" in pipeline_result
    assert "@<TRIPOS>BOND\n" in pipeline_result

    forcefield_documentation = _stdout("hotpot", "ff", "--doc")
    assert "hotpot ff" in forcefield_documentation
    assert "Standard input and pipelines" in forcefield_documentation

    forcefield_result = _stdout(
        "hotpot",
        "ff",
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
    )
    assert forcefield_result.startswith("@<TRIPOS>MOLECULE\n")
    assert "@<TRIPOS>ATOM\n" in forcefield_result
    assert "@<TRIPOS>BOND\n" in forcefield_result


if __name__ == "__main__":
    {
        "core": core,
        "mca": mca,
        "cbond": cbond,
        "inference": inference,
        "forcefield": forcefield,
    }[sys.argv[1]]()
