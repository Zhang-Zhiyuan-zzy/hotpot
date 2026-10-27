"""CLI contracts for the optional parameter-optimization workflow."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hotpot import __main__ as entry


def _arguments(input_path: Path, output_path: Path) -> list[str]:
    return [
        "optimize",
        str(input_path),
        str(output_path),
        "--mesh",
        "20",
        "-s",
        "Conc.:0:1.25,Current:5:20,FlowRate:5:30",
    ]


def test_optimize_command_forwards_dynamic_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "inputs.xlsx"
    output_path = tmp_path / "results"
    calls = []

    def optimize(excel_file, result_dir, arguments):
        calls.append((excel_file, result_dir, arguments.mesh, arguments.background))

    real_import = entry.importlib.import_module

    def import_module(name: str):
        if name == "hotpot.main.optimize":
            return SimpleNamespace(optimize=optimize)
        return real_import(name)

    monkeypatch.setattr(entry.importlib, "import_module", import_module)
    arguments = SimpleNamespace(
        show_version=False,
        works="optimize",
        excel_file=str(input_path),
        result_dir=str(output_path),
        mesh=20,
        background=["Conc.:0:1.25,Current:5:20,FlowRate:5:30"],
    )

    assert entry.run(arguments) == 0
    assert calls == [
        (str(input_path), str(output_path), 20, ["Conc.:0:1.25,Current:5:20,FlowRate:5:30"])
    ]


@pytest.mark.optional_dependency
@pytest.mark.high_compute
def test_optimize_example_writes_results_only_to_requested_directory(
    tmp_path: Path,
) -> None:
    pytest.importorskip("sklearn")
    source = Path(__file__).with_name("inputs") / "inputs.xlsx"
    output_path = tmp_path / "optimize"

    assert entry.main(_arguments(source, output_path)) == 0
    result = output_path / "next_params.csv"
    assert result.is_file()
    assert result.stat().st_size > 0
