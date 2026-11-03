from pathlib import Path

import numpy as np
import pytest

from hotpot import read_mol
from hotpot import __main__ as hotpot_main
from hotpot.cheminfo.AImodels.mca import cli
from hotpot.cheminfo.AImodels.mca.serialization import MoleculeDTO
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
        self.closed = False

    def iter_predict(self, molecules):
        return (_prediction(molecule.smiles) for molecule in molecules)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True
        return None


def test_single_prediction_table_contract():
    assert cli._format_table(_prediction()) == (
        "No.  Atom  MCA(kJ/mol)  is_Nuc_site\n"
        "1    C     102.12       False\n"
        "2    N     314.45       True"
    )


def test_top_level_mca_defaults_to_cpu_and_prints_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)

    assert hotpot_main.main(["mca", "CN"]) == 0

    output = capsys.readouterr().out
    assert "MCA(kJ/mol)" in output
    assert "2    N     314.45       True" in output
    assert "Done !!!" not in output


def test_top_level_debug_reconfigures_logging_at_debug_level(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)
    configured = []

    def record_logging(debug=False, to_stdout=True):
        configured.append((debug, to_stdout))

    monkeypatch.setattr(hotpot_main, "setup_logging", record_logging)

    assert hotpot_main.main(["--debug", "mca", "CN"]) == 0

    assert configured == [(False, False), (True, False)]
    assert "MCA(kJ/mol)" in capsys.readouterr().out


def test_top_level_mca_doc_prints_packaged_markdown_without_input(capsys):
    with pytest.raises(SystemExit) as exit_info:
        hotpot_main.main(["mca", "--doc"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert not output.startswith("# ")
    assert "hotpot mca" in output
    assert "Command synopsis" in output
    assert "$ hotpot mca inputs/*.mol2 --backend gpu -o results.txt" in output
    assert "Charged molecules and applicability" in output


def test_output_option_writes_table(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)
    output_path = tmp_path / "result.txt"

    assert hotpot_main.main(
        ["mca", "CN", "--backend", "cpu", "-o", str(output_path)]
    ) == 0

    assert capsys.readouterr().out == ""
    assert output_path.read_text(encoding="utf-8") == (
        f"{cli._format_table(_prediction())}\n"
    )


def test_multimolecule_file_is_read_and_reported(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)
    input_path = tmp_path / "molecules.smi"
    input_path.write_text("CN first\nCN second\n", encoding="utf-8")

    assert hotpot_main.main(["mca", str(input_path), "--backend", "cpu"]) == 0

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
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)
    observed = []

    def record_plot(molecules, predictions, path, all_sites):
        observed.append((len(molecules), len(predictions), Path(path), all_sites))

    monkeypatch.setattr(cli, "save_plot", record_plot)
    plot_path = tmp_path / "mca.png"

    assert hotpot_main.main(
        [
            "mca",
            "CN",
            "--backend",
            "cpu",
            "--plot",
            str(plot_path),
            "--all-site",
        ]
    ) == 0

    assert observed == [(1, 1, plot_path, True)]


def test_plot_backend_writes_png(tmp_path):
    plot_path = tmp_path / "mca.png"

    cli.save_plot([read_mol("CN")], [_prediction()], str(plot_path), False)

    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_range_expands_a_single_value():
    assert cli._plot_range([]) == (None, None)
    assert cli._plot_range([100.0]) == (95.0, 105.0)


def test_old_device_option_is_not_accepted():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["CN", "--device", "cpu"])


def test_cpu_backend_rejects_gpu_worker_options():
    args = cli.build_parser().parse_args(
        ["CN", "--backend", "cpu", "--site-cpu-per-gpu", "1"]
    )

    with pytest.raises(ValueError, match="GPU options"):
        cli.run(args)


def test_cpu_count_is_forwarded_only_to_cpu_backend(monkeypatch, capsys):
    predictors = []

    class RecordingPredictor(_Predictor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            predictors.append(self)

    monkeypatch.setattr(cli, "MCACpuPredictor", RecordingPredictor)

    assert hotpot_main.main(["mca", "CN", "--cpu-count", "3"]) == 0

    assert predictors[0].options["cpu_count"] == 3
    assert "MCA(kJ/mol)" in capsys.readouterr().out


def test_gpu_backend_rejects_cpu_count():
    args = cli.build_parser().parse_args(
        ["CN", "--backend", "gpu", "--cpu-count", "3"]
    )

    with pytest.raises(ValueError, match="--cpu-count"):
        cli.run(args)


def test_output_open_failure_closes_constructed_predictor(monkeypatch, tmp_path):
    predictors = []

    class RecordingPredictor(_Predictor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            predictors.append(self)

    monkeypatch.setattr(cli, "MCACpuPredictor", RecordingPredictor)
    args = cli.build_parser().parse_args(
        [
            "CN",
            "--backend",
            "cpu",
            "--output",
            str(tmp_path / "missing" / "result.txt"),
        ]
    )

    with pytest.raises(FileNotFoundError):
        cli.run(args)

    assert len(predictors) == 1
    assert predictors[0].closed


@pytest.mark.parametrize(
    "inputs,accepted_indices",
    [
        (["C[NH3+]", "CN", "CO"], [2, 3]),
        (["CN", "C[NH3+]", "CO"], [1, 3]),
        (["CN", "CO", "C[NH3+]"], [1, 2]),
        (["C[NH3+]", "CN", "C[NH3+]"], [2]),
        (["CN", "C[NH3+]"], [1]),
        (["C[NH3+]", "C[NH3+]"], []),
    ],
)
def test_domain_rejections_keep_original_numbers_and_aggregate_status(
    monkeypatch, capsys, inputs, accepted_indices
):
    calls = []

    class RecordingPredictor(_Predictor):
        def iter_predict(self, molecules):
            calls.append(True)
            for molecule in molecules:
                assert isinstance(molecule, MoleculeDTO)
                yield _prediction(molecule.smiles)

    monkeypatch.setattr(cli, "MCACpuPredictor", RecordingPredictor)
    assert cli.main(inputs) == 1
    output = capsys.readouterr()
    headings = [line for line in output.out.splitlines() if line.startswith("Molecule ")]
    assert [int(line.split()[1].rstrip(":")) for line in headings] == accepted_indices
    assert output.out.count("MCA(kJ/mol)") == len(accepted_indices)
    for index, value in enumerate(inputs, start=1):
        if index in accepted_indices:
            continue
        dto = cli.normalize_molecule(read_mol(value))
        with pytest.raises(ValueError) as error:
            cli.validate_dto(dto, index)
        assert f"MCA.INPUT_REJECTED: molecule {index}: {error.value}\n" in output.err
    assert calls == [True]


def test_rejected_molecules_are_excluded_from_plot_join(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)
    observed = []

    def record_plot(molecules, predictions, path, all_sites):
        observed.append(([mol.smiles for mol in molecules], [p.smiles for p in predictions]))

    monkeypatch.setattr(cli, "save_plot", record_plot)
    assert cli.main(["CN", "C[NH3+]", "CO", "--plot", str(tmp_path / "plot.png")]) == 1
    assert observed == [(["CN", "CO"], ["CN", "CO"])]


def test_all_rejected_writes_no_tables_or_plot(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)
    monkeypatch.setattr(cli, "save_plot", lambda *args: pytest.fail("No successful plots"))
    output = tmp_path / "results.txt"
    assert cli.main(["C[NH3+]", "-o", str(output), "--plot", str(tmp_path / "plot.png")]) == 1
    assert output.read_text() == ""
    assert "MCA.INPUT_REJECTED: molecule 1:" in capsys.readouterr().err


@pytest.mark.parametrize("count", [0, 1, 3])
def test_missing_or_extra_predictions_are_fatal(monkeypatch, count):
    class BrokenPredictor(_Predictor):
        def iter_predict(self, molecules):
            list(molecules)
            return iter([_prediction()] * count)

    monkeypatch.setattr(cli, "MCACpuPredictor", BrokenPredictor)
    reason = "extra prediction" if count == 3 else "missing prediction"
    with pytest.raises(RuntimeError, match=reason):
        cli.main(["CN", "CO"])


def test_unconsumed_accepted_inputs_are_detected_as_missing(monkeypatch):
    class BrokenPredictor(_Predictor):
        def iter_predict(self, molecules):
            return iter(())

    monkeypatch.setattr(cli, "MCACpuPredictor", BrokenPredictor)
    with pytest.raises(RuntimeError, match="missing prediction for molecule 1"):
        cli.main(["CN"])


@pytest.mark.parametrize("boundary", ["normalize", "validate", "infer", "parse"])
def test_only_domain_validation_value_errors_are_local(monkeypatch, capsys, boundary):
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)
    error_type = RuntimeError if boundary == "validate" else ValueError

    def fail(*args, **kwargs):
        raise error_type(f"fatal {boundary}")

    if boundary == "infer":
        monkeypatch.setattr(_Predictor, "iter_predict", fail)
    else:
        owner = {"normalize": "normalize_molecule", "validate": "validate_dto", "parse": "_read_molecules"}[boundary]
        monkeypatch.setattr(cli, owner, fail)
    with pytest.raises(error_type, match=f"fatal {boundary}"):
        cli.main(["CN"])
    assert "MCA.INPUT_REJECTED" not in capsys.readouterr().err


def test_allow_charged_preserves_all_success_single_table(monkeypatch, capsys):
    monkeypatch.setattr(cli, "MCACpuPredictor", _Predictor)
    assert cli.main(["C[NH3+]", "--allow-charged"]) == 0
    output = capsys.readouterr()
    assert output.out == cli._format_table(_prediction("C[NH3+]")) + "\n"
    assert output.err == ""


def test_nonplot_batch_does_not_retain_source_molecules(monkeypatch, capsys):
    import gc
    import weakref

    dto = cli.normalize_molecule(read_mol("CN"))
    references = []
    normalizations = []

    class Source:
        pass

    def sources(*args):
        for _ in range(300):
            molecule = Source()
            references.append(weakref.ref(molecule))
            yield molecule

    def normalize(molecule):
        normalizations.append(id(molecule))
        return dto

    class SpoolingPredictor(_Predictor):
        def iter_predict(self, molecules):
            accepted = list(molecules)
            gc.collect()
            assert all(reference() is None for reference in references)
            return (_prediction(value.smiles) for value in accepted)

    monkeypatch.setattr(cli, "_read_molecules", sources)
    monkeypatch.setattr(cli, "normalize_molecule", normalize)
    monkeypatch.setattr(cli, "MCACpuPredictor", SpoolingPredictor)
    assert cli.main(["unused"]) == 0
    assert len(normalizations) == 300
    assert capsys.readouterr().out.count("MCA(kJ/mol)") == 300


@pytest.mark.parametrize("consumed", [0, 1])
def test_early_predictor_stop_checks_only_first_missing_input(monkeypatch, consumed):
    def sources(*args):
        yield read_mol("CN")
        pytest.fail("Missing-result detection must not consume later inputs")

    class BrokenPredictor(_Predictor):
        def iter_predict(self, molecules):
            if consumed:
                next(molecules)
            return iter(())

    monkeypatch.setattr(cli, "_read_molecules", sources)
    monkeypatch.setattr(cli, "MCACpuPredictor", BrokenPredictor)
    with pytest.raises(RuntimeError, match="missing prediction for molecule 1"):
        cli.main(["unused"])
