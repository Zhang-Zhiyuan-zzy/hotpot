"""Shared spreadsheet selection honors each command's declared field names."""

import argparse
from types import SimpleNamespace

import pandas as pd
import pytest

from hotpot.commands import optimize


@pytest.fixture
def workbook(tmp_path):
    path = tmp_path / "inputs.xlsx"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6], "y": [7, 8]}).to_excel(path, index=False)
    return path


@pytest.mark.parametrize(
    "args,fields,expected",
    (
        (SimpleNamespace(params="b,c", exclude_params="c", by_index=False, target=None),
         ("params", "exclude_params"), ["b"]),
        (SimpleNamespace(features="b,c", exclude_features="c", by_index=False, target=None),
         ("features", "exclude_features"), ["b"]),
        (SimpleNamespace(features="0-2", exclude_features="0,2", by_index=True, target=None),
         ("features", "exclude_features"), ["b"]),
        (SimpleNamespace(features=None, exclude_features=None, by_index=False, target=None),
         ("features", "exclude_features"), ["a", "b", "c"]),
    ),
)
def test_command_field_names_select_and_exclude_columns(workbook, args, fields, expected):
    inputs, target = optimize.read_excel(
        workbook, args, input_field=fields[0], exclude_field=fields[1]
    )

    assert list(inputs.columns) == expected
    assert target.name == "y"
    assert target.tolist() == [7, 8]


@pytest.mark.optional_dependency
def test_training_passes_selected_feature_fields_to_workflow(workbook, tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from hotpot.commands import ml_train

    parser = argparse.ArgumentParser()
    ml_train.add_arguments(parser)
    args = parser.parse_args([
        str(workbook), "-f", "b,c", "-ef", "c", "-a", "LeastSquares",
        "-d", str(tmp_path / "training"),
    ])
    observed = {}

    class Workflow:
        def __init__(self, **options):
            observed.update(options)
            self.features = options["features"]
            self.valid_score = None

        def work(self):
            observed["worked"] = True

    monkeypatch.setattr(ml_train, "MachineLearning_", Workflow)
    ml_train.train(args)

    assert observed["features"] == ["b"]
    assert observed["target"] == "y"
    assert list(observed["data"].columns) == ["b", "y"]
    assert observed["worked"]
