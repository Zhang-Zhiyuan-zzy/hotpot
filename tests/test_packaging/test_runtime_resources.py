"""Exercise real package-data consumers in source and installed packages."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import hotpot
from hotpot.cheminfo.operations.solvent_properties import (
    get_medium_dataframe,
    get_solvent_dataframe,
    query_medium,
    query_solvent,
)


@pytest.mark.parametrize(
    ("reader", "query", "id_type", "identifiers"),
    (
        (get_medium_dataframe, query_medium, "CASs", ["64-19-7", "10097-32-2", "64-19-7"]),
        (get_medium_dataframe, query_medium, "Cid", [176, 260, 176]),
        (get_solvent_dataframe, query_solvent, "CASs", ["107-06-2", "107-21-1", "107-06-2"]),
        (get_solvent_dataframe, query_solvent, "Cid", [11, 174, 11]),
    ),
)
def test_clean_property_queries_preserve_requested_order_and_duplicates(
    reader, query, id_type, identifiers,
):
    table = reader()
    expected = table.set_index(id_type).loc[identifiers]
    result = query(identifiers, id_type=id_type)
    assert result.index.tolist() == identifiers
    assert_frame_equal(result, expected)


def test_property_query_reports_missing_identifier():
    with pytest.raises(ValueError, match="Undefined CASs"):
        query_medium(["not-in-the-property-table"])


def test_normal_ml_feature_import_reads_original_tables():
    from hotpot.plugins.ml import features

    elements = features.element_features.dataframe
    assert elements.shape == (118, 79)
    assert elements.loc[["C", "O", "Am"], "atomic_number"].tolist() == [6, 8, 95]

    solvent_names = features.get_all_solvents_names()
    medium_names = features.get_all_media_names()
    assert len(solvent_names) == 185
    assert len(medium_names) == 11
    for names, convert in (
        (solvent_names, features.to_solvent_features),
        (medium_names, features.to_media_features),
    ):
        requested = [names[1], names[0], names[1]]
        result = convert(requested, fill_miss=False)
        assert result.index.tolist() == requested
        assert_frame_equal(result.iloc[[0]], result.iloc[[2]])


def test_saved_colormap_leaf_reads_packaged_rgba_values():
    # Load the actual reader without executing unrelated plotting initializers.
    root = Path(hotpot.__file__).resolve().parent
    path = root / "plugins/plots/colors.py"
    spec = importlib.util.spec_from_file_location("hotpot_resource_colors", path)
    colors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colors)
    expected = np.load(root / "plugins/plots/cmaps/BluesSat.npy")
    assert expected.shape == (256, 4)
    assert "BluesSat" in colors.get_defined_cmap_name()
    listed = colors.load_defined_cmap("BluesSat", "list")
    np.testing.assert_array_equal(listed.colors, expected)
    linear = colors.load_defined_cmap("BluesSat", "linear")
    np.testing.assert_allclose(linear(0.0), [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(linear(1.0), [0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("reader", "query", "sheet", "shape", "identifiers"),
    (
        (get_medium_dataframe, query_medium, "fill", (96, 380),
         ["7778-77-0", "13138-45-9", "299-28-5", "7778-77-0"]),
        (get_solvent_dataframe, query_solvent, "total", (233, 86),
         ["100-41-4", "100-42-5", "100-44-7", "100-41-4"]),
    ),
)
def test_actual_nondefault_cas_queries_preserve_order_and_duplicates(
    reader, query, sheet, shape, identifiers,
):
    table = reader(sheet)
    assert table.shape == shape
    assert "CASs" in table.columns and "Cid" not in table.columns
    result = query(identifiers, "CASs", sheet)
    assert result.index.tolist() == identifiers
    assert_frame_equal(result, table.set_index("CASs").loc[identifiers])


@pytest.mark.parametrize(
    ("query", "sheet"),
    ((query_medium, "fill"), (query_solvent, "total")),
)
def test_nondefault_property_queries_expose_unavailable_cid(query, sheet):
    with pytest.raises(KeyError, match="Cid"):
        query([176], "Cid", sheet)


def test_solvent_fill_sheet_remains_unavailable():
    with pytest.raises(ValueError, match="fill"):
        get_solvent_dataframe("fill")
    with pytest.raises(ValueError, match="fill"):
        query_solvent(["100-41-4"], sub_sheet="fill")


@pytest.mark.parametrize(
    ("query", "sheet"),
    ((query_medium, "fill"), (query_solvent, "total")),
)
def test_nondefault_property_queries_report_missing_cas(query, sheet):
    with pytest.raises(ValueError, match="Undefined CASs"):
        query(["not-in-the-property-table"], sub_sheet=sheet)
