"""Bounded shipped-reference transport, independent of formal dataset stores."""

from collections import Counter
import hashlib
import importlib.util
from importlib.resources import as_file, files
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import hotpot
from hotpot.cheminfo.operations import solvent_properties


def test_packaged_reference_inventory_preserves_all_original_asset_bytes():
    package = files("hotpot.dataset")
    inventory = json.loads(package.joinpath("manifests", "packaged-reference-assets.json").read_text())
    assets = inventory["assets"]
    assert Counter(row["role"] for row in assets) == {
        "coformer_reference": 88, "solvent_reference": 74,
        "framework_smiles_reference": 1, "medium_properties": 1,
        "solvent_properties": 1,
    }
    assert len({row["path"] for row in assets}) == len(assets)
    product = files("hotpot")
    for row in assets:
        path = product.joinpath(*Path(row["path"]).parts[1:])
        content = path.read_bytes()
        assert len(content) == row["bytes"]
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
        assert not product.joinpath(*Path(row["original_path"]).parts[1:]).is_file()
    retired = inventory["retired"]
    assert {row["original_path"] for row in retired} == {
        "hotpot/cheminfo/ChemData/MolFragTemplate/FragDef.json",
        "hotpot/cheminfo/ChemData/MolFragTemplate/phosphonic_acid.json",
        "hotpot/cheminfo/ChemData/StructQueryTemplate/H_carbon.json",
        "hotpot/cheminfo/ChemData/StructQueryTemplate/sulfonic_acid.json",
    }
    assert all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in retired)
    assert all(not product.joinpath(*Path(row["original_path"]).parts[1:]).is_file()
               for row in retired)
    assert importlib.util.find_spec("hotpot.cheminfo.MolProps") is None


def test_framework_reference_is_distinct_from_the_sole_assembly_default():
    framework = files("hotpot.dataset").joinpath(
        "resources", "molecules", "frameworks", "phen_bipy_derived.json",
    ).read_bytes()
    assert hashlib.sha256(framework).hexdigest() == "92f5f61b6da110c9d8f8c7db1fa528b8890000c14481354ee111b480b7b16b70"
    collection = json.loads(framework)
    assert sum(map(len, collection.values())) == 36
    assert all(isinstance(smiles, str) for group in collection.values() for smiles in group)
    default = files("hotpot.cheminfo.operations.assembly").joinpath(
        "data", "fragment_template.json",
    ).read_bytes()
    assert hashlib.sha256(default).hexdigest() == "0ac6a05cd76c06d3a918bccf7c59acd053315c1b8629754153ad4b7185f08d52"
    definitions = json.loads(default)
    assert len(definitions) == 26
    assert {row["method"] for row in definitions} == {
        "EdgeShoulder", "AtomLink", "AtomReplace", "BondAdding", "AlkylGraft",
    }


@pytest.mark.parametrize("workbook, reader", (
    ("MedProp.xlsx", solvent_properties.get_medium_dataframe),
    ("SolProp.xlsx", solvent_properties.get_solvent_dataframe),
))
def test_workbook_readers_materialize_real_archive_resources(workbook, reader, tmp_path, monkeypatch):
    expected = reader()
    package = files("hotpot.dataset")
    archive_path = tmp_path / "references.zip"
    member = "hotpot/dataset/resources/properties/" + workbook
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, package.joinpath("resources", "properties", workbook).read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        owner = zipfile.Path(archive, at="hotpot/dataset/")
        requested = []
        def resource_files(name):
            requested.append(name)
            return owner
        monkeypatch.setattr(solvent_properties, "files", resource_files)
        assert_frame_equal(reader(), expected)
        assert requested == ["hotpot.dataset"]


@pytest.mark.parametrize("filename, elements, bond_count, pair_counts", (
    ("water.mol2", {1: 2, 8: 1}, 2, {(1, 8): 2}),
    ("methanol.mol2", {1: 4, 6: 1, 8: 1}, 5, {(1, 6): 3, (1, 8): 1, (6, 8): 1}),
    ("benzene.mol2", {1: 6, 6: 6}, 12, {(1, 6): 6, (6, 6): 6}),
))
def test_actual_packaged_mol2_chemistry_transport(filename, elements, bond_count, pair_counts):
    resource = files("hotpot.dataset").joinpath("resources", "molecules", "solvent", filename)
    with as_file(resource) as path:
        molecules = list(hotpot.MolReader(path, "mol2"))
    assert len(molecules) == 1
    molecule = molecules[0]
    assert Counter(atom.atomic_number for atom in molecule.atoms) == elements
    assert len(molecule.bonds) == bond_count
    assert Counter(tuple(sorted((bond.atom1.atomic_number, bond.atom2.atomic_number)))
                   for bond in molecule.bonds) == pair_counts
    assert molecule.coordinates.shape == (sum(elements.values()), 3)
    assert np.isfinite(molecule.coordinates).all()
    assert molecule.charge == 0
    if filename == "benzene.mol2":
        carbons = [atom for atom in molecule.atoms if atom.atomic_number == 6]
        carbon_bonds = [bond for bond in molecule.bonds
                        if bond.atom1.atomic_number == bond.atom2.atomic_number == 6]
        assert all(atom.is_aromatic for atom in carbons)
        assert all(bond.is_aromatic for bond in carbon_bonds)
    else:
        assert all(bond.bond_order == 1 for bond in molecule.bonds)
