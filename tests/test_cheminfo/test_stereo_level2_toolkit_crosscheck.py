"""Narrow external-toolkit observations for Hotpot Level-2 stereo.

RDKit independently observes Hotpot's ideal-coordinate templates.  Hotpot
descriptor numbers are deliberately mapped through frozen, version-specific
lookups: they are not RDKit ``_chiralPermutation`` numbers.  Open Babel only
provides a smaller TH/DB/SP parsing-and-distinction capability witness and runs
in a separate process because the third-party wheels have a known import-order
conflict.  Its raw ``refs`` are not treated as complete configuration objects
or compared with Hotpot semantics.

Neither toolkit is an oracle here for SPY, AXIS, chelation, enhanced groups,
or Hotpot's 12/2/3-degree acceptance policy.
"""

import json
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem, rdBase
from rdkit.Geometry import Point3D

from hotpot.cheminfo import stereo

from hotpot.cheminfo.rules.chemistry.stereochemistry import algebra as stereo_algebra
from hotpot.cheminfo.rules.chemistry.stereochemistry import coordinates as stereo_coordinates


_RDKIT_COORDINATION_LOOKUP = {
    stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR: (
        1, 3, 2,
    ),
    stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL: (
        7, 8, 5, 6, 3, 4, 1, 2, 13, 14,
        10, 12, 9, 11, 16, 19, 15, 20, 17, 18,
    ),
    stereo_algebra._Level2DescriptorFamily.OCTAHEDRAL: (
        27, 28, 25, 30, 26, 29, 21, 22, 19, 24,
        20, 23, 12, 13, 6, 18, 7, 17, 9, 11,
        3, 16, 5, 15, 8, 10, 1, 2, 4, 14,
    ),
}
_RDKIT_TAGS = {
    stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR: "CHI_SQUAREPLANAR",
    stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL: (
        "CHI_TRIGONALBIPYRAMIDAL"
    ),
    stereo_algebra._Level2DescriptorFamily.OCTAHEDRAL: "CHI_OCTAHEDRAL",
}
_CENTER_ATOMIC_NUMBERS = {
    stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR: 78,
    stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL: 33,
    stereo_algebra._Level2DescriptorFamily.OCTAHEDRAL: 27,
}
_LIGAND_ATOMIC_NUMBERS = (9, 17, 35, 53, 7, 15)
_OPENBABEL_CAPABILITY_SCOPE = frozenset({"TH", "DB", "SP"})
_EXCLUDED_ORACLE_SCOPE = frozenset({
    "SPY",
    "AXIS",
    "chelate partitions",
    "enhanced groups",
    "12/2/3-degree tolerances",
})


def _rdkit_perception(family, directions):
    editable = Chem.RWMol()
    center = Chem.Atom(_CENTER_ATOMIC_NUMBERS[family])
    center.SetNoImplicit(True)
    editable.AddAtom(center)
    for position, atomic_number in enumerate(
        _LIGAND_ATOMIC_NUMBERS[: len(directions)],
        start=1,
    ):
        ligand = Chem.Atom(atomic_number)
        ligand.SetNoImplicit(True)
        editable.AddAtom(ligand)
        editable.AddBond(0, position, Chem.BondType.SINGLE)

    molecule = editable.GetMol()
    molecule.UpdatePropertyCache(strict=False)
    conformer = Chem.Conformer(len(directions) + 1)
    conformer.Set3D(True)
    conformer.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    for position, direction in enumerate(directions, start=1):
        conformer.SetAtomPosition(
            position,
            Point3D(*(float(component) for component in direction)),
        )
    molecule.AddConformer(conformer)

    Chem.AssignAtomChiralTagsFromStructure(
        molecule,
        confId=0,
        replaceExistingTags=True,
    )
    perceived = molecule.GetAtomWithIdx(0)
    return (
        str(perceived.GetChiralTag()),
        perceived.GetIntProp("_chiralPermutation"),
    )


@pytest.fixture(scope="module", autouse=True)
def _enable_rdkit_nontetrahedral_chirality():
    previous = Chem.GetAllowNontetrahedralChirality()
    Chem.SetAllowNontetrahedralChirality(True)
    try:
        yield
    finally:
        Chem.SetAllowNontetrahedralChirality(previous)


_COORDINATION_CASES = tuple(
    (family, descriptor_index, rdkit_index)
    for family, rdkit_indices in _RDKIT_COORDINATION_LOOKUP.items()
    for descriptor_index, rdkit_index in enumerate(rdkit_indices, start=1)
)


def test_rdkit_oracle_is_the_pinned_release():
    assert rdBase.rdkitVersion == "2026.09.1"


@pytest.mark.parametrize(
    ("family", "descriptor_index", "rdkit_index"),
    _COORDINATION_CASES,
    ids=lambda value: value.value if hasattr(value, "value") else str(value),
)
def test_all_sp_tbp_oh_ideal_descriptors_match_frozen_rdkit_lookup(
    family,
    descriptor_index,
    rdkit_index,
):
    directions = stereo_coordinates._gate7_ideal_carrier_directions(
        stereo_algebra._Level2Descriptor(family, descriptor_index)
    )

    assert _rdkit_perception(family, directions) == (
        _RDKIT_TAGS[family],
        rdkit_index,
    )


@pytest.mark.parametrize(
    ("family", "descriptor_index"),
    (
        (stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR, 3),
        (stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL, 17),
        (stereo_algebra._Level2DescriptorFamily.OCTAHEDRAL, 29),
    ),
)
def test_rdkit_oracle_preserves_descriptor_under_proper_rotation(
    family,
    descriptor_index,
):
    directions = stereo_coordinates._gate7_ideal_carrier_directions(
        stereo_algebra._Level2Descriptor(family, descriptor_index)
    )
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )

    assert _rdkit_perception(family, directions @ rotation.T) == (
        _RDKIT_TAGS[family],
        _RDKIT_COORDINATION_LOOKUP[family][descriptor_index - 1],
    )


@pytest.mark.parametrize(
    ("family", "source_index", "reflected_index"),
    (
        (stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR, 1, 1),
        (stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL, 1, 2),
        (stereo_algebra._Level2DescriptorFamily.OCTAHEDRAL, 1, 2),
    ),
)
def test_rdkit_oracle_matches_frozen_mirror_witness(
    family,
    source_index,
    reflected_index,
):
    directions = stereo_coordinates._gate7_ideal_carrier_directions(
        stereo_algebra._Level2Descriptor(family, source_index)
    ).copy()
    directions[:, 0] *= -1.0

    assert _rdkit_perception(family, directions) == (
        _RDKIT_TAGS[family],
        _RDKIT_COORDINATION_LOOKUP[family][reflected_index - 1],
    )


@pytest.mark.parametrize(
    ("family", "reordered_index"),
    (
        (stereo_algebra._Level2DescriptorFamily.SQUARE_PLANAR, 2),
        (stereo_algebra._Level2DescriptorFamily.TRIGONAL_BIPYRAMIDAL, 2),
        (stereo_algebra._Level2DescriptorFamily.OCTAHEDRAL, 2),
    ),
)
def test_rdkit_oracle_matches_first_carrier_exchange(
    family,
    reordered_index,
):
    directions = stereo_coordinates._gate7_ideal_carrier_directions(
        stereo_algebra._Level2Descriptor(family, 1)
    ).copy()
    directions[[0, 1]] = directions[[1, 0]]

    assert _rdkit_perception(family, directions) == (
        _RDKIT_TAGS[family],
        _RDKIT_COORDINATION_LOOKUP[family][reordered_index - 1],
    )


def test_openbabel_th_db_sp_configuration_capability_witness():
    script = r"""
import json
from openbabel import openbabel as ob

def read(smiles):
    conversion = ob.OBConversion()
    conversion.SetInFormat("smi")
    molecule = ob.OBMol()
    if not conversion.ReadString(molecule, smiles):
        raise RuntimeError(smiles)
    return molecule

def tetrahedral(smiles):
    molecule = read(smiles)
    facade = ob.OBStereoFacade(molecule)
    for atom in ob.OBMolAtomIter(molecule):
        if facade.HasTetrahedralStereo(atom.GetId()):
            config = facade.GetTetrahedralStereo(atom.GetId()).GetConfig()
            return config.specified, list(config.refs)
    raise RuntimeError("missing tetrahedral stereo")

def double_bond(smiles):
    molecule = read(smiles)
    facade = ob.OBStereoFacade(molecule)
    for bond in ob.OBMolBondIter(molecule):
        if facade.HasCisTransStereo(bond.GetId()):
            config = facade.GetCisTransStereo(bond.GetId()).GetConfig()
            return config.specified, list(config.refs)
    raise RuntimeError("missing cis/trans stereo")

def square_planar(smiles):
    molecule = read(smiles)
    facade = ob.OBStereoFacade(molecule)
    for atom in ob.OBMolAtomIter(molecule):
        if facade.HasSquarePlanarStereo(atom.GetId()):
            config = facade.GetSquarePlanarStereo(atom.GetId()).GetConfig()
            return config.specified, list(config.refs)
    raise RuntimeError("missing square-planar stereo")

print(json.dumps({
    "version": ob.OBReleaseVersion(),
    "TH": [
        tetrahedral("F[C@](Cl)(Br)I"),
        tetrahedral("F[C@@](Cl)(Br)I"),
    ],
    "DB": [
        double_bond("F/C=C/F"),
        double_bond("F/C=C\\F"),
    ],
    "SP": [
        square_planar("[Pt@SP1](Cl)(Br)(I)F"),
        square_planar("[Pt@SP2](Cl)(Br)(I)F"),
        square_planar("[Pt@SP3](Cl)(Br)(I)F"),
    ],
}))
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)

    assert observed["version"].startswith("3.2.")
    for family in ("TH", "DB", "SP"):
        assert all(specified for specified, _ in observed[family])
        configurations = [tuple(refs) for _, refs in observed[family]]
        assert len(set(configurations)) == len(configurations)
        assert all(
            sorted(config) == sorted(configurations[0])
            for config in configurations[1:]
        )


def test_toolkit_observation_scope_excludes_hotpot_owned_semantics():
    rdkit_scope = frozenset(family.value for family in _RDKIT_COORDINATION_LOOKUP)

    assert rdkit_scope == frozenset({"SP", "TBP", "OH"})
    assert _OPENBABEL_CAPABILITY_SCOPE == frozenset({"TH", "DB", "SP"})
    assert _EXCLUDED_ORACLE_SCOPE.isdisjoint(
        rdkit_scope | _OPENBABEL_CAPABILITY_SCOPE
    )
