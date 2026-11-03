"""Direct-edge coordination-component regression tests."""

from pathlib import Path

import hotpot as hp
from hotpot import read_mol
from hotpot.cheminfo.adapters.structure.forcefields import (
    collect_coordination_environments,
)


def test_coordination_components_use_authoritative_bonds():
    molecule = read_mol("N.CN.[Zn]", "smi")
    molecule.add_bond(molecule.atoms[3], molecule.atoms[0])
    molecule.add_bond(molecule.atoms[3], molecule.atoms[2])

    (environment,) = collect_coordination_environments(molecule)

    assert environment.metal_idx == 3
    assert environment.donor_indices == (0, 2)
    assert environment.chelate_groups == ((0,), (2,))


def test_coordination_collector_owner_has_no_networkx_reference():
    source = (
        Path(hp.__file__).resolve().parent
        / "cheminfo"
        / "adapters"
        / "structure"
        / "forcefields.py"
    ).read_text()

    assert "networkx" not in source.lower()
    assert "nx." not in source
