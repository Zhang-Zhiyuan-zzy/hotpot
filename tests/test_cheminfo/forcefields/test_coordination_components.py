"""Direct-edge coordination-component regression tests."""

from pathlib import Path

from hotpot import read_mol
from hotpot.cheminfo.forcefields.coordination import (
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


def test_coordination_module_has_no_networkx_reference():
    source = (
        Path(__file__).resolve().parents[3]
        / "hotpot"
        / "cheminfo"
        / "forcefields"
        / "coordination.py"
    ).read_text()

    assert "networkx" not in source.lower()
    assert "nx." not in source
