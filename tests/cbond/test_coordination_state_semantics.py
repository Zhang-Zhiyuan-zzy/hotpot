"""CBond owns the chemical boundary of its newly assembled complexes."""

import pytest

from hotpot import read_mol
from hotpot.cheminfo.operations.inference.cbond import apply
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.cheminfo.structure.bond_kind import BondKind


def _constant_scores(context, donor_indices, runtime):
    return {index: 1.0 for index in context.candidate_indices}


@pytest.mark.parametrize("builder", ("auto", "one", "all"))
def test_predicted_coordination_is_explicit_without_changing_numeric_edges(monkeypatch, builder):
    monkeypatch.setattr(apply, "_predict_state", _constant_scores)
    ligand = read_mol("NCCO")
    if builder == "auto":
        molecules = (apply.auto_build_cbond(ligand, "Eu", return_details=True).molecule,)
    elif builder == "one":
        molecules = apply.build_one_cbond(ligand, "Eu", get_all=True)[0]
    else:
        molecules = tuple(result.molecule for result in apply.build_all_possible_cbond(
            ligand, "Eu", return_details=True))

    assert molecules
    for molecule in molecules:
        metal_bonds = tuple(bond for bond in molecule.bonds if bond.is_metal_ligand_bond)
        assert metal_bonds
        assert all(bond.bond_kind is BondKind.DATIVE for bond in metal_bonds)
        assert all(bond.bond_order == 1 for bond in metal_bonds)
        assert all(bond.bond_direction == "atom2_to_atom1" for bond in metal_bonds)
        facts = extract_molecular_facts(molecule)
        assert sum(bond.kind == "coordination" for bond in facts.semantic_bonds) == len(metal_bonds)
        assert len(facts.covalent_components) == 2


def test_new_metal_clears_ligand_declarations_but_preserves_atom_charge():
    ligand = read_mol("[NH3+]CCO")
    ligand.charge = 1
    ligand.spin_mult = 1
    molecule, metal = apply.init_metal_ligand_pair(ligand, "Eu")

    assert molecule is ligand
    assert metal.atomic_number == 63
    assert not molecule._charge_present
    assert not molecule._spin_present
    assert molecule.spin_mult is None
    assert sum(atom.formal_charge for atom in molecule.atoms if atom is not metal) == 1


def test_existing_metal_preserves_user_declarations_and_existing_bond_kind(monkeypatch):
    monkeypatch.setattr(apply, "_predict_state", _constant_scores)
    molecule = read_mol("[Eu].NCCO", "smi")
    molecule.add_bond(0, 1, bond_kind=BondKind.SINGLE)
    molecule.charge = 3
    molecule.spin_mult = 7
    result = apply.auto_build_cbond(molecule, "Eu", return_details=True)

    assert result.molecule.charge == 3
    assert result.molecule.spin_mult == 7
    assert result.molecule.bonds[0].bond_kind is BondKind.SINGLE


def test_new_metal_does_not_reuse_selected_ligand_frame_spin():
    import numpy as np
    from hotpot.cheminfo.structure.conformer import _import_frame_values

    ligand = read_mol("NCCO")
    values = (np.asarray([ligand.coordinates]), None, None, None, None, None,
              np.asarray([3]), None, None)
    _import_frame_values(ligand, values, 0, producer="test-spin")
    assert ligand.spin_mult == 3
    historical = ligand._source_frames
    apply.init_metal_ligand_pair(ligand, "Eu")
    assert ligand.spin_mult is None
    ligand.energy = -1.0
    assert ligand.energy == -1.0
    assert ligand.spin_mult is None
    assert ligand._source_frames == historical
