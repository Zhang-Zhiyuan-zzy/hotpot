"""Chemical values and demand-driven reads for the remaining Stage9A queries."""

import numpy as np
import pytest

from hotpot import Atom, Bond, Molecule
from hotpot.cheminfo.operations import chemistry
from tests.test_cheminfo._stage5_fixtures import factual_state, supplied_aromatic_flags


def _bonded(first, second, order=1):
    mol = Molecule()
    atoms = tuple(mol.create_atom(atomic_number=number) for number in (first, second))
    bond = mol.add_bond(*atoms, order)
    return mol, atoms, bond


@pytest.mark.parametrize('numbers, charge, unpaired, multiplicity', (
    ((1,), 0, 1, 2),
    ((1,), 1, 0, 1),
    ((8, 8), 0, 0, 1),
    ((6, 1, 1, 1), -1, 0, 1),
    ((), 1, 1, 2),
))
def test_molecular_defaults_follow_total_electron_parity(numbers, charge, unpaired, multiplicity):
    mol = Molecule()
    for number in numbers:
        mol.create_atom(atomic_number=number)
    mol.charge = charge
    before = factual_state(mol)
    assert mol.default_unpaired_electrons == unpaired
    assert mol.default_spin_mult == multiplicity
    assert factual_state(mol) == before


@pytest.mark.parametrize('number, expected, open_shell', (
    (0, (0, 0, [0, 0, 0, 0, 0]), 0),
    (1, (0, 1, [1, 0, 0, 0, 0]), 1),
    (6, (1, 2, [2, 2, 0, 0, 0]), 4),
    (8, (1, 2, [2, 4, 0, 0, 0]), 6),
    (26, (3, 2, [2, 6, 0, 0, 0]), 8),
))
def test_existing_shell_numbering_padding_and_open_shell_count(number, expected, open_shell):
    atom = Atom(atomic_number=number)
    before = factual_state(atom.mol)
    assert atom.calc_electron_config() == expected
    assert atom.open_shell_electrons == open_shell
    assert atom.n == expected[0]
    assert atom.l == expected[1]
    configuration = atom.calc_electron_config()[2]
    configuration[0] = 100
    assert atom.calc_electron_config() == expected
    assert factual_state(atom.mol) == before


def test_missing_electrons_are_the_element_vacancy_less_all_bond_orders():
    mol, (carbon, oxygen), bond = _bonded(6, 8, 2)
    assert carbon.missing_electrons_element == 4
    assert oxygen.missing_electrons_element == 2
    assert carbon.missing_electrons == 2
    assert oxygen.missing_electrons == 0
    bond.bond_order = 1
    assert carbon.missing_electrons == 3
    assert oxygen.missing_electrons == 1


def test_polar_hydrogen_oxygen_and_carbon_sites():
    _, (_, hydroxyl_h), _ = _bonded(8, 1)
    _, (_, carbon_h), _ = _bonded(6, 1)
    assert hydroxyl_h.is_polar_hydrogen
    assert not carbon_h.is_polar_hydrogen


def test_aromatic_nitrogen_is_a_polar_site_and_aliphatic_nitrogen_is_not():
    mol = Molecule()
    for number in (7, 6, 6, 6, 6, 6):
        mol.create_atom(atomic_number=number)
    for row in range(6):
        mol.add_bond(row, (row + 1) % 6, 1)
    aromatic = supplied_aromatic_flags(mol, (True,) * 6)
    assert aromatic.atoms[0].polar_hydrogen_site
    assert not Atom(atomic_number=7).polar_hydrogen_site


def test_polar_site_reads_aromaticity_only_for_nitrogen(monkeypatch):
    oxygen, carbon, nitrogen = (Atom(atomic_number=number) for number in (8, 6, 7))
    def unavailable(atom):
        raise ImportError('injected aromaticity import failure')
    monkeypatch.setattr(Atom, 'is_aromatic', property(unavailable))
    assert oxygen.polar_hydrogen_site
    assert not carbon.polar_hydrogen_site
    with pytest.raises(ImportError, match='injected aromaticity'):
        _ = nitrogen.polar_hydrogen_site


def test_polar_hydrogen_keeps_short_circuit_and_propagates_unexpected_import_failure(monkeypatch):
    carbon, hydrogen = Atom(atomic_number=6), Atom(atomic_number=1)
    def unavailable(atom):
        raise ImportError('injected neighbour import failure')
    monkeypatch.setattr(Atom, 'neighbours', property(unavailable))
    assert not carbon.is_polar_hydrogen
    with pytest.raises(ImportError, match='injected neighbour'):
        _ = hydrogen.is_polar_hydrogen


def test_isolated_hydrogen_keeps_the_missing_neighbour_error():
    with pytest.raises(IndexError):
        _ = Atom(atomic_number=1).is_polar_hydrogen


def test_electron_configuration_error_exempts_metals_before_missing_electron_read(monkeypatch):
    iron, carbon = Atom(atomic_number=26), Atom(atomic_number=6)
    def unavailable(atom):
        raise ImportError('injected missing-electron import failure')
    monkeypatch.setattr(Atom, 'missing_electrons_element', property(unavailable))
    assert not iron.is_error_electron_configure
    with pytest.raises(ImportError, match='injected missing-electron'):
        _ = carbon.is_error_electron_configure


def test_configuration_error_reports_element_vacancy_without_reinterpreting_bonds():
    assert Atom(atomic_number=6).is_error_electron_configure
    assert not Atom(atomic_number=10).is_error_electron_configure


@pytest.mark.parametrize('number, formal_charge, expected', (
    (26, 3, 3), (6, -1, 4), (7, 1, 3), (8, -1, 2),
    (1, 0, 1), (9, -1, 1), (17, -1, 1),
))
def test_legacy_oxidation_shortcuts_do_not_prepare_hydrogens(monkeypatch, number, formal_charge, expected):
    atom = Atom(atomic_number=number, formal_charge=formal_charge)
    def forbidden(*args, **kwargs):
        raise AssertionError('shortcut demanded hydrogens')
    monkeypatch.setattr(chemistry, '_require_implicit_hydrogens', forbidden)
    before = factual_state(atom.mol)
    assert atom.oxidation_state == expected
    assert factual_state(atom.mol) == before


@pytest.mark.parametrize('number, neighbours, hydrogens, expected', (
    (16, ((8, 2), (8, 2)), 0, 4),
    (16, ((15, 1), (6, 1)), 0, -2),
    (15, ((8, 2), (6, 1)), 0, 3),
    (15, ((6, 1),), 0, 0),
    (15, ((6, 1), (26, 1), (1, 1)), 1, 2),
))
def test_legacy_oxidation_keeps_signed_carbon_and_hydrogen_convention(number, neighbours, hydrogens, expected):
    mol = Molecule()
    site = mol.create_atom(atomic_number=number, implicit_hydrogens=hydrogens)
    for other_number, order in neighbours:
        other = mol.create_atom(atomic_number=other_number, implicit_hydrogens=0)
        mol.add_bond(site, other, order)
    assert site.oxidation_state == expected


def test_legacy_oxidation_demands_hydrogens_before_neighbour_electronegativity(monkeypatch):
    mol, (sulfur, _), _ = _bonded(16, 8)
    reads = []
    def unavailable(*args, **kwargs):
        reads.append('hydrogens')
        raise ValueError('injected unavailable H representation')
    def forbidden(atom):
        raise AssertionError('electronegativity read before H requirement')
    monkeypatch.setattr(chemistry, '_require_implicit_hydrogens', unavailable)
    monkeypatch.setattr(Atom, 'electronegativity', property(forbidden))
    with pytest.raises(ValueError, match='injected unavailable H'):
        _ = sulfur.oxidation_state
    assert reads == ['hydrogens']


def test_legacy_oxidation_compares_electronegativity_before_reading_bond_order(monkeypatch):
    mol = Molecule()
    sulfur = mol.create_atom(atomic_number=16, implicit_hydrogens=0)
    oxygen = mol.create_atom(atomic_number=8, implicit_hydrogens=0)
    mol.add_bond(sulfur, oxygen)
    monkeypatch.setattr(chemistry, '_require_implicit_hydrogens', lambda *args, **kwargs: (0,))
    def unavailable(atom):
        raise ImportError('injected electronegativity failure')
    def forbidden(bond):
        raise AssertionError('bond order read before electronegativity comparison')
    monkeypatch.setattr(Atom, 'electronegativity', property(unavailable))
    monkeypatch.setattr(Bond, 'bond_order', property(forbidden), raising=False)
    with pytest.raises(ImportError, match='injected electronegativity'):
        _ = sulfur.oxidation_state


@pytest.mark.parametrize('numbers, covalent, heavy, metal_ligand', (
    ((6, 8), True, True, False),
    ((6, 1), True, False, False),
    ((26, 7), False, False, True),
    ((7, 26), False, False, True),
    ((26, 29), False, False, False),
))
def test_bond_classification_uses_endpoint_elements(numbers, covalent, heavy, metal_ligand):
    _, _, bond = _bonded(*numbers)
    assert bond.is_covalent == covalent
    assert bond.is_heavy_covalent == heavy
    assert bond.is_metal_ligand_bond == metal_ligand


def test_bond_classification_stops_after_first_metal(monkeypatch):
    _, (iron, oxygen), bond = _bonded(26, 8)
    metal_property = Atom.is_metal
    def observe(atom):
        if atom is oxygen:
            raise AssertionError('classification inspected the unnecessary second endpoint')
        return metal_property.fget(atom)
    monkeypatch.setattr(Atom, 'is_metal', property(observe))
    assert not bond.is_covalent
    assert not bond.is_heavy_covalent


def test_rotatable_single_bond_and_ring_exclusion():
    mol, atoms, bond = _bonded(6, 6)
    assert bond.rotatable
    third = mol.create_atom(atomic_number=6)
    mol.add_bond(atoms[0], third)
    mol.add_bond(atoms[1], third)
    assert not bond.rotatable


def test_rotatable_non_single_bond_does_not_read_aromaticity(monkeypatch):
    _, _, bond = _bonded(6, 8, 2)
    def forbidden(bond):
        raise AssertionError('non-single bond demanded aromaticity')
    monkeypatch.setattr(Bond, 'is_aromatic', property(forbidden))
    assert not bond.rotatable


def test_rotatable_aromatic_bond_does_not_read_ring_membership(monkeypatch):
    _, _, bond = _bonded(6, 6)
    monkeypatch.setattr(Bond, 'is_aromatic', property(lambda bond: True))
    def forbidden(bond):
        raise AssertionError('aromatic bond demanded ring membership')
    monkeypatch.setattr(Bond, 'in_ring', property(forbidden))
    assert not bond.rotatable


def test_ideal_bond_length_keeps_the_reference_pair_and_order_lookup():
    _, _, carbon_carbon = _bonded(6, 6)
    _, _, carbon_oxygen = _bonded(6, 8)
    assert carbon_carbon.ideal_bond_length == pytest.approx(1.53)
    assert carbon_oxygen.ideal_bond_length == pytest.approx(1.414)


def test_ideal_bond_length_missing_pair_fails_before_reading_order(monkeypatch):
    _, _, bond = _bonded(0, 0)
    def forbidden(bond):
        raise AssertionError('missing reference pair demanded bond order')
    monkeypatch.setattr(Bond, 'bond_order', property(forbidden), raising=False)
    with pytest.raises(KeyError):
        _ = bond.ideal_bond_length


def test_detached_atom_projection_retains_values_and_returns_independent_buffers():
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6, implicit_hydrogens=2, formal_charge=-1,
                           coordinates=(1, 2, 3), id=91)
    mol.remove_atom(atom)
    values = atom.attrs
    assert values[0] == 6
    np.testing.assert_array_equal(values[1:7], (2, 2, 2, 0, 0, 0))
    assert values[7] == -1
    assert values[13] == 4
    assert values[15] == 91
    values[:] = -999
    assert atom.attrs[0] == 6
    assert atom.formal_charge == -1
    assert atom.implicit_hydrogens == 2
    assert atom.coordinates == (1, 2, 3)


def test_atom_hydrogen_completion_keeps_the_existing_return_and_membership():
    atom = Atom(atomic_number=8, implicit_hydrogens=0)
    added = atom.add_hydrogen(1)
    assert len(added) == 1
    hydrogen, = added
    assert hydrogen.atomic_number == 1
    assert hydrogen.mol is atom.mol
    assert hydrogen in atom.neighbours
    assert len(atom.mol.bonds) == 1


def test_atom_hydrogen_completion_rolls_back_a_late_topology_failure(monkeypatch):
    atom = Atom(atomic_number=8, implicit_hydrogens=0)
    mol = atom.mol
    before = factual_state(mol)
    original_atoms = tuple(mol.atoms)
    def unavailable(*args, **kwargs):
        raise RuntimeError('injected final topology publication failure')
    monkeypatch.setattr(mol, '_update_topology', unavailable)
    with pytest.raises(RuntimeError, match='injected final topology'):
        atom.add_hydrogen(1)
    assert tuple(mol.atoms) == original_atoms
    assert not mol.bonds
    assert factual_state(mol) == before


def test_molecular_implicit_hydrogen_sum_selects_heavy_atoms_only():
    mol = Molecule()
    mol.create_atom(atomic_number=1, implicit_hydrogens=9)
    mol.create_atom(atomic_number=6, implicit_hydrogens=3)
    mol.create_atom(atomic_number=8, implicit_hydrogens=1)
    assert mol.sum_implicit_hydrogens == 4


def test_charge_update_uses_atomic_charges_and_the_molecular_setter():
    mol = Molecule()
    mol.create_atom(atomic_number=7, formal_charge=1)
    mol.create_atom(atomic_number=8, formal_charge=-1)
    mol.charge = 3
    assert mol.update_mol_charge() is None
    assert mol.charge == 0
    assert [atom.formal_charge for atom in mol.atoms] == [1, -1]
