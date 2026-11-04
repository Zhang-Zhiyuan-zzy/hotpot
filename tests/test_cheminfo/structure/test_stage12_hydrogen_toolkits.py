"""Physical H inventories and registered toolkit maps for Stage12B.

These expectations count represented H vertices and the independent non-graph
inventory. Native execution cases require the rebuilt Stage12B RDKit extension.
"""

from dataclasses import replace

import numpy as np
import pytest
from openbabel import openbabel as ob
from rdkit import Chem

from hotpot import Molecule
from hotpot.cheminfo.operations import conversion
from hotpot.cheminfo.operations.native_packing import pack_molecule
from hotpot.cheminfo.adapters.native.molecule_buffers import (
    NativeHydrogenTransportFieldCode as H,
    NativeMoleculeField as F,
)
from hotpot.cheminfo.stereo import (
    AtomTransportRecord, HydrogenTransportState, StereoPayload,
    get_stereo_payload, set_stereo_payload,
)


_CASES = [
    ((6,), (0,), (4,)), ((6,), (1,), (4,)), ((6,), (4,), (4,)),
    ((6, 8), (0, 0), (3, 1)), ((6, 8), (1, 0), (3, 1)),
    ((6, 8), (3, 1), (3, 1)),
]
_ATOM_TRANSPORT = frozenset({F.ATOM_TRANSPORT})


def _rd_graph(numbers, graph_counts):
    mol = Chem.RWMol()
    for number in numbers:
        mol.AddAtom(Chem.Atom(number))
    if len(numbers) == 2:
        mol.AddBond(0, 1, Chem.BondType.SINGLE)
    for anchor, count in enumerate(graph_counts):
        for _ in range(count):
            atom = Chem.Atom(1)
            atom.SetNoImplicit(True)
            hydrogen = mol.AddAtom(atom)
            mol.AddBond(anchor, hydrogen, Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    return mol.GetMol()


def _ob_graph(numbers, graph_counts, targets):
    mol = ob.OBMol()
    mol.BeginModify()
    for number in numbers:
        mol.NewAtom().SetAtomicNum(number)
    if len(numbers) == 2:
        mol.AddBond(1, 2, 1)
    for anchor, count in enumerate(graph_counts):
        for _ in range(count):
            atom = mol.NewAtom()
            atom.SetAtomicNum(1)
            mol.AddBond(anchor + 1, atom.GetIdx(), 1)
    mol.EndModify()
    for row, atom in enumerate(ob.OBMolAtomIter(mol)):
        atom.SetImplicitHCount(targets[row] - graph_counts[row] if row < len(numbers) else 0)
    mol.SetAromaticPerceived(True)
    return mol


def _rd_counts(mol):
    return tuple(atom.GetNumImplicitHs() + atom.GetNumExplicitHs() for atom in mol.GetAtoms())


def _ob_counts(mol):
    return tuple(atom.GetImplicitHCount() for atom in ob.OBMolAtomIter(mol))


def _decode(table):
    data = table.data.tobytes()
    return tuple(data[int(offset):int(offset + length)].decode() if present else None
                 for offset, length, present in zip(table.offsets, table.lengths, table.present))


def _projected(mol):
    return pack_molecule(mol, _ATOM_TRANSPORT).atom_transport


@pytest.mark.parametrize('numbers,graph,targets', _CASES)
def test_rdkit_import_translates_target_and_retains_exact_decomposition(numbers, graph, targets):
    source = _rd_graph(numbers, graph)
    expected = tuple(target - represented for target, represented in zip(targets, graph))
    assert _rd_counts(source) == expected + (0,) * sum(graph)
    mol = conversion.from_rdmol(source, Molecule())
    assert tuple(atom.implicit_hydrogens for atom in mol.atoms) == targets + (0,) * sum(graph)
    payload = get_stereo_payload(mol)
    for fact, atom in zip(payload.atom_facts, source.GetAtoms()):
        state = HydrogenTransportState(atom.GetNumImplicitHs(), atom.GetNumExplicitHs(), atom.GetNoImplicit())
        assert fact.current_hydrogens == fact.source_hydrogens == state
    projected = conversion.to_rdmol(mol, sanitize=True)
    assert _rd_counts(projected) == _rd_counts(source)
    assert projected.GetNumAtoms() == source.GetNumAtoms()
    assert get_stereo_payload(mol) == payload
    packed = _projected(mol)
    assert packed.current_implicit_hydrogens.tolist() == [atom.GetNumImplicitHs() for atom in source.GetAtoms()]
    assert packed.current_bracket_explicit_hydrogens.tolist() == [atom.GetNumExplicitHs() for atom in source.GetAtoms()]
    assert packed.current_no_implicit.tolist() == [atom.GetNoImplicit() for atom in source.GetAtoms()]


@pytest.mark.parametrize('numbers,graph,targets', _CASES)
def test_openbabel_target_source_ledger_and_cross_toolkit_inventory(numbers, graph, targets):
    source = _ob_graph(numbers, graph, targets)
    expected = tuple(target - represented for target, represented in zip(targets, graph)) + (0,) * sum(graph)
    assert _ob_counts(source) == expected
    mol = conversion.obmol2mol(source, Molecule())
    assert tuple(atom.implicit_hydrogens for atom in mol.atoms) == targets + (0,) * sum(graph)
    payload = get_stereo_payload(mol)
    assert tuple(fact.source_namespace for fact in payload.atom_facts) == ('openbabel',) * len(mol.atoms)
    assert all(fact.current_hydrogens is None for fact in payload.atom_facts)
    assert tuple(fact.source_hydrogens for fact in payload.atom_facts) == tuple(
        HydrogenTransportState(count, None, None) for count in expected)
    assert _ob_counts(mol.to_obmol()) == expected
    rdmol = conversion.to_rdmol(mol, sanitize=True)
    assert _rd_counts(rdmol) == expected
    assert all(atom.GetNoImplicit() for atom in rdmol.GetAtoms())
    packed = _projected(mol)
    assert packed.current_implicit_hydrogens.tolist() == list(expected)
    assert packed.current_hydrogen_presence_masks.tolist() == [H.IMPLICIT] * len(expected)
    assert packed.source_implicit_hydrogens.tolist() == list(expected)
    assert _decode(packed.source_namespaces) == ('openbabel',) * len(expected)
    assert get_stereo_payload(mol) == payload


@pytest.mark.parametrize('text', ['[nH]1cccc1', '[NH4+]', '[CH3]', '[O-]', '[13CH3]O'])
def test_rdkit_registered_bracket_charge_radical_and_isotope_declarations(text):
    source = Chem.MolFromSmiles(text)
    mol = conversion.from_rdmol(source, Molecule())
    before = get_stereo_payload(mol)
    projected = conversion.to_rdmol(mol, sanitize=True)
    assert _rd_counts(projected) == _rd_counts(source)
    assert [(a.GetIsotope(), a.GetNumRadicalElectrons(), a.GetFormalCharge(), a.GetNoImplicit())
            for a in projected.GetAtoms()] == [
        (a.GetIsotope(), a.GetNumRadicalElectrons(), a.GetFormalCharge(), a.GetNoImplicit())
        for a in source.GetAtoms()]
    _projected(mol)
    assert get_stereo_payload(mol) == before


@pytest.mark.parametrize('isotope', [2, 3])
def test_openbabel_preserves_graph_hydrogen_isotope_in_both_exports_and_native_packing(isotope):
    source = _ob_graph((6,), (1,), (4,))
    source.GetAtom(1).SetIsotope(13)
    source.GetAtom(2).SetIsotope(isotope)
    mol = conversion.obmol2mol(source, Molecule())
    before = get_stereo_payload(mol)
    projected = mol.to_obmol()
    assert tuple(a.GetIsotope() for a in ob.OBMolAtomIter(projected)) == (13, isotope)
    assert tuple(a.GetIsotope() for a in mol.to_rdmol().GetAtoms()) == (13, isotope)
    assert _projected(mol).isotopes.tolist() == [13, isotope]
    assert get_stereo_payload(mol) == before


@pytest.mark.parametrize('source_namespace', [None, 'rdkit-2026.09.1', 'openbabel'])
def test_isotope_source_only_fact_uses_live_target_fallback(source_namespace):
    mol = Molecule()
    mol.create_atom(atomic_number=6, implicit_hydrogens=2)
    ledger = HydrogenTransportState(17, None, None)
    fact = AtomTransportRecord(0, 13, None, None, ledger, source_namespace)
    payload = StereoPayload(1, (), (fact,), ())
    set_stereo_payload(mol, payload)
    rdmol = conversion.to_rdmol(mol, sanitize=True)
    atom = rdmol.GetAtomWithIdx(0)
    assert _rd_counts(rdmol) == (2,)
    assert atom.GetNoImplicit() and atom.GetNumExplicitHs() == 2 and atom.GetIsotope() == 13
    assert _ob_counts(mol.to_obmol()) == (2,)
    packed = _projected(mol)
    actual = packed.current_implicit_hydrogens + packed.current_bracket_explicit_hydrogens
    assert actual.tolist() == [2]
    assert packed.source_implicit_hydrogens.tolist() == [17]
    assert get_stereo_payload(mol) == payload


@pytest.mark.parametrize('numbers,bonds', [((1,), ()), ((1, 1), ((0, 1),)), ((30,), ()), ((30, 1), ((0, 1),))])
def test_supplied_exceptional_hydrogen_and_metal_counts_use_fixed_physical_projection(numbers, bonds):
    mol = Molecule()
    for number in numbers:
        mol.create_atom(atomic_number=number, implicit_hydrogens=2)
    for pair in bonds:
        mol.add_bond(*pair, 1.)
    expected = tuple(0 if number == 1 else 2 for number in numbers)
    projected = conversion.to_rdmol(mol, sanitize=True)
    assert _rd_counts(projected) == expected
    assert all(atom.GetNoImplicit() for atom in projected.GetAtoms())
    assert _ob_counts(mol.to_obmol()) == expected
    assert _projected(mol).current_implicit_hydrogens.tolist() == list(expected)
    assert tuple(atom.implicit_hydrogens for atom in mol.atoms) == (2,) * len(numbers)


def test_imported_metal_retains_non_graph_count_without_adding_graph_hydride():
    rdmol = Chem.RWMol()
    metal = Chem.Atom(30)
    metal.SetNoImplicit(True)
    metal.SetNumExplicitHs(2)
    rdmol.AddAtom(metal)
    hydrogen = Chem.Atom(1)
    hydrogen.SetNoImplicit(True)
    rdmol.AddAtom(hydrogen)
    rdmol.AddBond(0, 1, Chem.BondType.SINGLE)
    rdmol.UpdatePropertyCache(strict=False)
    mol = conversion.from_rdmol(rdmol, Molecule())
    assert tuple(atom.implicit_hydrogens for atom in mol.atoms) == (2, 0)
    assert _rd_counts(mol.to_rdmol()) == (2, 0)
    assert _ob_counts(mol.to_obmol()) == (2, 0)


def test_supplied_zero_is_fixed_and_represented_hydrogen_is_not_removed():
    mol = Molecule()
    mol.create_atom(atomic_number=6, implicit_hydrogens=0)
    mol.create_atom(atomic_number=1, implicit_hydrogens=0)
    mol.add_bond(0, 1, 1.)
    projected = conversion.to_rdmol(mol, sanitize=True)
    assert _rd_counts(projected) == (0, 0)
    assert projected.GetNumAtoms() == 2
    assert projected.GetAtomWithIdx(0).GetNoImplicit()
    assert _ob_counts(mol.to_obmol()) == (0, 0)



def test_supplied_nonintegral_scalar_bytes_remain_unchanged_by_physical_exports():
    mol = Molecule()
    mol.create_atom(atomic_number=6, implicit_hydrogens=2.75)
    mol.create_atom(atomic_number=1, implicit_hydrogens=0)
    mol.add_bond(0, 1, 1.)
    before = mol._atom_table.column('implicit_hydrogens').tobytes()
    assert _rd_counts(mol.to_rdmol()) == (1, 0)
    assert _ob_counts(mol.to_obmol()) == (1, 0)
    assert _projected(mol).current_implicit_hydrogens.tolist() == [1, 0]
    assert mol._atom_table.column('implicit_hydrogens').tobytes() == before

def test_registered_current_count_mismatch_fails_final_rdkit_inventory_and_native_packing():
    mol = conversion.from_rdmol(Chem.MolFromSmiles('C'), Molecule())
    payload = get_stereo_payload(mol)
    mol.atoms[0].implicit_hydrogens = 2
    for operation in (mol.to_rdmol, lambda: _projected(mol)):
        with pytest.raises(ValueError, match=r'RDKIT.HYDROGEN_INVENTORY.*row 0.*requested 2'):
            operation()
    assert _ob_counts(mol.to_obmol()) == (2,)
    assert get_stereo_payload(mol) == payload


def test_current_bracket_conflict_is_shared_by_all_physical_exports():
    mol = conversion.from_rdmol(Chem.MolFromSmiles('[NH4+]'), Molecule())
    mol.atoms[0].implicit_hydrogens = 2
    payload = get_stereo_payload(mol)
    for operation in (mol.to_rdmol, mol.to_obmol, lambda: _projected(mol)):
        with pytest.raises(ValueError, match=r'HYDROGEN.DECLARATION_CONFLICT.*T=2.*E=0.*J=4'):
            operation()
    assert mol.atoms[0].implicit_hydrogens == 2
    assert get_stereo_payload(mol) == payload


def test_missing_current_implicit_output_is_expected_only_in_native_projection():
    mol = conversion.from_rdmol(Chem.MolFromSmiles('C'), Molecule())
    payload = get_stereo_payload(mol)
    current = replace(payload.atom_facts[0].current_hydrogens, implicit_hydrogens=None)
    payload = replace(payload, atom_facts=(replace(payload.atom_facts[0], current_hydrogens=current),))
    set_stereo_payload(mol, payload)
    assert _rd_counts(mol.to_rdmol()) == (4,)
    packed = _projected(mol)
    assert packed.current_implicit_hydrogens.tolist() == [4]
    assert packed.current_hydrogen_presence_masks.tolist() == [H.IMPLICIT | H.BRACKET_EXPLICIT | H.NO_IMPLICIT]
    assert get_stereo_payload(mol) == payload
    mol.atoms[0].implicit_hydrogens = 2
    with pytest.raises(ValueError, match='RDKIT.HYDROGEN_INVENTORY'):
        mol.to_rdmol()
    # The native verifier receives expected I=2 and must reject its reproduced I=4.
    assert _projected(mol).current_implicit_hydrogens.tolist() == [2]
    assert get_stereo_payload(mol) == payload


def test_current_implicit_output_verification_remains_strict():
    mol = conversion.from_rdmol(Chem.MolFromSmiles('C'), Molecule())
    payload = get_stereo_payload(mol)
    fact = payload.atom_facts[0]
    payload = replace(payload, atom_facts=(replace(fact, current_hydrogens=replace(
        fact.current_hydrogens, implicit_hydrogens=99)),))
    set_stereo_payload(mol, payload)
    with pytest.raises(ValueError, match='cannot reproduce current implicit hydrogen count'):
        mol.to_rdmol()
    assert get_stereo_payload(mol) == payload


def test_arbitrary_source_only_namespace_is_not_registered_as_rdkit_transport():
    mol = Molecule()
    mol.create_atom(atomic_number=6, implicit_hydrogens=2)
    set_stereo_payload(mol, StereoPayload(1, (), (
        AtomTransportRecord(0, 13, None, None, None, 'unregistered-toolkit'),), ()))
    with pytest.raises(ValueError, match='cannot represent atom transport namespace'):
        mol.to_rdmol()



@pytest.mark.parametrize('numbers,graph,targets', _CASES)
@pytest.mark.parametrize('toolkit', ['rdkit', 'openbabel'])
def test_toolkit_exports_after_complete_and_repeated_materialization(numbers, graph, targets, toolkit):
    if toolkit == 'rdkit':
        mol = conversion.from_rdmol(_rd_graph(numbers, graph), Molecule())
    else:
        mol = conversion.obmol2mol(_ob_graph(numbers, graph, targets), Molecule())
    before = get_stereo_payload(mol)
    expected_atoms = len(numbers) + sum(targets)
    for _ in range(2):
        mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(123))
        assert len(mol.atoms) == expected_atoms
        assert tuple(atom.implicit_hydrogens for atom in mol.atoms[:len(numbers)]) == targets
        assert _rd_counts(conversion.to_rdmol(mol, sanitize=True)) == (0,) * expected_atoms
        assert _ob_counts(mol.to_obmol()) == (0,) * expected_atoms
        packed = _projected(mol)
        assert not np.any(packed.current_implicit_hydrogens)
        assert not np.any(packed.current_bracket_explicit_hydrogens)
        after = get_stereo_payload(mol)
        assert tuple(fact.source_hydrogens for fact in after.atom_facts) == tuple(
            fact.source_hydrogens for fact in before.atom_facts)


@pytest.mark.parametrize('toolkit,bracket', [('rdkit', False), ('rdkit', True), ('openbabel', False)])
def test_toolkit_exports_after_partial_declared_materialization(toolkit, bracket):
    if toolkit == 'openbabel':
        mol = conversion.obmol2mol(_ob_graph((6,), (0,), (4,)), Molecule())
    else:
        source = Chem.MolFromSmiles('[CH4]' if bracket else 'C')
        mol = conversion.from_rdmol(source, Molecule())
    before = get_stereo_payload(mol)
    assert len(mol.atoms[0].add_hydrogen(1)) == 1
    assert len(mol.atoms) == 2
    assert mol.atoms[0].implicit_hydrogens == 4
    assert _rd_counts(conversion.to_rdmol(mol, sanitize=True)) == (3, 0)
    assert _ob_counts(mol.to_obmol()) == (3, 0)
    packed = _projected(mol)
    assert (packed.current_implicit_hydrogens + packed.current_bracket_explicit_hydrogens).tolist() == [3, 0]
    assert tuple(fact.source_hydrogens for fact in get_stereo_payload(mol).atom_facts) == tuple(
        fact.source_hydrogens for fact in before.atom_facts)

@pytest.mark.parametrize('source_namespace', [None, 'openbabel'])
def test_rebuilt_native_accepts_isotope_source_only_fallback(source_namespace):
    from hotpot.cheminfo import rdWrappers
    from hotpot.cheminfo.rdWrappers import native

    mol = Molecule()
    mol.create_atom(atomic_number=6, implicit_hydrogens=4)
    set_stereo_payload(mol, StereoPayload(1, (), (
        AtomTransportRecord(0, 13, None, None, HydrogenTransportState(4, None, None), source_namespace),), ()))
    result = native.generate(mol, rdWrappers.ETKDGOptions(seed=123), None)
    assert result.status is rdWrappers.GenerationStatus.SUCCESS
    assert result.coordinates.shape == (1, 1, 3)
    assert np.isfinite(result.coordinates).all()


def test_rebuilt_native_rejects_reproduced_implicit_count_when_expected_output_was_absent():
    from hotpot.cheminfo import rdWrappers
    from hotpot.cheminfo.rdWrappers import native

    mol = conversion.from_rdmol(Chem.MolFromSmiles('C'), Molecule())
    payload = get_stereo_payload(mol)
    current = replace(payload.atom_facts[0].current_hydrogens, implicit_hydrogens=None)
    set_stereo_payload(mol, replace(payload, atom_facts=(replace(payload.atom_facts[0], current_hydrogens=current),)))
    mol.atoms[0].implicit_hydrogens = 2
    result = native.generate(mol, rdWrappers.ETKDGOptions(seed=123), None)
    assert result.status is rdWrappers.GenerationStatus.UNSUPPORTED
    assert 'current implicit-hydrogen state' in result.diagnostic


def test_rebuilt_native_rejects_unregistered_source_only_namespace():
    from hotpot.cheminfo import rdWrappers
    from hotpot.cheminfo.rdWrappers import native

    mol = Molecule()
    mol.create_atom(atomic_number=6, implicit_hydrogens=4)
    set_stereo_payload(mol, StereoPayload(1, (), (
        AtomTransportRecord(0, 13, None, None, None, 'unregistered-toolkit'),), ()))
    result = native.generate(mol, rdWrappers.ETKDGOptions(seed=123), None)
    assert result.status is rdWrappers.GenerationStatus.UNSUPPORTED
    assert 'source_namespace' in result.diagnostic
