"""Named H graph edits, unchanged source ledgers and current persistence."""

from dataclasses import fields
import pickle

import numpy as np
import pytest
from rdkit import Chem

from hotpot import Molecule
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.operations.conversion import from_rdmol
from hotpot.cheminfo.operations.conformers.rdkit import _materialized_hydrogen_projection
from hotpot.cheminfo.structure._builder import _restore_molecule_record
from hotpot.cheminfo.structure._stereo import _get_stereo_payload


def _physical_rd_h(mol):
    rd = mol.to_rdmol()
    rd.UpdatePropertyCache(strict=False)
    return sum(atom.GetAtomicNum() == 1 for atom in rd.GetAtoms()) + sum(
        atom.GetNumImplicitHs() + atom.GetNumExplicitHs() for atom in rd.GetAtoms()
    )


def test_import_materialize_remove_readd_uses_current_target_without_resurrecting_history():
    mol = from_rdmol(Chem.MolFromSmiles('C'), Molecule())
    history = _get_stereo_payload(mol).atom_facts[0].source_hydrogens
    mol.add_hydrogens(rng=np.random.default_rng(42))
    assert len(mol.hydrogens) == 4 and _physical_rd_h(mol) == 4
    mol.remove_hydrogens()
    fact = _get_stereo_payload(mol).atom_facts[0]
    assert fact.current_hydrogens.implicit_hydrogens is None
    assert fact.current_hydrogens.bracket_explicit_hydrogens == 0
    assert fact.source_hydrogens == history
    assert mol.atoms[0].implicit_hydrogens == 4
    assert len(mol.hydrogens) == 0 and _physical_rd_h(mol) == 4
    mol.add_hydrogens(rng=np.random.default_rng(43))
    assert len(mol.hydrogens) == 4 and _physical_rd_h(mol) == 4
    assert _get_stereo_payload(mol).atom_facts[0].source_hydrogens == history


def test_remove_hydrogens_late_current_transport_failure_restores_graph_and_facts(monkeypatch):
    mol = from_rdmol(Chem.MolFromSmiles('C'), Molecule())
    mol.add_hydrogens(rng=np.random.default_rng(42))
    before = pickle.dumps(mol._capture_factual_record()[0])
    members = tuple(mol.atoms)
    def fail(*args, **kwargs):
        raise RuntimeError('injected removal publication failure')
    monkeypatch.setattr(chemistry, '_set_stereo_payload', fail)
    with pytest.raises(RuntimeError, match='injected removal publication failure'):
        mol.remove_hydrogens()
    assert pickle.dumps(mol._capture_factual_record()[0]) == before
    assert all(first is second for first, second in zip(members, mol.atoms))


@pytest.mark.parametrize('lifecycle', ['copy', 'pickle', 'component', 'merge'])
def test_normalized_partial_import_lifecycle_preserves_source_and_four_physical_h(lifecycle):
    rd = Chem.RWMol(Chem.MolFromSmiles('C'))
    index = rd.AddAtom(Chem.Atom(1))
    rd.AddBond(0, index, Chem.BondType.SINGLE)
    rd.UpdatePropertyCache(strict=True)
    mol = from_rdmol(rd, Molecule())
    history = _get_stereo_payload(mol).atom_facts[0].source_hydrogens
    if lifecycle == 'copy':
        clone = mol.copy()
    elif lifecycle == 'pickle':
        clone = pickle.loads(pickle.dumps(mol))
    elif lifecycle == 'component':
        clone = mol.components[0]
    else:
        clone = Molecule()
        clone.add_component(mol)
    assert clone.atoms[0].implicit_hydrogens == 4
    assert _physical_rd_h(clone) == 4
    assert _get_stereo_payload(clone).atom_facts[0].source_hydrogens == history


def test_current_private_record_generation_rejects_ambiguous_pre_normalization_pickle():
    record = Molecule()._capture_factual_record()[0]
    restore, (kind, generation, values) = record.__reduce__()
    assert restore is _restore_molecule_record and generation == 7
    assert len(fields(record)) == len(values) == 24
    with pytest.raises(ValueError, match='schema version 7 and 24 fields'):
        restore(kind, 5, values)
    assert pickle.dumps(restore(kind, generation, values)) == pickle.dumps(record)


def test_conformer_projection_still_exports_only_materialized_graph_h():
    mol = from_rdmol(Chem.MolFromSmiles('C'), Molecule())
    mol.add_hydrogens(rng=np.random.default_rng(42))
    before = pickle.dumps(mol._capture_factual_record()[0])
    projected = _materialized_hydrogen_projection(mol)
    assert projected.atoms[0].implicit_hydrogens == 0
    assert _physical_rd_h(projected) == 4
    assert pickle.dumps(mol._capture_factual_record()[0]) == before
