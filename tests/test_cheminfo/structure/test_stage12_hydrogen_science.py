"""Independent physical H/electron accounting across graph representations."""

import pickle
from types import SimpleNamespace

import pytest
from rdkit import Chem

from hotpot import Molecule
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.cheminfo.operations.conversion import from_rdmol, obmol2mol
from hotpot.cheminfo.rules.chemistry.scientific_state.charge.lewis import _bond_valences, solve_main_group_lewis
from hotpot.cheminfo.rules.chemistry.scientific_state.charge.net_charge import resolve_net_charge
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateProvenance, CoverageStatus, ScientificAdmissibility, StateValueAuthority,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.legality import _validate_oxidation_state_evidence
from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts import (
    CountingConvention, OxidationStateCandidateEvidence,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.anionic_x_ligands import _implicit_hydrogens


def _methane(graph_h, origin):
    if origin == 'inferred':
        mol = Molecule()
        mol.create_atom(atomic_number=6, formal_charge=0)
        for _ in range(graph_h):
            atom = mol.create_atom(atomic_number=1, formal_charge=0)
            mol.add_bond(0, atom.idx, 1.)
        return mol
    if origin == 'rdkit':
        native = Chem.RWMol()
        native.AddAtom(Chem.Atom(6))
        for _ in range(graph_h):
            index = native.AddAtom(Chem.Atom(1))
            native.AddBond(0, index, Chem.BondType.SINGLE)
        Chem.SanitizeMol(native)
        return from_rdmol(native, Molecule())
    from hotpot.cheminfo.adapters.openbabel._runtime import ob
    native = ob.OBMol()
    native.BeginModify()
    native.NewAtom().SetAtomicNum(6)
    for _ in range(graph_h):
        atom = native.NewAtom()
        atom.SetAtomicNum(1)
        native.AddBond(1, atom.GetIdx(), 1)
    native.EndModify()
    native.GetAtom(1).SetImplicitHCount(4 - graph_h)
    return obmol2mol(native, Molecule())


@pytest.mark.parametrize('origin', ['inferred', 'rdkit', 'openbabel'])
@pytest.mark.parametrize('graph_h', [0, 1, 4])
def test_methane_has_ten_nuclear_charges_and_four_carbon_bond_valences(origin, graph_h):
    mol = _methane(graph_h, origin)
    before = pickle.dumps(mol._capture_factual_record()[0])
    facts = extract_molecular_facts(mol)
    assert facts.hydrogen_inventory.explicit_counts[0] == graph_h
    assert facts.hydrogen_inventory.implicit_counts[0] == 4
    assert facts.nuclear_charge_sum == 10
    assert _bond_valences(facts) == (4,) + (1,) * graph_h
    assert _implicit_hydrogens(SimpleNamespace(facts=facts), 0) == 4 - graph_h
    charge = resolve_net_charge(facts, StateRequest(net_charge=0))[0]
    result = solve_main_group_lewis(facts, charge)[0]
    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.candidates[0].atom_formal_charges == (0,) * (1 + graph_h)
    assert pickle.dumps(mol._capture_factual_record()[0]) == before


@pytest.mark.parametrize('graph_h,expected', [(0, False), (1, False), (4, True)])
def test_complete_oxidation_states_require_every_non_graph_proton_to_be_covered(graph_h, expected):
    facts = extract_molecular_facts(_methane(graph_h, 'inferred'))
    candidate = OxidationStateCandidateEvidence(
        provenance=CandidateProvenance('h-count', StateValueAuthority.VERIFIED_EXPLICIT,
                                       source_identity='hand-counted-methane'),
        convention=CountingConvention.IONIC,
        atom_states=(-4,) + (1,) * graph_h,
        coverage=CoverageStatus.COMPLETE,
    )
    assessment = _validate_oxidation_state_evidence(facts, candidate)
    assert (assessment.admissibility is ScientificAdmissibility.ADMISSIBLE) is expected
    if not expected:
        assert assessment.findings[0].code == 'HARD-004.OS_IMPLICIT_HYDROGEN_UNCOVERED'


def test_target_below_graph_count_preserves_real_protons_and_supplied_target():
    mol = _methane(4, 'inferred')
    mol.atoms[0].implicit_hydrogens = 0
    facts = extract_molecular_facts(mol)
    assert facts.hydrogen_inventory.implicit_counts[0] == 0
    assert facts.nuclear_charge_sum == 10
    assert _bond_valences(facts) == (4, 1, 1, 1, 1)


@pytest.mark.parametrize('numbers,edges,expected', [
    ((1,), (), 1), ((1, 1), ((0, 1),), 2), ((30, 1), ((0, 1),), 31),
])
def test_real_hydrogen_and_metal_hydride_count_graph_vertices_once(numbers, edges, expected):
    mol = Molecule()
    for number in numbers:
        mol.create_atom(atomic_number=number)
    for first, second in edges:
        mol.add_bond(first, second, 1.)
    facts = extract_molecular_facts(mol)
    assert facts.nuclear_charge_sum == expected


def test_actual_bracket_target_conflict_cannot_become_a_successful_scientific_snapshot():
    mol = from_rdmol(Chem.MolFromSmiles('[CH4]'), Molecule())
    mol.atoms[0].implicit_hydrogens = 2
    before = pickle.dumps(mol._capture_factual_record()[0])
    with pytest.raises(ValueError, match='HYDROGEN.DECLARATION_CONFLICT'):
        extract_molecular_facts(mol)
    assert pickle.dumps(mol._capture_factual_record()[0]) == before
