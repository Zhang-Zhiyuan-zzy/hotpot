"""Independent physical-H regressions for scientific and graph-only consumers."""

import pickle

import numpy as np
import pytest
from rdkit import Chem

from hotpot import Molecule, read_mol
from hotpot.cheminfo.operations import conversion
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import StatePolicy
from hotpot.cheminfo.operations.calculation.scientific_state.defaults import _path_resolves_oxidation_state
from hotpot.cheminfo.operations.calculation.scientific_state.composition.derive import (
    _charge_from_complete_oxidation_state, _has_implicit_hydrogen, _ionic_component_charge,
)
from hotpot.cheminfo.operations.calculation.scientific_state.composition.expand import _merge_oxidation_states
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateProvenance, CoverageStatus, DEFAULT_CANDIDATE_BUDGET,
    ScientificQuantity, StateValueAuthority,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import (
    CandidateDomainScope, CandidateFactor, CandidateSearchStatus,
    CoupledScientificStateCandidate, FactorCandidateValue, FactorKey,
    MaterializationStatus, ScientificStateCandidateSpace, ScientificStateIdentity,
    StateValueOrigin, StateValueRef,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts import (
    CoordinationCountingCandidate, CountingConvention, OxidationStateCandidateEvidence,
)
from hotpot.cheminfo.operations.conformers.rdkit import _materialized_hydrogen_projection
from hotpot.cheminfo.rdWrappers import ETKDGOptions, GenerationStatus, native
from hotpot.cheminfo.adapters.native.molecule_buffers import NativeHydrogenTransportFieldCode as H
from hotpot.cheminfo.stereo import (
    AtomTransportRecord, HydrogenTransportState, StereoPayload,
    get_stereo_payload, set_stereo_payload,
)
from hotpot.plugins.xtb.adapter import molecule_to_xtb_geometry, prepare_xtb_input
from hotpot.plugins.xtb.contracts import XTBInputError, XTBMethod
from hotpot.plugins.xtb.stream import (
    XTBStreamError, XTBStreamMetadata, XTBStreamProvenance, _validate_metadata,
)


def _methane(graph_count):
    mol = Molecule()
    mol.create_atom(atomic_number=6, implicit_hydrogens=4, coordinates=(0., 0., 0.))
    for index in range(graph_count):
        atom = mol.create_atom(atomic_number=1, implicit_hydrogens=0,
                               coordinates=(float(index + 1), 0., 0.))
        mol.add_bond(0, atom.idx, 1.)
    return mol


def _provenance(identity):
    return CandidateProvenance(identity, StateValueAuthority.DERIVED_CONSTRAINT,
                               'literal-methane-oxidation-fixture')


def _oxidation_candidate(identity, states):
    return OxidationStateCandidateEvidence(_provenance(identity), CountingConvention.IONIC,
                                          states, CoverageStatus.PARTIAL)


def _metadata(unpaired, charge=0):
    return XTBStreamMetadata(charge, unpaired, XTBMethod.GFN2_XTB, -1.0,
                            XTBStreamProvenance('literal-review-version', None, '0' * 64))


def _space_and_path(candidate):
    factor = CandidateFactor('literal-oxidation-factor',
        FactorKey(ScientificQuantity.OXIDATION_STATE, 'literal-methane'), 'literal-domain',
        (FactorCandidateValue(candidate, ('literal-proposal',), ()),), (), None,
        CandidateDomainScope.EXHAUSTIVE, MaterializationStatus.COMPLETE)
    path = CoupledScientificStateCandidate(ScientificStateIdentity('literal-path'),
        StateValueRef(ScientificQuantity.NET_CHARGE, 0, StateValueOrigin.RESULT, 'neutral'),
        (), (), candidate.provenance.identity, 'literal-electronic-result', None,
        (), (candidate.provenance.identity,), ())
    space = ScientificStateCandidateSpace((factor,), (), DEFAULT_CANDIDATE_BUDGET,
        'literal-budget', 'literal-frontier', (path,), (), (), CandidateSearchStatus.DECISION_CERTIFIED)
    return space, path


@pytest.mark.parametrize('graph_count', [0, 1, 4])
def test_coverage_and_charge_leave_every_unrepresented_methane_hydrogen_uncovered(graph_count):
    facts = extract_molecular_facts(_methane(graph_count))
    assert facts.hydrogen_inventory.explicit_counts == (graph_count,) + (0,) * graph_count
    assert facts.hydrogen_inventory.implicit_counts == (4,) + (0,) * graph_count
    assert _has_implicit_hydrogen(facts, tuple(range(graph_count + 1))) is (graph_count < 4)
    first = _oxidation_candidate('carbon-state', (-4,) + (None,) * graph_count)
    second = _oxidation_candidate('hydrogen-states', (None,) + (1,) * graph_count)
    merged, rejection = _merge_oxidation_states(facts, (first, second))
    assert rejection is None
    assert merged.atom_states == (-4,) + (1,) * graph_count
    assert merged.coverage is (CoverageStatus.PARTIAL if graph_count < 4 else CoverageStatus.COMPLETE)
    charge = _charge_from_complete_oxidation_state(facts, merged)
    if graph_count < 4:
        assert charge is None
    else:
        assert charge.value == 0
    counting = CoordinationCountingCandidate(_provenance('ionic-counting'), CountingConvention.IONIC, ())
    ionic = _ionic_component_charge(facts, merged, (counting,))
    if graph_count < 4:
        assert ionic is None
    else:
        assert ionic.value == 0


@pytest.mark.parametrize('graph_count', [0, 1, 4])
def test_strict_default_resolution_requires_all_non_graph_hydrogen_oxidation_states(graph_count):
    facts = extract_molecular_facts(_methane(graph_count))
    candidate = _oxidation_candidate('all-graph-states', (-4,) + (1,) * graph_count)
    space, path = _space_and_path(candidate)
    assert _path_resolves_oxidation_state(path, StatePolicy.STRICT, space, facts) is (graph_count == 4)


@pytest.mark.parametrize('graph_count', [0, 1, 4])
@pytest.mark.parametrize('charge,valid_unpaired', [(0, 0), (1, 1)])
def test_xtb_stream_counts_methane_electrons_independent_of_graph_representation(graph_count, charge, valid_unpaired):
    mol = _methane(graph_count)
    before = pickle.dumps(mol._capture_factual_record(), protocol=5)
    _validate_metadata(mol, _metadata(valid_unpaired, charge))
    with pytest.raises(XTBStreamError, match='electron parity'):
        _validate_metadata(mol, _metadata(1 - valid_unpaired, charge))
    assert pickle.dumps(mol._capture_factual_record(), protocol=5) == before


def test_xtb_graph_geometry_rejects_remaining_supplied_metal_hydrogen():
    mol = Molecule()
    mol.create_atom(atomic_number=30, implicit_hydrogens=1, coordinates=(0., 0., 0.))
    mol.create_atom(atomic_number=1, implicit_hydrogens=0, coordinates=(1., 0., 0.))
    mol.add_bond(0, 1, 1.)
    with pytest.raises(XTBInputError, match='1 non-graph hydrogen') as caught:
        molecule_to_xtb_geometry(mol)
    assert caught.value.sample_failure_code == 'XTB-EXPLICIT-H-REQUIRED'
    # Zn30 + one represented H + one remaining H gives32electrons.
    _validate_metadata(mol, _metadata(0))
    with pytest.raises(XTBStreamError, match='electron parity'):
        _validate_metadata(mol, _metadata(1))


def test_prepare_xtb_input_rejects_actual_bracket_count_above_residual_without_mutation(tmp_path):
    mol = conversion.from_rdmol(Chem.MolFromSmiles('[NH4+]'), Molecule())
    mol.atoms[0].implicit_hydrogens = 0
    before = pickle.dumps(mol._capture_factual_record(), protocol=5)
    payload = get_stereo_payload(mol)
    assert payload.atom_facts[0].current_hydrogens.bracket_explicit_hydrogens == 4
    input_path = tmp_path / 'declaration-conflict.xyz'
    with pytest.raises(XTBInputError, match=r'HYDROGEN.DECLARATION_CONFLICT.*T=0.*E=0.*J=4') as caught:
        prepare_xtb_input(mol, input_path)
    assert caught.value.sample_failure_code == 'XTB-EXPLICIT-H-REQUIRED'
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value) == str(caught.value.__cause__)
    assert not input_path.exists()
    assert pickle.dumps(mol._capture_factual_record(), protocol=5) == before
    assert get_stereo_payload(mol) == payload


@pytest.mark.parametrize('target', [0, 1, 2])
def test_xtb_graph_geometry_accepts_lower_or_equal_target_without_deleting_graph_hydrogen(target):
    mol = _methane(2)
    mol.atoms[0].implicit_hydrogens = target
    before = pickle.dumps(mol._capture_factual_record(), protocol=5)
    assert molecule_to_xtb_geometry(mol).symbols == ('C', 'H', 'H')
    assert pickle.dumps(mol._capture_factual_record(), protocol=5) == before


@pytest.mark.parametrize('graph_count', [0, 1, 4])
def test_xtb_graph_geometry_requires_materialized_methane_residual(graph_count):
    mol = _methane(graph_count)
    if graph_count < 4:
        with pytest.raises(XTBInputError, match=f'{4 - graph_count} non-graph hydrogen'):
            molecule_to_xtb_geometry(mol)
    else:
        assert molecule_to_xtb_geometry(mol).symbols == ('C', 'H', 'H', 'H', 'H')


@pytest.mark.parametrize('count,valid_unpaired', [(1, 1), (2, 0)])
def test_xtb_stream_does_not_add_raw_counts_to_real_hydrogen_vertices(count, valid_unpaired):
    mol = Molecule()
    for index in range(count):
        mol.create_atom(atomic_number=1, implicit_hydrogens=42,
                        coordinates=(float(index), 0., 0.))
    if count == 2:
        mol.add_bond(0, 1, 1.)
    _validate_metadata(mol, _metadata(valid_unpaired))
    with pytest.raises(XTBStreamError, match='electron parity'):
        _validate_metadata(mol, _metadata(1 - valid_unpaired))
    assert molecule_to_xtb_geometry(mol).symbols == ('H',) * count
    assert tuple(atom.implicit_hydrogens for atom in mol.atoms) == (42,) * count


@pytest.mark.parametrize('namespace', [None, 'openbabel'])
def test_materialized_canonical_and_openbabel_current_zero_packs_logical_zero_and_runs_native(namespace):
    if namespace is None:
        mol = _methane(4)
        fact = AtomTransportRecord(0, 13, None, None, HydrogenTransportState(4, None, None), None)
        set_stereo_payload(mol, StereoPayload(1, (), (fact,), ()))
    else:
        mol = read_mol('[13CH4]', 'smi')
        mol.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(123))
    original = get_stereo_payload(mol)
    projected = _materialized_hydrogen_projection(mol)
    before = get_stereo_payload(projected)
    assert before.atom_facts[0].source_namespace == namespace
    assert before.atom_facts[0].current_hydrogens == HydrogenTransportState(0, 0, True)
    packed = native.pack_molecule(projected, native._RDKIT_ETKDG_FIELDS).atom_transport
    assert packed.current_hydrogen_presence_masks.tolist() == [H.IMPLICIT] * 5
    assert packed.current_implicit_hydrogens.tolist() == [0] * 5
    assert packed.isotopes[0] == 13
    assert packed.source_implicit_hydrogens[0] == 4
    result = native.generate(projected, ETKDGOptions(seed=123), None)
    assert result.status is GenerationStatus.SUCCESS
    assert result.coordinates.shape == (1, 5, 3)
    assert np.isfinite(result.coordinates).all()
    assert get_stereo_payload(projected) == before
    assert get_stereo_payload(mol) == original


@pytest.mark.parametrize('namespace,residual', [(None, 1), ('openbabel', 1), ('unregistered-toolkit', 0)])
def test_materialized_zero_fallback_is_limited_to_registered_maps_and_zero_residual(namespace, residual):
    mol = Molecule()
    mol.create_atom(atomic_number=6, implicit_hydrogens=residual)
    fact = AtomTransportRecord(0, 13, None, HydrogenTransportState(0, 0, True),
                               HydrogenTransportState(4, None, None), namespace)
    set_stereo_payload(mol, StereoPayload(1, (), (fact,), ()))
    before = get_stereo_payload(mol)
    packed = native.pack_molecule(mol, native._RDKIT_ETKDG_FIELDS).atom_transport
    assert packed.current_hydrogen_presence_masks.tolist() == [H.IMPLICIT | H.BRACKET_EXPLICIT | H.NO_IMPLICIT]
    result = native.generate(mol, ETKDGOptions(seed=123), None)
    assert result.status is GenerationStatus.UNSUPPORTED
    assert get_stereo_payload(mol) == before
