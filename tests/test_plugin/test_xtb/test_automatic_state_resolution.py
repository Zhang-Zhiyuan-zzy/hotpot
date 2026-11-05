"""One automatic source/profile route with method-specific, process-free projection."""

from dataclasses import replace
from pathlib import Path

import hotpot
import numpy as np
import pytest

from hotpot.cheminfo.operations.calculation.scientific_state import profiles
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.cheminfo.operations.calculation.scientific_state.profile_decision import _candidate_index, _molecular_two_s
from hotpot.cheminfo.stereo import get_stereo_payload, set_stereo_payload
from hotpot.plugins._harness.contracts import NativeProcessResult
from hotpot.plugins.xtb import cli, stage, state_resolution
from hotpot.plugins.xtb.stream import (
    XTBStreamMetadata, XTBStreamProvenance, XTBStreamRecord, read_sdf_records, write_sdf_records,
)
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import ScientificStateIdentity
from hotpot.plugins.xtb.contracts import XTBBackendInfo, XTBMethod, XTBTask
from tests.test_cheminfo.calculator.test_automatic_scientific_state import (
    candidate_profile, _molecule,
)


@pytest.fixture
def probe(monkeypatch):
    info = XTBBackendInfo(
        Path("/not-executed/xtb"), "6.7.1", None, "0" * 64,
        NativeProcessResult((), Path("/tmp"), 0, "stubbed capability evidence", "", 0.0),
        86, 86,
    )
    monkeypatch.setattr(state_resolution, "probe_xtb_backend", lambda *args, **kwargs: info)
    return info


def _resolve(mol, method=XTBMethod.GFN2_XTB, **kwargs):
    return state_resolution.resolve_xtb_state(
        mol, method, executable=None, environment={},
        charge=kwargs.pop("charge", None), uhf=kwargs.pop("uhf", None),
        policy=kwargs.pop("policy", None), **kwargs,
    )


def _two_s(bundle):
    path = next(path for path in bundle.candidate_space.expanded_paths
                if path.identity == bundle.execution_decision.selected_state_identity)
    return _molecular_two_s(path, _candidate_index(bundle.candidate_space.factors))


def test_automatic_euiii_retains_physical_spin_and_projects_active_uhf_zero(candidate_profile, probe):
    result = _resolve(_molecule())

    assert result.projection.charge == 3
    assert result.projection.uhf == 0
    assert result.projection.multiplicity is None
    assert _two_s(result.scientific_state) == 6
    assert result.scientific_state.execution_decision.static_profile_eligibility.profile_id == "coordination_screening_v1"


def test_gfnff_then_gfn2_reuses_same_scientific_bundle(candidate_profile, probe):
    mol = _molecule()
    first = _resolve(mol, XTBMethod.GFNFF)
    second = _resolve(mol, carried_state=first.scientific_state)

    assert first.projection.uhf is None
    assert second.projection.uhf == 0
    assert first.scientific_state is second.scientific_state
    assert first.projection.scientific_state_identity == second.projection.scientific_state_identity


def test_coordinate_only_change_preserves_carried_bundle(candidate_profile, probe):
    mol = _molecule()
    mol.coordinates = np.asarray(((0.0, 0.0, 0.0), (2.1, 0.0, 0.0)))
    first = _resolve(mol)
    mol.coordinates = mol.coordinates + 0.1
    second = _resolve(mol, carried_state=first.scientific_state)

    assert first.scientific_state is second.scientific_state
    assert first.projection.scientific_state_identity == second.projection.scientific_state_identity


@pytest.mark.parametrize("change", ("charge", "spin", "formal_charge_source", "radical_presence"))
def test_changed_supplied_state_requires_resolution_despite_same_structural_identity(candidate_profile, probe, change):
    mol = _molecule("[CH2]CO" if change == "radical_presence" else "O")
    first = _resolve(mol)
    identity = extract_molecular_facts(mol).identity
    if change == "charge":
        mol.charge = 2
    elif change == "spin":
        mol.spin_mult = 3
    elif change == "formal_charge_source":
        mol.atoms[0].formal_charge = 0
    else:
        payload = get_stereo_payload(mol)
        changed_facts = tuple(replace(fact, radical_electrons=None) if fact.atom == 0 else fact
                              for fact in payload.atom_facts)
        set_stereo_payload(mol, replace(payload, atom_facts=changed_facts))
    assert extract_molecular_facts(mol).identity == identity
    second = _resolve(mol, carried_state=first.scientific_state)

    assert second.scientific_state is not first.scientific_state
    assert first.projection.scientific_state_identity != second.projection.scientific_state_identity
    if change == "charge":
        assert second.projection.charge == 2
        assert _two_s(second.scientific_state) == 7
    if change == "spin":
        assert _two_s(second.scientific_state) == 2


def test_ordinary_input_uses_existing_strict_route_with_supplied_zero_charge(probe):
    result = _resolve(hotpot.read_mol("O", "smi"))

    assert result.projection.charge == 0
    assert result.projection.uhf == 0
    assert result.scientific_state.execution_decision.static_profile_eligibility.profile_id == "strict"


def test_explicit_backend_uhf_overrides_active_default_with_legal_parity(candidate_profile, probe):
    result = _resolve(_molecule(), uhf=2)

    assert result.projection.uhf == 2
    assert _two_s(result.scientific_state) == 6


def test_illegal_explicit_backend_uhf_remains_typed_local_failure(candidate_profile, probe):
    with pytest.raises(state_resolution.XTBStateResolutionError) as caught:
        _resolve(_molecule(), uhf=1)

    assert caught.value.sample_failure_code == "XTB-STATE-UNAVAILABLE"
    assert any("PARITY" in finding.code for finding in caught.value.projection_result.findings)
    assert _two_s(caught.value.scientific_state) == 6


def test_unsupported_element_is_not_made_supported_by_periodic_default(candidate_profile, probe):
    with pytest.raises(state_resolution.XTBStateResolutionError) as caught:
        _resolve(_molecule(metal=95))

    assert caught.value.projection_result.status.value == "backend_unsupported"
    assert any(finding.code == "HARD-008.UNSUPPORTED_ELEMENT" for finding in caught.value.projection_result.findings)


def test_cli_and_pipeline_default_to_context_selection_without_overriding_explicit_policy():
    assert cli.build_parser().parse_args(("-", "--input-format", "sdf")).state_policy is None
    assert stage._stage_parser().parse_args(()).state_policy is None
    assert stage._stage_parser().parse_args(("--state-policy", "strict")).state_policy == "strict"


def test_changed_profile_artifact_cannot_reuse_carried_bundle(candidate_profile, probe, monkeypatch):
    mol = _molecule()
    first = _resolve(mol)
    encoded, payload = profiles._profile_payload()
    monkeypatch.setattr(profiles, "_profile_payload", lambda: (encoded + b"\n", payload))
    profiles._coordination_screening_profile.cache_clear()
    second = _resolve(mol, carried_state=first.scientific_state)

    assert second.scientific_state is not first.scientific_state
    first_binding = next(finding.related_identities for result in first.scientific_state.net_charge_results
                         for finding in result.findings if finding.code == "SOURCE.MOLECULE_STATE_BINDING")
    second_binding = next(finding.related_identities for result in second.scientific_state.net_charge_results
                          for finding in result.findings if finding.code == "SOURCE.MOLECULE_STATE_BINDING")
    assert first_binding[0] == second_binding[0]
    assert first_binding[1] != second_binding[1]


def test_bundle_without_current_source_binding_is_resolved_again(candidate_profile, probe):
    mol = _molecule()
    first = _resolve(mol)
    unbound = replace(first.scientific_state, net_charge_results=tuple(
        replace(result, findings=tuple(finding for finding in result.findings
                                       if finding.code != "SOURCE.MOLECULE_STATE_BINDING"))
        for result in first.scientific_state.net_charge_results
    ))
    second = _resolve(mol, carried_state=unbound)

    assert second.scientific_state is not unbound
    assert second.projection.scientific_state_identity == first.projection.scientific_state_identity


def _stream_metadata(resolved, method):
    projection = resolved.projection
    return XTBStreamMetadata(
        projection.charge, projection.uhf, method, -1.0,
        XTBStreamProvenance("6.7.1", None, "0" * 64,
                            projection.scientific_state_identity, projection.facts_identity,
                            projection.projection_convention),
    )


def _cli_options():
    return cli._RunOptions(XTBMethod.GFN2_XTB, XTBTask.SINGLEPOINT, None, None,
                           None, "sdf", None, None, None, False, "off")


@pytest.mark.parametrize("original_charge", (None, 2, 3))
@pytest.mark.parametrize("previous_method", (XTBMethod.GFNFF, XTBMethod.GFN2_XTB))
def test_cli_stream_replay_preserves_source_or_explicit_charge_authority(
        candidate_profile, probe, monkeypatch, original_charge, previous_method):
    mol = _molecule()
    original = _resolve(mol, previous_method, charge=original_charge)
    metadata = _stream_metadata(original, previous_method)
    restored = read_sdf_records(write_sdf_records((XTBStreamRecord(mol, metadata),)))[0]
    input_record = cli._InputRecord(0, restored.mol, restored.metadata)
    calls, replay_charges = [], []

    def resolve(*args, **kwargs):
        replay_charges.append(kwargs["charge"])
        return state_resolution.resolve_xtb_state(*args, **kwargs)

    def workflow(molecule, **kwargs):
        calls.append((molecule, kwargs["state"]))
        return kwargs["state"]

    monkeypatch.setattr(cli, "resolve_xtb_state", resolve)
    monkeypatch.setattr(cli, "run_gfn_xtb", workflow)
    projected = cli._run_xtb_record(input_record, _cli_options())

    assert replay_charges == ([None] if original_charge is None else [None, original_charge])
    assert len(calls) == 1
    assert calls[0][0] is restored.mol
    assert projected.scientific_state_identity == original.projection.scientific_state_identity
    assert projected.charge == (3 if original_charge is None else original_charge)


def test_cli_stream_cannot_accept_a_different_scientific_identity(candidate_profile, probe, monkeypatch):
    mol = _molecule()
    original = _resolve(mol)
    metadata = _stream_metadata(original, XTBMethod.GFN2_XTB)
    metadata = replace(metadata, provenance=replace(
        metadata.provenance, scientific_state_identity=ScientificStateIdentity("f" * 64),
    ))
    record = read_sdf_records(write_sdf_records((XTBStreamRecord(mol, metadata),)))[0]
    calls = []
    monkeypatch.setattr(cli, "run_gfn_xtb", lambda *args, **kwargs: calls.append(kwargs))

    with pytest.raises(state_resolution.XTBInputError, match="does not match input stream provenance"):
        cli._run_xtb_record(cli._InputRecord(0, record.mol, record.metadata), _cli_options())

    assert calls == []


def test_cli_gfnff_does_not_promote_inherited_gfn_uhf_to_a_request(candidate_profile, probe, monkeypatch):
    mol = _molecule()
    original = _resolve(mol, uhf=2)
    metadata = _stream_metadata(original, XTBMethod.GFN2_XTB)
    restored = read_sdf_records(write_sdf_records((XTBStreamRecord(mol, metadata),)))[0]
    options = replace(_cli_options(), method=XTBMethod.GFNFF)
    calls = []

    def workflow(molecule, **kwargs):
        calls.append(kwargs["state"])
        return kwargs["state"]

    monkeypatch.setattr(cli, "run_gfnff", workflow)
    assert cli._state_values(cli._InputRecord(0, restored.mol, restored.metadata), options) == (None, None)
    projected = cli._run_xtb_record(cli._InputRecord(0, restored.mol, restored.metadata), options)

    assert len(calls) == 1
    assert projected.uhf is None
    assert projected.scientific_state_identity == original.projection.scientific_state_identity
    assert restored.metadata.unpaired_electrons == 2
