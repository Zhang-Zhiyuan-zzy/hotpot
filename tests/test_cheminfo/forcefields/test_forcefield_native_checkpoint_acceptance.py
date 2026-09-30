"""Acceptance-policy tests for precomputed native bond--ring checkpoints."""

from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.forcefields import acceptance as acceptance_policy
from hotpot.cheminfo.forcefields.native_reports import (
    NativeBondRingFinding,
    NativeRingCheckpointReport,
    NativeRingGraphScope,
)


def _carbon_bond() -> Molecule:
    mol = Molecule()
    mol.create_atom(atomic_number=6, coordinates=(0.0, 0.0, 0.0))
    mol.create_atom(atomic_number=6, coordinates=(1.52, 0.0, 0.0))
    mol.add_bond(0, 1, bond_order=1.0)
    mol.refresh_atom_id()
    return mol


def _native_checkpoint(
    *,
    findings: tuple[NativeBondRingFinding, ...] = (),
    excluded_ring_count: int = 0,
) -> NativeRingCheckpointReport:
    piercing_count = sum(
        finding.state is geo.PiercingState.PIERCES for finding in findings
    )
    undetermined_count = sum(
        finding.state is geo.PiercingState.UNDETERMINED
        for finding in findings
    )
    return NativeRingCheckpointReport(
        state=(
            geo.PiercingState.PIERCES
            if piercing_count
            else (
                geo.PiercingState.UNDETERMINED
                if undetermined_count
                else geo.PiercingState.DOES_NOT_PIERCE
            )
        ),
        scope=NativeRingGraphScope.FULL_GRAPH,
        maximum_actionable_ring_size=16,
        maximum_relevant_cycle_count=10000,
        relevant_cycle_count=len(findings) + excluded_ring_count,
        selected_ring_count=len(findings),
        excluded_ring_count=excluded_ring_count,
        active_bond_count=1,
        candidate_pair_count=len(findings),
        aabb_separated_pair_count=0,
        exact_pair_count=len(findings),
        piercing_pair_count=piercing_count,
        does_not_pierce_pair_count=0,
        undetermined_pair_count=undetermined_count,
        scan_complete=not undetermined_count,
        actionable_findings=findings,
    )


def test_native_checkpoint_acceptance_does_not_rescan_geometry(monkeypatch):
    def unexpected_scan(*args, **kwargs):
        raise AssertionError("native checkpoint acceptance must not rescan")

    monkeypatch.setattr(
        acceptance_policy.geo,
        "screen_bond_ring_relations",
        unexpected_scan,
    )

    result = (
        acceptance_policy.evaluate_structure_acceptance_at_native_checkpoint(
            _carbon_bond(),
            bond_ring_report=_native_checkpoint(),
            level="standard",
        )
    )

    assert result.passed
    assert result.metrics["bond_ring_scope"] == "full_graph"
    assert result.metrics["bond_ring_max_ring_size"] == 16
    assert result.metrics["bond_ring_scan_complete"] is True


def test_native_checkpoint_piercing_maps_to_existing_acceptance_check():
    finding = NativeBondRingFinding(
        ring_index=0,
        ring_atom_indices=(2, 3, 4),
        bond_key=(0, 1),
        state=geo.PiercingState.PIERCES,
        indeterminacy_causes=(),
        aabb_separated=False,
        surface_complete=True,
    )

    result = (
        acceptance_policy.evaluate_structure_acceptance_at_native_checkpoint(
            _carbon_bond(),
            bond_ring_report=_native_checkpoint(findings=(finding,)),
            level="basic",
        )
    )

    piercing = next(
        check
        for check in result.failures
        if check.name == "bond_ring_piercing"
    )
    assert piercing.measured == (2, 3, 4)
    assert piercing.atom_indices == (0, 1)
    assert piercing.bond_indices == (0,)
    assert result.metrics["bond_ring_piercing_count"] == 1


def test_native_checkpoint_undetermined_maps_causes_and_scope_warning():
    finding = NativeBondRingFinding(
        ring_index=0,
        ring_atom_indices=(2, 3, 4),
        bond_key=(0, 1),
        state=geo.PiercingState.UNDETERMINED,
        indeterminacy_causes=(
            geo.SegmentCycleIndeterminacy.NUMERIC_BAND,
        ),
        aabb_separated=False,
        surface_complete=False,
    )

    result = (
        acceptance_policy.evaluate_structure_acceptance_at_native_checkpoint(
            _carbon_bond(),
            bond_ring_report=_native_checkpoint(
                findings=(finding,),
                excluded_ring_count=2,
            ),
            level="standard",
        )
    )

    assert result.passed
    assert {
        warning.name for warning in result.warnings
    } == {"bond_ring_piercing", "bond_ring_scope_coverage"}
    relation_warning = next(
        warning
        for warning in result.warnings
        if warning.name == "bond_ring_piercing"
    )
    assert relation_warning.measured == ("numeric_band",)
    assert result.metrics["bond_ring_undetermined_count"] == 1
    assert result.metrics["bond_ring_excluded_ring_count"] == 2
