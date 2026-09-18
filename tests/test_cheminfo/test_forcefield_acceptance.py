from __future__ import annotations

import json
from copy import copy
from types import SimpleNamespace

from hotpot.cheminfo import forcefields as ff
from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.core import Molecule


def _carbon_bond() -> Molecule:
    mol = Molecule()
    mol.create_atom(atomic_number=6, coordinates=(0.0, 0.0, 0.0))
    mol.create_atom(atomic_number=6, coordinates=(1.52, 0.0, 0.0))
    mol.add_bond(0, 1, bond_order=1.0)
    mol.refresh_atom_id()
    return mol


def _bond_ring_report(state: geo.PiercingState):
    finding = SimpleNamespace(
        target=SimpleNamespace(
            ring=SimpleNamespace(key=(2, 3, 4)),
            bond=SimpleNamespace(key=(0, 1)),
        ),
        relation=SimpleNamespace(
            state=state,
            indeterminacy_causes=(
                geo.SegmentCycleIndeterminacy.NUMERIC_BAND,
            ),
        ),
    )
    piercings = (finding,) if state is geo.PiercingState.PIERCES else ()
    undetermined = (
        (finding,) if state is geo.PiercingState.UNDETERMINED else ()
    )
    return SimpleNamespace(
        piercings=piercings,
        undetermined=undetermined,
        piercing_pair_count=len(piercings),
        undetermined_pair_count=len(undetermined),
        scan_complete=True,
    )


def test_acceptance_report_is_owned_by_forcefields_and_serializable():
    report = ff.evaluate_structure_acceptance(_carbon_bond(), level="basic")

    assert isinstance(report, ff.ForceFieldValidationReport)
    assert report.passed
    assert ff.is_structure_accepted(_carbon_bond(), level="basic")
    json.dumps(report.to_dict())


def test_topology_reference_allows_only_appended_hydrogen():
    mol = _carbon_bond()
    reference = ff.capture_topology(mol, allow_added_hydrogens=True)
    candidate = copy(mol)
    hydrogen = candidate.create_atom(
        atomic_number=1,
        coordinates=(-1.0, 0.0, 0.0),
    )
    candidate.add_bond(candidate.atoms[0], hydrogen, bond_order=1.0)

    report = ff.evaluate_structure_acceptance(
        candidate,
        level="off",
        topology_reference=reference,
    )

    assert report.passed


def test_confirmed_bond_ring_piercing_fails_acceptance(monkeypatch):
    monkeypatch.setattr(
        ff.geo,
        "scan_bond_ring_relations",
        lambda *args, **kwargs: _bond_ring_report(geo.PiercingState.PIERCES),
    )

    report = ff.evaluate_structure_acceptance(
        _carbon_bond(),
        level="standard",
    )

    assert not report.passed
    assert any(
        check.name == "bond_ring_piercing" for check in report.failures
    )


def test_undetermined_bond_ring_relation_warns_without_rejection(monkeypatch):
    monkeypatch.setattr(
        ff.geo,
        "scan_bond_ring_relations",
        lambda *args, **kwargs: _bond_ring_report(
            geo.PiercingState.UNDETERMINED
        ),
    )

    report = ff.evaluate_structure_acceptance(
        _carbon_bond(),
        level="standard",
    )

    assert report.passed
    warning = next(
        check for check in report.warnings
        if check.name == "bond_ring_piercing"
    )
    assert warning.measured == ("numeric_band",)
