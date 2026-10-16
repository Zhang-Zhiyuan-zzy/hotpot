"""Pre-activation scientific-state resolver benchmark tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import hotpot
import pytest

from hotpot.cheminfo.core import BondKind

from .corpus import DryStateCase, DryStateSourceKind
from .runner import resolve_case


def _case(smiles: str = "[Eu].N") -> DryStateCase:
    return DryStateCase(
        index=1,
        case_id="eu-n",
        source_kind=DryStateSourceKind.SMILES,
        source=smiles,
        format="smi",
        source_sha256=hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
    )


def test_resolve_case_records_only_the_real_profile_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.benchmarks.scientific_state.runner as runner

    mol = hotpot.read_mol("[Eu].N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    monkeypatch.setattr(runner, "_load_molecule", lambda case: mol)

    record = resolve_case(_case())

    assert record.profile_activation_status == "planned"
    assert record.coordination_bond_count == 1
    assert record.profile_in_domain is False
    assert record.profile_finding_codes == (
        "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE",
    )
    assert record.static_profile_ready_except_activation is True
    assert record.python_peak_bytes > 0
    assert len(record.snapshot_sha256) == 64


def test_unknown_metal_bond_remains_an_explicit_profile_boundary() -> None:
    record = resolve_case(_case("[Eu]N"))

    assert record.coordination_bond_count == 0
    assert "PROFILE.COORDINATION_BOUNDARY_REQUIRED" in (
        record.profile_finding_codes
    )
    assert record.static_profile_ready_except_activation is False


def test_resolve_case_rejects_changed_structure_asset(tmp_path: Path) -> None:
    source = tmp_path / "structure.xyz"
    source.write_text("1\nrecord\nH 0 0 0\n", encoding="utf-8")
    case = DryStateCase(
        index=2,
        case_id="changed",
        source_kind=DryStateSourceKind.FILE,
        source=str(source),
        format="xyz",
        source_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="Source digest changed"):
        resolve_case(case)
