"""Reserved SDF metadata and molecular-stream contracts for xTB."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional, Protocol

import pytest

import hotpot
from hotpot.cheminfo.calculator.execution_selection import BackendStateProjection
from hotpot.cheminfo.core import Molecule
from hotpot.plugins.xtb.contracts import GFNXTBMethod, XTBMethod, XTBTask
from hotpot.plugins.xtb.stream import (
    XTBStreamError,
    XTBStreamMetadata,
    XTBStreamProvenance,
    XTBStreamRecord,
    metadata_from_report,
    read_sdf_records,
    write_sdf_records,
)
from hotpot.plugins.xtb.workflow import run_gfn_xtb, run_gfnff


FakeXTBFactory = Callable[[str, str], Path]


class XTBProjectionFactory(Protocol):
    def __call__(
        self,
        mol: Molecule,
        executable: Path,
        method: XTBMethod,
        charge: int = 0,
        uhf: Optional[int] = None,
    ) -> BackendStateProjection: ...


def _water():
    mol = hotpot.read_mol("[H]O[H]", "smi")
    for atom, coordinates in zip(
        mol.atoms,
        ((-0.75, 0.0, 0.0), (0.0, 0.5, 0.0), (0.75, 0.0, 0.0)),
    ):
        atom.coordinates = coordinates
    return mol


def _hydrogen_atom():
    mol = hotpot.read_mol("[H]", "smi")
    mol.atoms[0].coordinates = (0.0, 0.0, 0.0)
    return mol


def _metadata(
    *,
    method: XTBMethod = XTBMethod.GFN2_XTB,
    total_charge: int = 0,
    unpaired_electrons: int = 0,
    energy_hartree: float = -76.123,
) -> XTBStreamMetadata:
    return XTBStreamMetadata(
        total_charge=total_charge,
        unpaired_electrons=unpaired_electrons,
        method=method,
        energy_hartree=energy_hartree,
        provenance=XTBStreamProvenance(
            backend_version="6.7.1",
            backend_revision=None,
            executable_sha256="a" * 64,
        ),
    )


def _remove_property(text: str, name: str) -> str:
    return re.sub(
        rf">  <{name}>\n[^\n]*\n\n",
        "",
        text,
        count=1,
    )


def test_multiple_sdf_records_round_trip_molecules_and_metadata() -> None:
    records = (
        XTBStreamRecord(_water(), _metadata()),
        XTBStreamRecord(
            _hydrogen_atom(),
            _metadata(
                total_charge=0,
                unpaired_electrons=1,
                energy_hartree=-0.4,
            ),
        ),
    )

    restored = read_sdf_records(write_sdf_records(records))

    assert len(restored) == 2
    assert tuple(atom.symbol for atom in restored[0].mol.atoms) == ("H", "O", "H")
    assert tuple(atom.symbol for atom in restored[1].mol.atoms) == ("H",)
    assert restored[0].metadata == records[0].metadata
    assert restored[1].metadata == records[1].metadata


def test_plain_sdf_record_is_accepted_without_manufacturing_metadata() -> None:
    source = _water().write(fmt="sdf", write_single=True)

    record = read_sdf_records(source)[0]

    assert record.metadata is None


def test_gfnff_metadata_may_retain_an_existing_spin_state() -> None:
    record = XTBStreamRecord(
        _water(),
        _metadata(method=XTBMethod.GFNFF, unpaired_electrons=0),
    )

    restored = read_sdf_records(write_sdf_records((record,)))[0]

    assert restored.metadata == record.metadata


def test_missing_required_reserved_property_is_rejected() -> None:
    source = write_sdf_records((XTBStreamRecord(_water(), _metadata()),))
    incomplete = _remove_property(source, "HOTPOT_XTB_ENERGY_HARTREE")

    with pytest.raises(XTBStreamError, match="lacks required"):
        read_sdf_records(incomplete)


def test_duplicate_reserved_property_is_rejected() -> None:
    source = write_sdf_records((XTBStreamRecord(_water(), _metadata()),))
    duplicate = source.replace(
        ">  <HOTPOT_XTB_TOTAL_CHARGE>\n0\n\n",
        ">  <HOTPOT_XTB_TOTAL_CHARGE>\n0\n\n"
        ">  <HOTPOT_XTB_TOTAL_CHARGE>\n0\n\n",
        1,
    )

    with pytest.raises(XTBStreamError, match="Duplicate reserved"):
        read_sdf_records(duplicate)


@pytest.mark.parametrize(
    "property_name,replacement,error_match",
    (
        ("HOTPOT_XTB_TOTAL_CHARGE", "half", "not an integer"),
        ("HOTPOT_XTB_ENERGY_HARTREE", "NaN", "not finite"),
        ("HOTPOT_XTB_METHOD", "gfn42", "method is unknown"),
        ("HOTPOT_XTB_PROVENANCE", "{broken", "not valid JSON"),
    ),
)
def test_malformed_reserved_values_are_rejected(
    property_name: str,
    replacement: str,
    error_match: str,
) -> None:
    source = write_sdf_records((XTBStreamRecord(_water(), _metadata()),))
    malformed = re.sub(
        rf"(>  <{property_name}>\n)[^\n]*",
        rf"\g<1>{replacement}",
        source,
        count=1,
    )

    with pytest.raises(XTBStreamError, match=error_match):
        read_sdf_records(malformed)


def test_record_signature_rejects_metadata_moved_to_changed_geometry() -> None:
    source = write_sdf_records((XTBStreamRecord(_water(), _metadata()),))
    changed = source.replace("   -0.7500", "   -0.6500", 1)

    with pytest.raises(XTBStreamError, match="does not match"):
        read_sdf_records(changed)


def test_electron_parity_is_checked_against_the_molecular_record() -> None:
    with pytest.raises(XTBStreamError, match="electron parity"):
        write_sdf_records(
            (
                XTBStreamRecord(
                    _hydrogen_atom(),
                    _metadata(unpaired_electrons=0, energy_hartree=-0.4),
                ),
            )
        )


def test_non_sdf_text_is_rejected_by_the_narrow_codec() -> None:
    with pytest.raises(XTBStreamError, match="record terminator"):
        read_sdf_records("[H]O[H]\n")


def test_report_metadata_preserves_spin_through_gfnff(
    fake_xtb_factory: FakeXTBFactory,
    xtb_projection_factory: XTBProjectionFactory,
) -> None:
    mol = _water()
    gfn_xtb_executable = fake_xtb_factory("success", "gfn2_xtb")
    gfn_xtb_state = xtb_projection_factory(
        mol,
        gfn_xtb_executable,
        XTBMethod.GFN2_XTB,
        0,
        0,
    )
    xtb_report = run_gfn_xtb(
        mol,
        state=gfn_xtb_state,
        method=GFNXTBMethod.GFN2_XTB,
        task=XTBTask.SINGLEPOINT,
        executable=gfn_xtb_executable,
        environment=dict(os.environ),
    )
    electronic_metadata = metadata_from_report(xtb_report)
    gfnff_executable = fake_xtb_factory("success", "gfnff")
    gfnff_state = xtb_projection_factory(
        mol,
        gfnff_executable,
        XTBMethod.GFNFF,
    )
    gfnff_report = run_gfnff(
        mol,
        state=gfnff_state,
        task=XTBTask.SINGLEPOINT,
        executable=gfnff_executable,
        environment=dict(os.environ),
    )

    gfnff_metadata = metadata_from_report(gfnff_report, electronic_metadata)

    assert gfnff_metadata.method is XTBMethod.GFNFF
    assert gfnff_metadata.unpaired_electrons == 0
    assert (
        gfnff_metadata.provenance.scientific_state_identity
        == electronic_metadata.provenance.scientific_state_identity
    )
    assert (
        gfnff_metadata.provenance.projection_convention
        == "pure_spin_highest_ms"
    )
