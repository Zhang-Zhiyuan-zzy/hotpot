"""Reserved SDF metadata and molecular-stream contracts for xTB."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable

import pytest

import hotpot
from hotpot.cheminfo.calculator.electronic_state import (
    ChargeInferenceSource,
    SpinInferenceSource,
)
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
            charge_source=ChargeInferenceSource.EXPLICIT,
            spin_source=SpinInferenceSource.EXPLICIT,
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
) -> None:
    mol = _water()
    xtb_report = run_gfn_xtb(
        mol,
        method=GFNXTBMethod.GFN2_XTB,
        task=XTBTask.SINGLEPOINT,
        charge=0,
        unpaired_electrons=0,
        executable=fake_xtb_factory("success", "gfn2_xtb"),
        environment=dict(os.environ),
    )
    electronic_metadata = metadata_from_report(xtb_report)
    gfnff_report = run_gfnff(
        mol,
        task=XTBTask.SINGLEPOINT,
        charge=0,
        executable=fake_xtb_factory("success", "gfnff"),
        environment=dict(os.environ),
    )

    gfnff_metadata = metadata_from_report(gfnff_report, electronic_metadata)

    assert gfnff_metadata.method is XTBMethod.GFNFF
    assert gfnff_metadata.unpaired_electrons == 0
    assert gfnff_metadata.provenance.spin_source is SpinInferenceSource.EXPLICIT
