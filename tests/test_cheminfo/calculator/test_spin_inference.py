"""Target behavior for the named lowest-spin parity policy."""

from __future__ import annotations

import hotpot
import pytest


pytestmark = pytest.mark.xfail(
    strict=True,
    reason="spin inference is implemented in phase 6",
)


@pytest.mark.parametrize(
    ("smiles", "charge", "electron_count", "unpaired", "multiplicity"),
    (
        ("[He]", 0, 2, 0, 1),
        ("[H]", 0, 1, 1, 2),
        ("[Li+]", 1, 2, 0, 1),
    ),
)
def test_lowest_spin_parity_uses_serialized_nuclei_and_charge(
    smiles: str,
    charge: int,
    electron_count: int,
    unpaired: int,
    multiplicity: int,
) -> None:
    from hotpot.cheminfo.calculator import infer_lowest_spin

    mol = hotpot.read_mol(smiles, "smi")
    result = infer_lowest_spin(mol, charge)

    assert result.electron_count == electron_count
    assert result.unpaired_electrons == unpaired
    assert result.multiplicity == multiplicity
    assert result.source.value == "lowest-spin-parity"


def test_spin_inference_rejects_an_implicit_hydrogen_graph() -> None:
    from hotpot.cheminfo.calculator import infer_lowest_spin
    from hotpot.cheminfo.calculator.electronic_state.contracts import (
        IncompleteExplicitAtomError,
    )

    mol = hotpot.read_mol("N", "smi")

    with pytest.raises(IncompleteExplicitAtomError):
        infer_lowest_spin(mol, charge=0)

