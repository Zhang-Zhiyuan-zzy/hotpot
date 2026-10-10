"""Composition contracts for charge and spin state resolution."""

from __future__ import annotations

import hotpot
import pytest


def _explicit_molecule(smiles: str):
    mol = hotpot.read_mol(smiles, "smi")
    mol.add_hydrogens(rm_polar_hs=False)
    return mol


def test_resolver_composes_default_charge_and_lowest_spin() -> None:
    from hotpot.cheminfo.calculator import resolve_electronic_state

    state = resolve_electronic_state(_explicit_molecule("C"))

    assert state.charge == 0
    assert state.unpaired_electrons == 0
    assert state.multiplicity == 1
    assert state.charge_source.value == "valence"
    assert state.spin_source.value == "lowest-spin-parity"


def test_explicit_state_overrides_are_authoritative_and_recorded() -> None:
    from hotpot.cheminfo.calculator import resolve_electronic_state

    state = resolve_electronic_state(
        _explicit_molecule("[CH3+]"),
        charge=1,
        unpaired_electrons=2,
    )

    assert state.charge == 1
    assert state.unpaired_electrons == 2
    assert state.multiplicity == 3
    assert state.charge_source.value == "explicit"
    assert state.spin_source.value == "explicit"


def test_parity_inconsistent_explicit_spin_is_rejected() -> None:
    from hotpot.cheminfo.calculator import resolve_electronic_state

    with pytest.raises(ValueError, match="parity"):
        resolve_electronic_state(
            _explicit_molecule("C"),
            charge=0,
            unpaired_electrons=1,
        )


def test_resolver_accepts_independent_custom_estimators() -> None:
    from hotpot.cheminfo.calculator import resolve_electronic_state
    from hotpot.cheminfo.calculator.electronic_state.contracts import (
        ChargeInferenceResult,
        ChargeInferenceSource,
        FragmentCharge,
        SpinInferenceResult,
        SpinInferenceSource,
    )

    class FixedChargeEstimator:
        def infer(self, mol):
            return ChargeInferenceResult(
                atom_formal_charges=tuple(atom.formal_charge for atom in mol.atoms),
                fragments=(
                    FragmentCharge(
                        atom_indices=tuple(range(len(mol.atoms))),
                        charge=0,
                        source=ChargeInferenceSource.VALENCE,
                    ),
                ),
                total_charge=0,
                source=ChargeInferenceSource.VALENCE,
                assumptions=("test charge estimator",),
            )

    class FixedSpinEstimator:
        def infer(self, mol, charge):
            return SpinInferenceResult(
                unpaired_electrons=0,
                multiplicity=1,
                electron_count=sum(atom.atomic_number for atom in mol.atoms) - charge,
                source=SpinInferenceSource.LOWEST_SPIN_PARITY,
                assumptions=("test spin estimator",),
            )

    state = resolve_electronic_state(
        _explicit_molecule("C"),
        charge_estimator=FixedChargeEstimator(),
        spin_estimator=FixedSpinEstimator(),
    )

    assert state.assumptions == (
        "test charge estimator",
        "test spin estimator",
    )

