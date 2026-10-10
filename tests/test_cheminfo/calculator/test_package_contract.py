"""Public contracts for calculator electronic-state services."""


def test_electronic_state_package_exposes_typed_services() -> None:
    from hotpot.cheminfo.calculator import (
        infer_charge,
        infer_lowest_spin,
        resolve_electronic_state,
    )
    from hotpot.cheminfo.calculator.electronic_state import contracts

    assert callable(infer_charge)
    assert callable(infer_lowest_spin)
    assert callable(resolve_electronic_state)
    assert contracts.ChargeInferenceResult.__dataclass_params__.frozen
    assert contracts.SpinInferenceResult.__dataclass_params__.frozen
    assert contracts.ElectronicState.__dataclass_params__.frozen

