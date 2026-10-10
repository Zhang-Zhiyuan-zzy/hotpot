"""Validate method applicability against probed xTB backend facts."""

from __future__ import annotations

from typing import Iterable, Optional

from .contracts import XTBApplicabilityError, XTBBackendInfo, XTBMethod


__all__ = ["validate_element_support"]


def _method_limit(
    backend_info: XTBBackendInfo,
    method: XTBMethod,
) -> Optional[int]:
    if method is XTBMethod.GFNFF:
        return backend_info.gfnff_max_atomic_number
    return backend_info.gfn_xtb_max_atomic_number


def validate_element_support(
    backend_info: XTBBackendInfo,
    method: XTBMethod,
    atomic_numbers: Iterable[int],
) -> None:
    """Reject inputs outside the verified element domain before execution."""

    maximum_atomic_number = _method_limit(backend_info, method)
    if maximum_atomic_number is None:
        raise XTBApplicabilityError(
            f"GFN-FF element coverage is not verified for xTB "
            f"{backend_info.version}"
        )

    unsupported = tuple(
        atomic_number
        for atomic_number in atomic_numbers
        if atomic_number < 1 or atomic_number > maximum_atomic_number
    )
    if unsupported:
        raise XTBApplicabilityError(
            f"{method.value} with xTB {backend_info.version} supports atomic "
            f"numbers 1-{maximum_atomic_number}; unsupported input: "
            f"{unsupported!r}"
        )
