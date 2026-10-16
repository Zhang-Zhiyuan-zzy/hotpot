"""Exact xTB backend context for scientific-state projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from hotpot.cheminfo.calculator.contracts import (
    BackendIdentity,
    ScientificOperationRequirements,
    ScientificQuantity,
)

from .contracts import XTBBackendInfo, XTBMethod

__all__ = ["XTBBackendContext"]


_ALL_ELECTRON_DEFINITION = "all_electron_atomic_number_sum"
_CHARGE_ONLY_PROJECTION = "charge_only"
_HIGHEST_MS_PROJECTION = "pure_spin_highest_ms"
_BACKEND_ALLOWED_UNRESOLVED = frozenset(
    quantity
    for quantity in ScientificQuantity
    if quantity is not ScientificQuantity.NET_CHARGE
)


def _identity_digest(backend_info: XTBBackendInfo, method: XTBMethod) -> str:
    payload = {
        "backend": "xtb",
        "version": backend_info.version,
        "revision": backend_info.revision,
        "executable_sha256": backend_info.executable_sha256,
        "method": method.value,
        "gfn_xtb_max_atomic_number": backend_info.gfn_xtb_max_atomic_number,
        "gfnff_max_atomic_number": backend_info.gfnff_max_atomic_number,
        "electron_count_definition": _ALL_ELECTRON_DEFINITION,
        "projection_convention": (
            _CHARGE_ONLY_PROJECTION
            if method is XTBMethod.GFNFF
            else _HIGHEST_MS_PROJECTION
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class XTBBackendContext:
    """Exact method, capability and electron-count facts for one xTB run."""

    backend_info: XTBBackendInfo
    method: XTBMethod

    @property
    def identity(self) -> BackendIdentity:
        return BackendIdentity(_identity_digest(self.backend_info, self.method))

    @property
    def scientific_operation_requirements(
        self,
    ) -> ScientificOperationRequirements:
        return ScientificOperationRequirements(
            operation_id=f"xtb:{self.method.value}",
            required_resolved_quantities=frozenset(
                {ScientificQuantity.NET_CHARGE}
            ),
            allowed_unresolved_quantities=_BACKEND_ALLOWED_UNRESOLVED,
        )

    @property
    def projection_convention(self) -> str:
        return (
            _CHARGE_ONLY_PROJECTION
            if self.method is XTBMethod.GFNFF
            else _HIGHEST_MS_PROJECTION
        )

    @property
    def electron_count_definition(self) -> str:
        return _ALL_ELECTRON_DEFINITION

    @property
    def consumes_uhf(self) -> bool:
        return self.method is not XTBMethod.GFNFF

    @property
    def method_supported(self) -> bool:
        return (
            self.backend_info.gfnff_max_atomic_number is not None
            if self.method is XTBMethod.GFNFF
            else self.backend_info.gfn_xtb_max_atomic_number > 0
        )

    def supports_atomic_number(self, atomic_number: int) -> bool:
        limit = (
            self.backend_info.gfnff_max_atomic_number
            if self.method is XTBMethod.GFNFF
            else self.backend_info.gfn_xtb_max_atomic_number
        )
        return limit is not None and 0 < atomic_number <= limit

    def effective_nuclear_charge(self, atomic_number: int) -> int:
        return atomic_number
