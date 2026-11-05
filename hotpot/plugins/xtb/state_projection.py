"""Exact xTB backend context for scientific-state projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple

from hotpot.cheminfo.adapters.calculation.contracts import BackendIdentity
from hotpot.cheminfo.structure.snapshots import MolecularFacts
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    ScientificOperationRequirements,
    ScientificQuantity,
)

from .contracts import XTBBackendInfo, XTBMethod

__all__ = ["XTBBackendContext"]


_ALL_ELECTRON_DEFINITION = "all_electron_atomic_number_sum"
_GFN_VALENCE_ELECTRON_DEFINITION = "xtb_gfn_valence_atomic_number_sum_v1"
_GFN_SPIN_PROJECTION_DEFINITION = "xtb_gfn_closed_core_highest_ms_f_in_core_unmapped_v1"
# Official xTB 6.7.1 src/pqn.f90::ncore and src/prog/main.F90:
# GFN0/1/2 share Z-ncore, with Ce--Lu represented by three active electrons.
_GFN_CORE_BOUNDARIES = (
    (2, 0),
    (10, 2),
    (18, 10),
    (29, 18),
    (36, 28),
    (47, 36),
    (54, 46),
    (71, 54),
    (79, 68),
    (86, 78),
)
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
        "electron_count_definition": (
            _ALL_ELECTRON_DEFINITION
            if method is XTBMethod.GFNFF
            else _GFN_VALENCE_ELECTRON_DEFINITION
        ),
        "spin_projection_definition": (
            _CHARGE_ONLY_PROJECTION
            if method is XTBMethod.GFNFF
            else _GFN_SPIN_PROJECTION_DEFINITION
        ),
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
        return (
            _ALL_ELECTRON_DEFINITION
            if self.method is XTBMethod.GFNFF
            else _GFN_VALENCE_ELECTRON_DEFINITION
        )

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
        if self.method is XTBMethod.GFNFF:
            return atomic_number
        if 58 <= atomic_number <= 71:
            return 3
        core_electrons = next(
            core
            for limit, core in _GFN_CORE_BOUNDARIES
            if atomic_number <= limit
        )
        return atomic_number - core_electrons

    def project_scientific_spin(
        self,
        facts: MolecularFacts,
        two_s: Optional[int],
        multiplicity: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        """Map physical spin only when the omitted core has no active spin.

        Ordinary GFN0/1/2 place lanthanide f electrons in the core. Their
        physical molecular spin does not determine the active-electron UHF;
        explicit settings, exact-context history or an execution default own it.
        """
        if (
            self.method is XTBMethod.GFNFF
            or two_s is None
            or multiplicity is None
            or any(58 <= atomic_number <= 71 for atomic_number in facts.atomic_numbers)
        ):
            return None
        return two_s, multiplicity
