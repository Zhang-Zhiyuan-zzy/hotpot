"""xTB backend-context projection contracts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from hotpot.cheminfo.calculator.contracts import ScientificQuantity
from hotpot.plugins.xtb.backend import probe_xtb_backend
from hotpot.plugins.xtb.contracts import XTBMethod
from hotpot.plugins.xtb.state_projection import XTBBackendContext


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


def test_xtb_context_identity_changes_with_exact_method(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    backend_info = probe_xtb_backend(
        fake_xtb_factory(),
        environment=dict(os.environ),
    )
    gfnff = XTBBackendContext(backend_info, XTBMethod.GFNFF)
    gfn2 = XTBBackendContext(backend_info, XTBMethod.GFN2_XTB)

    assert gfnff.identity != gfn2.identity
    assert gfnff.identity == XTBBackendContext(
        backend_info,
        XTBMethod.GFNFF,
    ).identity


def test_gfnff_and_gfn2_declare_distinct_spin_transport(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    backend_info = probe_xtb_backend(
        fake_xtb_factory(),
        environment=dict(os.environ),
    )
    gfnff = XTBBackendContext(backend_info, XTBMethod.GFNFF)
    gfn2 = XTBBackendContext(backend_info, XTBMethod.GFN2_XTB)

    assert gfnff.consumes_uhf is False
    assert gfnff.projection_convention == "charge_only"
    assert gfn2.consumes_uhf is True
    assert gfn2.projection_convention == "pure_spin_highest_ms"
    assert gfn2.electron_count_definition == "all_electron_atomic_number_sum"


def test_xtb_context_requires_charge_but_allows_unresolved_scientific_spin(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    context = XTBBackendContext(
        probe_xtb_backend(
            fake_xtb_factory(),
            environment=dict(os.environ),
        ),
        XTBMethod.GFN2_XTB,
    )

    requirements = context.scientific_operation_requirements
    assert requirements.required_resolved_quantities == frozenset(
        {ScientificQuantity.NET_CHARGE}
    )
    assert (
        ScientificQuantity.ELECTRONIC_STATE
        in requirements.allowed_unresolved_quantities
    )


def test_xtb_context_exposes_verified_method_element_domain(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    context = XTBBackendContext(
        probe_xtb_backend(
            fake_xtb_factory(),
            environment=dict(os.environ),
        ),
        XTBMethod.GFNFF,
    )

    assert context.method_supported is True
    assert context.supports_atomic_number(86) is True
    assert context.supports_atomic_number(95) is False
    assert context.effective_nuclear_charge(95) == 95
