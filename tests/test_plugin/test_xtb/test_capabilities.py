"""Method and element-domain contracts for official xTB backends."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import pytest

from hotpot.plugins.xtb.backend import probe_xtb_backend
from hotpot.plugins.xtb.capabilities import validate_element_support
from hotpot.plugins.xtb.contracts import XTBApplicabilityError, XTBMethod


class FakeXTBFactory(Protocol):
    def __call__(
        self,
        scenario: str = "success",
        directory_name: str = "fake_xtb",
    ) -> Path: ...


@pytest.mark.parametrize(
    "method_name",
    ("GFNFF", "GFN0_XTB", "GFN1_XTB", "GFN2_XTB"),
)
def test_stable_671_backend_accepts_elements_through_radon(
    fake_xtb_factory: FakeXTBFactory,
    method_name: str,
) -> None:
    backend_info = probe_xtb_backend(
        executable=fake_xtb_factory(),
        environment=dict(os.environ),
    )

    validate_element_support(
        backend_info,
        getattr(XTBMethod, method_name),
        (1, 6, 86),
    )


@pytest.mark.parametrize(
    "method_name",
    ("GFNFF", "GFN0_XTB", "GFN1_XTB", "GFN2_XTB"),
)
def test_stable_671_backend_rejects_americium_before_execution(
    fake_xtb_factory: FakeXTBFactory,
    method_name: str,
) -> None:
    backend_info = probe_xtb_backend(
        executable=fake_xtb_factory(),
        environment=dict(os.environ),
    )

    with pytest.raises(XTBApplicabilityError):
        validate_element_support(
            backend_info,
            getattr(XTBMethod, method_name),
            (6, 95),
        )


def test_extended_gfnff_domain_does_not_expand_gfn_xtb_domain(
    fake_xtb_factory: FakeXTBFactory,
) -> None:
    stable_info = probe_xtb_backend(
        executable=fake_xtb_factory(),
        environment=dict(os.environ),
    )
    extended_info = replace(stable_info, gfnff_max_atomic_number=103)

    validate_element_support(extended_info, XTBMethod.GFNFF, (6, 95))
    with pytest.raises(XTBApplicabilityError):
        validate_element_support(
            extended_info,
            XTBMethod.GFN2_XTB,
            (6, 95),
        )
