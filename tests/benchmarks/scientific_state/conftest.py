"""Keep the historical pre-activation benchmark tests explicitly inactive."""

import pytest

from tests.test_cheminfo.calculator.conftest import planned_screening_profile


@pytest.fixture(autouse=True)
def historical_pre_activation_profile(planned_screening_profile):
    """Exercise the planned-profile boundary after production activation."""
