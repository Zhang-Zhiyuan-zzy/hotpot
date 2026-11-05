"""Scientific-state profile controls for explicit inactive-profile checks."""

import json

import pytest

from hotpot.cheminfo.operations.calculation.scientific_state import profiles


@pytest.fixture
def planned_screening_profile(monkeypatch):
    """Keep the inactive-profile gate covered after production activation."""
    _, payload = profiles._profile_payload()
    payload = {**payload, "activation_status": "planned"}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    monkeypatch.setattr(profiles, "_profile_payload", lambda: (encoded, payload))
    profiles._coordination_screening_profile.cache_clear()
    yield
    profiles._coordination_screening_profile.cache_clear()
