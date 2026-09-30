"""Cross-language fence for canonical geometry defaults."""

from dataclasses import asdict

from hotpot.cheminfo.geometry import _geometry_native
from hotpot.cheminfo.geometry.settings import DEFAULT_GEOMETRY_SETTINGS


def test_native_geometry_defaults_match_python_settings() -> None:
    assert _geometry_native._default_settings_snapshot() == asdict(
        DEFAULT_GEOMETRY_SETTINGS
    )


def test_native_geometry_default_snapshot_is_independent() -> None:
    changed = _geometry_native._default_settings_snapshot()
    changed["surface"]["maximum_cycle_vertices"] = 99

    assert (
        _geometry_native._default_settings_snapshot()["surface"][
            "maximum_cycle_vertices"
        ]
        == 8
    )
