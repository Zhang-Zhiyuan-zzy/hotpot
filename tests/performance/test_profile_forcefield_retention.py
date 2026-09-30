"""Lightweight tests for the force-field trajectory retention profiler."""

from __future__ import annotations

from tests.performance.profile_forcefield_retention import (
    DEFAULT_CASES,
    aggregate_samples,
    build_parser,
)


def _sample(case: int, save_movie: bool, rss: int, wall: float, retained: int):
    return {
        "case": case,
        "repeat": 0,
        "save_movie": save_movie,
        "status": "passed",
        "peak_rss_kib": rss,
        "wall_seconds": wall,
        "frame_count": 8 if save_movie else 1,
        "retained_coordinate_bytes": retained,
    }


def test_parser_exposes_representative_defaults_and_overrides() -> None:
    defaults = build_parser().parse_args(())
    selected = build_parser().parse_args(
        ("--cases", "54,109", "--repeats", "3", "--profile", "smoke")
    )

    assert defaults.cases == DEFAULT_CASES
    assert defaults.repeats == 1
    assert defaults.profile == "standard"
    assert selected.cases == (54, 109)
    assert selected.repeats == 3
    assert selected.profile == "smoke"


def test_aggregation_reports_modes_and_paired_deltas() -> None:
    report = aggregate_samples(
        (
            _sample(1, False, 1000, 2.0, 240),
            _sample(1, True, 1250, 3.0, 960),
            _sample(54, False, 1100, 4.0, 480),
            _sample(54, True, 1500, 5.0, 1440),
        )
    )

    assert report["modes"]["false"]["median_peak_rss_kib"] == 1050
    assert report["modes"]["true"]["median_frame_count"] == 8
    assert report["paired_deltas"] == [
        {
            "case": 1,
            "repeat": 0,
            "peak_rss_delta_kib": 250,
            "peak_rss_delta_percent": 25.0,
            "wall_seconds_delta": 1.0,
            "wall_seconds_delta_percent": 50.0,
            "retained_coordinate_bytes_delta": 720,
        },
        {
            "case": 54,
            "repeat": 0,
            "peak_rss_delta_kib": 400,
            "peak_rss_delta_percent": 400 / 11,
            "wall_seconds_delta": 1.0,
            "wall_seconds_delta_percent": 25.0,
            "retained_coordinate_bytes_delta": 960,
        },
    ]
