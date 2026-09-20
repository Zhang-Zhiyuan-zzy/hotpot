"""Executable examples used by the geometry package README files."""

from __future__ import annotations

import doctest
import re
from pathlib import Path

import pytest

from hotpot.cheminfo import geometry as geo

GEOMETRY_DOCUMENTATION_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "hotpot" / "cheminfo" / "geometry"
)
PYCON_BLOCK = re.compile(r"```pycon\r?\n(.*?)\r?\n```", re.DOTALL)


@pytest.mark.parametrize("readme_name", ("README.md", "README.zh.md"))
def test_every_documented_pycon_example_runs(readme_name: str) -> None:
    readme_path = GEOMETRY_DOCUMENTATION_DIRECTORY / readme_name
    blocks = PYCON_BLOCK.findall(readme_path.read_text(encoding="utf-8"))

    assert len(blocks) == 13

    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner()
    failed = 0
    attempted = 0
    for index, block in enumerate(blocks, start=1):
        test = parser.get_doctest(
            block,
            {},
            f"{readme_name} pycon block {index}",
            str(readme_path),
            0,
        )
        result = runner.run(test)
        failed += result.failed
        attempted += result.attempted

    assert attempted > 0
    assert failed == 0


def test_readme_quick_planar_cycle_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cycle = geo.Cycle(
        [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    )
    result = geo.measure_planarity(cycle)

    print(result.kind.value)
    print(result.maximum_deviation)

    assert capsys.readouterr().out == "planar\n0.0\n"


def test_readme_quick_nonplanar_cycle_piercing_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cycle = geo.Cycle(
        [(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)]
    )
    segment = geo.Segment((0.6, 0.8, -1), (0.6, 0.8, 1))
    planarity = geo.measure_planarity(cycle)
    relation = geo.determine_segment_cycle_relation(segment, cycle)

    print(planarity.kind.value)
    print(relation.state.value)
    print(relation.surface_model.value)
    print(relation.surface_evidence.embedded_surface_count)

    assert capsys.readouterr().out == (
        "nonplanar\n"
        "pierces\n"
        "vertex_triangulation_family\n"
        "2\n"
    )


def test_readme_measure_planarity_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cycle = geo.Cycle(
        [(0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)]
    )
    result = geo.measure_planarity(cycle)

    print(result.kind.value)
    print(result.maximum_deviation)
    print(tuple(round(value, 6) for value in result.normal))

    assert capsys.readouterr().out == "planar\n0.0\n(0.0, 0.0, 1.0)\n"


def test_readme_determine_line_relation_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = geo.Line((0, 0, 0), (1, 0, 0))
    second = geo.Line((0, 1, 1), (0, 1, 0))
    result = geo.determine_line_relation(first, second)

    print(result.kind.value)
    print(result.distance)
    print(result.parallel_measure)

    assert capsys.readouterr().out == "skew\n1.0\n1.0\n"


def test_readme_line_distance_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = geo.Line((0, 0, 0), (1, 0, 0))
    second = geo.Line((0, 1, 1), (0, 1, 0))

    print(geo.line_distance(first, second))

    assert capsys.readouterr().out == "1.0\n"


def test_readme_point_segment_distance_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    point = geo.Point((1, 1, 0))
    segment = geo.Segment((0, 0, 0), (2, 0, 0))

    print(geo.point_segment_distance(point, segment))

    assert capsys.readouterr().out == "1.0\n"


def test_readme_segment_segment_distance_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = geo.Segment((0, 0, 0), (1, 0, 0))
    second = geo.Segment((0.5, -1, 0), (0.5, 1, 0))

    print(geo.segment_segment_distance(first, second))

    assert capsys.readouterr().out == "0.0\n"


def test_readme_point_pair_distances_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    points = (
        geo.Point((0, 0, 0)),
        geo.Point((1, 0, 0)),
        geo.Point((3, 0, 0)),
    )
    result = geo.point_pair_distances(points)

    print(
        [
            (item.first_index, item.second_index, item.distance)
            for item in result
        ]
    )

    assert capsys.readouterr().out == (
        "[(0, 1, 1.0), (0, 2, 3.0), (1, 2, 2.0)]\n"
    )


def test_readme_find_point_pairs_below_distance_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    points = (
        geo.Point((0, 0, 0)),
        geo.Point((1, 0, 0)),
        geo.Point((3, 0, 0)),
    )
    result = geo.find_point_pairs_below_distance(points, threshold=3.0)

    print(
        [
            (item.first_index, item.second_index, item.distance)
            for item in result
        ]
    )

    assert capsys.readouterr().out == "[(0, 1, 1.0), (1, 2, 2.0)]\n"


def test_readme_locate_point_in_planar_cycle_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cycle = geo.Cycle(
        [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    )
    plane = geo.Plane((0, 0, 0), (0, 0, 1))
    result = geo.locate_point_in_planar_cycle(
        geo.Point((1, 1, 0)), cycle, plane
    )

    print(result.value)

    assert capsys.readouterr().out == "interior\n"


def test_readme_iter_segment_cycle_relations_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cycle = geo.Cycle(
        [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    )
    segments = (
        geo.Segment((1, 1, -1), (1, 1, 1)),
        geo.Segment((3, 3, -1), (3, 3, 1)),
    )
    result = geo.iter_segment_cycle_relations(segments, cycle)

    print([relation.state.value for relation in result])

    assert capsys.readouterr().out == "['pierces', 'does_not_pierce']\n"


def test_readme_determine_segment_cycle_relation_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cycle = geo.Cycle(
        [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    )
    segment = geo.Segment((1, 1, -1), (1, 1, 1))
    result = geo.determine_segment_cycle_relation(segment, cycle)

    print(result.state.value)
    print(result.surface_model.value)
    print([point.coordinates for point in result.intersection_points])

    assert capsys.readouterr().out == (
        "pierces\nplanar_polygon\n[(1.0, 1.0, 0.0)]\n"
    )


def test_readme_closest_cycle_edge_example(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cycle = geo.Cycle(
        [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    )
    segment = geo.Segment((-1, -1, -1), (-1, -1, 1))
    result = geo.closest_cycle_edge(cycle, segment)

    print(result.edge_index)
    print(result.edge.start.coordinates, result.edge.end.coordinates)
    print(round(result.distance, 6))

    assert capsys.readouterr().out == (
        "0\n(0.0, 0.0, 0.0) (2.0, 0.0, 0.0)\n1.414214\n"
    )
