"""Contract tests for the auditable Am benchmark gallery renderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import am_gallery


def _write_mol2(path: Path) -> None:
    path.write_text(
        """@<TRIPOS>MOLECULE
synthetic
3 2 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
1 Am  0.0000  0.0000  0.0000 Am 1 MOL 0.0
2 N   2.3000  0.1000  0.0000 N.3 1 MOL 0.0
3 H   2.8000  0.9000  0.2000 H   1 MOL 0.0
@<TRIPOS>BOND
1 1 2 1
2 2 3 1
""",
        encoding="utf-8",
    )


def _write_case(
    root: Path,
    index: int,
    *,
    cbond: bool,
) -> Path:
    case_dir = root / "cases" / f"{index:04d}"
    case_dir.mkdir(parents=True)
    report = {
        "index": index,
        "smiles": "N",
        "status": "passed" if cbond else "failed_cbond",
    }
    if cbond:
        report.update(
            cbond={"donor_count": 1, "donor_indices": [0]},
            output_frame_role="selected_success_frame",
        )
        _write_mol2(case_dir / "optimized.mol2")
    (case_dir / "input.smi").write_text("N\n", encoding="utf-8")
    (case_dir / "report.json").write_text(
        json.dumps(report) + "\n",
        encoding="utf-8",
    )
    return case_dir


def test_principal_axes_are_translation_invariant_and_right_handed() -> None:
    coordinates = np.asarray(
        [
            [-2.0, 0.2, 0.1],
            [1.2, -0.5, 0.4],
            [0.4, 2.2, -0.7],
            [0.1, -0.3, 1.8],
        ]
    )
    masses = np.asarray([12.011, 15.999, 14.007, 243.0])

    original = am_gallery.mass_weighted_principal_axes(coordinates, masses)
    translated = am_gallery.mass_weighted_principal_axes(
        coordinates + np.asarray([13.0, -7.5, 2.4]),
        masses,
    )

    assert np.allclose(original.moments, translated.moments)
    assert np.allclose(
        original.oriented_coordinates,
        translated.oriented_coordinates,
    )
    assert np.allclose(original.axes.T @ original.axes, np.eye(3))
    np.testing.assert_allclose(
        np.linalg.det(original.axes),
        1.0,
        atol=1.0e-12,
    )


def test_maximum_moment_axis_is_the_display_view_axis() -> None:
    coordinates = np.asarray(
        [
            [-3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.2],
        ]
    )
    masses = np.asarray([12.0, 12.0, 16.0])

    result = am_gallery.mass_weighted_principal_axes(coordinates, masses)
    displayed_tensor = result.axes.T @ result.inertia_tensor @ result.axes

    assert np.allclose(np.diag(displayed_tensor), result.moments)
    assert np.allclose(displayed_tensor - np.diag(np.diag(displayed_tensor)), 0.0)
    assert result.moments[2] >= result.moments[1] >= result.moments[0]


def test_principal_moments_and_view_axis_are_rotation_invariant() -> None:
    coordinates = np.asarray(
        [
            [-2.0, 0.2, 0.1],
            [1.2, -0.5, 0.4],
            [0.4, 2.2, -0.7],
            [0.1, -0.3, 1.8],
        ]
    )
    masses = np.asarray([12.011, 15.999, 14.007, 243.0])
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    original = am_gallery.mass_weighted_principal_axes(coordinates, masses)
    rotated = am_gallery.mass_weighted_principal_axes(
        coordinates @ rotation.T,
        masses,
    )

    assert np.allclose(original.moments, rotated.moments)
    expected_view_axis = rotation @ original.axes[:, 2]
    assert abs(float(np.dot(expected_view_axis, rotated.axes[:, 2]))) > 1.0 - 1e-12
    original_distances = np.linalg.norm(
        original.oriented_coordinates[:, None, :]
        - original.oriented_coordinates[None, :, :],
        axis=2,
    )
    rotated_distances = np.linalg.norm(
        rotated.oriented_coordinates[:, None, :]
        - rotated.oriented_coordinates[None, :, :],
        axis=2,
    )
    assert np.allclose(original_distances, rotated_distances)


def test_discovery_partitions_all_cases_by_actual_cbond_presence(
    tmp_path: Path,
) -> None:
    _write_case(tmp_path, 1, cbond=True)
    _write_case(tmp_path, 2, cbond=False)

    cases = am_gallery.discover_gallery_cases(tmp_path)
    cbond_indices = {case.index for case in cases if case.group == "cbond"}
    failed_indices = {
        case.index for case in cases if case.group == "failed_cbond"
    }

    assert cbond_indices == {1}
    assert failed_indices == {2}
    assert cbond_indices.isdisjoint(failed_indices)
    assert cbond_indices | failed_indices == {case.index for case in cases}
    assert cases[0].source_path.name == "optimized.mol2"
    assert cases[1].source_path.name == "input.smi"


def test_failed_cbond_display_builder_writes_explicit_hydrogen_3d_ligand(
    tmp_path: Path,
) -> None:
    case_dir = _write_case(tmp_path, 7, cbond=False)
    case = am_gallery.discover_gallery_cases(tmp_path)[0]

    structure = am_gallery._build_failed_cbond_ligand(case, tmp_path / "gallery", 31)
    geometry = am_gallery._read_mol2_geometry(structure)

    assert structure.is_file()
    assert geometry.elements.count("H") > 0
    assert geometry.coordinates.shape == (len(geometry.elements), 3)
    assert np.all(np.isfinite(geometry.coordinates))
    assert not (case_dir / "optimized.mol2").exists()


def test_gallery_writes_case_images_two_sheets_and_auditable_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark_root = tmp_path / "benchmark"
    output_root = tmp_path / "gallery"
    _write_case(benchmark_root, 1, cbond=True)
    _write_case(benchmark_root, 2, cbond=False)
    (benchmark_root / "manifest.json").write_text(
        json.dumps({"git": {"commit": "benchmark-commit"}}) + "\n",
        encoding="utf-8",
    )

    def fake_render(case, output, parameters):
        image_path = output / "cases" / f"{case.index:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (120, 90), "white").save(image_path)
        return {
            "index": case.index,
            "group": case.group,
            "benchmark_status": case.status,
            "output_frame_role": case.output_frame_role,
            "report_sha256": case.report_sha256,
            "input_source_sha256": case.source_sha256,
            "rendered_structure_sha256": case.source_sha256,
            "structure_origin": (
                "optimized_or_last_finite_benchmark_frame"
                if case.group == "cbond"
                else "visualization_only_explicit_hydrogen_3d_ligand"
            ),
            "image": str(image_path),
            "render_status": "rendered",
            "render_error": None,
            "explicit_hydrogen_count": 1,
            "americium_count": int(case.group == "cbond"),
            "inertia": {
                "mass_weighted": True,
                "view_axis": "maximum principal moment",
                "principal_moments_amu_angstrom2": [1.0, 2.0, 3.0],
                "basis_determinant": 1.0,
            },
        }

    monkeypatch.setattr(am_gallery, "_render_gallery_case", fake_render)
    monkeypatch.setattr(
        am_gallery.importlib.util,
        "find_spec",
        lambda name: object(),
    )

    report = am_gallery.render_am_gallery(
        benchmark_root,
        output_root,
        workers=1,
        parameters=am_gallery.GalleryParameters(columns=2),
    )

    assert report["benchmark_commit"] == "benchmark-commit"
    assert report["groups"]["cbond"]["case_indices"] == [1]
    assert report["groups"]["failed_cbond"]["case_indices"] == [2]
    assert report["groups"]["cbond"]["count"] == 1
    assert report["groups"]["failed_cbond"]["count"] == 1
    assert report["rendered_count"] == 2
    assert len(report["input_sha256"]) == 64
    assert len(report["renderer_sha256"]) == 64
    assert (output_root / "cases" / "0001.png").is_file()
    assert (output_root / "cases" / "0002.png").is_file()
    assert (output_root / "am_cbond_complexes.png").is_file()
    assert (output_root / "am_failed_cbond_ligands.png").is_file()
    evidence_path = output_root / "gallery_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["sample_count"] == 2
    assert evidence["quality_passed_count"] == 1
    assert evidence["placeholder_count"] == 0
    assert evidence["contact_sheets"]["cbond"]["filename"] == (
        "am_cbond_complexes.png"
    )
    assert str(tmp_path) not in evidence_path.read_text(encoding="utf-8")
    assert json.loads((output_root / "gallery.json").read_text())[
        "parameters"
    ]["view"] == "mass-weighted maximum principal moment of inertia axis"
