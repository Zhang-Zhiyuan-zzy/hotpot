"""Render reproducible Am coordination-complex benchmark galleries.

This module is deliberately a post-processing tool.  It reads an existing
benchmark tree and never rewrites the scientific ``optimized.mol2`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import periodictable
from PIL import Image, ImageChops, ImageDraw, ImageFont


__all__ = (
    "GalleryCase",
    "GalleryParameters",
    "PrincipalAxes",
    "discover_gallery_cases",
    "main",
    "mass_weighted_principal_axes",
    "render_am_gallery",
)


GROUP_CBOND = "cbond"
GROUP_FAILED_CBOND = "failed_cbond"
REGULAR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass(frozen=True)
class PrincipalAxes:
    """Mass-weighted inertia result with a right-handed display basis."""

    center_of_mass: np.ndarray
    inertia_tensor: np.ndarray
    moments: np.ndarray
    axes: np.ndarray
    oriented_coordinates: np.ndarray
    degenerate_pairs: tuple[bool, bool]


@dataclass(frozen=True)
class GalleryParameters:
    """Rendering parameters recorded verbatim in ``gallery.json``."""

    image_width: int = 1000
    image_height: int = 750
    ray_dpi: int = 180
    columns: int = 15
    cell_width: int = 460
    cell_height: int = 350
    sheet_header_height: int = 90
    ligand_build_seed: int = 20261009
    style: str = "Materials Studio-inspired orthographic ball-and-stick"
    view: str = "mass-weighted maximum principal moment of inertia axis"
    am_color_rgb: tuple[float, float, float] = (0.43, 0.38, 0.94)


@dataclass(frozen=True)
class GalleryCase:
    """Immutable rendering input derived from one benchmark case report."""

    index: int
    case_dir: Path
    group: str
    status: str
    smiles: str
    output_frame_role: Optional[str]
    report_sha256: str
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class _Mol2Geometry:
    coordinates: np.ndarray
    masses: np.ndarray
    elements: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cbond_exists(report: Mapping[str, object]) -> bool:
    cbond = report.get("cbond")
    if not isinstance(cbond, Mapping):
        return False
    donor_indices = cbond.get("donor_indices")
    donor_count = cbond.get("donor_count")
    return bool(donor_indices) or (
        donor_count is not None and int(donor_count) > 0
    )


def discover_gallery_cases(benchmark_root: Path) -> tuple[GalleryCase, ...]:
    """Discover and strictly partition every benchmark case by CBond presence."""

    cases_root = benchmark_root / "cases"
    report_paths = sorted(cases_root.glob("[0-9][0-9][0-9][0-9]/report.json"))
    cases = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        case_dir = report_path.parent
        cbond_exists = _cbond_exists(report)
        group = GROUP_CBOND if cbond_exists else GROUP_FAILED_CBOND
        source_path = (
            case_dir / "optimized.mol2"
            if cbond_exists
            else case_dir / "input.smi"
        )
        cases.append(
            GalleryCase(
                index=int(report.get("index", case_dir.name)),
                case_dir=case_dir,
                group=group,
                status=str(report.get("status", "unknown")),
                smiles=str(report.get("smiles", "")),
                output_frame_role=(
                    None
                    if report.get("output_frame_role") is None
                    else str(report["output_frame_role"])
                ),
                report_sha256=_sha256(report_path),
                source_path=source_path,
                source_sha256=_sha256(source_path) if source_path.is_file() else "",
            )
        )

    indices = [case.index for case in cases]
    if len(indices) != len(set(indices)):
        raise ValueError("benchmark case indices must be unique")
    if not cases:
        raise ValueError(f"no benchmark case reports found below {cases_root}")
    return tuple(cases)


def _canonical_axis_sign(axis: np.ndarray) -> np.ndarray:
    component = int(np.argmax(np.abs(axis)))
    return -axis if axis[component] < 0.0 else axis


def mass_weighted_principal_axes(
    coordinates: np.ndarray,
    masses: np.ndarray,
) -> PrincipalAxes:
    """Orient coordinates with the largest principal moment along view ``z``.

    The first and third eigenvector signs are fixed by their largest Cartesian
    component.  The second axis is their cross product, which makes the display
    basis deterministic and right handed.  Coordinates are not written back to
    the source molecular structure.
    """

    xyz = np.asarray(coordinates, dtype=np.float64)
    weights = np.asarray(masses, dtype=np.float64)
    center = np.sum(xyz * weights[:, np.newaxis], axis=0) / np.sum(weights)
    centered = xyz - center
    squared_radii = np.einsum("ij,ij->i", centered, centered)
    tensor = (
        np.eye(3) * np.sum(weights * squared_radii)
        - np.einsum("i,ij,ik->jk", weights, centered, centered)
    )
    moments, eigenvectors = np.linalg.eigh(tensor)
    order = np.argsort(moments)
    moments = moments[order]
    eigenvectors = eigenvectors[:, order]
    screen_x = _canonical_axis_sign(eigenvectors[:, 0])
    view_z = _canonical_axis_sign(eigenvectors[:, 2])
    screen_y = np.cross(view_z, screen_x)
    screen_y /= np.linalg.norm(screen_y)
    axes = np.column_stack((screen_x, screen_y, view_z))
    scale = max(float(np.max(np.abs(moments))), 1.0)
    degeneracy_tolerance = scale * 1.0e-10
    degenerate_pairs = (
        bool(abs(moments[1] - moments[0]) <= degeneracy_tolerance),
        bool(abs(moments[2] - moments[1]) <= degeneracy_tolerance),
    )
    return PrincipalAxes(
        center_of_mass=center,
        inertia_tensor=tensor,
        moments=moments,
        axes=axes,
        oriented_coordinates=centered @ axes,
        degenerate_pairs=degenerate_pairs,
    )


def _element_symbol(atom_name: str, atom_type: str) -> str:
    raw_symbol = atom_type.split(".", 1)[0]
    letters = "".join(character for character in raw_symbol if character.isalpha())
    if not letters:
        letters = "".join(character for character in atom_name if character.isalpha())
    symbol = letters[0].upper() + letters[1:].lower()
    periodictable.elements.symbol(symbol)
    return symbol


def _read_mol2_geometry(path: Path) -> _Mol2Geometry:
    coordinates = []
    masses = []
    elements = []
    in_atom_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "@<TRIPOS>ATOM":
            in_atom_block = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom_block = False
            continue
        if not in_atom_block or not line.strip():
            continue
        fields = line.split()
        symbol = _element_symbol(fields[1], fields[5])
        coordinates.append(tuple(float(value) for value in fields[2:5]))
        masses.append(float(periodictable.elements.symbol(symbol).mass))
        elements.append(symbol)
    if not coordinates:
        raise ValueError(f"no atoms found in {path}")
    return _Mol2Geometry(
        coordinates=np.asarray(coordinates, dtype=np.float64),
        masses=np.asarray(masses, dtype=np.float64),
        elements=tuple(elements),
    )


def _build_failed_cbond_ligand(
    case: GalleryCase,
    output_root: Path,
    seed: int,
) -> Path:
    from hotpot import read_mol
    from hotpot.cheminfo.obWrappers import build

    smiles = case.source_path.read_text(encoding="utf-8").strip().split()[0]
    ligand = read_mol(smiles, fmt="smi")
    ligand.add_hydrogens(
        rm_polar_hs=False,
        rng=np.random.default_rng(seed + case.index),
    )
    report = build(ligand)
    if not report.succeeded:
        raise RuntimeError("native Open Babel could not build the display ligand")
    structure_dir = output_root / "display_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    target = structure_dir / f"{case.index:04d}_failed_cbond_ligand.mol2"
    ligand.write(target, overwrite=True, write_single=True)
    return target


def _inertia_payload(result: PrincipalAxes) -> dict[str, object]:
    return {
        "mass_weighted": True,
        "center_of_mass_angstrom": result.center_of_mass.tolist(),
        "tensor_amu_angstrom2": result.inertia_tensor.tolist(),
        "principal_moments_amu_angstrom2": result.moments.tolist(),
        "display_axes_columns": result.axes.tolist(),
        "view_axis": "display_axes_columns[2] (maximum principal moment)",
        "basis_determinant": float(np.linalg.det(result.axes)),
        "degenerate_adjacent_moment_pairs": list(result.degenerate_pairs),
    }


def _configure_materials_studio_style(
    object_name: str,
    am_color: tuple[float, float, float],
) -> None:
    from pymol import cmd

    cmd.hide("everything", "all")
    cmd.show("sticks", object_name)
    cmd.show("spheres", object_name)
    cmd.set("stick_radius", 0.13)
    cmd.set("sphere_scale", 0.22, object_name)
    cmd.set("sphere_scale", 0.13, f"{object_name} and elem H")
    cmd.set("sphere_scale", 0.50, f"{object_name} and elem Am")
    cmd.set_color("gallery_carbon", [0.36, 0.36, 0.36])
    cmd.set_color("gallery_hydrogen", [1.0, 1.0, 1.0])
    cmd.set_color("gallery_nitrogen", [0.12, 0.26, 0.95])
    cmd.set_color("gallery_oxygen", [0.90, 0.08, 0.08])
    cmd.set_color("gallery_sulfur", [0.95, 0.82, 0.12])
    cmd.set_color("gallery_phosphorus", [0.95, 0.48, 0.10])
    cmd.set_color("gallery_halogen", [0.12, 0.72, 0.20])
    cmd.set_color("gallery_heavy_halogen", [0.55, 0.15, 0.08])
    cmd.set_color("gallery_am", list(am_color))
    cmd.color("gallery_carbon", f"{object_name} and elem C")
    cmd.color("gallery_hydrogen", f"{object_name} and elem H")
    cmd.color("gallery_nitrogen", f"{object_name} and elem N")
    cmd.color("gallery_oxygen", f"{object_name} and elem O")
    cmd.color("gallery_sulfur", f"{object_name} and elem S")
    cmd.color("gallery_phosphorus", f"{object_name} and elem P")
    cmd.color("gallery_halogen", f"{object_name} and elem F+Cl")
    cmd.color("gallery_heavy_halogen", f"{object_name} and elem Br+I")
    cmd.color("gallery_am", f"{object_name} and elem Am")
    cmd.set("bg_rgb", [1.0, 1.0, 1.0])
    cmd.set("orthoscopic", 1)
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.set("ambient", 0.38)
    cmd.set("direct", 0.62)
    cmd.set("specular", 0.24)
    cmd.set("shininess", 38)


def _render_pymol(
    source: Path,
    target: Path,
    oriented_coordinates: np.ndarray,
    parameters: GalleryParameters,
) -> None:
    from pymol import cmd

    object_name = "gallery_structure"
    cmd.reinitialize()
    cmd.set("retain_order", 1)
    cmd.load(str(source), object_name)
    cmd.load_coords(oriented_coordinates.tolist(), object_name, state=1)
    _configure_materials_studio_style(object_name, parameters.am_color_rgb)
    cmd.reset()
    cmd.zoom(object_name, buffer=1.6)
    cmd.ray(parameters.image_width, parameters.image_height)
    cmd.png(str(target), dpi=parameters.ray_dpi)


def _font(path: Path, size: int) -> ImageFont.ImageFont:
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (center_x - (bounds[2] - bounds[0]) // 2, y),
        text,
        font=font,
        fill=fill,
    )


def _write_placeholder(target: Path, case: GalleryCase, error: str) -> None:
    canvas = Image.new("RGB", (1000, 750), "white")
    draw = ImageDraw.Draw(canvas)
    _centered_text(
        draw,
        canvas.width // 2,
        285,
        f"Case {case.index:04d}",
        _font(BOLD_FONT, 42),
        (170, 45, 45),
    )
    _centered_text(
        draw,
        canvas.width // 2,
        350,
        error[:88],
        _font(REGULAR_FONT, 23),
        (65, 65, 65),
    )
    canvas.save(target)


def _render_gallery_case(
    case: GalleryCase,
    output_root: Path,
    parameters: GalleryParameters,
) -> dict[str, object]:
    images_dir = output_root / "cases"
    images_dir.mkdir(parents=True, exist_ok=True)
    target = images_dir / f"{case.index:04d}.png"
    source = case.source_path
    structure_origin = "optimized_or_last_finite_benchmark_frame"
    result: dict[str, object] = {
        "index": case.index,
        "group": case.group,
        "benchmark_status": case.status,
        "smiles": case.smiles,
        "output_frame_role": case.output_frame_role,
        "report_sha256": case.report_sha256,
        "input_source": str(case.source_path),
        "input_source_sha256": case.source_sha256,
        "rendered_structure": None,
        "rendered_structure_sha256": None,
        "structure_origin": structure_origin,
        "image": str(target),
        "render_status": "running",
        "render_error": None,
        "explicit_hydrogen_count": None,
        "americium_count": None,
        "inertia": None,
    }
    try:
        if case.group == GROUP_FAILED_CBOND:
            source = _build_failed_cbond_ligand(
                case,
                output_root,
                parameters.ligand_build_seed,
            )
            structure_origin = "visualization_only_explicit_hydrogen_3d_ligand"
        if not source.is_file():
            raise FileNotFoundError(source)
        geometry = _read_mol2_geometry(source)
        hydrogen_count = geometry.elements.count("H")
        if hydrogen_count == 0:
            raise ValueError("display structure contains no explicit hydrogen atoms")
        inertia = mass_weighted_principal_axes(
            geometry.coordinates,
            geometry.masses,
        )
        _render_pymol(
            source,
            target,
            inertia.oriented_coordinates,
            parameters,
        )
        result.update(
            rendered_structure=str(source),
            rendered_structure_sha256=_sha256(source),
            structure_origin=structure_origin,
            render_status="rendered",
            explicit_hydrogen_count=hydrogen_count,
            americium_count=geometry.elements.count("Am"),
            inertia=_inertia_payload(inertia),
        )
    except Exception as error:  # Each image and exact failure remain auditable.
        message = f"{type(error).__name__}: {error}"
        _write_placeholder(target, case, message)
        result.update(
            rendered_structure=str(source),
            rendered_structure_sha256=_sha256(source) if source.is_file() else None,
            structure_origin=structure_origin,
            render_status="placeholder",
            render_error=message,
        )
    return result


def _trim_image(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    bounds = ImageChops.difference(image, background).getbbox()
    if bounds is None:
        return image
    margin = 18
    left, top, right, bottom = bounds
    return image.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(image.width, right + margin),
            min(image.height, bottom + margin),
        )
    )


def _write_contact_sheet(
    output: Path,
    title: str,
    cases: Sequence[Mapping[str, object]],
    parameters: GalleryParameters,
) -> None:
    column_count = max(1, min(parameters.columns, len(cases)))
    row_count = max(1, (len(cases) + column_count - 1) // column_count)
    grid_width = column_count * parameters.cell_width
    canvas_width = max(1000, grid_width)
    grid_left = (canvas_width - grid_width) // 2
    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            parameters.sheet_header_height + row_count * parameters.cell_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(BOLD_FONT, 34)
    label_font = _font(BOLD_FONT, 18)
    status_font = _font(REGULAR_FONT, 14)
    _centered_text(draw, canvas.width // 2, 20, title, title_font, (25, 25, 25))
    molecule_height = parameters.cell_height - 58
    for position, item in enumerate(cases):
        column = position % column_count
        row = position // column_count
        x0 = grid_left + column * parameters.cell_width
        y0 = parameters.sheet_header_height + row * parameters.cell_height
        with Image.open(str(item["image"])) as image:
            molecule_image = _trim_image(image.convert("RGB"))
            molecule_image.thumbnail(
                (parameters.cell_width - 20, molecule_height - 8),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(
                molecule_image,
                (
                    x0 + (parameters.cell_width - molecule_image.width) // 2,
                    y0 + (molecule_height - molecule_image.height) // 2,
                ),
            )
        status = str(item["benchmark_status"])
        color = (32, 112, 70) if status == "passed" else (174, 76, 42)
        _centered_text(
            draw,
            x0 + parameters.cell_width // 2,
            y0 + molecule_height,
            f"Case {int(item['index']):04d}",
            label_font,
            color,
        )
        _centered_text(
            draw,
            x0 + parameters.cell_width // 2,
            y0 + molecule_height + 27,
            status,
            status_font,
            (60, 60, 60),
        )
        draw.rectangle(
            (
                x0 + 4,
                y0 + 3,
                x0 + parameters.cell_width - 5,
                y0 + parameters.cell_height - 5,
            ),
            outline=(210, 210, 210),
            width=2,
        )
    canvas.save(output)


def _repository_state() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _benchmark_commit(benchmark_root: Path) -> Optional[str]:
    manifest_path = benchmark_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    git = manifest.get("git")
    if not isinstance(git, Mapping) or git.get("commit") is None:
        return None
    return str(git["commit"])


def _aggregate_input_hash(cases: Sequence[GalleryCase]) -> str:
    payload = [
        {
            "index": case.index,
            "group": case.group,
            "report_sha256": case.report_sha256,
            "source_sha256": case.source_sha256,
        }
        for case in cases
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _contact_sheet_evidence(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "filename": path.name,
        "sha256": _sha256(path),
        "width": width,
        "height": height,
    }


def _public_evidence_payload(
    payload: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    """Return path-free evidence suitable for a tracked README asset."""
    cases = payload["cases"]
    public_cases = [
        {
            "index": item["index"],
            "group": item["group"],
            "benchmark_status": item["benchmark_status"],
            "output_frame_role": item["output_frame_role"],
            "report_sha256": item["report_sha256"],
            "input_source_sha256": item["input_source_sha256"],
            "rendered_structure_sha256": item["rendered_structure_sha256"],
            "structure_origin": item["structure_origin"],
            "render_status": item["render_status"],
            "explicit_hydrogen_count": item["explicit_hydrogen_count"],
            "americium_count": item["americium_count"],
            "principal_moments_amu_angstrom2": (
                item["inertia"]["principal_moments_amu_angstrom2"]
                if item["inertia"] is not None
                else None
            ),
            "basis_determinant": (
                item["inertia"]["basis_determinant"]
                if item["inertia"] is not None
                else None
            ),
        }
        for item in cases
    ]
    return {
        "schema_version": 1,
        "input_sha256": payload["input_sha256"],
        "benchmark_commit": payload["benchmark_commit"],
        "renderer_git": payload["renderer_git"],
        "renderer_sha256": payload["renderer_sha256"],
        "parameters": payload["parameters"],
        "sample_count": len(public_cases),
        "quality_passed_count": sum(
            item["benchmark_status"] == "passed" for item in public_cases
        ),
        "groups": {
            group: {
                "count": group_payload["count"],
                "case_indices": group_payload["case_indices"],
            }
            for group, group_payload in payload["groups"].items()
        },
        "rendered_count": payload["rendered_count"],
        "placeholder_count": payload["placeholder_count"],
        "contact_sheets": {
            GROUP_CBOND: _contact_sheet_evidence(
                output_root / "am_cbond_complexes.png"
            ),
            GROUP_FAILED_CBOND: _contact_sheet_evidence(
                output_root / "am_failed_cbond_ligands.png"
            ),
        },
        "cases": public_cases,
    }


def render_am_gallery(
    benchmark_root: Path,
    output_root: Path,
    *,
    workers: int = 1,
    parameters: GalleryParameters = GalleryParameters(),
) -> dict[str, object]:
    """Render case images, two exhaustive group sheets, and ``gallery.json``."""

    if importlib.util.find_spec("pymol") is None:
        raise RuntimeError("PyMOL is required to render the Am benchmark gallery")
    cases = discover_gallery_cases(benchmark_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rendered = []
    if workers == 1:
        for completed, case in enumerate(cases, start=1):
            rendered.append(_render_gallery_case(case, output_root, parameters))
            print(
                f"[Am gallery {completed}/{len(cases)}] case={case.index:04d}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                pool.submit(_render_gallery_case, case, output_root, parameters): case
                for case in cases
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                case = futures[future]
                rendered.append(future.result())
                print(
                    f"[Am gallery {completed}/{len(cases)}] case={case.index:04d}",
                    flush=True,
                )
    rendered.sort(key=lambda item: int(item["index"]))
    cbond_items = [item for item in rendered if item["group"] == GROUP_CBOND]
    failed_cbond_items = [
        item for item in rendered if item["group"] == GROUP_FAILED_CBOND
    ]
    _write_contact_sheet(
        output_root / "am_cbond_complexes.png",
        f"Am-ligand complexes with CBond (n={len(cbond_items)})",
        cbond_items,
        parameters,
    )
    _write_contact_sheet(
        output_root / "am_failed_cbond_ligands.png",
        f"Ligands without an inferred Am CBond (n={len(failed_cbond_items)})",
        failed_cbond_items,
        parameters,
    )
    payload = {
        "schema_version": 1,
        "benchmark_root": str(benchmark_root),
        "input_sha256": _aggregate_input_hash(cases),
        "benchmark_commit": _benchmark_commit(benchmark_root),
        "renderer_git": _repository_state(),
        "renderer_sha256": _sha256(Path(__file__)),
        "parameters": asdict(parameters),
        "groups": {
            GROUP_CBOND: {
                "count": len(cbond_items),
                "case_indices": [int(item["index"]) for item in cbond_items],
                "contact_sheet": str(output_root / "am_cbond_complexes.png"),
            },
            GROUP_FAILED_CBOND: {
                "count": len(failed_cbond_items),
                "case_indices": [
                    int(item["index"]) for item in failed_cbond_items
                ],
                "contact_sheet": str(output_root / "am_failed_cbond_ligands.png"),
            },
        },
        "rendered_count": sum(
            item["render_status"] == "rendered" for item in rendered
        ),
        "placeholder_count": sum(
            item["render_status"] == "placeholder" for item in rendered
        ),
        "cases": rendered,
    }
    (output_root / "gallery.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    public_evidence = _public_evidence_payload(payload, output_root)
    (output_root / "gallery_evidence.json").write_text(
        json.dumps(public_evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render an existing Am coordination benchmark as two auditable, "
            "Materials Studio-inspired galleries."
        )
    )
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    parser.add_argument("--columns", type=int, default=15)
    parser.add_argument("--image-width", type=int, default=1000)
    parser.add_argument("--image-height", type=int, default=750)
    parser.add_argument("--ray-dpi", type=int, default=180)
    parser.add_argument("--ligand-build-seed", type=int, default=20261009)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _build_parser().parse_args(argv)
    benchmark_root = arguments.benchmark_root.resolve()
    output_root = (
        arguments.output.resolve()
        if arguments.output is not None
        else benchmark_root / "am_gallery"
    )
    parameters = GalleryParameters(
        image_width=arguments.image_width,
        image_height=arguments.image_height,
        ray_dpi=arguments.ray_dpi,
        columns=arguments.columns,
        ligand_build_seed=arguments.ligand_build_seed,
    )
    report = render_am_gallery(
        benchmark_root,
        output_root,
        workers=arguments.workers,
        parameters=parameters,
    )
    print(
        f"Rendered {report['rendered_count']} cases; "
        f"placeholders={report['placeholder_count']}; output={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
