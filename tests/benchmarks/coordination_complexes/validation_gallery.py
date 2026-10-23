"""Exhaustive galleries and human-review evidence for Plan-17 validation.

The renderer consumes the normalized O0/F1/F2/C1/C2 result matrix.  It only
opens ``selected_structure`` for an automatically passed result; failed rows
and rows without a selected structure receive an explicit placeholder.  In
particular, this module never searches trajectory or last-finite-frame assets
for an alternative image.

Rendering creates a *pending* manual-review template.  A human reviewer must
fill and complete that template before :func:`validate_manual_review_artifact`
will accept it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from PIL import Image, ImageDraw

from .rendering import (
    BOLD_FONT,
    REGULAR_FONT,
    _centered_text,
    _font,
    _trim_render,
)
from .validation_dataset import FORMAL_CASE_INDICES, ROUTES


GALLERY_SCHEMA = "hotpot-plan17-validation-gallery/v1"
MANUAL_REVIEW_SCHEMA = "hotpot-plan17-manual-review/v1"


@dataclass(frozen=True)
class ValidationGalleryParameters:
    """Stable rendering and contact-sheet dimensions recorded in evidence."""

    image_width: int = 1000
    image_height: int = 750
    ray_dpi: int = 180
    columns: int = 8
    cell_width: int = 430
    cell_height: int = 350
    header_height: int = 90


StructureRenderer = Callable[[Path, Path, ValidationGalleryParameters], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _require_digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _normalized_matrix(
    records: Iterable[Mapping[str, object]],
    case_indices: Sequence[int],
) -> tuple[dict[str, object], ...]:
    if (
        not case_indices
        or len(case_indices) != len(set(case_indices))
        or any(
            not isinstance(case_index, int) or isinstance(case_index, bool)
            for case_index in case_indices
        )
    ):
        raise ValueError("gallery case_indices must be unique integers")
    expected = {(case_index, route) for route in ROUTES for case_index in case_indices}
    normalized: dict[tuple[int, str], dict[str, object]] = {}
    run_ids: dict[str, str] = {}
    for value in records:
        route = value.get("route")
        case_index = value.get("case_index")
        if route not in ROUTES:
            raise ValueError(f"gallery route must be one of {ROUTES}")
        if not isinstance(case_index, int) or isinstance(case_index, bool):
            raise ValueError("gallery case_index must be an integer")
        run_id = value.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("gallery run_id must be non-empty")
        if route in run_ids and run_ids[route] != run_id:
            raise ValueError(f"route {route} has more than one run_id")
        run_ids[route] = run_id
        if type(value.get("passed")) is not bool:
            raise ValueError("gallery passed must be Boolean")
        if not isinstance(value.get("status"), str) or not value["status"]:
            raise ValueError("gallery status must be non-empty")
        key = (case_index, str(route))
        if key in normalized:
            raise ValueError(f"duplicate gallery route-case result: {key}")
        normalized[key] = dict(value)
    actual = set(normalized)
    if actual != expected:
        raise ValueError(
            f"gallery matrix mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return tuple(
        normalized[(case_index, route)]
        for route in ROUTES
        for case_index in case_indices
    )


def _default_structure_renderer(
    source: Path,
    target: Path,
    parameters: ValidationGalleryParameters,
) -> None:
    if importlib.util.find_spec("pymol") is None:
        raise RuntimeError("PyMOL is unavailable")
    from pymol import cmd

    from .am_gallery import _configure_materials_studio_style

    cmd.reinitialize()
    cmd.set("retain_order", 1)
    cmd.load(str(source), "validation_structure")
    _configure_materials_studio_style(
        "validation_structure",
        (0.43, 0.38, 0.94),
    )
    cmd.orient("validation_structure")
    cmd.zoom("validation_structure", buffer=1.6)
    cmd.ray(parameters.image_width, parameters.image_height)
    cmd.png(str(target), dpi=parameters.ray_dpi)


def _placeholder(
    target: Path,
    *,
    route: str,
    case_index: int,
    status: str,
    reason: str,
) -> None:
    canvas = Image.new("RGB", (1000, 750), "white")
    draw = ImageDraw.Draw(canvas)
    _centered_text(
        draw,
        canvas.width // 2,
        235,
        f"{route} | Case {case_index:04d}",
        _font(BOLD_FONT, 42),
        (166, 48, 48),
    )
    _centered_text(
        draw,
        canvas.width // 2,
        315,
        "NO SELECTED STRUCTURE DISPLAYED",
        _font(BOLD_FONT, 27),
        (166, 48, 48),
    )
    _centered_text(
        draw,
        canvas.width // 2,
        370,
        f"status: {status}"[:88],
        _font(REGULAR_FONT, 23),
        (60, 60, 60),
    )
    _centered_text(
        draw,
        canvas.width // 2,
        410,
        reason[:88],
        _font(REGULAR_FONT, 21),
        (60, 60, 60),
    )
    canvas.save(target)


def _selected_source(
    record: Mapping[str, object],
    result_root: Path,
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    """Return a verified selected source, its digest, and an error code."""
    selected = record.get("selected_structure")
    if not isinstance(selected, Mapping):
        return None, None, "no_selected_structure"
    locator = selected.get("path")
    digest = selected.get("sha256")
    if not isinstance(locator, str) or not locator:
        return None, None, "no_selected_structure"
    try:
        expected_digest = _require_digest(digest, "selected_structure.sha256")
    except ValueError:
        return None, None, "selected_structure_digest_invalid"
    source = Path(locator).expanduser()
    source = source if source.is_absolute() else result_root / source
    source = source.resolve()
    if not source.is_file():
        return None, expected_digest, "selected_structure_unavailable"
    if _sha256(source) != expected_digest:
        return None, expected_digest, "selected_structure_digest_mismatch"
    return source, expected_digest, None


def _render_case(
    record: Mapping[str, object],
    output_root: Path,
    result_root: Path,
    renderer: StructureRenderer,
    parameters: ValidationGalleryParameters,
) -> dict[str, object]:
    route = str(record["route"])
    case_index = int(record["case_index"])
    status = str(record["status"])
    passed = bool(record["passed"])
    target = output_root / "routes" / route / "cases" / f"{case_index:04d}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    selected_digest: Optional[str] = None
    render_error: Optional[str] = None

    if not passed:
        render_status = "scientific_placeholder"
        placeholder_reason = f"route_failed:{status}"
        _placeholder(
            target,
            route=route,
            case_index=case_index,
            status=status,
            reason=placeholder_reason,
        )
    else:
        source, selected_digest, source_error = _selected_source(
            record,
            result_root,
        )
        if source_error == "no_selected_structure":
            render_status = "scientific_placeholder"
            placeholder_reason = source_error
            _placeholder(
                target,
                route=route,
                case_index=case_index,
                status=status,
                reason=placeholder_reason,
            )
        elif source_error is not None:
            render_status = "render_error_placeholder"
            placeholder_reason = source_error
            render_error = source_error
            _placeholder(
                target,
                route=route,
                case_index=case_index,
                status=status,
                reason=placeholder_reason,
            )
        else:
            assert source is not None
            try:
                renderer(source, target, parameters)
                if not target.is_file():
                    raise RuntimeError("renderer did not create the case image")
            except Exception as error:  # Preserve one auditable cell per result.
                render_status = "render_error_placeholder"
                placeholder_reason = "selected_structure_render_failed"
                render_error = f"{type(error).__name__}: {error}"
                _placeholder(
                    target,
                    route=route,
                    case_index=case_index,
                    status=status,
                    reason=render_error,
                )
            else:
                render_status = "rendered_selected_structure"
                placeholder_reason = None

    return {
        "route": route,
        "run_id": str(record["run_id"]),
        "case_index": case_index,
        "status": status,
        "passed": passed,
        "display_role": (
            "selected_structure"
            if render_status == "rendered_selected_structure"
            else "placeholder"
        ),
        "render_status": render_status,
        "placeholder_reason": placeholder_reason,
        "render_error": render_error,
        "selected_structure_sha256": selected_digest,
        "image": target.relative_to(output_root).as_posix(),
        "image_sha256": _sha256(target),
    }


def _write_route_sheet(
    target: Path,
    route: str,
    run_id: str,
    cases: Sequence[Mapping[str, object]],
    output_root: Path,
    parameters: ValidationGalleryParameters,
) -> None:
    columns = max(1, min(parameters.columns, len(cases)))
    rows = max(1, (len(cases) + columns - 1) // columns)
    canvas = Image.new(
        "RGB",
        (
            columns * parameters.cell_width,
            parameters.header_height + rows * parameters.cell_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    _centered_text(
        draw,
        canvas.width // 2,
        19,
        f"Plan-17 {route} | run {run_id} | n={len(cases)}",
        _font(BOLD_FONT, 32),
        (25, 25, 25),
    )
    image_height = parameters.cell_height - 70
    for position, item in enumerate(cases):
        column = position % columns
        row = position // columns
        x0 = column * parameters.cell_width
        y0 = parameters.header_height + row * parameters.cell_height
        with Image.open(output_root / str(item["image"])) as source_image:
            image = _trim_render(source_image.convert("RGB"))
            image.thumbnail(
                (parameters.cell_width - 22, image_height - 8),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(
                image,
                (
                    x0 + (parameters.cell_width - image.width) // 2,
                    y0 + (image_height - image.height) // 2,
                ),
            )
        selected = item["display_role"] == "selected_structure"
        color = (33, 116, 70) if selected else (174, 67, 51)
        label = (
            f"Case {int(item['case_index']):04d} | selected"
            if selected
            else f"Case {int(item['case_index']):04d} | placeholder"
        )
        _centered_text(
            draw,
            x0 + parameters.cell_width // 2,
            y0 + image_height,
            label,
            _font(BOLD_FONT, 17),
            color,
        )
        _centered_text(
            draw,
            x0 + parameters.cell_width // 2,
            y0 + image_height + 27,
            str(item["status"])[:48],
            _font(REGULAR_FONT, 14),
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
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def render_validation_gallery(
    records: Iterable[Mapping[str, object]],
    output_root: Path,
    *,
    validation_revision: str,
    result_root: Optional[Path] = None,
    case_indices: Sequence[int] = FORMAL_CASE_INDICES,
    renderer: Optional[StructureRenderer] = None,
    parameters: ValidationGalleryParameters = ValidationGalleryParameters(),
) -> dict[str, object]:
    """Render five exhaustive route galleries and a pending review template."""
    if not validation_revision.strip():
        raise ValueError("validation_revision must be non-empty")
    ordered = _normalized_matrix(records, case_indices)
    output_root = output_root.resolve()
    result_root = Path.cwd().resolve() if result_root is None else result_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_renderer = renderer or _default_structure_renderer
    cases = [
        _render_case(
            record,
            output_root,
            result_root,
            selected_renderer,
            parameters,
        )
        for record in ordered
    ]

    route_payloads = []
    for route in ROUTES:
        route_cases = [item for item in cases if item["route"] == route]
        run_id = str(route_cases[0]["run_id"])
        sheet = output_root / "routes" / route / "gallery.png"
        _write_route_sheet(
            sheet,
            route,
            run_id,
            route_cases,
            output_root,
            parameters,
        )
        route_payloads.append(
            {
                "route": route,
                "run_id": run_id,
                "case_count": len(route_cases),
                "rendered_selected_count": sum(
                    item["render_status"] == "rendered_selected_structure"
                    for item in route_cases
                ),
                "scientific_placeholder_count": sum(
                    item["render_status"] == "scientific_placeholder"
                    for item in route_cases
                ),
                "render_error_count": sum(
                    item["render_status"] == "render_error_placeholder"
                    for item in route_cases
                ),
                "contact_sheet": sheet.relative_to(output_root).as_posix(),
                "contact_sheet_sha256": _sha256(sheet),
            }
        )

    matrix_identity = [
        {
            "route": item["route"],
            "run_id": item["run_id"],
            "case_index": item["case_index"],
            "status": item["status"],
            "passed": item["passed"],
            "selected_structure_sha256": item["selected_structure_sha256"],
        }
        for item in cases
    ]
    payload = {
        "schema": GALLERY_SCHEMA,
        "validation_revision": validation_revision,
        "case_indices": list(case_indices),
        "input_matrix_sha256": hashlib.sha256(
            _canonical_bytes(matrix_identity)
        ).hexdigest(),
        "parameters": asdict(parameters),
        "routes": route_payloads,
        "cases": cases,
    }
    manifest_path = output_root / "gallery.json"
    _write_json(manifest_path, payload)
    create_manual_review_template(
        manifest_path,
        output_root / "manual_review.template.json",
    )
    return payload


_GALLERY_FIELDS = {
    "schema",
    "validation_revision",
    "case_indices",
    "input_matrix_sha256",
    "parameters",
    "routes",
    "cases",
}
_ROUTE_FIELDS = {
    "route",
    "run_id",
    "case_count",
    "rendered_selected_count",
    "scientific_placeholder_count",
    "render_error_count",
    "contact_sheet",
    "contact_sheet_sha256",
}
_CASE_FIELDS = {
    "route",
    "run_id",
    "case_index",
    "status",
    "passed",
    "display_role",
    "render_status",
    "placeholder_reason",
    "render_error",
    "selected_structure_sha256",
    "image",
    "image_sha256",
}


def _verified_asset(root: Path, locator: object, digest: object, field: str) -> Path:
    if not isinstance(locator, str) or not locator:
        raise ValueError(f"{field} path must be non-empty")
    relative = Path(locator)
    if relative.is_absolute():
        raise ValueError(f"{field} path must be relative to the gallery")
    asset = (root / relative).resolve()
    if not asset.is_relative_to(root):
        raise ValueError(f"{field} path escapes the gallery")
    expected = _require_digest(digest, f"{field} SHA-256")
    if not asset.is_file() or _sha256(asset) != expected:
        raise ValueError(f"{field} asset is missing or its SHA-256 differs")
    return asset


def _load_gallery(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _GALLERY_FIELDS:
        raise ValueError("gallery manifest has unexpected fields")
    if value["schema"] != GALLERY_SCHEMA:
        raise ValueError("gallery schema is unsupported")
    if (
        not isinstance(value["validation_revision"], str)
        or not value["validation_revision"]
    ):
        raise ValueError("gallery validation_revision must be non-empty")
    _require_digest(value["input_matrix_sha256"], "input_matrix_sha256")
    case_indices = value["case_indices"]
    if (
        not isinstance(case_indices, list)
        or not case_indices
        or len(case_indices) != len(set(case_indices))
        or any(
            not isinstance(case_index, int) or isinstance(case_index, bool)
            for case_index in case_indices
        )
    ):
        raise ValueError("gallery case_indices must be unique integers")
    routes = value["routes"]
    cases = value["cases"]
    if not isinstance(routes, list) or not isinstance(cases, list):
        raise ValueError("gallery routes and cases must be arrays")
    route_by_name: dict[str, Mapping[str, object]] = {}
    for route_value in routes:
        if not isinstance(route_value, Mapping) or set(route_value) != _ROUTE_FIELDS:
            raise ValueError("gallery route has unexpected fields")
        route = route_value["route"]
        if route not in ROUTES or route in route_by_name:
            raise ValueError("gallery route identities are invalid")
        if not isinstance(route_value["run_id"], str) or not route_value["run_id"]:
            raise ValueError("gallery route run_id must be non-empty")
        _require_digest(
            route_value["contact_sheet_sha256"],
            f"{route}.contact_sheet_sha256",
        )
        _verified_asset(
            path.parent,
            route_value["contact_sheet"],
            route_value["contact_sheet_sha256"],
            f"{route} contact sheet",
        )
        route_by_name[str(route)] = route_value
    if tuple(route_by_name) != ROUTES:
        raise ValueError("gallery must contain O0/F1/F2/C1/C2 in fixed order")

    keys: set[tuple[str, int]] = set()
    indices_by_route: dict[str, set[int]] = {route: set() for route in ROUTES}
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != _CASE_FIELDS:
            raise ValueError("gallery case has unexpected fields")
        route = case["route"]
        case_index = case["case_index"]
        if route not in ROUTES:
            raise ValueError("gallery case route is invalid")
        if not isinstance(case_index, int) or isinstance(case_index, bool):
            raise ValueError("gallery case_index must be an integer")
        key = (str(route), case_index)
        if key in keys:
            raise ValueError(f"duplicate gallery case: {key}")
        keys.add(key)
        if case["run_id"] != route_by_name[str(route)]["run_id"]:
            raise ValueError(f"gallery case run_id mismatch: {key}")
        _verified_asset(
            path.parent,
            case["image"],
            case["image_sha256"],
            f"gallery image {key}",
        )
        if type(case["passed"]) is not bool:
            raise ValueError(f"gallery passed must be Boolean for {key}")
        render_status = case["render_status"]
        if render_status not in {
            "rendered_selected_structure",
            "scientific_placeholder",
            "render_error_placeholder",
        }:
            raise ValueError(f"gallery render_status is invalid for {key}")
        expected_role = (
            "selected_structure"
            if render_status == "rendered_selected_structure"
            else "placeholder"
        )
        if case["display_role"] != expected_role:
            raise ValueError(f"gallery display_role is inconsistent for {key}")
        if not case["passed"] and (
            render_status != "scientific_placeholder"
            or case["selected_structure_sha256"] is not None
        ):
            raise ValueError(f"failed gallery case exposes a structure for {key}")
        if render_status == "rendered_selected_structure":
            _require_digest(
                case["selected_structure_sha256"],
                f"selected structure {key}",
            )
        indices_by_route[str(route)].add(case_index)
    expected_indices = set(case_indices)
    if any(indices_by_route[route] != expected_indices for route in ROUTES):
        raise ValueError("gallery route case coverage is not exhaustive and identical")
    for route in ROUTES:
        if route_by_name[route]["case_count"] != len(expected_indices):
            raise ValueError(f"gallery {route} case_count is inconsistent")
        route_cases = [case for case in cases if case["route"] == route]
        expected_counts = {
            "rendered_selected_count": sum(
                case["render_status"] == "rendered_selected_structure"
                for case in route_cases
            ),
            "scientific_placeholder_count": sum(
                case["render_status"] == "scientific_placeholder"
                for case in route_cases
            ),
            "render_error_count": sum(
                case["render_status"] == "render_error_placeholder"
                for case in route_cases
            ),
        }
        for field, expected_count in expected_counts.items():
            if route_by_name[route][field] != expected_count:
                raise ValueError(f"gallery {route} {field} is inconsistent")
    return value


def _manual_review_routes(gallery: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "route": route["route"],
            "run_id": route["run_id"],
            "gallery_sha256": route["contact_sheet_sha256"],
        }
        for route in gallery["routes"]
    ]


def _review_rows(gallery: Mapping[str, object]) -> list[dict[str, object]]:
    order = {route: offset for offset, route in enumerate(ROUTES)}
    cases = sorted(
        gallery["cases"],
        key=lambda item: (int(item["case_index"]), order[str(item["route"])]),
    )
    return [
        {
            "route": case["route"],
            "run_id": case["run_id"],
            "case_index": case["case_index"],
            "gallery_image_sha256": case["image_sha256"],
            "reviewed": False,
            "flagged": False,
            "reason": None,
        }
        for case in cases
    ]


def create_manual_review_template(
    gallery_manifest: Path,
    output: Optional[Path] = None,
) -> dict[str, object]:
    """Create an exhaustive template without claiming human completion."""
    gallery_manifest = gallery_manifest.resolve()
    gallery = _load_gallery(gallery_manifest)
    reviews = _review_rows(gallery)
    value = {
        "schema": MANUAL_REVIEW_SCHEMA,
        "validation_revision": gallery["validation_revision"],
        "gallery_manifest_sha256": _sha256(gallery_manifest),
        "status": "pending_human_review",
        "reviewer": None,
        "completed_at": None,
        "routes": _manual_review_routes(gallery),
        "reviews": reviews,
        "summary": {
            "expected_case_count": len(reviews),
            "reviewed_case_count": 0,
            "flagged_case_count": 0,
        },
    }
    if output is not None:
        _write_json(output, value)
    return value


_MANUAL_FIELDS = {
    "schema",
    "validation_revision",
    "gallery_manifest_sha256",
    "status",
    "reviewer",
    "completed_at",
    "routes",
    "reviews",
    "summary",
}
_MANUAL_ROUTE_FIELDS = {"route", "run_id", "gallery_sha256"}
_REVIEW_FIELDS = {
    "route",
    "run_id",
    "case_index",
    "gallery_image_sha256",
    "reviewed",
    "flagged",
    "reason",
}
_SUMMARY_FIELDS = {
    "expected_case_count",
    "reviewed_case_count",
    "flagged_case_count",
}


def _timezone_aware(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def validate_manual_review_artifact(
    value: Mapping[str, object],
    gallery_manifest: Path,
) -> dict[str, object]:
    """Validate one completed, attributable, exhaustive human review."""
    gallery_manifest = gallery_manifest.resolve()
    gallery = _load_gallery(gallery_manifest)
    if set(value) != _MANUAL_FIELDS:
        raise ValueError("manual review artifact has unexpected fields")
    if value["schema"] != MANUAL_REVIEW_SCHEMA:
        raise ValueError("manual review schema is unsupported")
    if value["validation_revision"] != gallery["validation_revision"]:
        raise ValueError("manual review validation_revision differs from gallery")
    if value["gallery_manifest_sha256"] != _sha256(gallery_manifest):
        raise ValueError("manual review gallery manifest hash differs")
    if value["status"] != "complete":
        raise ValueError("manual review status must be complete")
    if not isinstance(value["reviewer"], str) or not value["reviewer"].strip():
        raise ValueError("manual review requires an attributable reviewer")
    if not _timezone_aware(value["completed_at"]):
        raise ValueError("manual review completed_at must include a timezone")
    if any(route["render_error_count"] for route in gallery["routes"]):
        raise ValueError("gallery render errors prevent completed manual review")

    expected_routes = _manual_review_routes(gallery)
    if not isinstance(value["routes"], list):
        raise ValueError("manual review routes must be an array")
    for route in value["routes"]:
        if not isinstance(route, Mapping) or set(route) != _MANUAL_ROUTE_FIELDS:
            raise ValueError("manual review route has unexpected fields")
    if value["routes"] != expected_routes:
        raise ValueError("manual review route/run/gallery identities differ")

    expected_rows = _review_rows(gallery)
    expected = {
        (str(row["route"]), int(row["case_index"])): row for row in expected_rows
    }
    reviews = value["reviews"]
    if not isinstance(reviews, list):
        raise ValueError("manual review reviews must be an array")
    observed: dict[tuple[str, int], Mapping[str, object]] = {}
    flagged_count = 0
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != _REVIEW_FIELDS:
            raise ValueError("manual review row has unexpected fields")
        route = review["route"]
        case_index = review["case_index"]
        if (
            route not in ROUTES
            or not isinstance(case_index, int)
            or isinstance(case_index, bool)
        ):
            raise ValueError("manual review row identity is invalid")
        key = (str(route), case_index)
        if key in observed:
            raise ValueError(f"duplicate manual review row: {key}")
        observed[key] = review
        expected_row = expected.get(key)
        if expected_row is None:
            raise ValueError(f"unexpected manual review row: {key}")
        for field in ("run_id", "gallery_image_sha256"):
            if review[field] != expected_row[field]:
                raise ValueError(f"manual review {field} differs for {key}")
        if review["reviewed"] is not True:
            raise ValueError(f"manual review is incomplete for {key}")
        if type(review["flagged"]) is not bool:
            raise ValueError(f"manual review flagged must be Boolean for {key}")
        reason = review["reason"]
        if review["flagged"]:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"flagged manual review requires a reason for {key}")
            flagged_count += 1
        elif reason is not None:
            raise ValueError(f"unflagged manual review reason must be null for {key}")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise ValueError(f"manual review is missing route/case rows: {missing}")

    summary = value["summary"]
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_FIELDS:
        raise ValueError("manual review summary has unexpected fields")
    expected_summary = {
        "expected_case_count": len(expected),
        "reviewed_case_count": len(observed),
        "flagged_case_count": flagged_count,
    }
    if dict(summary) != expected_summary:
        raise ValueError("manual review summary is inconsistent")
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    return dict(value)


def write_manual_review_artifact(
    value: Mapping[str, object],
    gallery_manifest: Path,
    output: Path,
) -> dict[str, object]:
    """Validate and write a completed human-supplied review artifact."""
    validated = validate_manual_review_artifact(value, gallery_manifest)
    _write_json(output, validated)
    return validated


__all__ = (
    "GALLERY_SCHEMA",
    "MANUAL_REVIEW_SCHEMA",
    "ValidationGalleryParameters",
    "create_manual_review_template",
    "render_validation_gallery",
    "validate_manual_review_artifact",
    "write_manual_review_artifact",
)
