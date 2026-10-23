"""Failure diagnostics and efficiency reporting for Plan-17 formal results.

This module does not change route acceptance or manufacture accepted assets.
It extracts rejected trajectory frames when they exist and otherwise renders
the input ligand graph with an explicit ``no route 3D frame`` label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path

ROUTES = ("O0", "F1", "F2", "C1", "C2")
ANALYSIS_SCHEMA = "hotpot-plan17-failure-analysis/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_identity(path: Path) -> dict[str, object]:
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(file_path)))
        total_bytes += file_path.stat().st_size
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


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


def _failed_gates(record: Mapping[str, object]) -> tuple[str, ...]:
    acceptance = record["acceptance"]
    return tuple(
        str(gate["gate_id"])
        for gate in acceptance["gates"]
        if not gate["passed"]
    )


def classify_failure(record: Mapping[str, object]) -> tuple[str, str]:
    """Return the stable failure category and a concise direct reason."""
    error = record.get("error") or {}
    message = str(error.get("message", ""))
    if "triangle smoothing" in message:
        return (
            "c2_bounds_triangle_smoothing",
            (
                "The fixed C2 coordination-bounds matrix is triangle-inconsistent; "
                "ETKDG did not start."
            ),
        )
    if "no conformers" in message or "single conformer provider call" in message:
        scope = "ligand" if record["route"] in {"F1", "F2"} else "whole complex"
        return (
            "etkdg_no_conformer",
            f"The fixed-seed ETKDG {scope} proposal returned no conformer.",
        )

    evidence = record.get("evidence") or {}
    if evidence.get("unavailable_gradient_events"):
        return (
            "unrecovered_unavailable_gradient",
            (
                "Optimization reached a coordinate frame but no finite gradient "
                "direction was restored by the established workflow."
            ),
        )

    quality = evidence.get("quality") or {}
    failures = Counter(str(name) for name in quality.get("failure_names", ()))
    if "A08" in _failed_gates(record):
        details = ", ".join(
            f"{name} x{count}" if count > 1 else name
            for name, count in sorted(failures.items())
        )
        return (
            "a08_geometry_rejected",
            f"A finite selected frame failed the standard geometry gate: {details}.",
        )
    return (
        "other_explicit_failure",
        message or ", ".join(_failed_gates(record)),
    )


def _selected_values(record: Mapping[str, object]) -> dict[str, object]:
    evidence = record.get("evidence") or {}
    selected = next(
        (
            frame
            for frame in evidence.get("frames", ())
            if frame.get("role") == "selected"
        ),
        None,
    )
    if selected is None:
        return {
            "frame_id": None,
            "coordinates_finite": None,
            "energy": None,
            "rms_gradient": None,
            "max_gradient": None,
        }
    return {
        "frame_id": selected.get("frame_id"),
        "coordinates_finite": selected.get("coordinates_finite"),
        "energy": selected.get("energy", {}).get("value"),
        "rms_gradient": selected.get("rms_gradient", {}).get("value"),
        "max_gradient": selected.get("max_gradient", {}).get("value"),
    }


def failure_rows(
    records: Iterable[Mapping[str, object]],
    cohort: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build one explicit diagnostic row per failed primary route-case."""
    cases = {int(case["index"]): case for case in cohort["cases"]}
    rows = []
    for record in records:
        if record["passed"]:
            continue
        case_index = int(record["case_index"])
        case = cases[case_index]
        category, reason = classify_failure(record)
        evidence = record.get("evidence") or {}
        quality = evidence.get("quality") or {}
        unavailable = evidence.get("unavailable_gradient_events", ())
        error = record.get("error") or {}
        rows.append(
            {
                "route": str(record["route"]),
                "run_id": str(record["run_id"]),
                "case_index": case_index,
                "status": str(record["status"]),
                "failure_category": category,
                "failure_phase": error.get("phase") or "acceptance_evaluation",
                "direct_reason": reason,
                "error_type": error.get("type"),
                "error_message": error.get("message"),
                "failed_gates": list(_failed_gates(record)),
                "quality_failure_counts": dict(
                    sorted(
                        Counter(
                            str(name)
                            for name in quality.get("failure_names", ())
                        ).items()
                    )
                ),
                "unavailable_gradient_events": list(unavailable),
                "selected_frame": _selected_values(record),
                "evidence_frame_count": len(evidence.get("frames", ())),
                "trajectory_declared": record.get("trajectory_artifact") is not None,
                "source_smiles": str(record["source_smiles"]),
                "semantic_graph_sha256": str(record["semantic_graph_sha256"]),
                "atom_count": len(case["graph"]["atoms"]),
                "donor_count": len(case["coordination"]["donor_rows"]),
                "runner_elapsed_seconds": float(
                    record["timing"]["runner_elapsed_seconds"]
                ),
                "diagnostic_asset": None,
            }
        )
    order = {route: index for index, route in enumerate(ROUTES)}
    return sorted(rows, key=lambda row: (order[row["route"]], row["case_index"]))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def efficiency_rows(
    records: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarize route efficiency without treating child-excluding CPU as total."""
    records = tuple(records)
    primary_o0 = next(record for record in records if record["route"] == "O0")
    o0_run = str(primary_o0["run_id"])
    o0_records = tuple(record for record in records if record["run_id"] == o0_run)
    o0_elapsed = sum(
        float(record["timing"]["runner_elapsed_seconds"])
        for record in o0_records
    )
    rows = []
    records_by_run = {
        str(record["run_id"]): tuple(
            candidate
            for candidate in records
            if candidate["run_id"] == record["run_id"]
        )
        for record in records
    }
    run_ids = sorted(
        records_by_run,
        key=lambda run_id: (
            ROUTES.index(str(records_by_run[run_id][0]["route"])),
            "repeat" in run_id,
        ),
    )
    for run_id in run_ids:
        selected = records_by_run[run_id]
        elapsed = [
            float(record["timing"]["runner_elapsed_seconds"])
            for record in selected
        ]
        accepted = sum(bool(record["passed"]) for record in selected)
        end_times = [
            datetime.fromisoformat(str(record["recorded_at"]))
            for record in selected
        ]
        start_times = [
            end - timedelta(seconds=case_elapsed)
            for end, case_elapsed in zip(end_times, elapsed)
        ]
        observed_span = (max(end_times) - min(start_times)).total_seconds()
        stages = {}
        for field in (
            "provider_or_generation_seconds",
            "ligand_build_seconds",
            "coordination_restoration_seconds",
            "optimization_seconds",
            "acceptance_seconds",
        ):
            values = [
                record["timing"]["stages"].get(field) for record in selected
            ]
            stages[field] = {
                "observed_count": sum(value is not None for value in values),
                "sum_seconds": sum(
                    float(value) for value in values if value is not None
                ),
            }
        total = sum(elapsed)
        user_cpu = sum(
            float(record["resources"]["usage"]["user_cpu_seconds"])
            for record in selected
        )
        system_cpu = sum(
            float(record["resources"]["usage"]["system_cpu_seconds"])
            for record in selected
        )
        rows.append(
            {
                "route": str(selected[0]["route"]),
                "run_id": run_id,
                "repeat": selected[0]["route"] == "F1" and "repeat" in run_id,
                "case_count": len(selected),
                "accepted_count": accepted,
                "acceptance_rate": accepted / len(selected),
                "runner_elapsed_sum_seconds": total,
                "runner_elapsed_mean_seconds": total / len(selected),
                "runner_elapsed_per_accepted_seconds": total / accepted,
                "runner_elapsed_p50_seconds": _percentile(elapsed, 0.50),
                "runner_elapsed_p95_seconds": _percentile(elapsed, 0.95),
                "runner_elapsed_max_seconds": max(elapsed),
                "elapsed_sum_relative_to_o0": total / o0_elapsed,
                "observed_completion_span_seconds": observed_span,
                "observed_completion_span_case_throughput": (
                    len(selected) / observed_span
                ),
                "observed_completion_span_accepted_throughput": (
                    accepted / observed_span
                ),
                "parent_worker_user_cpu_seconds": user_cpu,
                "parent_worker_system_cpu_seconds": system_cpu,
                "parent_worker_user_cpu_to_elapsed_ratio": user_cpu / total,
                "maximum_observed_process_high_water_rss_kib": max(
                    float(
                        record["resources"]["usage"][
                            "process_max_rss_kib_at_case_end"
                        ]
                    )
                    for record in selected
                ),
                "stage_sums": stages,
            }
        )
    return rows


def _trajectory_path(
    record_path: Path,
    record: Mapping[str, object],
) -> Path | None:
    declared = record.get("trajectory_artifact")
    candidate = record_path.parent / "trajectory"
    if declared is None:
        return None
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    identity = _tree_identity(candidate)
    for field in ("sha256", "file_count", "total_bytes"):
        if declared[field] != identity[field]:
            raise ValueError(f"trajectory {field} differs for {record_path}")
    return candidate


def _extract_rejected_frame(
    record_path: Path,
    record: Mapping[str, object],
    target: Path,
) -> dict[str, object] | None:
    trajectory_path = _trajectory_path(record_path, record)
    if trajectory_path is None:
        return None

    from hotpot.cheminfo.forcefields.trajectory import (
        ForceFieldTrajectoryArchive,
        _TrajectoryWriter,
    )

    archive = ForceFieldTrajectoryArchive.read(trajectory_path)
    branches = [("main", archive.main)]
    branches.extend(
        (f"ligand_build_attempts/{index:04d}", trajectory)
        for index, trajectory in enumerate(archive.ligand_build_attempts)
    )
    total_frames = sum(len(trajectory.frames) for _, trajectory in branches)
    selected = next(
        (
            (branch, trajectory, trajectory.selected_frame)
            for branch, trajectory in branches
            if trajectory.selected_frame is not None
        ),
        None,
    )
    if selected is None:
        selected = next(
            (
                (branch, trajectory, trajectory.terminal_frame)
                for branch, trajectory in branches
                if trajectory.terminal_frame is not None
            ),
            None,
        )
    if selected is None:
        selected = next(
            (
                (branch, trajectory, trajectory.frames[-1])
                for branch, trajectory in reversed(branches)
                if trajectory.frames
            ),
            None,
        )
    if selected is None:
        return None
    branch, trajectory, frame = selected
    coordinates = trajectory.coordinates(frame.index)
    maximum_extent = float((coordinates.max(axis=0) - coordinates.min(axis=0)).max())
    nondegenerate = maximum_extent > 1.0e-8
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _TrajectoryWriter._sdf_record(trajectory, frame),
        encoding="utf-8",
    )
    if branch == "main":
        role = "rejected_selected_complex_frame"
    elif nondegenerate:
        role = "failed_ligand_attempt_frame"
    else:
        role = "failed_ligand_attempt_collapsed_frame"
    return {
        "role": role,
        "has_recorded_route_frame": True,
        "has_nondegenerate_route_3d_structure": nondegenerate,
        "trajectory_frame_count": total_frames,
        "maximum_coordinate_extent_angstrom": maximum_extent,
        "branch": branch,
        "frame_index": int(frame.index),
        "frame_stage": str(frame.stage.value),
        "frame_event": str(frame.event.value),
        "source": target.as_posix(),
        "source_sha256": _sha256(target),
        "source_format": "sdf",
        "image": None,
        "image_sha256": None,
    }


def _draw_input_graph(smiles: str, target: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor

    molecule = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(molecule)
    target.parent.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(molecule, str(target), size=(1000, 750))


def _record_paths(result_root: Path) -> dict[tuple[str, int], Path]:
    paths = {}
    for path in result_root.glob("runs/*/*/cases/*/result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "repeat" in str(record["run_id"]):
            continue
        key = (str(record["route"]), int(record["case_index"]))
        paths[key] = path
    return paths


def prepare_diagnostic_assets(
    analysis: dict[str, object],
    records: Mapping[tuple[str, int], Mapping[str, object]],
    record_paths: Mapping[tuple[str, int], Path],
    output_root: Path,
) -> None:
    """Extract rejected frames or create a clearly labeled input-graph view."""
    for row in analysis["failures"]:
        key = (str(row["route"]), int(row["case_index"]))
        record = records[key]
        route_root = output_root / "sources" / key[0]
        sdf_target = route_root / f"{key[1]:04d}.sdf"
        asset = _extract_rejected_frame(
            record_paths[key],
            record,
            sdf_target,
        )
        if asset is None:
            graph_target = route_root / f"{key[1]:04d}-input.png"
            _draw_input_graph(str(row["source_smiles"]), graph_target)
            asset = {
                "role": "input_ligand_graph_no_route_3d_frame",
                "has_recorded_route_frame": False,
                "has_nondegenerate_route_3d_structure": False,
                "trajectory_frame_count": 0,
                "maximum_coordinate_extent_angstrom": None,
                "branch": None,
                "frame_index": None,
                "frame_stage": None,
                "frame_event": None,
                "source": graph_target.as_posix(),
                "source_sha256": _sha256(graph_target),
                "source_format": "png",
                "image": None,
                "image_sha256": None,
            }
        elif not asset["has_nondegenerate_route_3d_structure"]:
            graph_target = route_root / f"{key[1]:04d}-input.png"
            _draw_input_graph(str(row["source_smiles"]), graph_target)
            asset["input_graph_source"] = graph_target.as_posix()
            asset["input_graph_source_sha256"] = _sha256(graph_target)
        row["diagnostic_asset"] = asset


def _render_pymol(source: Path, target: Path) -> None:
    from pymol import cmd

    cmd.reinitialize()
    cmd.set("retain_order", 1)
    cmd.load(str(source), "failed_structure")
    cmd.hide("everything", "failed_structure")
    cmd.show("sticks", "failed_structure")
    cmd.show("spheres", "failed_structure and metals")
    cmd.set("sphere_scale", 0.34, "failed_structure")
    cmd.set("stick_radius", 0.16, "failed_structure")
    cmd.orient("failed_structure")
    cmd.zoom("failed_structure", buffer=1.6)
    cmd.bg_color("white")
    cmd.ray(1000, 750)
    cmd.png(str(target), dpi=180)


def _annotated_image(
    source: Path,
    target: Path,
    row: Mapping[str, object],
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(source) as loaded:
        image = loaded.convert("RGB")
    canvas = Image.new("RGB", (1000, 900), "white")
    image.thumbnail((980, 735))
    canvas.paste(image, ((1000 - image.width) // 2, (740 - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    small = ImageFont.load_default(size=19)
    draw.rectangle(
        (0, 740, 999, 899),
        fill=(255, 238, 236),
        outline=(170, 45, 45),
        width=5,
    )
    asset = row["diagnostic_asset"]
    if asset["has_nondegenerate_route_3d_structure"]:
        frame_label = "REJECTED ROUTE 3D FRAME"
    elif asset["has_recorded_route_frame"]:
        frame_label = "DEGENERATE ATTEMPT FRAME + INPUT GRAPH"
    else:
        frame_label = "NO ROUTE 3D FRAME - INPUT LIGAND GRAPH ONLY"
    draw.text(
        (22, 758),
        f"{row['route']} case {int(row['case_index']):04d} | {frame_label}",
        fill=(145, 35, 35),
        font=font,
    )
    draw.text(
        (22, 800),
        str(row["failure_category"]),
        fill=(35, 35, 35),
        font=small,
    )
    draw.text(
        (22, 832),
        str(row["direct_reason"])[:112],
        fill=(35, 35, 35),
        font=small,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def _collapsed_frame_comparison(
    frame_image: Path,
    graph_image: Path,
    target: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1000, 750), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    for offset, source, label in (
        (0, frame_image, "recorded collapsed frame"),
        (500, graph_image, "input ligand graph"),
    ):
        with Image.open(source) as loaded:
            image = loaded.convert("RGB")
        image.thumbnail((480, 665))
        canvas.paste(image, (offset + (500 - image.width) // 2, 45))
        draw.text((offset + 18, 715), label, fill=(145, 35, 35), font=font)
    draw.line((500, 0, 500, 750), fill=(190, 190, 190), width=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def _contact_sheet(
    rows: Sequence[Mapping[str, object]],
    output_root: Path,
    target: Path,
    title: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    columns = min(4, len(rows))
    cell_width = 460
    cell_height = 480
    header = 70
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * cell_width, header + row_count * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), title, fill=(20, 20, 20), font=ImageFont.load_default(size=30))
    for position, row in enumerate(rows):
        column = position % columns
        sheet_row = position // columns
        x0 = column * cell_width
        y0 = header + sheet_row * cell_height
        image_path = output_root / str(row["diagnostic_asset"]["image"])
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        image.thumbnail((440, 410))
        canvas.paste(image, (x0 + (cell_width - image.width) // 2, y0 + 5))
        draw.text(
            (x0 + 12, y0 + 425),
            f"case {int(row['case_index']):04d} | {row['failure_category']}",
            fill=(145, 35, 35),
            font=ImageFont.load_default(size=18),
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def render_diagnostic_gallery(
    analysis: dict[str, object],
    output_root: Path,
) -> None:
    """Render only failed route-cases without promoting their coordinates."""
    for row in analysis["failures"]:
        asset = row["diagnostic_asset"]
        source = Path(str(asset["source"]))
        raw = (
            output_root
            / "rendered"
            / str(row["route"])
            / f"{int(row['case_index']):04d}-raw.png"
        )
        raw.parent.mkdir(parents=True, exist_ok=True)
        if asset["source_format"] == "sdf":
            _render_pymol(source, raw)
        else:
            raw.write_bytes(source.read_bytes())
        display_source = raw
        if (
            asset["has_recorded_route_frame"]
            and not asset["has_nondegenerate_route_3d_structure"]
        ):
            comparison = raw.with_name(raw.stem + "-comparison.png")
            _collapsed_frame_comparison(
                raw,
                Path(str(asset["input_graph_source"])),
                comparison,
            )
            display_source = comparison
        target = (
            output_root
            / "rendered"
            / str(row["route"])
            / f"{int(row['case_index']):04d}.png"
        )
        _annotated_image(display_source, target, row)
        asset["image"] = target.relative_to(output_root).as_posix()
        asset["image_sha256"] = _sha256(target)

    galleries = []
    for route in ROUTES:
        rows = [row for row in analysis["failures"] if row["route"] == route]
        target = output_root / "galleries" / f"{route}.png"
        _contact_sheet(rows, output_root, target, f"Plan-17 {route} failed route-cases")
        galleries.append(
            {
                "route": route,
                "failure_count": len(rows),
                "path": target.relative_to(output_root).as_posix(),
                "sha256": _sha256(target),
            }
        )
    combined = output_root / "galleries" / "all-failures.png"
    _contact_sheet(
        analysis["failures"],
        output_root,
        combined,
        "Plan-17 all failed route-cases",
    )
    analysis["diagnostic_gallery"] = {
        "route_galleries": galleries,
        "combined": {
            "path": combined.relative_to(output_root).as_posix(),
            "sha256": _sha256(combined),
        },
    }


def _failure_markdown(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Plan 17 formal failure analysis",
        "",
        "Rejected coordinates below remain diagnostic evidence only. They are not",
        (
            "accepted selected structures and cannot enter the governed coordinate "
            "dataset."
        ),
        "",
        "## Failure inventory",
        "",
        (
            "| Route | Case | Status | Direct cause | Route 3D frame | Frames | "
            "Failed gates |"
        ),
        "| --- | ---: | --- | --- | --- | ---: | --- |",
    ]
    for row in analysis["failures"]:
        asset = row["diagnostic_asset"]
        lines.append(
            (
                "| {route} | {case} | {status} | {cause} | {frame} | "
                "{count} | {gates} |"
            ).format(
                route=row["route"],
                case=row["case_index"],
                status=row["status"],
                cause=str(row["direct_reason"]).replace("|", "/"),
                frame=(
                    "nondegenerate 3D"
                    if asset["has_nondegenerate_route_3d_structure"]
                    else (
                        "degenerate attempt"
                        if asset["has_recorded_route_frame"]
                        else "none"
                    )
                ),
                count=asset["trajectory_frame_count"],
                gates=", ".join(row["failed_gates"]),
            )
        )
    lines.extend(
        [
            "",
            "## Route efficiency",
            "",
            (
                "| Run | Pass | Rate | Mean s/case | p50 s | p95 s | Max s | "
                "Sum s | O0 ratio |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["efficiency"]:
        lines.append(
            "| {run} | {passed}/{count} | {rate:.3%} | {mean:.3f} | {p50:.3f} | "
            "{p95:.3f} | {maximum:.3f} | {total:.3f} | {ratio:.3f} |".format(
                run=row["run_id"],
                passed=row["accepted_count"],
                count=row["case_count"],
                rate=row["acceptance_rate"],
                mean=row["runner_elapsed_mean_seconds"],
                p50=row["runner_elapsed_p50_seconds"],
                p95=row["runner_elapsed_p95_seconds"],
                maximum=row["runner_elapsed_max_seconds"],
                total=row["runner_elapsed_sum_seconds"],
                ratio=row["elapsed_sum_relative_to_o0"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Observed completion spans overlap at route boundaries in the shared "
                "16-worker pool and are not additive. Parent-worker CPU excludes "
                "spawned ligand-worker CPU for O0/F1/F2. RSS is a process high-water "
                "observation, not a per-case increment. C1/C2 early generation "
                "failures also shorten their elapsed sums."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _primary_records(result_root: Path) -> tuple[dict[str, object], ...]:
    manifest = json.loads((result_root / "manifest.json").read_text(encoding="utf-8"))
    specs = manifest["scientific_configuration"]["run_specs"]
    primary_ids = {
        str(spec["run_id"])
        for spec in specs
        if spec.get("repeat_of") is None
    }
    records = []
    for path in result_root.glob("runs/*/*/cases/*/result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["run_id"] in primary_ids:
            records.append(record)
    return tuple(records)


def build_analysis(
    result_root: Path,
    cohort_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Build and materialize the complete primary-route failure analysis."""
    result_root = result_root.resolve()
    cohort_path = cohort_path.resolve()
    output_root = output_root.resolve()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    records = _primary_records(result_root)
    if len(records) != 910:
        raise ValueError(f"expected 910 primary results, found {len(records)}")
    paths = _record_paths(result_root)
    records_by_key = {
        (str(record["route"]), int(record["case_index"])): record
        for record in records
    }
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "source": {
            "manifest_sha256": _sha256(result_root / "manifest.json"),
            "summary_sha256": _sha256(result_root / "summary.json"),
            "cohort_file_sha256": _sha256(cohort_path),
            "cohort_canonical_sha256": cohort["canonical_sha256"],
        },
        "primary_result_count": 910,
        "failed_result_count": 0,
        "failures_with_nondegenerate_route_3d_structure": 0,
        "failures_with_only_degenerate_attempt_frame": 0,
        "failures_without_recorded_route_frame": 0,
        "failures": failure_rows(records, cohort),
        "efficiency": efficiency_rows(records),
        "diagnostic_gallery": None,
    }
    prepare_diagnostic_assets(
        analysis,
        records_by_key,
        paths,
        output_root,
    )
    analysis["failed_result_count"] = len(analysis["failures"])
    analysis["failures_with_nondegenerate_route_3d_structure"] = sum(
        row["diagnostic_asset"]["has_nondegenerate_route_3d_structure"]
        for row in analysis["failures"]
    )
    analysis["failures_with_only_degenerate_attempt_frame"] = sum(
        row["diagnostic_asset"]["has_recorded_route_frame"]
        and not row["diagnostic_asset"]["has_nondegenerate_route_3d_structure"]
        for row in analysis["failures"]
    )
    analysis["failures_without_recorded_route_frame"] = sum(
        not row["diagnostic_asset"]["has_recorded_route_frame"]
        for row in analysis["failures"]
    )
    _write_json(output_root / "failure-analysis.json", analysis)
    (output_root / "failure-analysis.md").write_text(
        _failure_markdown(analysis),
        encoding="utf-8",
    )
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--result-root", type=Path, required=True)
    prepare.add_argument("--cohort", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render")
    render.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        build_analysis(arguments.result_root, arguments.cohort, arguments.output)
    else:
        analysis_path = arguments.output.resolve() / "failure-analysis.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        render_diagnostic_gallery(analysis, arguments.output.resolve())
        _write_json(analysis_path, analysis)
        (arguments.output.resolve() / "failure-analysis.md").write_text(
            _failure_markdown(analysis),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ANALYSIS_SCHEMA",
    "build_analysis",
    "classify_failure",
    "efficiency_rows",
    "failure_rows",
    "main",
    "prepare_diagnostic_assets",
    "render_diagnostic_gallery",
)
