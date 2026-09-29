"""Optional PyMOL rendering isolated from scientific benchmark execution."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

REGULAR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
COLUMNS = 8
CELL_WIDTH = 560
CELL_HEIGHT = 455
HEADER_HEIGHT = 100
MOLECULE_WIDTH = 520
MOLECULE_HEIGHT = 335


def pymol_available() -> bool:
    """Return whether the optional PyMOL renderer can be imported."""
    return importlib.util.find_spec("pymol") is not None


def _read_metadata(case_dir: Path, structure_name: str) -> dict[str, object]:
    report = json.loads((case_dir / "report.json").read_text(encoding="utf-8"))
    optimization = report.get("optimization") or {}
    energy = optimization.get("best_energy", report.get("energy"))
    return {
        "case": int(case_dir.name),
        "status": str(report["status"]),
        "phase": str(report.get("phase", "unknown")),
        "energy": None if energy is None else float(energy),
        "energy_unit": str(optimization.get("energy_unit", "kJ/mol")),
        "structure": str(case_dir / structure_name),
        "image": None,
        "hydrogen_count": None,
        "metal_count": None,
        "render_error": None,
        "render_is_placeholder": False,
    }


def _mol2_element_counts(source: Path) -> tuple[int, int]:
    in_atom_block = False
    hydrogen_count = 0
    metal_count = 0
    metal_elements = {
        "LI",
        "BE",
        "NA",
        "MG",
        "AL",
        "K",
        "CA",
        "SC",
        "TI",
        "V",
        "CR",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "GA",
        "RB",
        "SR",
        "Y",
        "ZR",
        "NB",
        "MO",
        "TC",
        "RU",
        "RH",
        "PD",
        "AG",
        "CD",
        "IN",
        "SN",
        "CS",
        "BA",
        "LA",
        "CE",
        "PR",
        "ND",
        "PM",
        "SM",
        "EU",
        "GD",
        "TB",
        "DY",
        "HO",
        "ER",
        "TM",
        "YB",
        "LU",
        "HF",
        "TA",
        "W",
        "RE",
        "OS",
        "IR",
        "PT",
        "AU",
        "HG",
        "TL",
        "PB",
        "BI",
        "FR",
        "RA",
        "AC",
        "TH",
        "PA",
        "U",
        "NP",
        "PU",
        "AM",
        "CM",
        "BK",
        "CF",
        "ES",
        "FM",
        "MD",
        "NO",
        "LR",
    }
    for line in source.read_text(encoding="utf-8").splitlines():
        if line == "@<TRIPOS>ATOM":
            in_atom_block = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atom_block = False
            continue
        if not in_atom_block or not line.strip():
            continue
        element = line.split()[5].split(".", 1)[0].upper()
        hydrogen_count += element == "H"
        metal_count += element in metal_elements
    return hydrogen_count, metal_count


def _render_case(case_dir_text: str, structure_name: str) -> dict[str, object]:
    from pymol import cmd

    case_dir = Path(case_dir_text)
    item = _read_metadata(case_dir, structure_name)
    source = Path(str(item["structure"]))
    target = case_dir / "final.png"
    if not source.is_file():
        item["render_error"] = f"{structure_name} is unavailable"
        _write_case_placeholder(target, item, str(item["render_error"]))
        return item

    hydrogen_count, metal_count = _mol2_element_counts(source)
    item["hydrogen_count"] = hydrogen_count
    item["metal_count"] = metal_count
    if hydrogen_count == 0:
        item["render_error"] = "optimized structure has no explicit hydrogen atoms"
        _write_case_placeholder(target, item, str(item["render_error"]))
        return item
    if metal_count == 0:
        item["render_error"] = "optimized structure has no metal atom"
        _write_case_placeholder(target, item, str(item["render_error"]))
        return item

    try:
        cmd.reinitialize()
        cmd.load(str(source), "complex")
        cmd.hide("everything", "all")
        cmd.show("sticks", "complex")
        cmd.show("spheres", "complex")
        cmd.set("stick_radius", 0.13)
        cmd.set("sphere_scale", 0.19, "complex")
        cmd.set("sphere_scale", 0.10, "complex and elem H")
        cmd.set("sphere_scale", 0.45, "complex and metals")
        cmd.color("gray70", "complex and elem C")
        cmd.color("marine", "complex and elem N")
        cmd.color("red", "complex and elem O")
        cmd.color("yellow", "complex and elem S")
        cmd.color("orange", "complex and elem P")
        cmd.color("green", "complex and elem F+Cl")
        cmd.color("firebrick", "complex and elem Br+I")
        cmd.color("white", "complex and elem H")
        cmd.color("tv_orange", "complex and metals")
        cmd.set_bond(
            "stick_color",
            "tv_orange",
            "complex and metals",
            "complex and not metals",
        )
        cmd.set("bg_rgb", [1.0, 1.0, 1.0])
        cmd.set("ray_opaque_background", 1)
        cmd.set("ray_trace_mode", 1)
        cmd.set("ray_shadows", 0)
        cmd.set("antialias", 2)
        cmd.set("ambient", 0.36)
        cmd.set("direct", 0.64)
        cmd.set("specular", 0.20)
        cmd.set("shininess", 32)
        cmd.set("orthoscopic", 1)
        cmd.orient("complex")
        cmd.zoom("complex", buffer=1.8)
        cmd.ray(1000, 750)
        cmd.png(str(target), dpi=180)
    except Exception as error:
        item["render_error"] = f"{type(error).__name__}: {error}"
        _write_case_placeholder(target, item, str(item["render_error"]))
    else:
        item["image"] = str(target)
    return item


def _font(path: Path, size: int) -> ImageFont.ImageFont:
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _write_case_placeholder(
    target: Path,
    item: dict[str, object],
    message: str,
) -> None:
    canvas = Image.new("RGB", (1000, 750), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(BOLD_FONT, 42)
    detail_font = _font(REGULAR_FONT, 26)
    _centered_text(
        draw,
        canvas.width // 2,
        260,
        f"Case {int(item['case']):04d} | {item['status']}",
        title_font,
        _status_color(str(item["status"])),
    )
    _centered_text(
        draw,
        canvas.width // 2,
        330,
        message[:80],
        detail_font,
        (70, 70, 70),
    )
    canvas.save(target)
    item["image"] = str(target)
    item["render_is_placeholder"] = True


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


def _status_color(status: str) -> tuple[int, int, int]:
    if status == "passed":
        return 35, 125, 72
    if status == "failed_quality":
        return 205, 126, 32
    return 185, 61, 51


def _trim_render(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, image.getpixel((0, 0)))
    bounds = ImageChops.difference(image, background).getbbox()
    if bounds is None:
        return image
    left, top, right, bottom = bounds
    margin = 24
    return image.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(image.width, right + margin),
            min(image.height, bottom + margin),
        )
    )


def _energy_label(item: dict[str, object]) -> str:
    energy = item["energy"]
    if energy is None:
        return "Energy unavailable"
    value = float(energy)
    formatted = f"{value:,.2f}" if abs(value) < 1.0e6 else f"{value:.3e}"
    return f"E = {formatted} {item['energy_unit']}"


def _write_contact_sheet(
    experiment: Path,
    title: str,
    rendered: list[dict[str, object]],
) -> None:
    column_count = max(1, min(COLUMNS, len(rendered)))
    rows = (len(rendered) + column_count - 1) // column_count
    canvas = Image.new(
        "RGB",
        (column_count * CELL_WIDTH, HEADER_HEIGHT + rows * CELL_HEIGHT),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(BOLD_FONT, 38)
    label_font = _font(BOLD_FONT, 21)
    detail_font = _font(REGULAR_FONT, 17)
    _centered_text(draw, canvas.width // 2, 22, title, title_font, (25, 25, 25))

    for cell, item in enumerate(rendered):
        column = cell % column_count
        row = cell // column_count
        x0 = column * CELL_WIDTH
        y0 = HEADER_HEIGHT + row * CELL_HEIGHT
        color = _status_color(str(item["status"]))
        image_path = item["image"]
        if image_path is not None:
            with Image.open(str(image_path)) as source_image:
                molecule_image = _trim_render(source_image.convert("RGB"))
                molecule_image.thumbnail(
                    (MOLECULE_WIDTH, MOLECULE_HEIGHT),
                    Image.Resampling.LANCZOS,
                )
                canvas.paste(
                    molecule_image,
                    (
                        x0 + (CELL_WIDTH - molecule_image.width) // 2,
                        y0 + (MOLECULE_HEIGHT - molecule_image.height) // 2,
                    ),
                )
        else:
            _centered_text(
                draw,
                x0 + CELL_WIDTH // 2,
                y0 + 145,
                "No optimized structure",
                label_font,
                color,
            )

        caption_y = y0 + MOLECULE_HEIGHT + 4
        _centered_text(
            draw,
            x0 + CELL_WIDTH // 2,
            caption_y,
            f"Case {int(item['case']):04d} | {item['status']}",
            label_font,
            color,
        )
        _centered_text(
            draw,
            x0 + CELL_WIDTH // 2,
            caption_y + 31,
            _energy_label(item),
            detail_font,
            (45, 45, 45),
        )
        draw.rounded_rectangle(
            (x0 + 8, y0 + 4, x0 + CELL_WIDTH - 8, y0 + CELL_HEIGHT - 7),
            radius=12,
            outline=color,
            width=3,
        )
    canvas.save(experiment / "final.png")


def render_experiment(
    experiment: Path,
    *,
    title: str,
    workers: int,
    mode: str,
    structure_name: str = "optimized.mol2",
) -> list[dict[str, object]]:
    """Render all output structures when explicitly enabled."""
    available = pymol_available()
    if mode == "required" and not available:
        raise RuntimeError("PyMOL is required by --render required but is unavailable")
    if mode == "off" or not available:
        payload = {
            "mode": mode,
            "pymol_available": available,
            "rendered_count": 0,
            "status": "disabled" if mode == "off" else "unavailable",
            "cases": [],
        }
        (experiment / "render_report.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return []

    case_dirs = sorted(
        case_dir
        for case_dir in (experiment / "cases").iterdir()
        if (case_dir / "report.json").is_file()
    )
    rendered = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(_render_case, str(case_dir), structure_name): case_dir
            for case_dir in case_dirs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            case_dir = futures[future]
            try:
                item = future.result()
            except Exception as error:
                if mode == "required":
                    raise
                item = _read_metadata(case_dir, structure_name)
                item["render_error"] = f"{type(error).__name__}: {error}"
                _write_case_placeholder(
                    case_dir / "final.png",
                    item,
                    str(item["render_error"]),
                )
            rendered.append(item)
            outcome = (
                f"placeholder: {item['render_error']}"
                if item["render_is_placeholder"]
                else "rendered"
            )
            print(
                f"[render {completed}/{len(futures)}] "
                f"case={int(item['case']):04d} result={outcome}",
                flush=True,
            )

    rendered.sort(key=lambda item: int(item["case"]))
    _write_contact_sheet(experiment, title, rendered)
    payload = {
        "mode": mode,
        "pymol_available": True,
        "rendered_count": sum(
            item["image"] is not None and not item["render_is_placeholder"]
            for item in rendered
        ),
        "placeholder_count": sum(
            bool(item["render_is_placeholder"]) for item in rendered
        ),
        "status": "complete",
        "cases": rendered,
    }
    (experiment / "render_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return rendered
