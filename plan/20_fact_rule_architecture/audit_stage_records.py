#!/usr/bin/env python3
"""Read-only checks for stage filenames, indexes, links and path migration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[1]
STAGES = PLAN / "stages"
LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\((?:<([^>\n]+)>|([^\s)\n]+))\)")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _prose(text):
    fence = None
    for line in text.splitlines():
        marker = FENCE.match(line)
        if marker:
            kind = marker.group(1)[0]
            fence = None if fence == kind else kind if fence is None else fence
        elif fence is None:
            yield line


def _local_links(path):
    for number, line in enumerate(_prose(path.read_text()), 1):
        for match in LINK.finditer(line):
            target = match.group(1) or match.group(2)
            parsed = urlsplit(target)
            if not parsed.scheme and not parsed.netloc and parsed.path:
                yield number, target, (path.parent / unquote(parsed.path)).resolve()


def _fenced_blocks(text):
    blocks, block, fence = [], [], None
    for line in text.splitlines(keepends=True):
        marker = FENCE.match(line)
        if marker and fence is None:
            fence = marker.group(1)[0]
            block = [line]
        elif fence is not None:
            block.append(line)
            if marker and marker.group(1)[0] == fence:
                blocks.append("".join(block))
                fence = None
    return blocks


def audit(*, check_migration_bytes=False):
    migration = json.loads((STAGES / "path_migration.json").read_text())
    catalog = (STAGES / "README.md").read_text()
    allowed = set(re.findall(r"^\| `([^`]+\.md)` \|", catalog, re.MULTILINE))
    errors = []
    folders = sorted(path for path in STAGES.iterdir() if path.is_dir())
    stage_dirs = [path for path in folders if re.fullmatch(r"stage\d{2}", path.name)]
    for path in folders:
        if path not in stage_dirs:
            errors.append(f"Unknown stage directory: {path.relative_to(ROOT)}")
    markdown = sorted(STAGES.rglob("*.md"))
    records = sorted(STAGES.rglob("*.json"))
    if not allowed:
        errors.append("The catalog defines no Markdown filenames")
    for path in markdown:
        if path.name not in allowed:
            errors.append(f"Unknown Markdown type: {path.relative_to(ROOT)}")
    for folder in stage_dirs:
        if not (folder / "README.md").is_file():
            errors.append(f"Missing stage index: {folder.relative_to(ROOT)}")
    index_links = {}
    for path in records + markdown:
        relative = path.relative_to(STAGES)
        index = STAGES / relative.parts[0] / "README.md" if len(relative.parts) > 1 else STAGES / "README.md"
        if len(relative.parts) > 1 and not re.fullmatch(r"stage\d{2}", relative.parts[0]):
            errors.append(f"Unknown stage directory: {path.relative_to(ROOT)}")
        if index.is_file():
            targets = index_links.setdefault(index, {target for _, _, target in _local_links(index)})
            if path != index and path.resolve() not in targets:
                errors.append(f"Record absent from its README: {path.relative_to(ROOT)}")
    for path in records:
        try:
            json.loads(path.read_text())
        except json.JSONDecodeError as error:
            errors.append(f"Invalid JSON: {path.relative_to(ROOT)}: {error}")
    for path in PLAN.iterdir():
        if path.is_file() and re.match(r"stage\d+_.*\.(?:md|json)$", path.name):
            errors.append(f"Stage record remains in root: {path.name}")
    for path in markdown + sorted(PLAN.glob("*.md")):
        for number, target, resolved in _local_links(path):
            if not resolved.exists():
                errors.append(f"Missing local target: {path.relative_to(ROOT)}:{number}: {target}")
    destinations = [item["new"] for item in migration["entries"]]
    if len(set(destinations)) != len(destinations):
        errors.append("Migration destinations collide")
    old_names = {item["old"] for item in migration["entries"]}
    for item in migration["entries"]:
        if (PLAN / item["old"]).exists():
            errors.append(f"Old stage path still exists: {item['old']}")
        if not (PLAN / item["new"]).is_file():
            errors.append(f"Missing migrated record: {item['new']}")
    for relative in migration["active_root_references_updated"]:
        path = ROOT / relative
        prose = "\n".join(_prose(path.read_text())) if path.suffix == ".md" else path.read_text()
        for old in old_names:
            # A label can describe the old name; a link/code/file target may not.
            if re.search(r"(?:\(|`|\")" + re.escape(old) + r"(?:\)|`|\")", prose):
                errors.append(f"Obsolete active filename: {relative}: {old}")
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode().split("\0")
    external_checked = 0
    for relative in tracked:
        path = ROOT / relative
        if not relative or not path.is_file() or path.is_relative_to(PLAN):
            continue
        if path.suffix not in {".md", ".py", ".json", ".toml", ".yaml", ".yml", ".sh", ".txt"}:
            continue
        external_checked += 1
        text = path.read_text(errors="replace")
        for old in old_names:
            obsolete = "plan/20_fact_rule_architecture/" + old
            if obsolete in text:
                errors.append(f"Obsolete external path: {relative}: {obsolete}")
    identity_checks = {"enabled": check_migration_bytes, "json_snapshots": 0,
                       "markdown_command_and_identity_records": 0,
                       "implementation_log_prefix": False}
    if check_migration_bytes:
        for item in migration["entries"]:
            path = PLAN / item["new"]
            if item["kind"] == "json":
                if hashlib.sha256(path.read_bytes()).hexdigest() != item["original_sha256"]:
                    errors.append(f"Historical JSON bytes changed: {item['new']}")
                identity_checks["json_snapshots"] += 1
            else:
                original = subprocess.check_output(
                    ["git", "show", f"{migration['source_commit']}:plan/20_fact_rule_architecture/{item['old']}"],
                    cwd=ROOT,
                ).decode()
                current = path.read_text()
                if _fenced_blocks(original) != _fenced_blocks(current):
                    errors.append(f"Historical command fence changed: {item['new']}")
                identity_rows = lambda text: [line for line in text.splitlines()
                                              if re.search(r"\b[0-9a-f]{64}\b", line)]
                if identity_rows(original) != identity_rows(current):
                    errors.append(f"Historical digest identity changed: {item['new']}")
                identity_checks["markdown_command_and_identity_records"] += 1
        old_log = subprocess.check_output(
            ["git", "show", f"{migration['source_commit']}:plan/20_fact_rule_architecture/implementation_log.md"],
            cwd=ROOT,
        ).decode()
        identity_checks["implementation_log_prefix"] = (PLAN / "implementation_log.md").read_text().startswith(old_log)
        if not identity_checks["implementation_log_prefix"]:
            errors.append("Existing implementation-log text was changed")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "scope": "Document layout and navigation only; no implementation/test/scientific/performance acceptance",
        "stages": [path.name for path in stage_dirs],
        "allowed_markdown_types": sorted(allowed),
        "markdown_files_checked": len(markdown),
        "json_files_checked": len(records),
        "migrated_records": len(migration["entries"]),
        "external_text_files_scanned": external_checked,
        "historical_identity_checks": identity_checks,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-migration-bytes", action="store_true",
                        help="Check the migration checkpoint's JSON bytes, historical Markdown and log prefix")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(check_migration_bytes=args.check_migration_bytes)
    result["command"] = [sys.executable, *sys.argv]
    result["cwd"] = str(ROOT)
    result["python"] = sys.version
    result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    result["audit_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
