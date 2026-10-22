from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

WRAPPER_ROOT = "hotpot/cheminfo/rdWrappers/"
REQUIRED_WRAPPER_MEMBERS = {
    f"{WRAPPER_ROOT}__init__.py",
    f"{WRAPPER_ROOT}contracts.py",
    f"{WRAPPER_ROOT}native.py",
    f"{WRAPPER_ROOT}sampler.py",
    f"{WRAPPER_ROOT}_rd_native.pyi",
    f"{WRAPPER_ROOT}_native/THIRD_PARTY_NOTICES.md",
    f"{WRAPPER_ROOT}_native/licenses/COPYING.MPL2",
    f"{WRAPPER_ROOT}_native/licenses/LICENSE",
    f"{WRAPPER_ROOT}_native/licenses/LICENSE.md",
    f"{WRAPPER_ROOT}_native/licenses/LICENSE.txt",
    f"{WRAPPER_ROOT}_native/licenses/LICENSE_1_0.txt",
    f"{WRAPPER_ROOT}_native/licenses/THIRD_PARTY_NOTICES.md",
    f"{WRAPPER_ROOT}_native/licenses/license.txt",
}
FORBIDDEN_PREFIXES = (
    "hotpot/cheminfo/_native/include/",
    "third_party/rdkit/",
)


def audit_wheel(path: Path) -> None:
    with ZipFile(path) as archive:
        members = {item.filename for item in archive.infolist() if not item.is_dir()}

    extensions = {
        member
        for member in members
        if member.startswith(f"{WRAPPER_ROOT}_rd_native.") and member.endswith(".so")
    }
    unexpected_wrapper_members = {
        member
        for member in members
        if member.startswith(WRAPPER_ROOT)
        and member not in REQUIRED_WRAPPER_MEMBERS
        and member not in extensions
    }
    forbidden_members = {
        member
        for member in members
        if member.startswith(FORBIDDEN_PREFIXES)
        or PurePosixPath(member).name.startswith(
            ("libhotpot_rdwrappers", "HotpotRDWrappersConfig")
        )
    }
    project_licenses = {
        member
        for member in members
        if member.startswith("hotpot_zzy-")
        and ".dist-info/" in member
        and member.endswith(".dist-info/licenses/LICENSE")
    }

    problems = []
    missing = REQUIRED_WRAPPER_MEMBERS - members
    if missing:
        problems.append(f"missing required members: {sorted(missing)}")
    if len(extensions) != 1:
        problems.append(
            "expected exactly one rdWrapper native extension, "
            f"found {sorted(extensions)}"
        )
    if unexpected_wrapper_members:
        problems.append(
            f"unexpected rdWrapper members: {sorted(unexpected_wrapper_members)}"
        )
    if forbidden_members:
        problems.append(f"forbidden SDK/build members: {sorted(forbidden_members)}")
    if len(project_licenses) != 1:
        problems.append(
            "expected exactly one project distribution license, "
            f"found {sorted(project_licenses)}"
        )

    if problems:
        raise ValueError("; ".join(problems))


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print("expected exactly one wheel path", file=sys.stderr)
        return 2

    wheel = Path(arguments[0])
    if not wheel.is_file():
        print(f"wheel does not exist: {wheel}", file=sys.stderr)
        return 2

    try:
        audit_wheel(wheel)
    except (BadZipFile, ValueError) as error:
        print(f"wheel content audit failed: {error}", file=sys.stderr)
        return 1

    print(f"wheel content audit passed: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
