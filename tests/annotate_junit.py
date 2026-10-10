"""Publish JUnit failures as GitHub Actions workflow annotations."""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree


def _escape(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def main(junit_path: Path) -> None:
    root = ElementTree.parse(junit_path).getroot()
    for case in root.iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if failure is None:
            continue
        title = "::".join(
            part
            for part in (case.get("classname"), case.get("name"))
            if part
        )
        detail = (failure.text or failure.get("message") or "Test failed").strip()
        print(f"::error title={_escape(title)}::{_escape(detail[-6000:])}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
