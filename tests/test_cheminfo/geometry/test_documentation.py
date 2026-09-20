"""Completeness checks for the installed geometry API reference."""

from __future__ import annotations

import re
from importlib.resources import files

import pytest

from hotpot.cheminfo import geometry

API_REFERENCE = files("hotpot.cheminfo.geometry").joinpath("relation.md")


def test_api_reference_is_available_as_package_data():
    documentation = API_REFERENCE.read_text(encoding="utf-8")

    assert documentation.startswith("# `hotpot.cheminfo.geometry` API 参考")
    assert "## 1. Package introduction" in documentation


@pytest.mark.parametrize("public_name", geometry.__all__)
def test_every_public_export_has_a_dedicated_api_section(public_name):
    documentation = API_REFERENCE.read_text(encoding="utf-8")
    heading = re.compile(
        r"^### \d+\.\d+ `" + re.escape(public_name) + r"`$",
        re.MULTILINE,
    )

    assert heading.search(documentation), public_name
