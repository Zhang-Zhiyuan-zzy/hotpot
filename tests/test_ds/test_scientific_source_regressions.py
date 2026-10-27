"""Source-fact sentinels that do not depend on a removed storage model."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from hotpot.ds._paths import SOURCES_ROOT

pytestmark = pytest.mark.scientific_boundary


def test_csd_oxidation_disagreement_sentinels_remain_visible() -> None:
    source = SOURCES_ROOT / "csd-electronic-state" / "electronic_states.sqlite3"
    labels_path = SOURCES_ROOT / "oxidation" / "gold" / "sco95" / "labels.csv"
    if not source.is_file() or not labels_path.is_file():
        pytest.skip("local CSD evidence and curated SCO95 labels are not installed")

    with labels_path.open(newline="", encoding="utf-8") as handle:
        curated = {
            row["structure_refcode"]: int(row["metal_oxidation_state"])
            for row in csv.DictReader(handle)
        }
    connection = sqlite3.connect(source)
    try:
        extracted = {
            refcode: {
                row[0]
                for row in connection.execute(
                    "SELECT oxidation_state FROM entry_oxidation_states "
                    "WHERE refcode=? AND element='Fe'",
                    (refcode,),
                )
            }
            for refcode in ("ACAHUJ", "ZEKDOK", "ACAHUJ02", "ZEKDOK01")
        }
    finally:
        connection.close()

    assert curated["ACAHUJ"] == curated["ZEKDOK"] == 2
    assert extracted["ACAHUJ"] == extracted["ZEKDOK"] == {3}
    assert extracted["ACAHUJ02"] == extracted["ZEKDOK01"] == {2}
