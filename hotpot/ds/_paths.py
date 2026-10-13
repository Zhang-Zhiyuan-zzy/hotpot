"""Canonical local paths for Hotpot dataset resources."""

from pathlib import Path


HOTPOT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = HOTPOT_PACKAGE_ROOT / "dataset"
DATA_ROOT = DATASET_ROOT / "data"
STORE_ROOT = DATA_ROOT / "store"
SOURCES_ROOT = DATA_ROOT / "sources"
STRUCTURES_ROOT = DATA_ROOT / "structures"
AUDITS_ROOT = DATA_ROOT / "audits"
DEFAULT_STORE_PATH = STORE_ROOT / "datasets.sqlite3"
