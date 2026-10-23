from hashlib import sha256
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTRACTANT_SOURCE = (
    REPOSITORY_ROOT / "molecules" / "extractant" / "extractants.smi"
)
EXPECTED_COUNT = 187
EXPECTED_SHA256 = (
    "e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7"
)


def test_extractant_source_is_the_canonical_tracked_corpus() -> None:
    source = EXTRACTANT_SOURCE.read_bytes()

    assert len(source.splitlines()) == EXPECTED_COUNT
    assert sha256(source).hexdigest() == EXPECTED_SHA256
