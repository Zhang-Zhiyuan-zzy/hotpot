import json
import sqlite3
from pathlib import Path

import hotpot.ds as ds


ROOT = Path(__file__).resolve().parents[2]
API_DOCUMENTATION = ROOT / "hotpot" / "ds" / "README.md"
CLI_DOCUMENTATION = ROOT / "hotpot" / "ds" / "cli_doc.md"
DATASET_DOCUMENTATION = ROOT / "hotpot" / "dataset" / "README.md"
PUBLIC_API = [
    "datasets_list",
    "load",
    "Dataset",
    "Data",
    "BaseSearcher",
    "DataSearcher",
    "DatasetSearcher",
    "SmartsSearcher",
    "load_bundle",
    "Bundle",
    "write_certificate",
]


def test_api_documentation_covers_the_supported_public_surface() -> None:
    text = API_DOCUMENTATION.read_text(encoding="utf-8")

    assert ds.__all__ == PUBLIC_API
    for name in PUBLIC_API:
        assert name in text
    assert "hotpot.ds.detail" not in text
    assert "ds.store" not in text
    for obsolete in (
        "ds.Certificate",
        "ds.Store",
        "ds.Structures",
        "ds.QualityLevel",
        "ds.EvidenceLevel",
        "ds.load_certificate",
        "ds.search",
    ):
        assert obsolete not in text

    headings = [
        "## 1. Query datasets",
        "## 2. `Dataset` and `Data`",
        "## 3. Search",
        "## 4. Build and manage a Bundle",
        "## 5. Gates and levels",
        "## 6. Intake organization",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_cli_documentation_uses_the_installed_console_entrypoint() -> None:
    text = CLI_DOCUMENTATION.read_text(encoding="utf-8")
    project_configuration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'hotpot = "hotpot.__main__:main"' in project_configuration
    assert "hotpot ds --help" in text
    assert "python -m hotpot ds" not in text
    for command in (
        "list",
        "info",
        "get",
        "labels",
        "subset",
        "search",
        "certify",
        "ingest",
        "detail",
    ):
        assert f"hotpot ds {command}" in text


def test_python_documentation_blocks_are_valid_python() -> None:
    text = API_DOCUMENTATION.read_text(encoding="utf-8")

    for index, fragment in enumerate(text.split("```python")[1:], start=1):
        source = fragment.split("```", 1)[0]
        compile(source, f"hotpot/ds/README.md:block-{index}", "exec")


def test_submission_examples_are_valid_json_and_sql() -> None:
    text = API_DOCUMENTATION.read_text(encoding="utf-8")

    for fragment in text.split("```json")[1:]:
        json.loads(fragment.split("```", 1)[0])

    connection = sqlite3.connect(":memory:")
    for fragment in text.split("```sql")[1:]:
        connection.executescript(fragment.split("```", 1)[0])
    connection.close()


def test_official_dataset_documentation_is_english() -> None:
    for path in (
        API_DOCUMENTATION,
        CLI_DOCUMENTATION,
        DATASET_DOCUMENTATION,
    ):
        text = path.read_text(encoding="utf-8")
        assert not any("\u3400" <= character <= "\u9fff" for character in text)


def test_documentation_uses_the_canonical_dataset_namespace() -> None:
    texts = (
        API_DOCUMENTATION.read_text(encoding="utf-8"),
        CLI_DOCUMENTATION.read_text(encoding="utf-8"),
        DATASET_DOCUMENTATION.read_text(encoding="utf-8"),
    )

    assert "hotpot/dataset/data/" in "\n".join(texts)
    for text in texts:
        assert "datasets/database/" not in text
        assert "benchmark/database/" not in text
        assert "hotpot.benchmark_data" not in text
