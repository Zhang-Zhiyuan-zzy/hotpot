import json
import re
import sqlite3
from pathlib import Path

import hotpot.ds as ds
from hotpot.ds.cli import load_cli_documentation


ROOT = Path(__file__).resolve().parents[2]
API_DOCUMENTATION = ROOT / "hotpot" / "ds" / "README.md"
CLI_DOCUMENTATION = ROOT / "hotpot" / "ds" / "cli_doc.md"
DATASET_DOCUMENTATION = ROOT / "hotpot" / "dataset" / "README.md"
RECORD_STANDARD = (
    ROOT / "hotpot" / "dataset" / "schemas" / "record-standard.json"
)
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
    assert "hotpot ds --doc" in text
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
        "copy",
        "detail",
    ):
        assert f"hotpot ds {command}" in text


def test_cli_documentation_contains_a_complete_direct_ingestion_guide() -> None:
    text = CLI_DOCUMENTATION.read_text(encoding="utf-8")

    for required in (
        "## Fastest complete publication example",
        "hotpot ds bundle build submission",
        "hotpot ds bundle --root submission synchronize",
        "hotpot ds certify submission",
        "hotpot ds ingest submission",
        "## 5. Bundle intake layout",
        "bundle.json",
        "cases.jsonl",
        "intake.sqlite3",
        "structures[].representations",
        "## 6. Troubleshooting",
    ):
        assert required in text

    assert load_cli_documentation() == text


def test_cli_documentation_json_examples_are_valid() -> None:
    text = CLI_DOCUMENTATION.read_text(encoding="utf-8")

    blocks = text.split("```json")[1:]
    assert blocks
    for fragment in blocks:
        json.loads(fragment.split("```", 1)[0])


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


def test_record_standard_is_the_current_case_level_information_fence() -> None:
    standard = json.loads(RECORD_STANDARD.read_text(encoding="utf-8"))
    serialized = json.dumps(standard, ensure_ascii=False, sort_keys=True)
    manifest_schema = json.loads(
        (
            ROOT / "hotpot" / "ds" / "schemas" / "bundle" / "manifest.schema.json"
        ).read_text(encoding="utf-8")
    )
    case_schema = json.loads(
        (
            ROOT / "hotpot" / "ds" / "schemas" / "bundle" / "case.schema.json"
        ).read_text(encoding="utf-8")
    )

    assert standard["standard"] == "hotpot-dataset-information-fence/current"
    assert set(standard["bundle_manifest"]["required_fields"]) == set(
        manifest_schema["required"]
    )
    assert set(standard["bundle_manifest"]["optional_fields"]) == (
        set(manifest_schema["properties"]) - set(manifest_schema["required"])
    )
    assert set(standard["case_facts"]["required_intake_fields"]) == set(
        case_schema["required"]
    )
    assert set(standard["property_policy"]["definition_fields"]) == set(
        manifest_schema["$defs"]["propertyDefinition"]["properties"]
    )
    assert set(standard["property_policy"]["comparator_fields"]) == set(
        manifest_schema["$defs"]["propertyComparator"]["properties"]
    )
    assert set(standard["quality_levels"]) == {
        "scope",
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
    }
    assert set(standard["evidence_levels"]) == {
        "scope",
        "E0",
        "E1",
        "E2",
        "E3",
        "E4",
    }
    assert set(standard["archive_classes"]) == {
        "incomplete",
        "ordinary",
        "benchmark",
        "visibility",
    }
    assert set(standard["gates"]) >= {
        "EvidenceGate",
        "QualityGate",
        "BenchmarkGate",
    }
    assert set(standard["case_facts"]["forbidden_operative_input_fields"]) == {
        "quality_level",
        "evidence_level",
        "archive_class",
        "benchmark",
    }
    assert "public_index" in standard["case_facts"]["canonical_identity_fields"]
    assert standard["certificate_and_storage"]["ingestion_api"].startswith(
        "Bundle.ingest("
    )
    assert "negative_declaration" in standard["scientific_assets"]
    assert "property_comparators" in standard["bundle_manifest"]["optional_fields"]

    assert re.search(r"\b[BE][0-4]_[A-Za-z]", serialized) is None
    for obsolete in (
        "Certificate.Configure",
        "submission",
        "store.ingest",
        "assessor_version",
        "required_admission_flow",
        "quality_level_scope",
        "release_quality_level",
        "release_evidence_level",
    ):
        assert obsolete not in serialized
