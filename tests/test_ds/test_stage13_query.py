from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hotpot.ds.public.catalog import datasets_list, load
from hotpot.ds.public._implementation import runtime_policy
from hotpot.ds.public.search import DataSearcher, DatasetSearcher, SmartsSearcher
from hotpot.ds.public.structures import (
    AmbiguousStructureError,
    AssetIntegrityError,
)
from hotpot.ds.storage import (
    TARGET_SCHEMA_ID,
    TargetStore,
    asset_inventory_digest,
    case_content_digest,
    case_set_digest,
)


def _json_digest(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _asset(path: Path, structure_id: str, fmt: str, *, primary: bool) -> dict:
    content = path.read_bytes()
    return {
        "structure_id": structure_id,
        "format": fmt,
        "role": "primary" if primary else "alternative",
        "primary": primary,
        "source_path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def _case(
    case_id: str,
    index: int,
    archive_class: str,
    spin: int,
    family: str,
    structures: list[dict],
) -> dict:
    quality = {
        "incomplete": "B0",
        "ordinary": "B2",
        "benchmark": "B4",
    }[archive_class]
    evidence = {
        "incomplete": "E0",
        "ordinary": "E2",
        "benchmark": "E4",
    }[archive_class]
    return {
        "id": case_id,
        "public_index": index,
        "local_id": f"local-{case_id}",
        "facts": {
            "id": case_id,
            "local_id": f"local-{case_id}",
            "metadata": {"family": family},
            "core_info": {"sample": case_id},
            "observations": [
                {
                    "observation_id": f"observation-{case_id}",
                    "property": "spin",
                    "value": spin,
                    "status": "active",
                }
            ],
        },
        "assessment": {
            "archive_class": archive_class,
            "quality_level": quality,
            "evidence_level": evidence,
            "benchmark": archive_class == "benchmark",
            "benchmark_status": (
                "admitted" if archive_class == "benchmark" else "not_admitted"
            ),
            "reasons": [],
        },
        "identifiers": [{"namespace": "CSD", "value": case_id.upper()}],
        "structures": structures,
    }


def _database(
    tmp_path: Path,
    *,
    primary: bool = True,
    shared_identifier: bool = False,
) -> tuple[Path, list[Path]]:
    water = tmp_path / "water.smi"
    nitrogen = tmp_path / "nitrogen.smi"
    alternative = tmp_path / "alternative.mol2"
    evidence_note = tmp_path / "chemist.txt"
    water.write_text("O\n", encoding="utf-8")
    nitrogen.write_text("N\n", encoding="utf-8")
    alternative.write_text("@<TRIPOS>MOLECULE\nalternative\n", encoding="utf-8")
    evidence_note.write_text("chemist assigned spin = 1\n", encoding="utf-8")

    cases = [
        _case(
            "aaaksi",
            2,
            "ordinary",
            1,
            "oxygen",
            [_asset(water, "water", "smi", primary=True)],
        ),
        _case(
            "bbbbbb",
            7,
            "benchmark",
            3,
            "nitrogen",
            [
                _asset(nitrogen, "nitrogen-primary", "smi", primary=primary),
                _asset(alternative, "nitrogen-alternative", "mol2", primary=False),
            ],
        ),
        _case("broken", 11, "incomplete", 5, "unknown", []),
    ]
    cases[0]["facts"]["evidence"] = [
        {
            "evidence_id": "evidence-aaaksi",
            "source_kind": "expert_statement",
            "path": "evidence/original/chemist.txt",
            "quote": "chemist assigned spin = 1",
        }
    ]
    if shared_identifier:
        for case in cases[:2]:
            case["identifiers"].append(
                {"namespace": "shared", "value": "DUPLICATE"}
            )
    for case in cases:
        indexes: dict[str, int] = {}
        scientific_assets = []
        for structure in case["structures"]:
            structure_id = structure["structure_id"]
            asset_index = indexes.get(structure_id, 0)
            indexes[structure_id] = asset_index + 1
            scientific_assets.append(
                {
                    "collection": "structures",
                    "declaration_id": structure_id,
                    "asset_index": asset_index,
                    "role": structure["role"],
                    "status": "active",
                    "source_path": structure["source_path"],
                    "sha256": structure["sha256"],
                    "size": structure["size"],
                }
            )
        if case["id"] == "aaaksi":
            content = evidence_note.read_bytes()
            scientific_assets.append(
                {
                    "collection": "evidence",
                    "declaration_id": "evidence-aaaksi",
                    "asset_index": 0,
                    "role": "supplier_text",
                    "status": "active",
                    "source_path": evidence_note,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                }
            )
        case["assets"] = scientific_assets

    database = tmp_path / "datasets.sqlite3"
    store = TargetStore(database)
    policy = {
        "schema": "hotpot-ds-policy/current",
        "gates": {"quality": "test"},
        "runtime_policy": runtime_policy(),
    }
    assets = [asset for case in cases for asset in case["assets"]]
    snapshot = {
        "target_context": {
            "canonical_path": str(database.resolve()),
            "store_state": "absent",
            "store_uuid": None,
            "schema_id": TARGET_SCHEMA_ID,
            "dataset_state": "absent",
            "dataset_policy_revision": None,
            "dataset_policy_digest": None,
        },
        "bundle": {
            "dataset": "spin",
            "release": "spin-r1",
            "version": "1.0",
            "release_serial": 1,
            "title": "Spin states",
            "description": "Query surface fixture",
            "metadata": {
                "provenance": {"citation": "Curated spin source"},
                "core_labels": ["spin"],
                "core_info": ["sample"],
            },
        },
        "policy": {
            "revision": 1,
            "digest": _json_digest(policy),
            "value": policy,
        },
        "digests": {
            "bundle": hashlib.sha256(b"bundle").hexdigest(),
            "certificate": hashlib.sha256(b"certificate").hexdigest(),
            "generation": hashlib.sha256(b"generation").hexdigest(),
            "case_content": case_content_digest(cases),
            "included_cases": case_set_digest(case["id"] for case in cases),
            "excluded_cases": case_set_digest([]),
            "asset_inventory": asset_inventory_digest(assets),
        },
        "cases": cases,
    }
    store.ingest(
        snapshot,
        include_incomplete=True,
        reason="Retain an intentionally incomplete query fixture.",
    )
    return database, [water, nitrogen, alternative, evidence_note]


def test_catalog_identifier_subset_and_archive_opt_in(tmp_path: Path) -> None:
    database, _ = _database(tmp_path)

    assert datasets_list(db=database) == ["spin"]
    dataset = load("spin", db=database)

    assert len(dataset) == 2
    assert [item.id for item in dataset] == ["hp-spin[001]-2", "hp-spin[001]-7"]
    assert dataset.identifier_namespaces == ("CSD",)
    assert dataset.CSD_identifier["AAAKSI"].local_id == "local-aaaksi"
    with pytest.raises(AttributeError):
        _ = dataset.CSD
    assert dataset.labels == {"spin": 2}
    assert dataset.core_labels == ["spin"]
    assert dataset.handle == ["Curated spin source"]
    assert dataset.quality_levels == {"B2": 1, "B4": 1}
    assert dataset.evidence_levels == {"E2": 1, "E4": 1}

    assert [item.local_id for item in dataset.subset(spin={1, 9})] == [
        "local-aaaksi"
    ]
    assert [item.local_id for item in dataset.subset(family="nitrogen")] == [
        "local-bbbbbb"
    ]
    assert len(dataset.subset(archive_class="incomplete")) == 0

    mixed = load(
        "spin",
        db=database,
        archive_class={"incomplete", "ordinary", "benchmark"},
    )
    assert len(mixed) == 3
    assert mixed.CSD_identifier["BROKEN"].archive_class == "incomplete"
    assert [item.id for item in mixed.subset(archive_class="ordinary")] == [
        "hp-spin[001]-2"
    ]


def test_identifier_lookup_uses_index_and_respects_the_current_view(
    tmp_path: Path,
) -> None:
    database, _ = _database(tmp_path, shared_identifier=True)
    dataset = load("spin", db=database)
    loaded_case_ids = []
    get_case = dataset._store.get_case

    def track_get_case(dataset_name, release_id, case_id):
        loaded_case_ids.append(case_id)
        return get_case(dataset_name, release_id, case_id)

    dataset._store.get_case = track_get_case

    assert dataset.CSD_identifier["AAAKSI"].record_id == "aaaksi"
    assert loaded_case_ids == ["aaaksi"]

    loaded_case_ids.clear()
    duplicate = dataset.shared_identifier["DUPLICATE"]
    assert isinstance(duplicate, type(dataset))
    assert loaded_case_ids == []
    assert [item.record_id for item in duplicate] == ["aaaksi", "bbbbbb"]

    view = dataset[1:]
    loaded_case_ids.clear()
    assert view.shared_identifier["DUPLICATE"].record_id == "bbbbbb"
    assert loaded_case_ids == ["bbbbbb"]
    with pytest.raises(KeyError):
        _ = view.CSD_identifier["AAAKSI"]

    reversed_view = dataset[::-1]
    duplicate = reversed_view.shared_identifier["DUPLICATE"]
    assert [item.record_id for item in duplicate] == ["bbbbbb", "aaaksi"]

    oxygen = dataset.subset(family="oxygen")
    loaded_case_ids.clear()
    assert oxygen.shared_identifier["DUPLICATE"].record_id == "aaaksi"
    assert loaded_case_ids == ["aaaksi"]


def test_data_info_is_json_and_contains_full_fact_projection(tmp_path: Path) -> None:
    database, sources = _database(tmp_path)
    sources[-1].unlink()
    data = load("spin", db=database)[0]

    assert data.record_id == "aaaksi"
    assert data.label == {"spin": 1}
    assert data.metadata == {"family": "oxygen"}
    assert data.info["facts"]["core_info"] == {"sample": "aaaksi"}
    assert data.info["structures"][0]["sha256"]
    assert isinstance(data.info["structures"][0]["path"], str)
    assert data.info["facts"]["evidence"][0]["path"] == (
        "evidence/original/chemist.txt"
    )
    evidence_asset = data.info["evidence_assets"][0]
    assert evidence_asset["evidence_id"] == "evidence-aaaksi"
    assert Path(evidence_asset["path"]).read_text(encoding="utf-8") == (
        "chemist assigned spin = 1\n"
    )
    assert Path(evidence_asset["path"]).is_relative_to(
        database.with_name(database.name + ".assets")
    )
    json.dumps(data.info)


def test_structure_access_uses_target_assets_and_selects_primary_first(
    tmp_path: Path,
) -> None:
    database, sources = _database(tmp_path)
    for source in sources:
        source.unlink()

    data = load("spin", db=database).CSD_identifier["BBBBBB"]
    assert data.structures() == ["mol2", "smi"]
    assert data.structures.smi == "N\n"
    assert data.structures.mol2 is None
    molecule = data.molecule
    assert (
        getattr(molecule, "data", None) is data
        or getattr(molecule, "ds_data", None) is data
    )

    target_path = data.structures.path("smi")
    target_path.write_text("O\n", encoding="utf-8")
    with pytest.raises(AssetIntegrityError, match="fails integrity"):
        _ = data.structures.smi


def test_multiple_structures_without_unique_primary_are_ambiguous(
    tmp_path: Path,
) -> None:
    database, _ = _database(tmp_path, primary=False)
    data = load("spin", db=database).CSD_identifier["BBBBBB"]

    assert data.structures() == ["mol2", "smi"]
    with pytest.raises(AmbiguousStructureError, match="exactly one primary"):
        _ = data.structures.smi


def test_field_and_smarts_searchers_preserve_dataset_views(tmp_path: Path) -> None:
    database, _ = _database(tmp_path)
    dataset = load("spin", db=database)
    is_spin_one = DataSearcher(labels__spin=1)
    is_oxygen = DataSearcher(metadata__family="oxygen")

    assert [item.record_id for item in (is_spin_one & is_oxygen).search(dataset)] == [
        "aaaksi"
    ]
    assert [item.record_id for item in (~is_spin_one).search(dataset)] == ["bbbbbb"]
    assert [item.record_id for item in SmartsSearcher("O").search(dataset)] == [
        "aaaksi"
    ]
    assert [
        item.name
        for item in DatasetSearcher(name="spin").search((dataset,))
    ] == ["spin"]


def test_dataset_searcher_resolves_public_attributes_before_info() -> None:
    class DatasetProbe:
        name = "spin"

        @property
        def info(self):
            raise AssertionError("dataset.info must not be hydrated for name search")

    assert DatasetSearcher(name="spin").match(DatasetProbe())

