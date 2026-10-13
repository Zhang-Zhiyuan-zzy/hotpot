from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import hotpot.ds as ds
import hotpot.ds.detail as detail
from hotpot.ds.public.structures import AmbiguousStructureError


def _demo_bundle(root: Path) -> detail.DatasetBundle:
    smiles = root / "water.smi"
    smiles.write_text("O\n", encoding="utf-8")
    digest = hashlib.sha256(smiles.read_bytes()).hexdigest()
    oxidation_state = next(
        item
        for item in detail.builtin_property_types()
        if item.property_type_id == "oxidation_state"
    )
    return detail.DatasetBundle(
        standard_version=detail.STANDARD_VERSION,
        datasets=(detail.Dataset("demo", "Demo", "Public API test dataset"),),
        releases=(
            detail.DatasetRelease(
                "demo.v1",
                "demo",
                "1.0",
                "B0_collection",
                "CC0-1.0",
                "open",
                {"created_by": "test"},
                {"core_labels": ["oxidation_state"], "core_info": []},
            ),
        ),
        subsets=(),
        records=(
            detail.DatasetRecord("demo:r1", "demo.v1", "r1", {"family": "a"}),
            detail.DatasetRecord("demo:r2", "demo.v1", "r2", {"family": "b"}),
        ),
        memberships=(),
        observation_memberships=(),
        claim_memberships=(),
        entities=(
            detail.Entity("demo:e1", "molecular_structure", None, {}),
            detail.Entity("demo:e2", "molecular_structure", None, {}),
        ),
        external_identifiers=(
            detail.ExternalIdentifier("demo:e1", "CSD_refcode", "AAAKSI", None),
            detail.ExternalIdentifier("demo:e2", "CSD_refcode", "BBBBBB", None),
        ),
        record_entities=(
            detail.RecordEntity("demo:r1", "demo:e1", "structure"),
            detail.RecordEntity("demo:r2", "demo:e2", "subject"),
        ),
        entity_relations=(),
        assets=(
            detail.Asset(
                "demo:water-smi",
                "water.smi",
                "chemical/x-daylight-smiles",
                digest,
                "open",
                {},
            ),
        ),
        structures=(detail.Structure("demo:s1", "demo:e1", None, 3, 0, {}),),
        structure_assets=(
            detail.StructureAsset("demo:s1", "demo:water-smi", "coordinates"),
        ),
        descriptors=(),
        property_types=(oxidation_state,),
        observations=(
            detail.Observation(
                "demo:o1",
                "demo:r1",
                "demo:e1",
                "oxidation_state",
                2,
                None,
                {},
                {},
                "active",
                "E3_primary_source_supported",
                {"reported_by": "test"},
                {},
            ),
            detail.Observation(
                "demo:o2",
                "demo:r2",
                "demo:e2",
                "oxidation_state",
                3,
                None,
                {},
                {},
                "active",
                "E2_explicit_reported",
                {"reported_by": "test"},
                {},
            ),
        ),
        observation_participants=(),
        sources=(
            detail.Source(
                "demo:source", "publication", "Test source", None, "CC0-1.0", "open", {}
            ),
        ),
        evidence=(
            detail.EvidenceItem(
                "demo:ev1", "demo:source", "reported_value", "page 1", None, "2 and 3", {}
            ),
        ),
        observation_evidence=(
            detail.ObservationEvidence("demo:o1", "demo:ev1", "supports"),
            detail.ObservationEvidence("demo:o2", "demo:ev1", "supports"),
        ),
        claims=(),
        claim_evidence=(),
        claim_resolutions=(),
    )


def _multi_structure_bundle(
    root: Path, *, primary_structure: bool
) -> detail.DatasetBundle:
    bundle = _demo_bundle(root)
    mol2 = root / "alternate.mol2"
    mol2.write_text("@<TRIPOS>MOLECULE\nalternate\n", encoding="utf-8")
    digest = hashlib.sha256(mol2.read_bytes()).hexdigest()
    first_role = "structure" if primary_structure else "variant_a"
    return replace(
        bundle,
        entities=bundle.entities
        + (detail.Entity("demo:e3", "molecular_structure", None, {}),),
        record_entities=(
            detail.RecordEntity("demo:r1", "demo:e1", first_role),
            detail.RecordEntity("demo:r1", "demo:e3", "variant_b"),
            *bundle.record_entities[1:],
        ),
        assets=bundle.assets
        + (
            detail.Asset(
                "demo:alternate-mol2",
                "alternate.mol2",
                "chemical/x-mol2",
                digest,
                "open",
                {},
            ),
        ),
        structures=bundle.structures
        + (detail.Structure("demo:s2", "demo:e3", None, 3, 0, {}),),
        structure_assets=bundle.structure_assets
        + (
            detail.StructureAsset(
                "demo:s2", "demo:alternate-mol2", "coordinates"
            ),
        ),
    )


@pytest.fixture()
def public_database(tmp_path: Path) -> Path:
    bundle = _demo_bundle(tmp_path)
    path = tmp_path / "datasets.sqlite3"
    backend = detail.DatasetStore(path)
    backend.ingest(bundle, detail.assess_bundle(bundle, tmp_path), tmp_path)
    return path


def test_top_level_api_is_compact_and_backend_remains_in_detail() -> None:
    expected = [
        "datasets_list",
        "load",
        "Dataset",
        "Data",
        "BaseSearcher",
        "DataSearcher",
        "DatasetSearcher",
        "SmartsSearcher",
        "write_certificate",
        "load_bundle",
        "Bundle",
        "store",
    ]
    assert ds.__all__ == expected
    assert set(dir(ds)) == set(expected)
    fresh_names = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import hotpot.ds as ds, json; "
                "print(json.dumps(sorted(k for k in vars(ds) if not k.startswith('_'))))",
            ],
            text=True,
        )
    )
    assert fresh_names == sorted(expected)
    for name in (
        "Certificate",
        "Store",
        "Structures",
        "QualityLevel",
        "EvidenceLevel",
        "load_certificate",
        "search",
        "DatasetStore",
        "DatasetBundle",
    ):
        with pytest.raises(AttributeError):
            getattr(ds, name)
    assert detail.DatasetStore is not None
    assert detail.DatasetBundle is not None
    assert detail.assess_bundle is not None
    assert detail.Certificate is not None
    assert detail.Store is not None


def test_dataset_catalog_load_sequence_ids_and_identifier_index(
    public_database: Path,
) -> None:
    assert "demo" in ds.datasets_list(db=public_database)

    dataset = ds.load(ds_name="demo", db=public_database)

    assert dataset.info["name"] == "demo"
    assert dataset.info["size"] == 2
    assert len(dataset) == 2
    assert dataset[0].id == "hp-demo[001]-0"
    assert dataset[1].id == "hp-demo[001]-1"
    assert ds.load(ds_name="demo", db=public_database)[0].id == dataset[0].id
    assert dataset.CSD_identifier["AAAKSI"].id == dataset[0].id
    assert dataset.handle == ["Test source"]


def test_default_catalog_does_not_depend_on_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert list(ds.datasets_list) == ["csd-electronic-state", "oxidation", "spin"]
    assert [
        dataset.name
        for dataset in ds.DatasetSearcher(name="oxidation").search(ds.datasets_list)
    ] == ["oxidation"]


def test_labels_core_labels_subset_and_json_information(public_database: Path) -> None:
    dataset = ds.load(ds_name="demo", db=public_database)

    assert dataset.labels == {"oxidation_state": 2}
    assert dataset.core_labels == ["oxidation_state"]
    assert [item.local_id for item in dataset.subset(oxidation_state=[2, 9])] == ["r1"]
    assert [item.local_id for item in dataset.subset(family={"a"})] == ["r1"]

    data = dataset[0]
    assert data.label == {"oxidation_state": 2}
    assert data.quality_level == "B0_collection"
    assert data.evidence_level == "E3_primary_source_supported"
    assert data.info["id"] == data.id
    assert data.info["identifiers"]["CSD_refcode"] == "AAAKSI"
    json.dumps(data.info)


def test_structures_missing_formats_and_molecule_back_reference(
    public_database: Path,
) -> None:
    dataset = ds.load(ds_name="demo", db=public_database)
    data = dataset[0]

    assert data.structures() == ["smi"]
    assert data.structures.smi == "O\n"
    assert data.structures.mol2 is None
    assert dataset[1].structures() == []
    assert dataset[1].structures.smi is None

    molecule = data.molecule
    assert getattr(molecule, "ds_data", getattr(molecule, "data", None)) is data


def test_structure_format_access_resolves_primary_before_format(
    tmp_path: Path,
) -> None:
    bundle = _multi_structure_bundle(tmp_path, primary_structure=True)
    database = tmp_path / "primary.sqlite3"
    detail.DatasetStore(database).ingest(
        bundle, detail.assess_bundle(bundle, tmp_path), tmp_path
    )

    structures = ds.load(ds_name="demo", db=database)[0].structures

    assert structures() == ["mol2", "smi"]
    assert structures.smi == "O\n"
    assert structures.mol2 is None


def test_structure_format_access_rejects_variants_without_primary(
    tmp_path: Path,
) -> None:
    bundle = _multi_structure_bundle(tmp_path, primary_structure=False)
    database = tmp_path / "ambiguous.sqlite3"
    detail.DatasetStore(database).ingest(
        bundle, detail.assess_bundle(bundle, tmp_path), tmp_path
    )

    structures = ds.load(ds_name="demo", db=database)[0].structures

    with pytest.raises(AmbiguousStructureError, match="multiple structures"):
        _ = structures.mol2


def test_field_searchers_and_all_logical_operators(public_database: Path) -> None:
    dataset = ds.load(ds_name="demo", db=public_database)
    is_two = ds.DataSearcher(label__oxidation_state=2)
    is_a = ds.DataSearcher(metadata__family="a")

    assert [item.local_id for item in (is_two & is_a).search(dataset)] == ["r1"]
    assert [item.local_id for item in (is_two | ds.DataSearcher(label__oxidation_state=3)).search(dataset)] == [
        "r1",
        "r2",
    ]
    assert [item.local_id for item in (is_two ^ is_a).search(dataset)] == []
    assert [item.local_id for item in (~is_two).search(dataset)] == ["r2"]
    assert [item.local_id for item in is_two.not_().search(dataset)] == ["r2"]
    with pytest.raises(TypeError, match="truth value"):
        bool(is_two)
    with pytest.raises(TypeError, match="data and dataset"):
        is_two & ds.DatasetSearcher(name="demo")

    assert [
        item.name
        for item in ds.DatasetSearcher(name="demo").search((dataset,))
    ] == ["demo"]
    assert [item.local_id for item in ds.SmartsSearcher("O").search(dataset)] == ["r1"]


def test_certificate_configure_empty_core_requirement_and_envelope(
    tmp_path: Path,
) -> None:
    bundle = ds.load_bundle(_demo_bundle(tmp_path), asset_root=tmp_path)
    configure = {
        "schema_version": "hotpot-ds-certificate-config/current",
        "generic": {"min_records": 3},
        "datasets": {"demo": {"min_records": 1}},
    }

    with pytest.raises(ValueError, match="core_labels.*core_info"):
        ds.write_certificate(bundle, tmp_path / "missing-declaration.json")

    destination = ds.write_certificate(
        bundle,
        tmp_path / "certificate.json",
        core_labels=[],
        configure=configure,
    )

    envelope = json.loads(destination.read_text(encoding="utf-8"))
    assert envelope["schema_version"].startswith("hotpot-ds-public-certificate/")
    assert envelope["mode"] == "new"
    loaded = detail.Certificate.load(destination)
    assert loaded.passed
    loaded.verify_integrity()


def test_store_requires_public_certificate(tmp_path: Path) -> None:
    bundle = ds.load_bundle(_demo_bundle(tmp_path), asset_root=tmp_path)
    target = tmp_path / "stored.sqlite3"

    with pytest.raises(TypeError):
        ds.store.ingest(bundle)
    with pytest.raises(TypeError, match="certificate"):
        ds.store.ingest(bundle, certificate=None, db=target)

    certificate = ds.write_certificate(
        bundle,
        tmp_path / "certificate.json",
        core_labels=[],
    )
    with pytest.raises(TypeError, match="db"):
        ds.store.ingest(bundle, certificate=certificate)
    result = ds.store.ingest(bundle, certificate=certificate, db=target)
    assert result["records"] == 2


def test_store_rejects_new_certificate_without_admission_declaration(
    tmp_path: Path,
) -> None:
    bundle = _demo_bundle(tmp_path)
    certificate = detail.Certificate(
        detail.assess_bundle(bundle, tmp_path),
        configure=detail.Configure(),
        ds_name="demo",
        mode="new",
        declarations={},
        source_kind="bundle",
    )

    with pytest.raises(detail.QualityGateError, match="quality_level|core_labels|core_info"):
        detail.Store(tmp_path / "undeclared.sqlite3").ingest(
            bundle,
            certificate=certificate,
            asset_root=tmp_path,
        )
