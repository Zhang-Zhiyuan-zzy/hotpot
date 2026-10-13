from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import hotpot.ds as ds
import hotpot.ds.cli as ds_cli
from hotpot.ds.cli import build_parser, run


def _write_submission(root: Path, *, second_identifier: bool = False) -> Path:
    root.mkdir()
    structures = root / "structures"
    structures.mkdir()
    (structures / "water.smi").write_text("O\n", encoding="utf-8")
    (structures / "water.xyz").write_text(
        "3\nwater\n"
        "O 0.000000 0.000000 0.000000\n"
        "H 0.757000 0.586000 0.000000\n"
        "H -0.757000 0.586000 0.000000\n",
        encoding="utf-8",
    )
    (structures / "methane.smi").write_text("C\n", encoding="utf-8")
    first_identifiers = {"CSD_refcode": "WATER1"}
    if second_identifier:
        first_identifiers["local_code"] = "local-water"
    manifest = {
        "schema_version": "hotpot-ds-submission/current",
        "dataset": {
            "name": "demo",
            "version": "1.0",
            "title": "CLI export demo",
            "description": "Small governed dataset for CLI export tests",
            "license": "CC0-1.0",
            "access_policy": "open",
            "provenance": {"created_by": "test"},
        },
        "records": [
            {
                "id": "water-smiles",
                "identifiers": first_identifiers,
                "metadata": {"family": "alpha"},
                "structures": [
                    {
                        "role": "reported_structure",
                        "primary": True,
                        "representations": {
                            "smi": {"path": "structures/water.smi"}
                        },
                    }
                ],
                "labels": {},
            },
            {
                "id": "water-with-variant",
                "identifiers": {"CSD_refcode": "WATER2"},
                "metadata": {"family": "beta"},
                "structures": [
                    {
                        "role": "reported_structure",
                        "primary": True,
                        "representations": {
                            "xyz": {"path": "structures/water.xyz"}
                        },
                    },
                    {
                        "role": "computed_variant",
                        "primary": False,
                        "representations": {
                            "smi": {"path": "structures/methane.smi"}
                        },
                    },
                ],
                "labels": {},
            },
        ],
        "sources": {},
    }
    if second_identifier:
        manifest["records"].append(
            {
                "id": "label-only",
                "identifiers": {"CSD_refcode": "LABEL1"},
                "metadata": {"family": "label-only"},
                "structures": [],
                "labels": {"description": "no molecular structure supplied"},
            }
        )
    (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture()
def cli_database(tmp_path: Path) -> Path:
    source = _write_submission(tmp_path / "submission")
    bundle = ds.load_bundle(source)
    certificate = ds.write_certificate(
        bundle,
        tmp_path / "certificate.json",
        core_labels=[],
        core_info=[],
    )
    database = tmp_path / "datasets.sqlite3"
    ds.store.ingest(bundle, certificate=certificate, db=database)
    return database


@pytest.fixture()
def multi_identifier_database(tmp_path: Path) -> Path:
    source = _write_submission(
        tmp_path / "submission-with-two-identifiers", second_identifier=True
    )
    bundle = ds.load_bundle(source)
    certificate = ds.write_certificate(
        bundle,
        tmp_path / "two-identifiers.certificate.json",
        core_labels=[],
        core_info=[],
    )
    database = tmp_path / "two-identifiers.sqlite3"
    ds.store.ingest(bundle, certificate=certificate, db=database)
    return database


def _run_cli(arguments: list[str], database: Path) -> int:
    parsed = build_parser().parse_args([*arguments, "--db", str(database)])
    return run(parsed)


def _envelope(capsys: pytest.CaptureFixture[str], key: str) -> tuple[dict, str]:
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert set(envelope) == {"key", "channel", "protocol_sha256", "payload"}
    assert envelope["key"] == key
    return envelope, captured.err


def _aligned_multi_files(destination: Path) -> tuple[list[Path], list[Path]]:
    structures = sorted(destination.glob("struct-*"))
    information = sorted(destination.glob("info-*.json"))
    structure_keys = {
        path.name.removeprefix("struct-").rsplit(".", 1)[0] for path in structures
    }
    information_keys = {
        path.name.removeprefix("info-").removesuffix(".json")
        for path in information
    }
    assert structure_keys == information_keys
    return structures, information


@pytest.mark.parametrize(
    "arguments,key,channel",
    (
        (["list"], "hotpot.ds.list", "default"),
        (["info", "demo"], "hotpot.ds.info", "default"),
        (["labels", "demo"], "hotpot.ds.labels", "default"),
        (["subset", "demo"], "hotpot.ds.subset", "default"),
        (["search", "demo"], "hotpot.ds.search", "data"),
        (["search", "--scope", "dataset"], "hotpot.ds.search", "dataset"),
    ),
)
def test_query_commands_emit_declared_protocol_envelopes(
    arguments: list[str],
    key: str,
    channel: str,
    cli_database: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run_cli(arguments, cli_database) == 0
    envelope, diagnostics = _envelope(capsys, key)
    assert envelope["channel"] == channel
    assert diagnostics == ""


def test_get_positional_selector_modes_and_raw_structures(
    cli_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(["get", "demo", "0"], cli_database) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["items"][0]["local_id"] == "water-smiles"

    assert _run_cli(
        ["get", "demo", "hp-demo[001]-1", "--id"], cli_database
    ) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["items"][0]["local_id"] == "water-with-variant"

    assert _run_cli(["get", "demo", "WATER1", "--identifier"], cli_database) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["items"][0]["local_id"] == "water-smiles"

    assert _run_cli(
        ["get", "demo", "WATER2", "--identifier", "CSD"], cli_database
    ) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["items"][0]["local_id"] == "water-with-variant"

    assert _run_cli(["get", "demo", "0", "--smi"], cli_database) == 0
    envelope, presentation = _envelope(capsys, "hotpot.ds.get")
    assert envelope["channel"] == "structure"
    assert envelope["payload"]["representations"][0]["content"] == "O\n"
    assert presentation == ""

    assert _run_cli(["get", "demo", "1", "--smi"], cli_database) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["representations"][0]["content"].strip() == "O"

    assert _run_cli(["get", "demo", "1", "--xyz"], cli_database) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    xyz = envelope["payload"]["representations"][0]["content"]
    assert xyz.startswith("3\nwater\n")
    assert xyz.count("\n") == 5


def test_get_selects_one_of_multiple_structures_explicitly(
    cli_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = ds.load(ds_name="demo", db=cli_database)[1]
    structures = data.structures.info
    computed_index, computed = next(
        (index, structure)
        for index, structure in enumerate(structures)
        if "computed_variant" in structure["roles"]
    )
    _, reported = next(
        (index, structure)
        for index, structure in enumerate(structures)
        if any(role.endswith("reported_structure") for role in structure["roles"])
    )

    assert _run_cli(
        ["get", "demo", "1", "--smi", "--structure-role", "computed_variant"],
        cli_database,
    ) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["representations"][0]["content"] == "C\n"

    assert _run_cli(
        ["get", "demo", "1", "--smi", "--structure-index", str(computed_index)],
        cli_database,
    ) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["representations"][0]["content"] == "C\n"

    assert _run_cli(
        ["get", "demo", "1", "--xyz", "--structure-id", reported["structure_id"]],
        cli_database,
    ) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["representations"][0]["content"].startswith(
        "3\nwater\n"
    )

    assert _run_cli(
        ["get", "demo", "1", "--structure-index", "0"], cli_database
    ) == 2
    assert "require --smi or --xyz" in capsys.readouterr().err

    assert _run_cli(
        ["get", "demo", "1", "--smi", "--structure-role", "missing-role"],
        cli_database,
    ) == 2
    assert "available roles" in capsys.readouterr().err

    assert _run_cli(
        ["get", "demo", "1", "--smi", "--structure-index", "99"],
        cli_database,
    ) == 2
    assert "out of range" in capsys.readouterr().err


def test_bare_identifier_requires_a_unique_dataset_identifier_kind(
    multi_identifier_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(
        ["get", "demo", "WATER1", "--identifier"], multi_identifier_database
    ) == 2
    assert "identifier" in capsys.readouterr().err.lower()

    assert _run_cli(
        ["get", "demo", "WATER1", "--identifier", "CSD"],
        multi_identifier_database,
    ) == 0
    envelope, _ = _envelope(capsys, "hotpot.ds.get")
    assert envelope["payload"]["items"][0]["local_id"] == "water-smiles"

    for representation in ("--smi", "--xyz"):
        assert _run_cli(
            ["get", "demo", "2", representation], multi_identifier_database
        ) == 2
        assert "structure" in capsys.readouterr().err.lower()


def test_xyz_requires_a_coordinate_bearing_source(
    cli_database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(["get", "demo", "0", "--xyz"], cli_database) == 2
    assert "coordinate" in capsys.readouterr().err.lower()

    destination = tmp_path / "invented-xyz"
    assert _run_cli(
        ["export", "demo", str(destination), "0", "--format", "xyz"],
        cli_database,
    ) == 2
    assert not destination.exists()
    assert "coordinate" in capsys.readouterr().err.lower()


def test_export_single_native_and_converted_structure(
    cli_database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    native = tmp_path / "native"
    assert _run_cli(["export", "demo", str(native), "0"], cli_database) == 0
    capsys.readouterr()

    assert {path.name for path in native.iterdir()} == {"struct.smi", "info.json"}
    assert (native / "struct.smi").read_bytes() == (
        tmp_path / "submission" / "structures" / "water.smi"
    ).read_bytes()
    assert json.loads((native / "info.json").read_text(encoding="utf-8"))["id"] == (
        "hp-demo[001]-0"
    )

    converted = tmp_path / "converted"
    assert _run_cli(
        [
            "export",
            "demo",
            str(converted),
            "hp-demo[001]-0",
            "--id",
            "--format",
            "mol2",
        ],
        cli_database,
    ) == 0
    capsys.readouterr()

    assert {path.name for path in converted.iterdir()} == {
        "struct.mol2",
        "info.json",
    }
    assert "@<TRIPOS>MOLECULE" in (converted / "struct.mol2").read_text(
        encoding="utf-8"
    )


def test_export_all_subset_and_half_open_range_layouts(
    cli_database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    complete = tmp_path / "complete"
    assert _run_cli(["export", "demo", str(complete)], cli_database) == 0
    capsys.readouterr()
    structures, information = _aligned_multi_files(complete)
    assert len(structures) == len(information) == 3
    assert (complete / "summary.json").is_file()
    summary = json.loads((complete / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {
        "dataset",
        "selection",
        "record_count",
        "structure_count",
        "requested_format",
        "files",
    }
    assert summary["dataset"]["name"] == "demo"
    assert summary["record_count"] == 2
    assert summary["structure_count"] == 3
    assert summary["requested_format"] is None
    assert len(summary["files"]) == 3
    assert all(
        set(item)
        == {
            "data_id",
            "record_id",
            "structure_id",
            "structure_file",
            "info_file",
            "source_format",
            "output_format",
        }
        for item in summary["files"]
    )
    assert not (complete / "struct").exists()
    assert not (complete / "info").exists()

    selected = tmp_path / "selected"
    assert _run_cli(
        ["export", "demo", str(selected), "--where", "family=beta"],
        cli_database,
    ) == 0
    capsys.readouterr()
    structures, information = _aligned_multi_files(selected)
    assert len(structures) == len(information) == 2
    assert (selected / "summary.json").is_file()
    summary = json.loads((selected / "summary.json").read_text(encoding="utf-8"))
    assert summary["record_count"] == 1
    assert summary["structure_count"] == 2

    sliced = tmp_path / "sliced"
    assert _run_cli(
        ["export", "demo", str(sliced), "--range", "0", "1"], cli_database
    ) == 0
    capsys.readouterr()
    assert {path.name for path in sliced.iterdir()} == {"struct.smi", "info.json"}
    assert json.loads((sliced / "info.json").read_text(encoding="utf-8"))["local_id"] == (
        "water-smiles"
    )


def test_export_rejects_existing_target_and_failed_conversion_atomically(
    cli_database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    assert _run_cli(["export", "demo", str(existing), "0"], cli_database) == 2
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert {path.name for path in existing.iterdir()} == {"keep.txt"}
    assert capsys.readouterr().err

    failed = tmp_path / "failed"
    assert _run_cli(
        [
            "export",
            "demo",
            str(failed),
            "0",
            "--format",
            "definitely-not-a-format",
        ],
        cli_database,
    ) == 2
    assert not failed.exists()
    assert capsys.readouterr().err


def test_native_export_reports_asset_integrity_failure_without_partial_output(
    cli_database: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "submission" / "structures" / "water.smi"
    source.write_text("tampered\n", encoding="utf-8")
    destination = tmp_path / "integrity-failure"

    assert _run_cli(["export", "demo", str(destination), "0"], cli_database) == 2
    assert not destination.exists()
    assert "sha-256 mismatch" in capsys.readouterr().err.lower()


def test_subset_get_and_export_form_a_json_protocol_pipeline(
    cli_database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run_cli(["subset", "demo"], cli_database) == 0
    subset_document = capsys.readouterr().out

    monkeypatch.setattr(ds_cli.sys, "stdin", io.StringIO(subset_document))
    get_args = build_parser().parse_args(["get", "demo", "-"])
    assert run(get_args) == 0
    get_document = capsys.readouterr().out
    envelope = json.loads(get_document)
    assert envelope["key"] == "hotpot.ds.get"
    assert envelope["payload"]["dataset"]["database"] == str(
        cli_database.resolve()
    )
    assert [item["local_id"] for item in envelope["payload"]["items"]] == [
        "water-smiles",
        "water-with-variant",
    ]

    destination = tmp_path / "protocol-export"
    monkeypatch.setattr(ds_cli.sys, "stdin", io.StringIO(get_document))
    export_args = build_parser().parse_args(
        ["export", "demo", str(destination), "-"]
    )
    assert run(export_args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["key"] == "hotpot.ds.export"
    assert receipt["payload"]["record_count"] == 2
    assert destination.is_dir()


def test_protocol_get_structure_keeps_stdout_json_and_stderr_noninteractive(
    cli_database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run_cli(["subset", "demo", "--where", "family=alpha"], cli_database) == 0
    upstream = capsys.readouterr().out
    monkeypatch.setattr(ds_cli.sys, "stdin", io.StringIO(upstream))

    assert run(build_parser().parse_args(["get", "demo", "-", "--smi"])) == 0
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope["channel"] == "structure"
    assert envelope["payload"]["representations"][0]["content"] == "O\n"
    assert captured.err == ""


def test_protocol_structure_has_a_separate_interactive_presentation_stream(
    cli_database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InteractiveStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    presentation = InteractiveStream()
    monkeypatch.setattr(ds_cli.sys, "stderr", presentation)
    arguments = build_parser().parse_args(
        ["get", "demo", "0", "--smi", "--db", str(cli_database)]
    )

    assert run(arguments) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["payload"]["representations"][0]["content"] == "O\n"
    assert presentation.getvalue() == "O\n"


def test_failed_interactive_presentation_does_not_change_machine_result(
    cli_database: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosedPresentation:
        def isatty(self) -> bool:
            return True

        def write(self, value: str) -> None:
            raise ValueError("presentation stream is closed")

    monkeypatch.setattr(ds_cli.sys, "stderr", ClosedPresentation())
    arguments = build_parser().parse_args(
        ["get", "demo", "0", "--smi", "--db", str(cli_database)]
    )

    assert run(arguments) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["key"] == "hotpot.ds.get"
    assert envelope["payload"]["representations"][0]["content"] == "O\n"


def test_empty_protocol_selection_is_valid_for_get_but_not_export(
    cli_database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _run_cli(
        ["subset", "demo", "--where", "family=not-present"], cli_database
    ) == 0
    upstream = capsys.readouterr().out

    monkeypatch.setattr(ds_cli.sys, "stdin", io.StringIO(upstream))
    assert run(build_parser().parse_args(["get", "demo", "-", "--smi"])) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["channel"] == "structure"
    assert envelope["payload"]["count"] == 0
    assert envelope["payload"]["items"] == []
    assert envelope["payload"]["representations"] == []

    monkeypatch.setattr(ds_cli.sys, "stdin", io.StringIO(upstream))
    destination = tmp_path / "empty-export"
    arguments = build_parser().parse_args(
        ["export", "demo", str(destination), "-"]
    )
    assert run(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "at least 1" in captured.err
    assert not destination.exists()


def test_protocol_stdin_is_explicit_and_context_mismatches_fail_without_stdout(
    cli_database: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnreadableInput(io.StringIO):
        def read(self, *args, **kwargs):
            raise AssertionError("ambient stdin was read")

    monkeypatch.setattr(ds_cli.sys, "stdin", UnreadableInput("ignored"))
    assert _run_cli(["get", "demo", "0"], cli_database) == 0
    _envelope(capsys, "hotpot.ds.get")

    assert _run_cli(["get", "demo", "-", "--id"], cli_database) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot be combined" in captured.err

    assert _run_cli(["subset", "demo"], cli_database) == 0
    envelope = json.loads(capsys.readouterr().out)
    envelope["protocol_sha256"] = "0" * 64
    monkeypatch.setattr(ds_cli.sys, "stdin", io.StringIO(json.dumps(envelope)))
    arguments = build_parser().parse_args(["get", "demo", "-"])
    assert run(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fingerprint" in captured.err

    envelope["protocol_sha256"] = json.loads(
        _protocol_document(cli_database, capsys)
    )["protocol_sha256"]
    monkeypatch.setattr(ds_cli.sys, "stdin", io.StringIO(json.dumps(envelope)))
    wrong_store = tmp_path / "wrong.sqlite3"
    arguments = build_parser().parse_args(
        ["get", "demo", "-", "--db", str(wrong_store)]
    )
    assert run(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--db does not match" in captured.err


def _protocol_document(
    database: Path, capsys: pytest.CaptureFixture[str]
) -> str:
    assert _run_cli(["subset", "demo"], database) == 0
    return capsys.readouterr().out
