import argparse
import json
from types import SimpleNamespace

import pytest

from hotpot.__main__ import build_parser as build_hotpot_parser
from hotpot.ds import cli as ds_cli
from hotpot.ds.cli import HELP_FORMATTER, build_parser as build_ds_parser


def _hotpot_help(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    parser = build_hotpot_parser(selected_command="ds")
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["ds", *arguments])
    assert exit_info.value.code == 0
    return " ".join(capsys.readouterr().out.split())


def _walk_parsers(parser: argparse.ArgumentParser):
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                yield from _walk_parsers(child)


def test_every_ds_command_and_argument_has_help() -> None:
    for parser in _walk_parsers(build_ds_parser()):
        assert parser.description
        assert parser.formatter_class is HELP_FORMATTER
        for action in parser._actions:
            if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
                continue
            assert action.help not in (None, argparse.SUPPRESS)


def test_hotpot_ds_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    output = _hotpot_help(["--help"], capsys)

    assert output.startswith("usage: hotpot ds [-h] COMMAND")
    assert "datasets" in output.lower()
    for command in (
        "list",
        "info",
        "get",
        "labels",
        "subset",
        "search",
        "export",
        "bundle",
        "certify",
        "ingest",
        "detail",
    ):
        assert command in output


@pytest.mark.parametrize(
    "command",
    (
        "list",
        "info",
        "get",
        "labels",
        "subset",
        "search",
        "export",
        "bundle",
        "certify",
        "ingest",
        "detail",
    ),
)
def test_each_concise_public_command_has_help(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    output = _hotpot_help([command, "--help"], capsys)

    assert output.startswith(f"usage: hotpot ds {command}")
    assert "options:" in output


def test_dataset_commands_use_positional_dataset_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "info": "DATASET",
        "get": "DATASET SELECTOR",
        "labels": "DATASET",
        "subset": "DATASET",
        "search": "[DATASET]",
        "export": "DATASET OUTPUT",
    }
    for command, positionals in expected.items():
        output = _hotpot_help([command, "--help"], capsys)
        assert "--db PATH" in output
        assert positionals in output
        assert "--dataset" not in output


def test_get_selector_modes_and_structure_flags_are_unambiguous() -> None:
    parser = build_ds_parser()

    for arguments in (
        ["get", "demo", "3"],
        ["get", "demo", "3", "--index"],
        ["get", "demo", "hp-demo[001]-3", "--id"],
        ["get", "demo", "AAAKSI", "--identifier"],
        ["get", "demo", "AAAKSI", "--identifier", "CSD"],
        ["get", "demo", "3", "--smi"],
        ["get", "demo", "3", "--xyz"],
        ["get", "demo", "3", "--smi", "--structure-index", "0"],
        ["get", "demo", "3", "--smi", "--structure-role", "high_temperature"],
        ["get", "demo", "3", "--xyz", "--structure-id", "structure-id"],
    ):
        parsed = parser.parse_args(arguments)
        assert parsed.dataset == "demo"

    with pytest.raises(SystemExit):
        parser.parse_args(["get", "demo", "3", "--smi", "--xyz"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "get",
                "demo",
                "3",
                "--smi",
                "--structure-index",
                "0",
                "--structure-role",
                "high_temperature",
            ]
        )


def test_subset_quality_and_evidence_gates_accept_only_levels_zero_through_four() -> None:
    parser = build_ds_parser()

    defaults = parser.parse_args(["subset", "demo"])
    assert defaults.quality == 0
    assert defaults.evidence == 0

    explicit = parser.parse_args(["subset", "demo", "-q", "4", "-e", "3"])
    assert explicit.quality == 4
    assert explicit.evidence == 3

    for option, value in (("--quality", "-1"), ("-q", "5"), ("--evidence", "5")):
        with pytest.raises(SystemExit):
            parser.parse_args(["subset", "demo", option, value])


def test_export_selection_modes_are_exclusive() -> None:
    parser = build_ds_parser()

    for arguments in (
        ["export", "demo", "destination"],
        ["export", "demo", "destination", "0"],
        ["export", "demo", "destination", "hp-demo[001]-0", "--id"],
        ["export", "demo", "destination", "AAAKSI", "--identifier"],
        ["export", "demo", "destination", "AAAKSI", "--identifier", "CSD"],
        ["export", "demo", "destination", "--range", "0", "10"],
        ["export", "demo", "destination", "--where", "family=alpha"],
        ["export", "demo", "destination", "--format", "mol2"],
    ):
        parsed = parser.parse_args(arguments)
        assert parsed.dataset == "demo"

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["export", "demo", "destination", "0", "--range", "0", "1"]
        )


def test_certify_and_ingest_expose_the_certificate_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    certify = _hotpot_help(["certify", "--help"], capsys)
    ingest = _hotpot_help(["ingest", "--help"], capsys)

    assert "BUNDLE_ROOT" in certify
    assert "--dataset NAME" in certify
    assert "--core-label" not in certify
    assert "--quality-level" not in certify
    assert "--certificate" not in certify
    assert "BUNDLE_ROOT" in ingest
    assert "--db PATH" in ingest

    default_store = build_ds_parser().parse_args(["ingest", "submission"])
    assert default_store.bundle_root == "submission"
    assert not hasattr(default_store, "db")

    explicit_store = build_ds_parser().parse_args(
        ["ingest", "submission", "--db", "store.sqlite3"]
    )
    assert explicit_store.db == "store.sqlite3"


def test_ingest_handler_uses_the_active_default_store_consistently(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = tmp_path / "submission"
    bundle_root.mkdir()
    (bundle_root / "bundle.json").write_text(
        json.dumps({"ds_name": "demo"}), encoding="utf-8"
    )
    default_store = (tmp_path / "package" / "datasets.sqlite3").resolve()
    observed: dict[str, object] = {}

    class FakeBundle:
        root = bundle_root.resolve()
        name = "demo"
        _database = default_store
        _state = SimpleNamespace(manifest={"release_id": "demo.v001"})

        def ingest(self, **kwargs):
            observed["ingest_kwargs"] = kwargs
            return {
                "receipt_digest": "receipt",
                "included_count": 1,
                "excluded_count": 0,
            }

    class FakeTarget:
        def __init__(self, path):
            observed["target_path"] = path
            self.path = path

        def metadata(self):
            return {"store_uuid": "uuid", "schema_id": "schema"}

        def load_release_metadata(self, dataset, release_id):
            return {
                "receipt": {
                    "included_cases_digest": "included",
                    "excluded_cases_digest": "excluded",
                }
            }

    def fake_load_bundle(root, name, *, db):
        observed["load_bundle"] = (root, name, db)
        return FakeBundle()

    emitted: dict[str, object] = {}
    monkeypatch.setattr(ds_cli, "DEFAULT_STORE_PATH", default_store)
    monkeypatch.setattr(ds_cli, "load_bundle", fake_load_bundle)
    monkeypatch.setattr(ds_cli, "TargetStore", FakeTarget)
    monkeypatch.setattr(
        ds_cli,
        "emit_envelope",
        lambda key, payload, **kwargs: emitted.update(key=key, payload=payload),
    )

    args = build_ds_parser().parse_args(["ingest", str(bundle_root)])
    assert ds_cli.run(args) == 0
    assert observed["load_bundle"] == (bundle_root.resolve(), "demo", default_store)
    assert observed["target_path"] == default_store
    assert emitted["key"] == "hotpot.ds.ingest"
    assert emitted["payload"]["store"]["canonical_path"] == str(default_store)


@pytest.mark.parametrize(
    "option",
    (
        ("--quality-level", "B0_collection"),
        ("--core-label", "oxidation_state"),
        ("--core-info", "metadata.name"),
        ("--no-core-fields",),
        ("--configure", "policy.json"),
    ),
)
def test_removed_certification_policy_options_are_not_parseable(
    option: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        build_ds_parser().parse_args(["certify", "submission", *option])


def test_bundle_locator_and_transitions_have_one_current_layout() -> None:
    parser = build_ds_parser()
    default_build = parser.parse_args(["bundle", "build"])
    assert default_build.bundle_root == "."
    build = parser.parse_args(
        ["bundle", "build", "submission", "--dataset", "demo", "--db", "store.sqlite3"]
    )
    assert build.bundle_root == "submission"
    concise = parser.parse_args(["bundle", "status"])
    assert concise.bundle_action == "status"
    explicit = parser.parse_args(
        ["bundle", "--root", "submission", "--dataset", "demo", "status"]
    )
    assert explicit.bundle_root_option == "submission"
    assert explicit.bundle_dataset == "demo"
    with pytest.raises(SystemExit):
        parser.parse_args(["bundle", "status", "--root", "submission"])
