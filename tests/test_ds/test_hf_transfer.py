from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import HfHubHTTPError

from hotpot.ds import _hf
from hotpot.ds._store_copy import StoreCopyError
from hotpot.ds.cli import build_parser, run
from hotpot.ds.public.catalog import load
from tests.test_ds.test_store_copy import (
    _only_cas_path,
    _source_package,
    _target_package,
)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """An offline Hub transport; snapshots and installed stores remain real."""
    state = SimpleNamespace(
        archive=tmp_path / "hub-store.tar.gz",
        private=True,
        sha="a" * 40,
        calls=[],
    )

    class Api:
        def __init__(self, *, token):
            state.calls.append(("auth", token))

        def create_repo(self, **kwargs):
            state.calls.append(("create", kwargs))

        def repo_info(self, **kwargs):
            state.calls.append(("info", kwargs))
            return SimpleNamespace(private=state.private, sha=state.sha)

        def upload_file(self, **kwargs):
            state.calls.append(("upload", kwargs))
            assert kwargs["parent_commit"] == state.sha
            shutil.copyfile(kwargs["path_or_fileobj"], state.archive)
            state.sha = "b" * 40
            return SimpleNamespace(oid=state.sha)

    def download(**kwargs):
        state.calls.append(("download", kwargs))
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True)
        shutil.copyfile(state.archive, target)
        return str(target)

    monkeypatch.setattr(_hf, "HfApi", Api)
    monkeypatch.setattr(_hf, "hf_hub_download", download)
    return state


@pytest.mark.parametrize("credentials", ["automatic", "explicit"])
def test_cli_round_trip_preserves_queryable_store_and_uses_separate_tokens(
    tmp_path, monkeypatch, capsys, hub, credentials
):
    _, source, source_generation, structure_bytes = _source_package(tmp_path)
    target = _target_package(tmp_path)
    token_file = tmp_path / "data-write.token"
    token_file.write_text("dataset-write-secret\n")
    monkeypatch.setenv("HF_TOKEN", "model-secret")
    if credentials == "automatic":
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("HOTPOT_DATA_TOKEN", raising=False)
        directory = Path(_hf.TOKEN_ROOT).expanduser() / _hf.DEFAULT_REPO_ID
        directory.mkdir(parents=True)
        (directory / "read.token").write_text("dataset-read-secret\n")
        (directory / "write.token").write_text("dataset-write-secret\n")
        push_flags = []
    else:
        monkeypatch.setenv("HOTPOT_DATA_TOKEN", "dataset-read-secret")
        push_flags = ["--token-file", str(token_file)]
    # Unreferenced local files and credentials must never enter the archive.
    (source_generation.parent / "private-draft.txt").write_text("unpublished")
    (Path(str(source_generation) + ".assets") / "data.token").write_text("private")

    parser = build_parser()
    assert run(parser.parse_args(["push", "--db", str(source), *push_flags])) == 0
    push_output = capsys.readouterr()
    pushed = json.loads(push_output.out)
    assert pushed["key"] == "hotpot.ds.push"
    assert pushed["payload"]["counts"] == {
        "datasets": 1, "releases": 1, "cases": 1, "assets": 2,
        "asset_bytes": len(structure_bytes) + len(b"chemist-verified spin assignment\n"),
    }
    assert pushed["payload"]["revision"] == "b" * 40
    with tarfile.open(hub.archive) as archive:
        names = archive.getnames()
    assert names[0] == "datasets.sqlite3"
    assert len(names) == 3
    assert all("private" not in name and "token" not in name for name in names)

    assert run(parser.parse_args(["download", "--target-package", str(target)])) == 0
    download_output = capsys.readouterr()
    downloaded = json.loads(download_output.out)
    assert downloaded["key"] == "hotpot.ds.download"
    assert downloaded["payload"]["revision"] == pushed["payload"]["revision"]
    assert downloaded["payload"]["store_uuid"] == pushed["payload"]["store_uuid"]
    assert downloaded["payload"]["counts"] == pushed["payload"]["counts"]
    database = Path(downloaded["payload"]["destination_database"])
    assert database.parent.is_symlink()
    assert _only_cas_path(database).read_bytes() == structure_bytes
    dataset = load("spin", db=database)
    assert len(dataset) == 1
    assert dataset[0].label == {"spin": 3}
    assert [value for action, value in hub.calls if action == "auth"] == [
        "dataset-write-secret", "dataset-read-secret",
    ]
    upload = next(value for action, value in hub.calls if action == "upload")
    assert upload["repo_type"] == "dataset"
    assert upload["path_in_repo"] == _hf.ARCHIVE_PATH
    create = next(value for action, value in hub.calls if action == "create")
    assert create == {"repo_id": _hf.DEFAULT_REPO_ID, "repo_type": "dataset", "private": True, "exist_ok": True}
    download = next(value for action, value in hub.calls if action == "download")
    assert download["token"] == "dataset-read-secret"
    assert download["revision"] == "b" * 40
    assert download["repo_type"] == "dataset"
    assert not Path(download["local_dir"]).exists()
    assert Path(download["cache_dir"]).parent == Path(download["local_dir"]).parent
    assert not Path(download["cache_dir"]).exists()
    for output in (push_output, download_output):
        assert "secret" not in output.out + output.err
    assert os.environ["HF_TOKEN"] == "model-secret"


def test_data_token_sources_are_explicit_and_never_use_model_login(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "model-token")
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "model.token"))
    monkeypatch.delenv("HOTPOT_DATA_TOKEN", raising=False)
    with pytest.raises(StoreCopyError, match="dataset token"):
        _hf.data_token()
    default = Path(_hf.TOKEN_ROOT).expanduser() / _hf.DEFAULT_REPO_ID / "read.token"
    default.parent.mkdir(parents=True)
    default.write_text("file-token\n")
    assert _hf.data_token() == "file-token"
    monkeypatch.setenv("HOTPOT_DATA_TOKEN", "env-token")
    assert _hf.data_token() == "env-token"
    assert _hf.data_token(token_file=str(default)) == "file-token"
    monkeypatch.setenv("CUSTOM_DATA_TOKEN", "custom-token")
    assert _hf.data_token(token_env="CUSTOM_DATA_TOKEN") == "custom-token"
    with pytest.raises(StoreCopyError, match="empty or missing"):
        _hf.data_token(token_env="MISSING_DATA_TOKEN")
    with pytest.raises(StoreCopyError, match="cannot read"):
        _hf.data_token(token_file=str(tmp_path / "absent"))
    default.write_text(" \n")
    with pytest.raises(StoreCopyError, match="empty or missing"):
        _hf.data_token(token_file=str(default))


def test_default_token_search_distinguishes_access_and_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HOTPOT_DATA_TOKEN", raising=False)
    directory = Path(_hf.TOKEN_ROOT).expanduser() / _hf.DEFAULT_REPO_ID
    directory.mkdir(parents=True)
    read = directory / "read.token"
    write = directory / "write.token"
    read.write_text("read-secret\n")
    assert _hf.data_token() == "read-secret"
    with pytest.raises(StoreCopyError, match="with write access"):
        _hf.data_token(write=True)
    write.write_text("write-secret\n")
    assert _hf.data_token() == "read-secret"
    assert _hf.data_token(write=True) == "write-secret"
    read.write_text(" \n")
    assert _hf.data_token() == "write-secret"
    read.unlink()
    assert _hf.data_token() == "write-secret"
    other_repo = "another-owner/private-data"
    with pytest.raises(StoreCopyError, match="another-owner/private-data/read.token"):
        _hf.data_token(repo_id=other_repo)
    other = Path(_hf.TOKEN_ROOT).expanduser() / other_repo
    other.mkdir(parents=True)
    (other / "read.token").write_text("other-read-secret")
    assert _hf.data_token(repo_id=other_repo) == "other-read-secret"


@pytest.mark.parametrize("command,access", [("push", "write"), ("download", "read")])
def test_missing_default_credentials_explain_setup_before_network(
    command, access, tmp_path, monkeypatch, capsys, hub
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "unrelated-model-secret")
    monkeypatch.delenv("HOTPOT_DATA_TOKEN", raising=False)
    assert run(build_parser().parse_args([command])) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "https://huggingface.co/settings/tokens" in output.err
    assert f"with {access} access" in output.err
    assert f"{_hf.TOKEN_ROOT}/{_hf.DEFAULT_REPO_ID}/{access}.token" in output.err
    assert "file permissions: 600" in output.err
    assert "unrelated-model-secret" not in output.err
    if command == "download":
        assert "request a read-only dataset token from its owner" in output.err
    assert hub.calls == []


@pytest.mark.parametrize("command", ["push", "download"])
def test_cli_missing_credentials_fails_before_network(command, tmp_path, capsys, hub):
    assert run(build_parser().parse_args([
        command, "--token-file", str(tmp_path / "missing.token"),
    ])) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "dataset token" in output.err
    assert hub.calls == []


def test_push_refuses_public_repository(tmp_path, hub):
    _, source, _, _ = _source_package(tmp_path)
    hub.private = False
    with pytest.raises(StoreCopyError, match="must be private"):
        _hf.push_store(source, token="write-token")
    assert not hub.archive.exists()
    assert not any(action == "upload" for action, _ in hub.calls)


def test_download_refuses_public_repository_before_fetch(tmp_path, hub):
    target = _target_package(tmp_path)
    hub.private = False
    with pytest.raises(StoreCopyError, match="must be private"):
        _hf.download_store(target, token="read-token")
    assert not any(action == "download" for action, _ in hub.calls)
    assert not (target / "dataset" / "data" / "store" / "active").exists()


@pytest.mark.parametrize("damage", ["receipt", "asset", "empty-store"])
def test_push_refuses_unverified_store_before_network(tmp_path, hub, damage):
    _, source, generation, _ = _source_package(tmp_path)
    if damage == "receipt":
        with sqlite3.connect(generation) as connection:
            connection.execute("UPDATE datasets SET title='tampered'")
    elif damage == "asset":
        _only_cas_path(generation).write_bytes(b"corrupt")
    else:
        from hotpot.ds.storage import TargetStore

        empty = tmp_path / "empty.sqlite3"
        TargetStore(empty).initialize()
        source = empty
    with pytest.raises(StoreCopyError):
        _hf.push_store(source, token="write-token")
    assert hub.calls == []


def test_download_pins_requested_revision_and_repository(tmp_path, hub):
    _, source, _, _ = _source_package(tmp_path)
    _hf.push_store(source, token="write-token", repo_id="owner/private", revision="release")
    target = _target_package(tmp_path)
    result = _hf.download_store(target, token="read-token", repo_id="owner/private", revision="v1")
    info_calls = [value for action, value in hub.calls if action == "info"]
    assert info_calls[-1] == {"repo_id": "owner/private", "repo_type": "dataset", "revision": "v1"}
    download = next(value for action, value in hub.calls if action == "download")
    assert download["repo_id"] == "owner/private"
    assert download["revision"] == result["revision"] == "b" * 40
    upload = next(value for action, value in hub.calls if action == "upload")
    assert upload["revision"] == "release"


@pytest.mark.parametrize("occupied", ["directory", "dangling-link"])
def test_download_refuses_occupied_target_before_network(tmp_path, hub, occupied):
    target = _target_package(tmp_path)
    active = target / "dataset" / "data" / "store" / "active"
    active.parent.mkdir(parents=True)
    if occupied == "directory":
        active.mkdir()
        (active / "marker").write_text("preserve")
    else:
        active.symlink_to("missing-generation")
    with pytest.raises(StoreCopyError, match="already exists"):
        _hf.download_store(target, token="read-token")
    assert hub.calls == []
    if occupied == "directory":
        assert (active / "marker").read_text() == "preserve"
    else:
        assert active.readlink() == Path("missing-generation")


@pytest.mark.parametrize("damage", ["asset", "receipt", "missing-asset"])
def test_download_rejects_corruption_without_activating(tmp_path, hub, damage):
    _, source, _, _ = _source_package(tmp_path)
    _hf.push_store(source, token="write-token")
    extracted = tmp_path / "corrupt-snapshot"
    extracted.mkdir()
    generation = _hf._read_archive(hub.archive, extracted)
    if damage == "receipt":
        with sqlite3.connect(generation) as connection:
            connection.execute("UPDATE datasets SET title='tampered'")
    elif damage == "asset":
        _only_cas_path(generation).write_bytes(b"corrupt")
    else:
        _only_cas_path(generation).unlink()
    _hf._write_archive(generation.parent, hub.archive)
    target = _target_package(tmp_path)
    with pytest.raises(StoreCopyError, match="asset|formal release content"):
        _hf.download_store(target, token="read-token")
    store = target / "dataset" / "data" / "store"
    assert not (store / "active").exists()
    assert not tuple((store / "generations").iterdir())
    download = next(value for action, value in hub.calls if action == "download")
    assert not Path(download["local_dir"]).exists()


@pytest.mark.parametrize("name,kind", [
    ("../../escape", tarfile.REGTYPE),
    ("/tmp/escape", tarfile.REGTYPE),
    ("datasets.sqlite3", tarfile.SYMTYPE),
    ("datasets.sqlite3", tarfile.LNKTYPE),
    ("datasets.sqlite3", tarfile.FIFOTYPE),
    ("credentials.token", tarfile.REGTYPE),
    ("datasets.sqlite3.assets/objects/sha256/aa/" + "b" * 64, tarfile.REGTYPE),
])
def test_download_rejects_unsafe_archive_members(tmp_path, hub, name, kind):
    with tarfile.open(hub.archive, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../../escape" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
        member.size = 1 if kind == tarfile.REGTYPE else 0
        archive.addfile(member, io.BytesIO(b"x") if member.size else None)
    target = _target_package(tmp_path)
    with pytest.raises(StoreCopyError, match="unexpected path or file type"):
        _hf.download_store(target, token="read-token")
    assert not (target / "dataset" / "data" / "store" / "active").exists()
    assert not (tmp_path / "escape").exists()


def test_push_snapshot_includes_committed_wal_data(tmp_path, hub):
    _, source, generation, _ = _source_package(tmp_path)
    with sqlite3.connect(generation) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("PRAGMA user_version=123")
        connection.commit()
        assert Path(str(generation) + "-wal").is_file()
        _hf.push_store(source, token="write-token")
        extract = tmp_path / "extract"
        extract.mkdir()
        snapshot = _hf._read_archive(hub.archive, extract)
        with sqlite3.connect(snapshot) as copied:
            assert copied.execute("PRAGMA user_version").fetchone()[0] == 123


@pytest.mark.parametrize("failure_at", ["metadata", "archive"])
def test_cli_hub_access_denied_never_installs_a_store(
    tmp_path, monkeypatch, capsys, hub, failure_at
):
    target = _target_package(tmp_path)
    token_file = tmp_path / "read.token"
    token_file.write_text("dataset-read-secret")
    hub.archive.write_bytes(b"previously cached archive")

    def denied(**kwargs):
        raise HfHubHTTPError(
            "401 Unauthorized", response=SimpleNamespace(headers={}, request=None)
        )

    if failure_at == "metadata":
        monkeypatch.setattr(_hf.HfApi, "repo_info", lambda self, **kwargs: denied())
    else:
        monkeypatch.setattr(_hf, "hf_hub_download", denied)
    result = run(build_parser().parse_args([
        "download", "--target-package", str(target), "--token-file", str(token_file)
    ]))
    assert result == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "401 Unauthorized" in output.err
    assert "dataset-read-secret" not in output.err
    assert not (target / "dataset" / "data" / "store" / "active").exists()


def test_push_reports_commit_conflict_without_retry(tmp_path, monkeypatch, capsys, hub):
    _, source, _, _ = _source_package(tmp_path)
    token_file = tmp_path / "write.token"
    token_file.write_text("dataset-write-secret")
    attempts = []

    def conflicting_upload(self, **kwargs):
        attempts.append(kwargs["parent_commit"])
        raise HfHubHTTPError(
            "409 Conflict", response=SimpleNamespace(headers={}, request=None)
        )

    monkeypatch.setattr(_hf.HfApi, "upload_file", conflicting_upload)
    assert run(build_parser().parse_args([
        "push", "--db", str(source), "--token-file", str(token_file)
    ])) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "409 Conflict" in output.err
    assert "dataset-write-secret" not in output.err
    assert attempts == ["a" * 40]
    assert not hub.archive.exists()
    assert load("spin", db=source)[0].label == {"spin": 3}
