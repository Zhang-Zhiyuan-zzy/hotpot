"""Authenticated transfer of verified formal stores through Hugging Face."""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import validate_repo_id

from ._store_copy import StoreCopyError, _package_root, _stage_store, copy_store


DEFAULT_REPO_ID = "Zhang-Zhiyuan-zzy/hotpot-data"
TOKEN_ROOT = "~/.config/hotpot/huggingface/datasets"
ARCHIVE_PATH = "data/store.tar.gz"
_ASSET_PATH = re.compile(
    r"datasets\.sqlite3\.assets/objects/sha256/([0-9a-f]{2})/([0-9a-f]{64})"
)


def _token_guidance(repo_id: str, *, write: bool) -> str:
    access = "write" if write else "read"
    path = f"{TOKEN_ROOT}/{repo_id}/{access}.token"
    account_note = (
        "Your HF account must have write access to this repository."
        if write
        else "Your HF account must have access to this private repository, "
        "or request a read-only dataset token from its owner."
    )
    return (
        "Create a fine-grained Hugging Face token at "
        f"https://huggingface.co/settings/tokens with {access} access to "
        f"the dataset {repo_id}.\n"
        f"{account_note}\n"
        f"Create the parent directories and save only the token text in {path} "
        "(file permissions: 600).\n"
        "Use --token-file PATH or --token-env NAME to select another credential source."
    )


def data_token(
    *,
    write: bool = False,
    repo_id: str = DEFAULT_REPO_ID,
    token_file: str | None = None,
    token_env: str | None = None,
) -> str:
    """Find a repository-scoped read/write token independently of model login."""

    validate_repo_id(repo_id)
    guidance = _token_guidance(repo_id, write=write)
    if token_env is not None:
        token = os.environ.get(token_env, "").strip()
        if token:
            return token
        raise StoreCopyError(
            f"dataset token environment variable {token_env} is empty or missing.\n{guidance}"
        )
    if token_file is None and os.environ.get("HOTPOT_DATA_TOKEN", "").strip():
        return os.environ["HOTPOT_DATA_TOKEN"].strip()

    directory = Path(TOKEN_ROOT).expanduser() / repo_id
    names = ("write.token",) if write else ("read.token", "write.token")
    paths = (
        [Path(token_file).expanduser()]
        if token_file is not None
        else [directory / name for name in names]
    )
    for path in paths:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            if token_file is not None:
                raise StoreCopyError(
                    f"cannot read dataset token file {path}.\n{guidance}"
                ) from exc
            continue
        except OSError as exc:
            raise StoreCopyError(
                f"cannot read dataset token file {path}.\n{guidance}"
            ) from exc
        if token:
            return token
    searched = ", ".join(str(path) for path in paths)
    raise StoreCopyError(
        f"dataset token is empty or missing. Searched: {searched}.\n{guidance}"
    )


def _write_archive(staging: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz", compresslevel=6) as output:
        output.add(
            staging / "datasets.sqlite3", arcname="datasets.sqlite3", recursive=False
        )
        for path in sorted((staging / "datasets.sqlite3.assets").rglob("*")):
            if path.is_file():
                output.add(
                    path, arcname=path.relative_to(staging).as_posix(), recursive=False
                )


def _read_archive(archive: Path, staging: Path) -> Path:
    # Only regular database/CAS files belong to this transport. In particular,
    # do not let an archive create links or choose paths outside staging.
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            match = _ASSET_PATH.fullmatch(member.name)
            if not member.isfile() or not (
                member.name == "datasets.sqlite3"
                or (match and match[2].startswith(match[1]))
            ):
                raise StoreCopyError(
                    "dataset archive contains an unexpected path or file type"
                )
            destination = staging / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as input_stream, destination.open("xb") as output:
                shutil.copyfileobj(input_stream, output)
    return staging / "datasets.sqlite3"


def push_store(
    source_db: str | Path,
    *,
    token: str,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
) -> dict[str, Any]:
    """Publish one consistent audited snapshot to a private dataset repo."""

    source = Path(source_db).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="hotpot-ds-push-") as temporary:
        root = Path(temporary)
        staging = root / "store"
        staging.mkdir()
        metadata, counts = _stage_store(source, staging)
        archive = root / "store.tar.gz"
        _write_archive(staging, archive)

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
        if not info.private:
            raise StoreCopyError("dataset repository must be private before pushing data")
        commit = api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            parent_commit=info.sha,
            path_or_fileobj=archive,
            path_in_repo=ARCHIVE_PATH,
            commit_message="Publish verified Hotpot dataset store",
        )
        return {
            "repo_id": repo_id,
            "revision": commit.oid,
            "path_in_repo": ARCHIVE_PATH,
            "source_database": str(source),
            "store_uuid": metadata["store_uuid"],
            "schema_id": metadata["schema_id"],
            "counts": counts,
            "audit": {"passed": True},
        }


def download_store(
    target_package: str | Path,
    *,
    token: str,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
) -> dict[str, Any]:
    """Download a pinned snapshot and activate it only after store audits."""

    package, _ = _package_root(target_package)
    if os.path.lexists(package / "dataset" / "data" / "store" / "active"):
        raise StoreCopyError(
            "target active store already exists; download never overwrites or merges"
        )
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
    if not info.private:
        raise StoreCopyError("dataset repository must be private before downloading data")
    with tempfile.TemporaryDirectory(prefix="hotpot-ds-download-") as temporary:
        root = Path(temporary)
        archive = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=ARCHIVE_PATH,
            revision=info.sha,
            token=token,
            local_dir=root / "download",
            cache_dir=root / "cache",
        )
        staging = root / "store"
        staging.mkdir()
        source = _read_archive(Path(archive), staging)
        result = copy_store(source, package)
        result.pop("source_database")
        return {
            "repo_id": repo_id,
            "revision": info.sha,
            "path_in_repo": ARCHIVE_PATH,
            **result,
        }
