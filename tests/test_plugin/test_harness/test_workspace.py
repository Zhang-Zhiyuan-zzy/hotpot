"""Workspace-lifecycle contracts for the external-process harness."""

from __future__ import annotations

from pathlib import Path

def test_workspaces_are_isolated_and_removed_by_default(tmp_path: Path) -> None:
    from hotpot.plugins._harness.workspace import isolated_workspace

    with isolated_workspace(parent=tmp_path, prefix="record-") as first:
        with isolated_workspace(parent=tmp_path, prefix="record-") as second:
            assert first != second
            assert first.parent == tmp_path
            assert second.parent == tmp_path
            assert first.is_dir()
            assert second.is_dir()
            (first / "result.dat").write_text("first", encoding="utf-8")
            assert not (second / "result.dat").exists()
        assert not second.exists()
        assert first.exists()

    assert not first.exists()


def test_workspace_can_be_retained_explicitly(tmp_path: Path) -> None:
    from hotpot.plugins._harness.workspace import isolated_workspace

    with isolated_workspace(
        parent=tmp_path,
        prefix="retained-",
        retain=True,
    ) as workspace:
        (workspace / "native.log").write_text("evidence", encoding="utf-8")

    assert workspace.is_dir()
    assert (workspace / "native.log").read_text(encoding="utf-8") == "evidence"
