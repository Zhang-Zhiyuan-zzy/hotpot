from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path
from typing import Callable

import pytest


@pytest.fixture
def fake_xtb_factory(tmp_path: Path) -> Callable[[str, str], Path]:
    """Create independently configured fake xTB executables."""

    source_path = Path(__file__).parent / "fixtures" / "fake_xtb.py"

    def create(scenario: str = "success", directory_name: str = "fake_xtb") -> Path:
        executable_dir = tmp_path / directory_name
        executable_dir.mkdir()
        executable_path = executable_dir / "xtb"
        shutil.copyfile(source_path, executable_path)
        executable_path.chmod(
            executable_path.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )
        (executable_dir / "scenario.json").write_text(
            json.dumps({"scenario": scenario}),
            encoding="utf-8",
        )
        return executable_path

    return create
