from __future__ import annotations

import json
import shutil
import stat
import os
from pathlib import Path
from typing import Callable, Optional

import pytest

from hotpot.cheminfo.calculator.execution_selection import (
    BackendProjectionStatus,
    BackendStateProjection,
    BackendStateRequest,
    resolve_and_project_state,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)
from hotpot.cheminfo.core import Molecule
from hotpot.plugins.xtb.backend import probe_xtb_backend
from hotpot.plugins.xtb.contracts import XTBMethod
from hotpot.plugins.xtb.state_projection import XTBBackendContext


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


@pytest.fixture
def xtb_projection_factory() -> Callable[
    [Molecule, Path, XTBMethod, int, Optional[int]],
    BackendStateProjection,
]:
    """Create a validated projection for one fake executable and method."""

    def create(
        mol: Molecule,
        executable: Path,
        method: XTBMethod,
        charge: int = 0,
        uhf: Optional[int] = None,
    ) -> BackendStateProjection:
        backend_info = probe_xtb_backend(
            executable,
            environment=dict(os.environ),
        )
        result = resolve_and_project_state(
            extract_molecular_facts(mol),
            XTBBackendContext(backend_info, method),
            StateRequest(net_charge=charge),
            backend_request=BackendStateRequest(uhf=uhf),
        )
        if (
            result.backend_projection.status
            is not BackendProjectionStatus.PROJECTED
            or result.backend_projection.projection is None
        ):
            raise AssertionError(result.backend_projection)
        return result.backend_projection.projection

    return create
