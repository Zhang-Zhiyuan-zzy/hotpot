"""Command modules are loaded only for the selected Hotpot subcommand."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).parents[2]
_COMMAND_MODULES = (
    "hotpot.cheminfo.AImodels.cbond.cli",
    "hotpot.cheminfo.AImodels.mca.cli",
    "hotpot.cheminfo.AImodels.artifacts.cli",
    "hotpot.cheminfo.forcefields.cli",
    "hotpot.plugins.xtb.cli",
)


def _loaded_after(arguments) -> tuple[int, dict[str, bool]]:
    script = """
import json
import sys
from hotpot import __main__ as entry

arguments = json.loads(sys.argv[1])
try:
    status = entry.main(arguments)
except SystemExit as error:
    status = int(error.code or 0)
modules = json.loads(sys.argv[2])
print(json.dumps({"status": status, "loaded": {name: name in sys.modules for name in modules}}))
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            json.dumps(list(arguments)),
            json.dumps(_COMMAND_MODULES),
        ],
        cwd=_PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout.splitlines()[-1])
    return payload["status"], payload["loaded"]


def test_version_does_not_load_any_command_module() -> None:
    status, loaded = _loaded_after(("--version",))

    assert status == 0
    assert loaded == {name: False for name in _COMMAND_MODULES}


def test_xtb_help_loads_only_the_selected_command_module() -> None:
    status, loaded = _loaded_after(("xtb", "--help"))

    assert status == 0
    assert loaded["hotpot.plugins.xtb.cli"] is True
    assert all(
        not is_loaded
        for name, is_loaded in loaded.items()
        if name != "hotpot.plugins.xtb.cli"
    )


def test_forcefield_help_does_not_load_xtb_or_model_commands() -> None:
    status, loaded = _loaded_after(("ff", "--help"))

    assert status == 0
    assert loaded["hotpot.cheminfo.forcefields.cli"] is True
    assert loaded["hotpot.plugins.xtb.cli"] is False
    assert loaded["hotpot.cheminfo.AImodels.cbond.cli"] is False
    assert loaded["hotpot.cheminfo.AImodels.mca.cli"] is False
