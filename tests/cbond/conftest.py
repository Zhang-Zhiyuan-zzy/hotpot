from pathlib import Path
import sys

import pytest


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "hotpot" / "cheminfo" / "AImodels"),
)


@pytest.fixture(scope="session")
def runtime():
    from cbond.runtime import CBondRuntime

    return CBondRuntime(device="cpu")
