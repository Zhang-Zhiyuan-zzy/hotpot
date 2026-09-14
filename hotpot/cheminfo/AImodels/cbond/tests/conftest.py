from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session")
def runtime():
    from cbond.runtime import CBondRuntime

    return CBondRuntime(device="cpu")
