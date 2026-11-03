import pytest


@pytest.fixture(scope="session")
def runtime():
    from hotpot.cheminfo.operations.inference.cbond.runtime import CBondRuntime

    return CBondRuntime(device="cpu")
