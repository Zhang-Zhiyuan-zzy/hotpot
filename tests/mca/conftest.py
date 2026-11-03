
import pytest


@pytest.fixture(scope="session")
def predictor():
    from hotpot.cheminfo.operations.inference.mca import MCACpuPredictor

    with MCACpuPredictor() as predictor:
        yield predictor
