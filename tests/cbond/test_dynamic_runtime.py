import numpy as np
import pytest

from cbond.runtime import padding_rings


def test_runtime_executes_graph_and_dynamic_ring_head(runtime):
    atoms = np.array([6, 7, 8, 26], dtype=np.int32)
    edges = np.array(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
        dtype=np.int64,
    )
    xg = runtime.embed_graph(atoms, edges)
    padded_rings = xg[[0, 1, 2]].reshape(1, 3, 128)
    rings_mask = np.zeros((1, 3), dtype=bool)
    cbond_index = np.array([[3, 3], [0, 1]], dtype=np.int64)

    prediction = runtime.predict(xg, padded_rings, rings_mask, cbond_index)

    assert runtime.device == "cpu"
    assert prediction[:, 0] == pytest.approx(
        [-5.365810871124268, -1.006901741027832], abs=1e-6
    )


@pytest.mark.parametrize("rings,size", [(1, 1), (3, 7), (12, 6), (16, 64), (32, 64)])
def test_dynamic_ring_dimensions(runtime, rings, size):
    rng = np.random.default_rng(rings * 100 + size)
    xg = rng.normal(size=(41, 128)).astype(np.float32)
    padded_rings = rng.normal(size=(rings, size, 128)).astype(np.float32)
    rings_mask = np.zeros((rings, size), dtype=bool)
    cbond_index = np.array([[0, 1], [2, 3]], dtype=np.int64)

    prediction = runtime.predict(xg, padded_rings, rings_mask, cbond_index)

    assert prediction.shape == (2, 1)
    assert np.isfinite(prediction).all()


def test_padding_uses_exact_dimensions():
    xg = np.arange(6 * 128, dtype=np.float32).reshape(6, 128)
    padded, mask = padding_rings(
        xg,
        np.array([0, 1, 2, 3, 4], dtype=np.int64),
        np.array([2, 3], dtype=np.int32),
    )

    assert padded.shape == (2, 3, 128)
    assert mask.tolist() == [[False, False, True], [False, False, False]]


def test_zero_ring_input_remains_finite(runtime):
    atoms = np.array([6, 7, 26], dtype=np.int32)
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    xg = runtime.embed_graph(atoms, edges)
    padded_rings, rings_mask = padding_rings(
        xg,
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int32),
    )

    prediction = runtime.predict(
        xg,
        padded_rings,
        rings_mask,
        np.array([[2], [1]], dtype=np.int64),
    )

    assert padded_rings.shape == (1, 1, 128)
    assert rings_mask.tolist() == [[True]]
    assert prediction[:, 0] == pytest.approx([-1.0473889112472534], abs=1e-6)


def test_padding_rejects_dimensions_outside_validated_domain():
    xg = np.zeros((65, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="exceed the supported limits"):
        padding_rings(
            xg,
            np.arange(65, dtype=np.int64),
            np.array([65], dtype=np.int32),
        )
