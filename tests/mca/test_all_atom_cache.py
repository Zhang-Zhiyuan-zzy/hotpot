"""Synthetic cache responsibility tests; these do not validate the MCA transform."""

import errno
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import pytest
from hotpot.cheminfo.adapters.inference import all_atom_model as cache
from onnx import TensorProto, helper, numpy_helper


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    source = source_directory / "selected.onnx"
    weights = np.array([1.0, 2.0], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"])],
        "cache-fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
        [numpy_helper.from_array(weights, name="weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    onnx.save_model(
        model,
        source,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    source_bytes = source.read_bytes()
    external_bytes = source.with_name("weights.bin").read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    external_digest = hashlib.sha256(b"weights.bin\0" + external_bytes).hexdigest()
    cache_root = tmp_path / "cache"
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir()
    monkeypatch.setenv("HOTPOT_DERIVED_MODEL_HOME", str(cache_root))
    monkeypatch.setattr(cache.tempfile, "tempdir", str(temporary_root))
    transforms = []
    real_transform = cache._all_atom_graph

    def transform(model):
        transforms.append(model)
        return model

    # Cache tests substitute only graph transformation, keeping real ONNX I/O,
    # external-data copies, hashing, locks, staging, publication and cleanup.
    monkeypatch.setattr(cache, "_all_atom_graph", transform)
    yield {
        "source": source,
        "source_bytes": source_bytes,
        "external_bytes": external_bytes,
        "source_digest": source_digest,
        "external_digest": external_digest,
        "cache_root": cache_root,
        "temporary_root": temporary_root,
        "lock": cache_root / "mca" / source_digest / ".all-atoms.lock",
        "transforms": transforms,
        "real_transform": real_transform,
    }
    assert source.read_bytes() == source_bytes
    assert source.with_name("weights.bin").read_bytes() == external_bytes


def _assert_complete(path, bundle):
    assert path.is_file()
    assert path.with_name("weights.bin").read_bytes() == bundle["external_bytes"]
    marker = json.loads(path.with_name("complete.json").read_text(encoding="utf-8"))
    assert marker == {
        "transform_version": cache._TRANSFORM_VERSION,
        "source_sha256": bundle["source_digest"],
        "external_data_sha256": bundle["external_digest"],
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "external_data": ["weights.bin"],
    }
    assert cache._complete_model(
        path.parent, bundle["source_digest"], bundle["external_digest"]
    ) == path
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path))
    np.testing.assert_array_equal(
        numpy_helper.to_array(model.graph.initializer[0]),
        np.array([1.0, 2.0], dtype=np.float32),
    )


def _fail_lock(bundle, monkeypatch, error):
    real_open = cache.os.open
    attempts = []

    def open_file(path, flags, mode=0o777, *, dir_fd=None):
        if Path(path) == bundle["lock"]:
            attempts.append(path)
            raise error
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(cache.os, "open", open_file)
    return attempts


@pytest.mark.parametrize("provided_digest", [False, True])
def test_persistent_bundle_is_complete_and_reused(bundle, provided_digest):
    digest = bundle["external_digest"] if provided_digest else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with cache.materialize_all_atom_model(bundle["source"], digest) as path:
            _assert_complete(path, bundle)
            model_stat = path.stat()
            external_stat = path.with_name("weights.bin").stat()
        assert path.is_file()
        with cache.materialize_all_atom_model(bundle["source"], digest) as reused:
            assert reused == path
            _assert_complete(reused, bundle)
            assert reused.stat().st_ino == model_stat.st_ino
            assert reused.stat().st_mtime_ns == model_stat.st_mtime_ns
            assert reused.with_name("weights.bin").stat().st_ino == external_stat.st_ino
            assert reused.with_name("weights.bin").stat().st_mtime_ns == external_stat.st_mtime_ns
    assert not caught
    assert len(bundle["transforms"]) == (1 if provided_digest else 2)
    assert not list(bundle["temporary_root"].iterdir())
    assert sorted(item.name for item in path.parent.parent.iterdir()) == [
        ".all-atoms.lock", path.parent.name
    ]


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EROFS, errno.ENOSPC, errno.EDQUOT])
def test_known_cache_error_warns_and_materializes_temporary_bundle(
    bundle, monkeypatch, error_number
):
    error = OSError(error_number, "injected cache lock failure", str(bundle["lock"]))
    attempts = _fail_lock(bundle, monkeypatch, error)
    with pytest.warns(RuntimeWarning) as caught:
        with cache.materialize_all_atom_model(bundle["source"]) as path:
            _assert_complete(path, bundle)
            temporary = path.parent.parent
            assert temporary.parent == bundle["temporary_root"]
            assert temporary.name.startswith("hotpot-mca-all-atoms-")
    assert len(caught) == 1
    message = str(caught[0].message)
    assert str(error) in message
    assert f"[Errno {error_number}]" in message
    assert str(bundle["lock"]) in message
    assert "identical temporary copy for this context" in message
    assert attempts == [bundle["lock"]]
    assert not temporary.exists()
    assert not cache._cache_directory(
        bundle["source_digest"], bundle["external_digest"]
    ).exists()


@pytest.mark.parametrize("error_number", [errno.EIO, errno.EINVAL, None])
def test_other_cache_error_is_reraised_unchanged(bundle, monkeypatch, error_number):
    error = OSError(error_number, "unexplained cache failure", str(bundle["lock"]))
    attempts = _fail_lock(bundle, monkeypatch, error)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(OSError) as raised:
            with cache.materialize_all_atom_model(bundle["source"]):
                pytest.fail("unexpected materialization")
    assert raised.value is error
    assert attempts == [bundle["lock"]]
    assert not caught
    assert not list(bundle["temporary_root"].iterdir())


def test_temporary_bundle_is_removed_when_caller_raises(bundle, monkeypatch):
    cache_error = OSError(errno.EROFS, "cache is read-only", str(bundle["lock"]))
    _fail_lock(bundle, monkeypatch, cache_error)
    caller_error = OSError(errno.EACCES, "caller failure", "caller-output")
    with pytest.warns(RuntimeWarning) as caught:
        with pytest.raises(OSError) as raised:
            with cache.materialize_all_atom_model(bundle["source"]) as path:
                _assert_complete(path, bundle)
                temporary = path.parent.parent
                raise caller_error
    assert raised.value is caller_error
    assert len(caught) == 1
    assert str(cache_error) in str(caught[0].message)
    assert not temporary.exists()


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO])
def test_persistent_caller_error_propagates_without_fallback(bundle, error_number):
    caller_error = OSError(error_number, "caller failure", "caller-output")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(OSError) as raised:
            with cache.materialize_all_atom_model(bundle["source"]) as path:
                _assert_complete(path, bundle)
                raise caller_error
    assert raised.value is caller_error
    assert not caught
    assert path.is_file()
    assert not list(bundle["temporary_root"].iterdir())


def test_temporary_build_failure_propagates_and_cleans_up(bundle, monkeypatch):
    cache_error = OSError(errno.EACCES, "cache access denied", str(bundle["lock"]))
    _fail_lock(bundle, monkeypatch, cache_error)
    copy_error = OSError(errno.ENOSPC, "temporary storage exhausted", "weights.bin")
    real_copy = cache.shutil.copy2
    destinations = []

    def copy_file(source, destination, *args, **kwargs):
        real_copy(source, destination, *args, **kwargs)
        destinations.append(Path(destination))
        raise copy_error

    monkeypatch.setattr(cache.shutil, "copy2", copy_file)
    with pytest.warns(RuntimeWarning) as caught:
        with pytest.raises(OSError) as raised:
            with cache.materialize_all_atom_model(bundle["source"]):
                pytest.fail("unexpected materialization")
    assert raised.value is copy_error
    assert len(caught) == 1
    assert str(cache_error) in str(caught[0].message)
    assert len(destinations) == 1
    assert not destinations[0].parent.parent.exists()
    assert not list(bundle["temporary_root"].iterdir())


@pytest.mark.parametrize("temporary", [False, True])
def test_external_copy_integrity_failure_propagates(bundle, monkeypatch, temporary):
    if temporary:
        cache_error = OSError(errno.EDQUOT, "cache quota exceeded", str(bundle["lock"]))
        _fail_lock(bundle, monkeypatch, cache_error)
    real_copy = cache.shutil.copy2
    destinations = []

    def copy_corrupt(source, destination, *args, **kwargs):
        result = real_copy(source, destination, *args, **kwargs)
        destinations.append(Path(destination))
        Path(destination).write_bytes(b"corrupt copied external data")
        return result

    monkeypatch.setattr(cache.shutil, "copy2", copy_corrupt)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="bundle changed while deriving"):
            with cache.materialize_all_atom_model(bundle["source"]):
                pytest.fail("unexpected materialization")
    assert len(caught) == int(temporary)
    if temporary:
        assert caught[0].category is RuntimeWarning
        assert str(cache_error) in str(caught[0].message)
    assert len(destinations) == 1
    assert not destinations[0].parent.parent.exists()
    assert not list(bundle["temporary_root"].iterdir())
    assert not cache._cache_directory(
        bundle["source_digest"], bundle["external_digest"]
    ).exists()


@pytest.mark.parametrize("corruption", ["model", "external", "marker", "missing-external"])
def test_corrupt_persistent_bundle_is_rejected_and_rebuilt(bundle, corruption):
    with cache.materialize_all_atom_model(bundle["source"]) as path:
        _assert_complete(path, bundle)
    if corruption == "model":
        path.write_bytes(b"invalid ONNX bytes")
    elif corruption == "external":
        path.with_name("weights.bin").write_bytes(b"corrupt weights")
    elif corruption == "marker":
        path.with_name("complete.json").write_text("{", encoding="utf-8")
    else:
        path.with_name("weights.bin").unlink()
    assert cache._complete_model(
        path.parent, bundle["source_digest"], bundle["external_digest"]
    ) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with cache.materialize_all_atom_model(
            bundle["source"], bundle["external_digest"]
        ) as rebuilt:
            assert rebuilt == path
            _assert_complete(rebuilt, bundle)
    assert not caught
    assert sorted(item.name for item in path.parent.parent.iterdir()) == [
        ".all-atoms.lock", path.parent.name
    ]
    assert not list(bundle["temporary_root"].iterdir())


def test_graph_error_propagates_without_creating_cache(bundle, monkeypatch):
    error = cache.UnsupportedAllAtomOptimization("unsupported synthetic graph")

    def reject_graph(model):
        raise error

    monkeypatch.setattr(cache, "_all_atom_graph", reject_graph)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(cache.UnsupportedAllAtomOptimization) as raised:
            with cache.materialize_all_atom_model(bundle["source"]):
                pytest.fail("unexpected materialization")
    assert raised.value is error
    assert not caught
    assert not bundle["cache_root"].exists()
    assert not list(bundle["temporary_root"].iterdir())


def test_source_only_path_does_not_create_cache(bundle, monkeypatch):
    # This fixture has no atom_index input, so use the real no-transform path.
    monkeypatch.setattr(cache, "_all_atom_graph", bundle["real_transform"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with cache.materialize_all_atom_model(bundle["source"]) as path:
            assert path == bundle["source"]
            onnx.checker.check_model(str(path))
    assert not caught
    assert not bundle["cache_root"].exists()
    assert not list(bundle["temporary_root"].iterdir())
