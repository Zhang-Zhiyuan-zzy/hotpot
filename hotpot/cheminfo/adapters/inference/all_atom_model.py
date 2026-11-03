"""Derive an all-atom MCA graph from the pinned selected-atom ONNX graph."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import TensorProto, helper

_TRANSFORM_VERSION = 2
_DERIVED_MODEL_NAME = "mecap_mca_all_atom.onnx"
_MARKER_NAME = "complete.json"


class UnsupportedAllAtomOptimization(RuntimeError):
    """The selected-atom graph is not the pinned MCA graph contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _initializer_by_name(model) -> dict[str, object]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _external_locations(model) -> tuple[str, ...]:
    locations = set()
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        metadata = {entry.key: entry.value for entry in initializer.external_data}
        location = metadata["location"]
        location_path = Path(location)
        if (
            location_path.is_absolute()
            or location in {".", ".."}
            or location_path.name != location
        ):
            raise RuntimeError(f"Unsafe MCA external-data location: {location}")
        locations.add(location)
    return tuple(sorted(locations))


def _all_atom_graph(model):
    input_names = {input_.name for input_ in model.graph.input}
    if "atom_index" not in input_names:
        return None

    producers = {
        output: (index, node)
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }
    dependent = {"atom_index"}
    for node in model.graph.node:
        if any(input_ in dependent for input_ in node.input):
            dependent.update(node.output)

    gathers = [
        node
        for node in model.graph.node
        if node.op_type == "GatherElements"
        and any(input_ in dependent for input_ in node.input)
        and any(input_ not in dependent for input_ in node.input)
    ]
    if len(gathers) != 1:
        raise UnsupportedAllAtomOptimization(
            "The MCA atom-index gather could not be identified uniquely"
        )
    gather = gathers[0]
    gather_attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in gather.attribute
    }
    if gather_attributes.get("axis", 0) != 1 or len(gather.input) != 2:
        raise UnsupportedAllAtomOptimization(
            "The MCA atom-index gather has an unexpected orientation"
        )
    encoder_inputs = [input_ for input_ in gather.input if input_ not in dependent]
    if (
        len(encoder_inputs) != 1
        or encoder_inputs[0] not in producers
        or gather.input[0] != encoder_inputs[0]
        or gather.input[1] not in dependent
    ):
        raise UnsupportedAllAtomOptimization(
            "The MCA all-atom encoder output could not be identified"
        )
    encoder_output = encoder_inputs[0]
    encoder_node_index = producers[encoder_output][0]

    consumers = {}
    for node in model.graph.node:
        for input_ in node.input:
            consumers.setdefault(input_, []).append(node)

    index_tail = []
    index_tensor = "atom_index"
    for expected_op in ("Reshape", "Expand", "Cast"):
        nodes = consumers.get(index_tensor, ())
        if len(nodes) != 1 or nodes[0].op_type != expected_op:
            raise UnsupportedAllAtomOptimization(
                f"Unexpected MCA atom-index path after {index_tensor!r}"
            )
        node = nodes[0]
        if len(node.output) != 1:
            raise UnsupportedAllAtomOptimization(
                f"Unexpected multi-output MCA {expected_op} node"
            )
        index_tail.append(node)
        index_tensor = node.output[0]
    index_cast = index_tail[-1]
    index_cast_attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in index_cast.attribute
    }
    if (
        index_tensor != gather.input[1]
        or consumers.get(index_tensor, ()) != [gather]
        or index_cast_attributes.get("to") != TensorProto.INT64
    ):
        raise UnsupportedAllAtomOptimization(
            "The MCA atom-index expansion has an unexpected contract"
        )

    expected_tail = ("Squeeze", "Identity", "Gemm", "Reshape", "Mul", "Add")
    tail = []
    tail_tensor = gather.output[0]
    for expected_op in expected_tail:
        nodes = consumers.get(tail_tensor, ())
        if len(nodes) != 1 or nodes[0].op_type != expected_op:
            raise UnsupportedAllAtomOptimization(
                f"Unexpected MCA prediction tail after {tail_tensor!r}"
            )
        node = nodes[0]
        if len(node.output) != 1:
            raise UnsupportedAllAtomOptimization(
                f"Unexpected multi-output MCA {expected_op} node"
            )
        tail.append(node)
        tail_tensor = node.output[0]
    _squeeze, _identity, head, output_reshape, scale, unscale = tail
    if len(model.graph.output) != 1 or tail_tensor != model.graph.output[0].name:
        raise UnsupportedAllAtomOptimization(
            "The MCA prediction tail does not terminate at the graph output"
        )
    if len(head.input) != 3:
        raise UnsupportedAllAtomOptimization(
            "The MCA single-atom prediction head is not affine"
        )

    attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in head.attribute
    }
    if attributes.get("transA", 0) != 0 or attributes.get("transB", 0) != 1:
        raise UnsupportedAllAtomOptimization(
            "The MCA single-atom prediction head has an unexpected orientation"
        )

    initializers = _initializer_by_name(model)
    weight = initializers.get(head.input[1])
    if weight is None or len(weight.dims) != 2 or int(weight.dims[0]) != 1:
        raise UnsupportedAllAtomOptimization(
            "The MCA single-atom head has an unexpected weight shape"
        )
    hidden_size = int(weight.dims[1])
    if "target_std" not in initializers or "target_mean" not in initializers:
        raise UnsupportedAllAtomOptimization("The MCA output scaling tensors are missing")
    if set(scale.input) != {output_reshape.output[0], "target_std"}:
        raise UnsupportedAllAtomOptimization("The MCA output scale is unexpected")
    if set(unscale.input) != {scale.output[0], "target_mean"}:
        raise UnsupportedAllAtomOptimization("The MCA output offset is unexpected")

    output = model.graph.output[0]
    output_name = output.name
    output_type = output.type.tensor_type.elem_type
    del model.graph.node[encoder_node_index + 1 :]

    flat_shape = "mca_all_atom_flat_shape"
    model.graph.initializer.append(
        helper.make_tensor(
            flat_shape,
            TensorProto.INT64,
            [2],
            [-1, hidden_size],
        )
    )
    flat_hidden = "mca_all_atom_flat_hidden"
    flat_prediction = "mca_all_atom_flat_prediction"
    output_shape = "mca_all_atom_output_shape"
    normalized = "mca_all_atom_normalized"
    scaled = "mca_all_atom_scaled"
    model.graph.node.append(
        helper.make_node(
            "Reshape",
            [encoder_output, flat_shape],
            [flat_hidden],
            name="mca_all_atom_flatten",
        )
    )
    all_atom_head = deepcopy(head)
    all_atom_head.name = "mca_all_atom_head"
    all_atom_head.input[0] = flat_hidden
    all_atom_head.output[0] = flat_prediction
    model.graph.node.append(all_atom_head)
    model.graph.node.extend(
        (
            helper.make_node(
                "Shape",
                ["atom_mask"],
                [output_shape],
                name="mca_all_atom_shape",
            ),
            helper.make_node(
                "Reshape",
                [flat_prediction, output_shape],
                [normalized],
                name="mca_all_atom_unflatten",
            ),
        )
    )
    all_atom_scale = deepcopy(scale)
    all_atom_scale.name = "mca_all_atom_scale"
    all_atom_scale.input[:] = [
        normalized if input_ == output_reshape.output[0] else input_
        for input_ in scale.input
    ]
    all_atom_scale.output[0] = scaled
    model.graph.node.append(all_atom_scale)
    all_atom_unscale = deepcopy(unscale)
    all_atom_unscale.name = "mca_all_atom_unscale"
    all_atom_unscale.input[:] = [
        scaled if input_ == scale.output[0] else input_ for input_ in unscale.input
    ]
    all_atom_unscale.output[0] = output_name
    model.graph.node.append(all_atom_unscale)

    retained_inputs = [
        deepcopy(input_)
        for input_ in model.graph.input
        if input_.name != "atom_index"
    ]
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(
            output_name,
            output_type,
            ["batch_size", "atom_count"],
        )
    )
    retained_value_info = [
        deepcopy(value)
        for value in model.graph.value_info
        if value.name not in dependent
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)
    del model.metadata_props[:]
    return model


def _derived_root() -> Path:
    configured = os.environ.get("HOTPOT_DERIVED_MODEL_HOME")
    if configured:
        return Path(configured).expanduser()
    model_home = Path(
        os.environ.get(
            "HOTPOT_MODEL_HOME",
            Path.home() / ".cache" / "hotpot" / "models",
        )
    ).expanduser()
    return model_home / "derived"


def _marker_payload(
    source_digest: str,
    external_digest: str,
    model_digest: str,
    external_locations: tuple[str, ...],
) -> dict:
    return {
        "transform_version": _TRANSFORM_VERSION,
        "source_sha256": source_digest,
        "external_data_sha256": external_digest,
        "model_sha256": model_digest,
        "external_data": list(external_locations),
    }


def _read_marker(directory: Path) -> dict | None:
    try:
        return json.loads((directory / _MARKER_NAME).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _complete_model(
    directory: Path,
    source_digest: str,
    external_digest: str,
) -> Path | None:
    marker = _read_marker(directory)
    if (
        marker is None
        or marker.get("transform_version") != _TRANSFORM_VERSION
        or marker.get("source_sha256") != source_digest
        or marker.get("external_data_sha256") != external_digest
    ):
        return None
    model_path = directory / _DERIVED_MODEL_NAME
    if model_path.is_symlink() or not model_path.is_file():
        return None
    external_data = marker.get("external_data")
    if not isinstance(external_data, list) or not all(
        isinstance(name, str)
        and name not in {"", ".", ".."}
        and not Path(name).is_absolute()
        and Path(name).name == name
        for name in external_data
    ):
        return None
    external_paths = tuple(directory / name for name in external_data)
    if not all(
        not path.is_symlink() and path.is_file() for path in external_paths
    ):
        return None
    try:
        if _sha256(model_path) != marker.get("model_sha256"):
            return None
        if _bundle_sha256(external_paths) != external_digest:
            return None
    except OSError:
        return None
    return model_path


def _build_derived_directory(
    source: Path,
    transformed,
    source_digest: str,
    external_digest: str,
    external_locations: tuple[str, ...],
    directory: Path,
) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / _DERIVED_MODEL_NAME
    onnx.save_model(transformed, str(model_path), save_as_external_data=False)
    for location in external_locations:
        external_source = source.parent / location
        if not external_source.is_file():
            raise FileNotFoundError(f"MCA external data not found: {external_source}")
        shutil.copy2(external_source, directory / location)
    derived_external_paths = tuple(directory / name for name in external_locations)
    if _bundle_sha256(derived_external_paths) != external_digest:
        raise RuntimeError("MCA external-data bundle changed while deriving the model")
    payload = _marker_payload(
        source_digest,
        external_digest,
        _sha256(model_path),
        external_locations,
    )
    (directory / _MARKER_NAME).write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return model_path


def _cache_directory(source_digest: str, external_digest: str) -> Path:
    external_key = hashlib.sha256(external_digest.encode()).hexdigest()
    return (
        _derived_root()
        / "mca"
        / source_digest
        / f"all-atoms-v{_TRANSFORM_VERSION}-{external_key}"
    )


def _discard_stale_cache(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _cache_lock(parent: Path):
    lock_path = parent / ".all-atoms.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "Invalid derived MCA cache lock", lock_path)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _persistent_derived_model(
    source: Path,
    transformed,
    source_digest: str,
    external_digest: str,
    external_locations: tuple[str, ...],
) -> Path:
    parent = _derived_root() / "mca" / source_digest
    destination = _cache_directory(source_digest, external_digest)
    parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock(parent):
        complete = _complete_model(destination, source_digest, external_digest)
        if complete is not None:
            return complete

        stale = None
        if destination.exists() or destination.is_symlink():
            stale = parent / (
                f".{destination.name}.stale-{os.getpid()}-{uuid.uuid4().hex}"
            )
            try:
                os.rename(destination, stale)
            except FileNotFoundError:
                stale = None
        staging = Path(tempfile.mkdtemp(prefix=".all-atoms-", dir=parent))
        try:
            model_path = _build_derived_directory(
                source,
                transformed,
                source_digest,
                external_digest,
                external_locations,
                staging / "bundle",
            )
            try:
                os.rename(model_path.parent, destination)
            except OSError as exc:
                if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise
                winner = _complete_model(
                    destination,
                    source_digest,
                    external_digest,
                )
                if winner is None:
                    raise RuntimeError(
                        f"Incomplete derived MCA model cache: {destination}"
                    )
                return winner
            return destination / _DERIVED_MODEL_NAME
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if stale is not None:
                _discard_stale_cache(stale)


@contextmanager
def materialize_all_atom_model(
    source: Path,
    external_digest: str | None = None,
):
    """Yield a cached all-atom graph, retaining a temporary fallback if needed."""

    source = Path(source)
    source_digest = _sha256(source)
    if external_digest is not None:
        cached = _complete_model(
            _cache_directory(source_digest, external_digest),
            source_digest,
            external_digest,
        )
        if cached is not None:
            yield cached
            return

    model = onnx.load(str(source), load_external_data=False)
    transformed = _all_atom_graph(model)
    if transformed is None:
        yield source
        return

    external_locations = _external_locations(transformed)
    external_paths = tuple(source.parent / name for name in external_locations)
    for path in external_paths:
        if not path.is_file():
            raise FileNotFoundError(f"MCA external data not found: {path}")
    if external_digest is None:
        external_digest = _bundle_sha256(external_paths)
    cached = _complete_model(
        _cache_directory(source_digest, external_digest),
        source_digest,
        external_digest,
    )
    if cached is not None:
        yield cached
        return

    try:
        persistent = _persistent_derived_model(
            source,
            transformed,
            source_digest,
            external_digest,
            external_locations,
        )
    except OSError:
        persistent = None
    if persistent is not None:
        yield persistent
        return

    with tempfile.TemporaryDirectory(prefix="hotpot-mca-all-atoms-") as temporary:
        directory = Path(temporary) / "bundle"
        yield _build_derived_directory(
            source,
            transformed,
            source_digest,
            external_digest,
            external_locations,
            directory,
        )
