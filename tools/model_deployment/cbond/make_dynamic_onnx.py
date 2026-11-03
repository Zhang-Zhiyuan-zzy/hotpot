"""Convert the released fixed-shape CBond head into one dynamic ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


_SHAPE_CONSTANTS = {
    "node_Constant_13": lambda rings, size: [size, rings, 3, 128],
    "node_Constant_27": lambda rings, size: [size, 2 * rings, 64],
    "node_Constant_37": lambda rings, size: [rings, 1, 1, size],
    "node_Constant_45": lambda rings, size: [2 * rings, 1, size],
    "node_Constant_48": lambda rings, size: [rings, 2, -1, size],
    "node_Constant_51": lambda rings, size: [rings, 2, size, 64],
    "node_Constant_89": lambda rings, size: [rings * size, 128],
    "node_Constant_93": lambda rings, size: [size, rings, 128],
    "node_Constant_128": lambda rings, size: rings + 3,
}


def _input_dimensions(model: onnx.ModelProto) -> tuple[int, int]:
    inputs = {value.name: value for value in model.graph.input}
    dimensions = inputs["padded_Xr"].type.tensor_type.shape.dim
    return dimensions[0].dim_value, dimensions[1].dim_value


def _validate_shape_constants(model: onnx.ModelProto) -> None:
    rings, size = _input_dimensions(model)
    nodes = {node.name: node for node in model.graph.node}
    for name, expected in _SHAPE_CONSTANTS.items():
        node = nodes[name]
        value = next(attribute.t for attribute in node.attribute if attribute.name == "value")
        np.testing.assert_array_equal(numpy_helper.to_array(value), expected(rings, size))


def _initializer_fingerprint(model: onnx.ModelProto) -> str:
    digest = hashlib.sha256()
    for tensor in sorted(model.graph.initializer, key=lambda item: item.name):
        array = numpy_helper.to_array(tensor)
        digest.update(tensor.name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_model_matrix(matrix_dir: Path) -> list[Path]:
    paths = sorted(matrix_dir.glob("opset21_cbond(*-*).onnx"))
    models = [onnx.load(path) for path in paths]
    fingerprints = {_initializer_fingerprint(model) for model in models}
    if len(fingerprints) != 1:
        raise RuntimeError("CBond model matrix does not contain one shared parameter set")
    for model in models:
        _validate_shape_constants(model)
    return paths


def _tensor(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def convert(source: Path, output: Path) -> None:
    model = onnx.load(source)
    _validate_shape_constants(model)

    inputs = {value.name: value for value in model.graph.input}
    for input_name in ("padded_Xr", "rings_mask"):
        dimensions = inputs[input_name].type.tensor_type.shape.dim
        dimensions[0].ClearField("dim_value")
        dimensions[0].dim_param = "rings_num"
        dimensions[1].ClearField("dim_value")
        dimensions[1].dim_param = "ring_size"
    del model.graph.value_info[:]

    constants = {
        "__idx0_vector": [0],
        "__idx1_vector": [1],
        "__idx0_scalar": np.asarray(0, dtype=np.int64),
        "__one": [1],
        "__minus_one": [-1],
        "__two": [2],
        "__three": [3],
        "__three_scalar": np.asarray(3, dtype=np.int64),
        "__sixty_four": [64],
        "__one_twenty_eight": [128],
    }
    model.graph.initializer.extend(_tensor(name, value) for name, value in constants.items())

    helper_nodes = [
        helper.make_node("Shape", ["padded_Xr"], ["__ring_shape"], name="dynamic_ring_shape"),
        helper.make_node("Gather", ["__ring_shape", "__idx0_vector"], ["__rings"], axis=0, name="dynamic_rings"),
        helper.make_node("Gather", ["__ring_shape", "__idx1_vector"], ["__ring_size"], axis=0, name="dynamic_ring_size"),
        helper.make_node("Gather", ["__ring_shape", "__idx0_scalar"], ["__rings_scalar"], axis=0, name="dynamic_rings_scalar"),
        helper.make_node("Mul", ["__rings", "__two"], ["__two_rings"], name="dynamic_two_rings"),
        helper.make_node("Mul", ["__rings", "__ring_size"], ["__rings_times_size"], name="dynamic_rings_times_size"),
        helper.make_node("Add", ["__rings_scalar", "__three_scalar"], ["val_80"], name="dynamic_rings_plus_three"),
        helper.make_node("Concat", ["__ring_size", "__rings", "__three", "__one_twenty_eight"], ["val_8"], axis=0, name="dynamic_shape_8"),
        helper.make_node("Concat", ["__ring_size", "__two_rings", "__sixty_four"], ["val_14"], axis=0, name="dynamic_shape_14"),
        helper.make_node("Concat", ["__rings", "__one", "__one", "__ring_size"], ["val_18"], axis=0, name="dynamic_shape_18"),
        helper.make_node("Concat", ["__two_rings", "__one", "__ring_size"], ["val_23"], axis=0, name="dynamic_shape_23"),
        helper.make_node("Concat", ["__rings", "__two", "__minus_one", "__ring_size"], ["val_25"], axis=0, name="dynamic_shape_25"),
        helper.make_node("Concat", ["__rings", "__two", "__ring_size", "__sixty_four"], ["val_27"], axis=0, name="dynamic_shape_27"),
        helper.make_node("Concat", ["__rings_times_size", "__one_twenty_eight"], ["val_60"], axis=0, name="dynamic_shape_60"),
        helper.make_node("Concat", ["__ring_size", "__rings", "__one_twenty_eight"], ["val_62"], axis=0, name="dynamic_shape_62"),
    ]
    fixed_nodes = [node for node in model.graph.node if node.name not in _SHAPE_CONSTANTS]
    del model.graph.node[:]
    model.graph.node.extend(helper_nodes)
    model.graph.node.extend(fixed_nodes)

    model.doc_string = "CBond head with dynamic ring-count and ring-size dimensions"
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path)
    args = parser.parse_args()

    if args.matrix_dir is not None:
        paths = validate_model_matrix(args.matrix_dir)
        print(f"validated {len(paths)} fixed-shape CBond models")
    convert(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
