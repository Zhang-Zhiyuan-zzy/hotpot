import json
from pathlib import Path

import onnx
from mca.all_atom_model import materialize_all_atom_model
from mca.model_store import ModelStore
from onnx import TensorProto, helper


def test_unsupported_graph_keeps_the_legacy_contract(tmp_path, monkeypatch):
    source = tmp_path / "unsupported.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["atom_index"], ["prediction"])],
        "unsupported",
        [helper.make_tensor_value_info("atom_index", TensorProto.INT64, [None])],
        [helper.make_tensor_value_info("prediction", TensorProto.INT64, [None])],
    )
    onnx.save_model(helper.make_model(graph), source)
    monkeypatch.setenv("HOTPOT_DERIVED_MODEL_HOME", str(tmp_path / "cache"))

    with materialize_all_atom_model(source) as materialized:
        assert materialized == source


def test_corrupt_external_cache_is_rebuilt(tmp_path, monkeypatch):
    store = ModelStore()
    source, variant, _, _ = store.resolve("cpu", None)
    external_digest = store.manifest["models"][variant]["external_data"]["sha256"]
    monkeypatch.setenv("HOTPOT_DERIVED_MODEL_HOME", str(tmp_path / "cache"))

    with materialize_all_atom_model(source, external_digest) as materialized:
        cached_model = Path(materialized)
        marker = json.loads(
            cached_model.with_name("complete.json").read_text(encoding="utf-8")
        )
        external_path = cached_model.with_name(marker["external_data"][0])

    external_path.unlink()
    external_path.write_bytes(b"corrupt")

    with materialize_all_atom_model(source, external_digest) as rebuilt:
        assert rebuilt == cached_model
        assert external_path.stat().st_size > len(b"corrupt")
        model = onnx.load(str(rebuilt), load_external_data=False)
        assert "atom_index" not in {input_.name for input_ in model.graph.input}
