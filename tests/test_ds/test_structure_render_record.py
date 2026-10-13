from __future__ import annotations

from types import SimpleNamespace

from hotpot.ds import _export


class _Asset:
    format = "smi"

    def text(self) -> str:
        return "O\n"


def test_render_structure_record_preserves_direct_content_and_identity(monkeypatch) -> None:
    structure = SimpleNamespace(
        structure_id="structure-1",
        roles=("primary",),
        asset=lambda fmt: _Asset() if fmt == "smi" else None,
    )
    monkeypatch.setattr(_export, "_select_structure", lambda *args, **kwargs: structure)
    data = SimpleNamespace(id="hp-demo[001]-0")

    record = _export.render_structure_record(data, "smiles", structure_index=0)

    assert record == {
        "data_id": "hp-demo[001]-0",
        "structure_id": "structure-1",
        "structure_roles": ["primary"],
        "selection": {
            "mode": "index",
            "structure_index": 0,
            "structure_role": None,
            "structure_id": None,
        },
        "format": "smi",
        "source_format": "smi",
        "converted": False,
        "content": "O\n",
    }


def test_render_structure_record_performs_one_conversion(monkeypatch) -> None:
    structure = SimpleNamespace(
        structure_id="structure-2",
        roles=("reported",),
        asset=lambda fmt: None,
    )
    calls = []
    monkeypatch.setattr(_export, "_select_structure", lambda *args, **kwargs: structure)

    def convert(selected, fmt, data_id):
        calls.append((selected, fmt, data_id))
        return "3\nwater\n...\n", "mol2"

    monkeypatch.setattr(_export, "_convert_structure", convert)
    data = SimpleNamespace(id="hp-demo[001]-1")

    record = _export.render_structure_record(data, "xyz", structure_role="reported")

    assert record["content"] == "3\nwater\n...\n"
    assert record["source_format"] == "mol2"
    assert record["converted"] is True
    assert calls == [(structure, "xyz", "hp-demo[001]-1")]

