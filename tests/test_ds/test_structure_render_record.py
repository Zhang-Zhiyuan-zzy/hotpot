from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hotpot

from hotpot.ds._export import render_structure_record


class _Structures:
    def __init__(self, rows, paths):
        self._values = tuple(rows)
        self._paths = paths

    def _rows(self):
        return self._values

    def _verified_path(self, row):
        return self._paths[row["structure_id"]]


def _data(rows, paths):
    return SimpleNamespace(
        id="hp-demo[001]-7",
        structures=_Structures(rows, paths),
    )


def test_render_structure_record_preserves_direct_bytes_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "record.smi"
    path.write_text("[NH4+]\n", encoding="utf-8")
    data = _data(
        [{"structure_id": "structure-1", "format": "smi", "role": "primary", "primary": True}],
        {"structure-1": path},
    )

    result = render_structure_record(data, "smi")

    assert result == {
        "data_id": "hp-demo[001]-7",
        "structure_id": "structure-1",
        "structure_roles": ["primary"],
        "selection": {
            "mode": "primary",
            "structure_index": None,
            "structure_role": None,
            "structure_id": None,
        },
        "format": "smi",
        "source_format": "smi",
        "converted": False,
        "content": "[NH4+]\n",
    }


def test_render_structure_record_converts_once_from_selected_current_asset(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "record.xyz"
    path.write_text("1\nrecord\nH 0 0 0\n", encoding="utf-8")
    data = _data(
        [{"structure_id": "structure-2", "format": "xyz", "role": "reported", "primary": True}],
        {"structure-2": path},
    )
    calls = []

    class Molecule:
        smiles = "[H]"

        @classmethod
        def read_one(cls, source, fmt):
            calls.append((source, fmt))
            return cls()

    monkeypatch.setattr(hotpot, "Molecule", Molecule)

    result = render_structure_record(data, "smi", structure_role="reported")

    assert calls == [(str(path), "xyz")]
    assert result["content"] == "[H]\n"
    assert result["source_format"] == "xyz"
    assert result["converted"] is True
    assert result["selection"]["mode"] == "role"
