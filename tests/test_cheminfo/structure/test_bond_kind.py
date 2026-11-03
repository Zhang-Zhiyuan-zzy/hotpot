"""The shared bond category contract has one identity and no execution dependency."""

from pathlib import Path
import subprocess
import sys

import pytest


FIXTURE = Path(__file__).with_name("fixtures") / "bond_kind_core_protocol4.json"


def test_public_and_carrier_bond_categories_share_the_canonical_contract():
    from hotpot import BondKind, Molecule
    from hotpot.cheminfo.structure.bond_kind import BondKind as canonical_kind

    assert BondKind is canonical_kind
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=8)
    bond = mol.add_bond(0, 1, bond_order=2., bond_kind=BondKind.DOUBLE)
    assert bond.bond_kind is canonical_kind.DOUBLE
    assert bond.bond_order == 2.
    assert mol.copy().bonds[0].bond_kind is canonical_kind.DOUBLE


@pytest.mark.parametrize("first_import", ["hotpot", "hotpot.cheminfo.structure.bond_kind"])
def test_old_core_pickles_are_rejected_and_current_enum_pickles_round_trip_cold(first_import):
    subprocess.run([sys.executable, "-B", "-c", """
import base64
import importlib
import json
import pickle
import sys
from unittest import TestCase
importlib.import_module(sys.argv[1])
from hotpot import BondKind
from hotpot.cheminfo.structure.bond_kind import BondKind as canonical
with open(sys.argv[2]) as source:
    captured = json.load(source)
assert captured["module"] == "hotpot.cheminfo.core"
assert BondKind is canonical
assert canonical.__module__ == "hotpot.cheminfo.structure.bond_kind"
for row in captured["members"]:
    with TestCase().assertRaises(ModuleNotFoundError) as rejected:
        pickle.loads(base64.b64decode(row["pickle_protocol4_base64"]))
    assert rejected.exception.name == captured["module"]
    value = canonical[row["name"]]
    assert value.value == row["value"]
    for protocol in (4, 5):
        assert pickle.loads(pickle.dumps(value, protocol=protocol)) is value
""", first_import, str(FIXTURE)], check=True)


def test_contract_leaf_loads_without_importing_the_hotpot_execution_tree():
    from hotpot.cheminfo.structure import bond_kind

    source = Path(bond_kind.__file__).resolve()
    subprocess.run([sys.executable, "-I", "-B", "-c", """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_bond_category_contract", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.BondKind.DATIVE.value == "dative"
assert not any(name == "hotpot" or name.startswith("hotpot.") for name in sys.modules)
""", str(source)], check=True)
