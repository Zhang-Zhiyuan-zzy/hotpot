"""Collection identity and real graph export behavior after the owner move."""

from copy import deepcopy
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import typing as tp

import pytest

import hotpot
from hotpot import Molecule, MolBundle, to_hotpot_mol, to_pyg_dataset


@pytest.fixture
def molecules():
    ethanol = to_hotpot_mol("CCO")
    benzene = to_hotpot_mol("c1ccccc1")
    ethanol.identifier = "ethanol"
    benzene.identifier = "benzene"
    return ethanol, benzene


@pytest.fixture
def pyg_runtime():
    torch = pytest.importorskip("torch")
    data = pytest.importorskip("torch_geometric.data")
    return torch, data.Data


def _assert_chemical_graph(graph, name, runtime):
    torch, data_type = runtime
    assert isinstance(graph, data_type)
    assert graph.hotpot_schema == ["hotpot.molecule/1"]
    count = 3 if name == "ethanol" else 6
    assert graph.num_nodes == count
    assert graph.z.dtype == torch.int64
    assert graph.z.tolist() == ([6, 6, 8] if name == "ethanol" else [6] * 6)
    assert graph.formal_charge.tolist() == [0] * count
    assert graph.total_h.tolist() == ([3, 2, 1] if name == "ethanol" else [1] * count)
    assert graph.aromatic.tolist() == ([False] * count if name == "ethanol" else [True] * count)
    bonds = {(0, 1), (1, 2)} if name == "ethanol" else {
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5),
    }
    assert {tuple(sorted(row)) for row in graph.edge_index.T.tolist()} == bonds
    assert graph.edge_index.shape == (2, len(bonds))
    assert graph.bond_aromatic.tolist() == ([False] * 2 if name == "ethanol" else [True] * 6)
    assert sorted(graph.bond_order.tolist()) == ([1] * 2 if name == "ethanol" else [1] * 3 + [2] * 3)
    assert "x" not in graph and "edge_attr" not in graph


def test_public_collection_and_operation_have_one_current_owner():
    cheminfo = importlib.import_module("hotpot.cheminfo")
    structure = importlib.import_module("hotpot.cheminfo.structure")
    owner = importlib.import_module("hotpot.cheminfo.structure.bundle")
    operation = importlib.import_module("hotpot.cheminfo.operations.collections")
    assert hotpot.MolBundle is cheminfo.MolBundle is owner.MolBundle is MolBundle
    assert hotpot.to_pyg_dataset is cheminfo.to_pyg_dataset is operation.to_pyg_dataset
    assert MolBundle.__module__ == owner.__name__
    assert MolBundle.__qualname__ == "MolBundle"
    assert to_pyg_dataset.__module__ == operation.__name__
    assert Path(inspect.getfile(MolBundle)).resolve() == Path(owner.__file__).resolve()
    assert structure.__all__ == ("Molecule", "Atom", "Bond", "BondKind", "Crystal")
    legacy = "hotpot.cheminfo.bundle"
    assert legacy not in sys.modules
    assert importlib.util.find_spec(legacy) is None


@pytest.mark.parametrize("function,names,annotations", [
    (MolBundle.__init__, ("self", "list_mol"), {"list_mol": tp.Iterable[Molecule]}),
    (MolBundle.to_pyg_dataset, ("self", "dataset_root"),
     {"dataset_root": os.PathLike}),
    (to_pyg_dataset, ("list_mol", "dataset_root"),
     {"list_mol": tp.Iterable[Molecule], "dataset_root": os.PathLike}),
])
def test_public_signatures_keep_eager_annotations_and_defaults(function, names, annotations):
    signature = inspect.signature(function)
    assert tuple(signature.parameters) == names
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for p in signature.parameters.values())
    assert {name: p.default for name, p in signature.parameters.items()
            if p.default is not inspect.Parameter.empty} == {}
    assert function.__annotations__ == annotations
    assert tp.get_type_hints(function) == annotations
    assert signature.return_annotation is inspect.Signature.empty


def test_constructor_materializes_order_once_and_retains_shared_members(molecules):
    ethanol, benzene = molecules
    visited = []

    def members():
        for mol in (benzene, ethanol, benzene):
            visited.append(mol)
            yield mol

    bundle = MolBundle(members())
    assert visited == bundle.list_mol == [benzene, ethanol, benzene]
    assert bundle.list_mol[0] is bundle.list_mol[2] is benzene
    original = [ethanol, benzene]
    other = MolBundle(original)
    original.reverse()
    assert other.list_mol == [ethanol, benzene]
    ethanol.identifier = "updated"
    assert other.list_mol[0].identifier == "updated"


@pytest.mark.parametrize("copy_graph", [deepcopy, pytest.param(
    lambda value: pickle.loads(pickle.dumps(value)), id="current-pickle",
)])
def test_current_graph_roundtrip_preserves_collection_and_member_sharing(molecules, copy_graph):
    ethanol, benzene = molecules
    bundle = MolBundle([ethanol, benzene, ethanol])
    restored, mol, atom, bond, members = copy_graph((
        bundle, ethanol, ethanol.atoms[0], ethanol.bonds[0], bundle.list_mol,
    ))
    assert type(restored) is MolBundle
    assert restored is not bundle
    assert restored.list_mol is members
    assert members[0] is members[2] is mol
    assert mol is not ethanol and members[1] is not benzene
    assert atom is mol.atoms[0] and atom.mol is mol
    assert bond is mol.bonds[0] and bond.mol is mol
    assert bond.atom1 is mol.atoms[0] and bond.atom2 is mol.atoms[1]
    assert [a.atomic_number for a in mol.atoms] == [6, 6, 8]
    assert len(members[1].atoms) == 6 and len(members[1].bonds) == 6


@pytest.mark.parametrize("module", [
    "hotpot.cheminfo.structure.bundle",
    "hotpot.cheminfo.operations.collections",
    "hotpot.cheminfo.adapters.pyg",
])
def test_cold_import_loads_current_owner_without_eager_torch(module, tmp_path):
    package = Path(hotpot.__file__).resolve().parent
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if not environment.get("HOTPOT_TEST_INSTALLED_PACKAGE"):
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(package.parent) + (os.pathsep + previous if previous else "")
    program = '''
import importlib
import json
from pathlib import Path
import sys
module = importlib.import_module(sys.argv[1])
import hotpot
package = Path(hotpot.__file__).resolve().parent
assert package == Path(sys.argv[2]).resolve()
assert Path(module.__file__).resolve() == package.joinpath(*module.__name__.split(".")[1:]).with_suffix(".py")
assert not any(name == "torch" or name.startswith("torch.") or name == "torch_geometric" or name.startswith("torch_geometric.") for name in sys.modules)
assert "hotpot.cheminfo.bundle" not in sys.modules
assert hotpot.MolBundle.__module__ == "hotpot.cheminfo.structure.bundle"
print(json.dumps({"package": str(package), "module": str(Path(module.__file__).resolve())}))
'''
    result = subprocess.run([sys.executable, "-s", "-B", "-c", program, module, str(package)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["package"] == str(package)


@pytest.mark.parametrize("entrypoint", ["method", "function"])
def test_real_pyg_roundtrip_has_independent_ethanol_and_benzene_semantics(
        tmp_path, molecules, pyg_runtime, entrypoint):
    root = tmp_path / "graphs"
    if entrypoint == "method":
        result = MolBundle(molecules).to_pyg_dataset(root)
    else:
        result = to_pyg_dataset(iter(molecules), root)
    assert result is None
    assert {p.name for p in root.iterdir()} == {"ethanol.pt", "benzene.pt"}
    torch, _ = pyg_runtime
    for name in ("ethanol", "benzene"):
        graph = torch.load(root / (name + ".pt"), weights_only=False)
        _assert_chemical_graph(graph, name, pyg_runtime)


def test_export_binds_serializer_once_then_converts_and_saves_in_member_order(
        tmp_path, molecules, pyg_runtime, monkeypatch):
    from hotpot.cheminfo.adapters import pyg
    torch, _ = pyg_runtime
    ethanol, benzene = molecules
    root = tmp_path / "graphs"
    events = []
    convert = Molecule.to_pyg

    def serializer():
        assert root.is_dir()
        events.append("serializer")
        return torch.save

    def observed_conversion(self):
        events.append(self.identifier)
        if self is benzene:
            _assert_chemical_graph(torch.load(root / "ethanol.pt", weights_only=False),
                                   "ethanol", pyg_runtime)
        return convert(self)

    monkeypatch.setattr(pyg, "_torch_save", serializer)
    monkeypatch.setattr(Molecule, "to_pyg", observed_conversion)
    assert to_pyg_dataset([ethanol, benzene], root) is None
    assert events == ["serializer", "ethanol", "benzene"]
    _assert_chemical_graph(torch.load(root / "benzene.pt", weights_only=False),
                           "benzene", pyg_runtime)


def test_empty_collection_creates_only_the_leaf_and_returns_none(tmp_path, pyg_runtime):
    root = tmp_path / "empty"
    assert MolBundle([]).to_pyg_dataset(root) is None
    assert root.is_dir() and list(root.iterdir()) == []
    assert to_pyg_dataset(iter(()), root) is None
    assert list(root.iterdir()) == []


def test_missing_parent_is_not_created_recursively(tmp_path):
    root = tmp_path / "missing-parent" / "graphs"
    with pytest.raises(FileNotFoundError):
        to_pyg_dataset([], root)
    assert not root.parent.exists()


def test_input_iterable_failure_precedes_directory_creation(tmp_path, molecules):
    root = tmp_path / "graphs"
    failure = ValueError("input iteration failed")

    def members():
        yield molecules[0]
        raise failure

    with pytest.raises(ValueError) as caught:
        to_pyg_dataset(members(), root)
    assert caught.value is failure
    assert not root.exists()


@pytest.mark.parametrize("entrypoint", ["method", "function"])
def test_duplicate_identifier_overwrites_with_the_later_real_graph(
        tmp_path, molecules, pyg_runtime, entrypoint):
    torch, _ = pyg_runtime
    for mol in molecules:
        mol.identifier = "same"
    if entrypoint == "method":
        result = MolBundle(molecules).to_pyg_dataset(tmp_path)
    else:
        result = to_pyg_dataset(iter(molecules), tmp_path)
    assert result is None
    assert {p.name for p in tmp_path.iterdir()} == {"same.pt"}
    _assert_chemical_graph(torch.load(tmp_path / "same.pt", weights_only=False),
                           "benzene", pyg_runtime)


@pytest.mark.parametrize("entrypoint", ["method", "function"])
def test_empty_identifiers_save_real_graphs_with_member_ordinals(
        tmp_path, molecules, pyg_runtime, entrypoint):
    torch, _ = pyg_runtime
    for mol in molecules:
        mol.identifier = ""
    if entrypoint == "method":
        result = MolBundle(molecules).to_pyg_dataset(tmp_path)
    else:
        result = to_pyg_dataset(iter(molecules), tmp_path)
    assert result is None
    assert {p.name for p in tmp_path.iterdir()} == {"0.pt", "1.pt"}
    for ordinal, name in enumerate(("ethanol", "benzene")):
        graph = torch.load(tmp_path / f"{ordinal}.pt", weights_only=False)
        _assert_chemical_graph(graph, name, pyg_runtime)


@pytest.mark.parametrize("entrypoint", ["method", "function"])
def test_mixed_identifiers_keep_names_and_count_all_members_for_ordinals(
        tmp_path, molecules, pyg_runtime, entrypoint):
    torch, _ = pyg_runtime
    molecules[1].identifier = ""
    if entrypoint == "method":
        result = MolBundle(molecules).to_pyg_dataset(tmp_path)
    else:
        result = to_pyg_dataset(iter(molecules), tmp_path)
    assert result is None
    assert {p.name for p in tmp_path.iterdir()} == {"ethanol.pt", "1.pt"}
    for filename, name in (("ethanol.pt", "ethanol"), ("1.pt", "benzene")):
        _assert_chemical_graph(torch.load(tmp_path / filename, weights_only=False),
                               name, pyg_runtime)


@pytest.mark.parametrize("identifiers,filename", [(("1", ""), "1.pt"), (("", "0"), "0.pt")])
@pytest.mark.parametrize("entrypoint", ["method", "function"])
def test_named_and_ordinal_collision_keeps_the_later_real_graph(
        tmp_path, molecules, pyg_runtime, identifiers, filename, entrypoint):
    torch, _ = pyg_runtime
    for mol, identifier in zip(molecules, identifiers):
        mol.identifier = identifier
    if entrypoint == "method":
        result = MolBundle(molecules).to_pyg_dataset(tmp_path)
    else:
        result = to_pyg_dataset(iter(molecules), tmp_path)
    assert result is None
    assert {p.name for p in tmp_path.iterdir()} == {filename}
    _assert_chemical_graph(torch.load(tmp_path / filename, weights_only=False),
                           "benzene", pyg_runtime)


@pytest.mark.parametrize("empty", [False, True])
def test_existing_file_root_preserves_its_contents(tmp_path, molecules, pyg_runtime, empty):
    root = tmp_path / "file-root"
    root.write_bytes(b"existing contents")
    if empty:
        assert to_pyg_dataset([], root) is None
    else:
        with pytest.raises((OSError, RuntimeError)):
            to_pyg_dataset([molecules[0]], root)
    assert root.read_bytes() == b"existing contents"
    assert {p.name for p in tmp_path.iterdir()} == {"file-root"}


@pytest.mark.parametrize("entrypoint", ["method", "function"])
def test_later_conversion_failure_propagates_and_keeps_earlier_saved_graph(
        tmp_path, molecules, pyg_runtime, monkeypatch, entrypoint):
    torch, _ = pyg_runtime
    ethanol, benzene = molecules
    convert = Molecule.to_pyg
    ethanol.identifier = ""
    failure = ValueError("later conversion failed")
    visited = []

    def observed_conversion(self):
        visited.append(self)
        if self is benzene:
            raise failure
        return convert(self)

    monkeypatch.setattr(Molecule, "to_pyg", observed_conversion)
    with pytest.raises(ValueError) as caught:
        if entrypoint == "method":
            MolBundle([ethanol, benzene, ethanol]).to_pyg_dataset(tmp_path)
        else:
            to_pyg_dataset(iter((ethanol, benzene, ethanol)), tmp_path)
    assert caught.value is failure
    assert visited == [ethanol, benzene]
    assert {p.name for p in tmp_path.iterdir()} == {"0.pt"}
    _assert_chemical_graph(torch.load(tmp_path / "0.pt", weights_only=False),
                           "ethanol", pyg_runtime)


@pytest.mark.parametrize("entrypoint", ["method", "function"])
def test_later_save_failure_propagates_and_keeps_earlier_saved_graph(
        tmp_path, molecules, pyg_runtime, monkeypatch, entrypoint):
    torch, _ = pyg_runtime
    ethanol, benzene = molecules
    save = torch.save
    ethanol.identifier = ""
    failure = OSError("later save failed")
    visited = []

    def observed_save(graph, path):
        visited.append(Path(path).name)
        if len(visited) == 2:
            raise failure
        save(graph, path)

    monkeypatch.setattr(torch, "save", observed_save)
    with pytest.raises(OSError) as caught:
        if entrypoint == "method":
            MolBundle([ethanol, benzene, ethanol]).to_pyg_dataset(tmp_path)
        else:
            to_pyg_dataset(iter((ethanol, benzene, ethanol)), tmp_path)
    assert caught.value is failure
    assert visited == ["0.pt", "benzene.pt"]
    assert {p.name for p in tmp_path.iterdir()} == {"0.pt"}
    _assert_chemical_graph(torch.load(tmp_path / "0.pt", weights_only=False),
                           "ethanol", pyg_runtime)


@pytest.mark.parametrize("entrypoint", ["method", "function"])
def test_export_preserves_entrypoint_member_materialization(
        tmp_path, molecules, pyg_runtime, monkeypatch, entrypoint):
    ethanol, benzene = molecules
    bundle = MolBundle([ethanol])
    convert = Molecule.to_pyg
    visited = []

    def observed_conversion(self):
        visited.append(self)
        if self is ethanol:
            bundle.list_mol.append(benzene)
        return convert(self)

    monkeypatch.setattr(Molecule, "to_pyg", observed_conversion)
    if entrypoint == "method":
        result = bundle.to_pyg_dataset(tmp_path)
        assert visited == [ethanol, benzene]
        assert {path.name for path in tmp_path.iterdir()} == {"ethanol.pt", "benzene.pt"}
    else:
        result = to_pyg_dataset(bundle.list_mol, tmp_path)
        assert visited == [ethanol]
        assert {path.name for path in tmp_path.iterdir()} == {"ethanol.pt"}
    assert result is None
