"""Physical carrier owners, fresh initialization and current-version graph roots."""

from collections.abc import Mapping
from copy import deepcopy
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
import typing

import numpy as np
import pytest

import hotpot
from hotpot import Atom, Bond, BondKind, Crystal, Molecule
from hotpot.cheminfo.structure._fields import MolBlock
from hotpot.cheminfo.structure._source_identity import _SourceIdentity
from hotpot.cheminfo.structure.conformer import Conformers
from hotpot.cheminfo.structure.errors import NotInSameMolecule, ObjNotInMolecule
from hotpot.cheminfo.views.geometry import Angle, Torsion
from hotpot.cheminfo.views.pairs import AtomPair, AtomPairs
from hotpot.cheminfo.views.sequence import AtomSeq
from hotpot.cheminfo.views.topology import JointRing, Ring
from tests.test_cheminfo._stage5_fixtures import immutable_snapshot


_PUBLIC_OWNERS = {
    "Molecule": "hotpot.cheminfo.structure.molecule",
    "Atom": "hotpot.cheminfo.structure.atom",
    "Bond": "hotpot.cheminfo.structure.bond",
    "BondKind": "hotpot.cheminfo.structure.bond_kind",
    "Crystal": "hotpot.cheminfo.structure.crystal",
}
_OTHER_OWNERS = {
    "MolBlock": "hotpot.cheminfo.structure._fields",
    "_SourceIdentity": "hotpot.cheminfo.structure._source_identity",
    "NotInSameMolecule": "hotpot.cheminfo.structure.errors",
    "ObjNotInMolecule": "hotpot.cheminfo.structure.errors",
    "Conformers": "hotpot.cheminfo.structure.conformer",
    "AtomSeq": "hotpot.cheminfo.views.sequence",
    "AtomPair": "hotpot.cheminfo.views.pairs",
    "AtomPairs": "hotpot.cheminfo.views.pairs",
    "Angle": "hotpot.cheminfo.views.geometry",
    "Torsion": "hotpot.cheminfo.views.geometry",
    "JointRing": "hotpot.cheminfo.views.topology",
    "Ring": "hotpot.cheminfo.views.topology",
}
_FIRST_IMPORTS = tuple(dict.fromkeys((*_PUBLIC_OWNERS.values(), *_OTHER_OWNERS.values(),
                                    "hotpot.cheminfo.views._source")))
_WRAPPED_METHODS = (
    "add_atom", "_replace_atom", "replace_atom", "add_bond", "add_bonds",
    "remove_atom", "remove_atoms", "remove_bond", "remove_bonds", "add_hydrogens",
    "calc_atom_valence", "calc_implicit_hydrogens", "refresh_atom_id",
    "conformer_add", "conformer_clear", "clear_constraints", "constraint_bonds_angles",
)


_CHILD_ORIGINS = r'''
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys


def origin_record():
    import hotpot
    package = Path(hotpot.__file__).resolve().parent
    assert package == Path(sys.argv[1]).resolve()
    assert "hotpot.cheminfo.core" not in sys.modules
    assert importlib.util.find_spec("hotpot.cheminfo.core") is None
    assert not (package / "cheminfo/core.py").exists()
    origins = {}
    for name in (
        "hotpot.cheminfo.structure.molecule", "hotpot.cheminfo.structure.atom",
        "hotpot.cheminfo.structure.bond", "hotpot.cheminfo.structure.bond_kind",
        "hotpot.cheminfo.structure.crystal", "hotpot.cheminfo.structure.errors",
        "hotpot.cheminfo.structure._fields", "hotpot.cheminfo.structure.conformer",
        "hotpot.cheminfo.views.sequence", "hotpot.cheminfo.views.geometry",
        "hotpot.cheminfo.views.pairs", "hotpot.cheminfo.views.topology",
        "hotpot.cheminfo.views._source",
        "hotpot.cheminfo.structure._source_identity",
    ):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        assert origin == package.joinpath(*name.split(".")[1:]).with_suffix(".py")
        origins[name] = str(origin)
    native = sys.modules.get("hotpot.cheminfo.structure._fact_native")
    if native is not None:
        origin = Path(native.__file__).resolve()
        assert origin.parent == package / "cheminfo/structure"
        assert any(str(origin).endswith(suffix)
                   for suffix in importlib.machinery.EXTENSION_SUFFIXES)
        origins[native.__name__] = {
            "path": str(origin), "sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        }
    return {"package": str(package), "implementation": sys.implementation.cache_tag,
            "origins": origins}
'''


def _run_child(tmp_path, program, *arguments):
    package = Path(hotpot.__file__).resolve().parent
    checkout_package = Path(__file__).resolve().parents[3] / "hotpot"
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if environment.get("HOTPOT_TEST_INSTALLED_PACKAGE"):
        assert package != checkout_package
    else:
        assert package == checkout_package
        previous = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(package.parent) + (os.pathsep + previous if previous else "")
    script = tmp_path / "child.py"
    script.write_text(_CHILD_ORIGINS + "\n" + textwrap.dedent(program), encoding="utf-8")
    argv = [sys.executable, "-B", str(script), str(package), *(str(arg) for arg in arguments)]
    (tmp_path / "invocation.json").write_text(json.dumps({
        "argv": argv, "cwd": str(tmp_path.resolve()), "expected_package": str(package),
        "environment": {name: environment.get(name) for name in (
            "PYTHONPATH", "PYTHONSAFEPATH", "PYTHONDONTWRITEBYTECODE",
            "HOTPOT_TEST_INSTALLED_PACKAGE", "HOTPOT_TEST_WHEEL",
        )},
    }, indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(argv, cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=60)
    (tmp_path / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (tmp_path / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    (tmp_path / "result.json").write_text(json.dumps({
        "returncode": result.returncode,
    }) + "\n", encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize("name,owner", _PUBLIC_OWNERS.items())
def test_public_exports_are_the_exact_leaf_classes(name, owner):
    cheminfo = importlib.import_module("hotpot.cheminfo")
    structure = importlib.import_module("hotpot.cheminfo.structure")
    module = importlib.import_module(owner)
    cls = getattr(module, name)
    assert getattr(hotpot, name) is getattr(cheminfo, name) is getattr(structure, name) is cls
    assert cls.__module__ == owner and cls.__qualname__ == name
    assert Path(inspect.getfile(cls)).resolve() == Path(module.__file__).resolve()


@pytest.mark.parametrize("name,owner", _OTHER_OWNERS.items())
def test_support_and_view_classes_have_their_real_defining_paths(name, owner):
    cls = globals()[name]
    module = importlib.import_module(owner)
    assert getattr(module, name) is cls
    assert cls.__module__ == owner and cls.__qualname__ == name
    assert Path(inspect.getfile(cls)).resolve() == Path(module.__file__).resolve()
    assert importlib.import_module("hotpot.cheminfo.views").__all__ == ()


def test_exact_mro_and_real_subclasses_keep_handle_and_view_membership():
    assert Atom.__mro__ == (Atom, MolBlock, object)
    assert Bond.__mro__ == (Bond, AtomSeq, MolBlock, object)
    assert all(cls.__bases__ == (AtomSeq,) for cls in (Angle, Torsion, Ring))
    assert issubclass(AtomPairs, Mapping)

    class DerivedAtom(Atom):
        pass

    class DerivedBond(Bond):
        pass

    class DerivedSequence(AtomSeq):
        pass

    class DerivedRing(Ring):
        pass

    mol = Molecule()
    first = DerivedAtom(mol, atomic_number=6)
    second = Atom(mol, atomic_number=6, coordinates=(1., 0., 0.))
    third = Atom(mol, atomic_number=6, coordinates=(0., 1., 0.))
    for start, end in ((first, second), (second, third), (third, first)):
        mol.add_bond(start, end)
    bond = DerivedBond(first, second, bond_order=2.)
    assert bond.atoms == (first, second) and bond.bonds == [bond]
    assert bond in bond and first in bond and "_view_source" not in bond.__dict__
    first.x = .25
    assert bond.bond_kind is BondKind.DOUBLE and first.x == .25
    sequence = DerivedSequence(first, second, third)
    ring = DerivedRing(first, second, third)
    assert sequence.atoms == ring.atoms == (first, second, third)
    assert mol.bonds[0] in sequence and mol.bonds[0] in ring
    assert "_view_source" in sequence.__dict__ and "_view_representation" in ring.__dict__
    with pytest.raises(AttributeError, match="read-only"):
        sequence._replace_atom(first, third)


def test_eager_annotations_quoted_names_and_existing_defaults_are_preserved():
    geometry = importlib.import_module("hotpot.cheminfo.rules.mathematics.geometry")
    reports = importlib.import_module("hotpot.cheminfo.adapters.structure.geometry")
    ff = importlib.import_module("hotpot.cheminfo.operations.forcefields")
    graph = importlib.import_module("hotpot.cheminfo.rules.mathematics.graph")

    assert Atom.__init__.__annotations__ == {"mol": Molecule}
    assert inspect.signature(Atom.__init__).parameters["mol"].default is None
    assert Bond.__init__.__annotations__ == {"atom1": Atom, "atom2": Atom}
    assert AtomSeq.__init__.__annotations__ == {"atoms": Atom}
    assert typing.get_type_hints(Atom.__init__) == {"mol": Molecule}
    assert typing.get_type_hints(Bond.__init__) == {"atom1": Atom, "atom2": Atom}
    assert typing.get_type_hints(AtomSeq.__init__) == {"atoms": Atom}
    for getter in (Atom.bonds.fget, AtomSeq.bonds.fget):
        assert getter.__annotations__ == {"return": list["Bond"]}
        assert typing.get_type_hints(getter) == {"return": list[Bond]}
    assert Molecule.angles.fget.__annotations__ == {"return": list["Angle"]}
    assert typing.get_type_hints(Molecule.angles.fget) == {"return": list[Angle]}
    assert Bond.bond_segment.fget.__annotations__ == {"return": geometry.Segment}
    assert Ring.geometry_cycle.fget.__annotations__ == {"return": geometry.Cycle}
    assert Ring.relation_to_bond.__annotations__ == {
        "bond": Bond, "return": "geometry_adapter.BondRingFinding[Ring, Bond]",
    }
    assert typing.get_type_hints(Ring.relation_to_bond) == {
        "bond": Bond, "return": reports.BondRingFinding[Ring, Bond],
    }
    assert typing.get_type_hints(JointRing.__contains__) == {
        "item": typing.Union[Atom, Bond, Ring],
    }
    for method in (Molecule.build3d, Molecule.optimize):
        parameter = inspect.signature(method).parameters["convergence_level"]
        assert parameter.annotation == "ff.ConvergenceLevel"
        assert parameter.default is ff.DEFAULT_CONVERGENCE_LEVEL
    assert (inspect.signature(Molecule.rings_for_scope).parameters["max_cycles"].default
            is graph.DEFAULT_RELEVANT_CYCLE_LIMIT)


def test_source_identity_is_shared_by_snapshots_and_native_fact_updates():
    from hotpot.cheminfo.structure._identity import _capture_source
    from hotpot.cheminfo.structure._mutation import _normalize_atom_facts
    from hotpot.cheminfo.structure.snapshots import _BondingSnapshot, _AromaticAssignment

    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    source = _capture_source(mol)
    assert type(source) is _SourceIdentity and source.owner is mol._owner_token
    assert typing.get_type_hints(_BondingSnapshot)["source"] is _SourceIdentity
    assert typing.get_type_hints(_AromaticAssignment)["source"] is _SourceIdentity
    update = _normalize_atom_facts(source, (0,), (atom._uid,), {"x": 3.})
    assert update.source is source
    mol.updata_atoms(atoms=[atom], x=4.)
    assert atom.x == 4. and type(_capture_source(mol)) is _SourceIdentity


def test_conformers_retains_the_existing_missing_torch_type_hint_behavior():
    from hotpot.utils import types

    assert Conformers.add.__annotations__ == {
        "coords": types.ArrayLike, "energy": typing.Union[types.ArrayLike, float],
    }
    assert "torch" not in Conformers.add.__globals__
    with pytest.raises(NameError, match="torch"):
        typing.get_type_hints(Conformers.add)


@pytest.mark.parametrize("first_import", _FIRST_IMPORTS)
def test_every_owner_leaf_imports_cold_without_core_or_reinstalling_descriptors(first_import, tmp_path):
    program = '''
import inspect
first_import = sys.argv[2]
importlib.import_module(first_import)
from hotpot import Atom, Molecule
from hotpot.cheminfo.structure._fields import ATOM_FIELDS
from hotpot.cheminfo.views.geometry import Angle, Torsion
modules = json.loads(sys.argv[3])
wrapped = json.loads(sys.argv[4])
descriptors = {("atom", name): Atom.__dict__[name] for name in ATOM_FIELDS}
for name in ("energy", "partial_charge", "gibbs", "force", "zero_point", "thermo", "capacity"):
    descriptors[("molecule", name)] = Molecule.__dict__[name]
descriptors[("angle", "constraint")] = Angle.constraint
descriptors[("torsion", "constraint")] = Torsion.constraint
assert all(isinstance(value, property) for value in descriptors.values())
assert Angle.constraint.fset is not None and Torsion.constraint.fset is not None
methods = {name: getattr(Molecule, name) for name in wrapped}
for name, method in methods.items():
    assert method.__module__ == "hotpot.cheminfo.structure.molecule"
    assert method.__qualname__ == "Molecule." + name
    assert inspect.unwrap(method) is method.__wrapped__
    assert not hasattr(method.__wrapped__, "__wrapped__")
for name in reversed(modules):
    importlib.import_module(name)
for (owner, name), descriptor in descriptors.items():
    cls = {"atom": Atom, "molecule": Molecule, "angle": Angle, "torsion": Torsion}[owner]
    assert cls.__dict__[name] is descriptor
assert all(getattr(Molecule, name) is method for name, method in methods.items())
print(json.dumps({**origin_record(), "first_import": first_import,
                  "descriptors": len(descriptors), "wrappers": len(methods)}))
'''
    receipt = _run_child(tmp_path, program, first_import,
                         json.dumps(_FIRST_IMPORTS), json.dumps(_WRAPPED_METHODS))
    assert receipt["first_import"] == first_import
    assert receipt["descriptors"] == 28 and receipt["wrappers"] == 17


def test_first_scalar_write_installs_one_native_dispatch_for_all_real_atom_types(tmp_path):
    program = '''
atom_owner = importlib.import_module("hotpot.cheminfo.structure.atom")
commit_owner = importlib.import_module("hotpot.cheminfo.structure._commit")
errors = importlib.import_module("hotpot.cheminfo.structure.errors")
from hotpot import Atom, Molecule
from hotpot.cheminfo.structure._fields import MolBlock
from hotpot.cheminfo.structure._source_identity import _SourceIdentity
assert atom_owner._ATOM_FACT_ENGINE is None
assert "__setattr__" not in Atom.__dict__
engine = commit_owner._fact_engine()
assert engine is sys.modules["hotpot.cheminfo.structure._fact_native"]
install = engine._install_atom_dispatch
calls = []
def counted(*arguments):
    calls.append(arguments)
    return install(*arguments)
engine._install_atom_dispatch = counted
try:
    first = Atom(Molecule(), atomic_number=6)
    assert calls == [] and atom_owner._ATOM_FACT_ENGINE is None
    first.x = 1.25
    assert calls == [(Atom, atom_owner._atomic_number_setter,
                      commit_owner._commit_atom_sparse,
                      errors.NotInSameMolecule, errors.ObjNotInMolecule)]
    dispatch = Atom.__dict__["__setattr__"]
    first.atomic_number = 8
    second = Atom(Molecule(), atomic_number=7)
    second.y = -2.
    class DerivedAtom(Atom):
        pass
    third = DerivedAtom(Molecule(), atomic_number=6)
    third.z = 3.
    third.isotope = 13
    MolBlock.__setattr__(third, "x", 4.)
    assert first.atomic_number == 8 and first.x == 1.25
    assert second.y == -2. and third.coordinates == (4., 0., 3.) and third.isotope == 13
    assert atom_owner._ATOM_FACT_ENGINE is engine
    assert Atom.__dict__["__setattr__"] is dispatch and len(calls) == 1
    assert DerivedAtom.__setattr__ is Atom.__setattr__
finally:
    engine._install_atom_dispatch = install
print(json.dumps({**origin_record(), "installations": len(calls),
                  "callbacks": [item.__module__ + "." + item.__qualname__ for item in calls[0]]}))
'''
    receipt = _run_child(tmp_path, program)
    assert receipt["installations"] == 1
    assert receipt["callbacks"] == [
        "hotpot.cheminfo.structure.atom.Atom",
        "hotpot.cheminfo.structure.atom._atomic_number_setter",
        "hotpot.cheminfo.structure._commit._commit_atom_sparse",
        "hotpot.cheminfo.structure.errors.NotInSameMolecule",
        "hotpot.cheminfo.structure.errors.ObjNotInMolecule",
    ]
    assert "hotpot.cheminfo.structure._fact_native" in receipt["origins"]


def _joint_roots():
    mol = Molecule()
    for coordinates in ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (1., 1., 1.)):
        mol.create_atom(atomic_number=6, coordinates=coordinates)
    first, second, third, fourth = mol.atoms
    for start, end in ((first, second), (second, third), (third, first), (third, fourth)):
        mol.add_bond(start, end)
    detached = mol.create_atom(atomic_number=1, coordinates=(2., 1., 1.), constraint=True)
    detached.isotope = 2
    removed_bond = mol.add_bond(fourth, detached, bond_order=1., constraint=True, id=91)
    mol.remove_atom(detached)
    unregistered = Bond(first, third, bond_order=2., id=83)
    bond = mol.bonds[0]
    opaque = [b"annotation", {"value": 7}]
    bond._set_bond_metadata(bond_kind=BondKind.SINGLE,
                            bond_source_metadata={"a": opaque, "b": opaque})
    bond._bond_source_metadata["member"] = first
    annotation = {"values": [b"shared", 6]}
    first.annotation = annotation
    other = Molecule()
    foreign = other.create_atom(atomic_number=8)
    mol.properties.update(owner=mol, other=other, annotation=annotation)
    other.properties.update(other=mol, member=first, annotation=annotation)
    mol.charge = 0
    mol.spin_mult = 1
    mol.conformer_add(mol.coordinates, energy=-2.)
    angle, torsion = Angle(first, second, third), Torsion(first, second, third, fourth)
    angle.constraint = torsion.constraint = True
    table = mol.atom_pairs
    pair = table[(first, fourth)]
    pair.wiberg_bond_order = .25
    ring = Ring(first, second, third)
    joint = JointRing(ring)
    roots = {
        "view": angle, "molecule": mol, "other": other, "atom": first, "foreign": foreign,
        "bond": bond, "detached": detached, "removed_bond": removed_bond,
        "unregistered": unregistered, "annotation": annotation,
        "views": (angle, torsion, table, pair, ring, joint),
    }
    expected = {
        "record": immutable_snapshot(mol._capture_factual_record()[0]),
        "other_record": immutable_snapshot(other._capture_factual_record()[0]),
        "detached": immutable_snapshot(detached._detached_table.__getstate__()),
        "detached_cell": detached._constraint_values[detached._uid],
        "uids": tuple(member._uid for member in (*mol.atoms, *mol.bonds)),
        "high_water": (mol._next_atom_uid, mol._next_bond_uid),
        "detached_uid": detached._uid, "removed_bond_uid": removed_bond._uid,
        "unregistered_uid": unregistered._uid,
        "view_geometry": tuple(getattr(view, "_view_geometry", getattr(view, "_geometry", None))
                               for view in roots["views"]),
    }
    return roots, expected


def _assert_joint_roots(roots, expected):
    from hotpot import Atom, Bond
    from hotpot.cheminfo.views.geometry import Angle, Torsion
    from hotpot.cheminfo.views.pairs import AtomPair, AtomPairs
    from hotpot.cheminfo.views.topology import JointRing, Ring

    mol, other = roots["molecule"], roots["other"]
    first, foreign, bond = roots["atom"], roots["foreign"], roots["bond"]
    detached, removed, unregistered = roots["detached"], roots["removed_bond"], roots["unregistered"]
    assert first is mol.atoms[0] and foreign is other.atoms[0] and bond is mol.bonds[0]
    assert first._uid == foreign._uid and first is not foreign
    assert mol._owner_token is not other._owner_token and mol._mutation_lock is not other._mutation_lock
    assert mol.properties["owner"] is mol and mol.properties["other"] is other
    assert other.properties["other"] is mol and other.properties["member"] is first
    assert mol.properties["annotation"] is other.properties["annotation"] is roots["annotation"]
    assert first.annotation is roots["annotation"]
    assert all(member._uid_owner is mol._owner_token for member in
               (*mol.atoms, *mol.bonds, detached, removed, unregistered))
    assert foreign._uid_owner is other._owner_token
    assert bond.atom1 is first and bond.atom2 is mol.atoms[1]
    metadata = bond._bond_source_metadata
    assert metadata["a"] is metadata["b"] and metadata["member"] is first
    assert metadata["a"] == [b"annotation", {"value": 7}]
    assert detached.mol is removed.mol is unregistered.mol is mol
    assert detached._uid == expected["detached_uid"] and detached.atomic_number == 1 and detached.isotope == 2
    assert all(member is not detached for member in mol.atoms)
    assert removed._uid == expected["removed_bond_uid"] and removed.atom2 is detached
    assert removed.atom1 is mol.atoms[3] and removed.constraint and removed.id == 91
    assert unregistered._uid == expected["unregistered_uid"]
    assert unregistered.atom1 is first and unregistered.atom2 is mol.atoms[2]
    assert all(member is not removed and member is not unregistered for member in mol.bonds)
    assert unregistered.bond_order == 2. and unregistered.id == 83
    assert immutable_snapshot(detached._detached_table.__getstate__()) == expected["detached"]
    assert detached._constraint_values[detached._uid] == expected["detached_cell"]
    assert all(not block.flags.writeable for block in detached._detached_table._blocks)
    assert tuple(member._uid for member in (*mol.atoms, *mol.bonds)) == expected["uids"]
    assert (mol._next_atom_uid, mol._next_bond_uid) == expected["high_water"]
    assert immutable_snapshot(mol._capture_factual_record()[0]) == expected["record"]
    assert immutable_snapshot(other._capture_factual_record()[0]) == expected["other_record"]
    views = roots["views"]
    assert tuple(type(view) for view in views) == (Angle, Torsion, AtomPairs, AtomPair, Ring, JointRing)
    assert roots["view"] is views[0] and all(view.mol is mol for view in views)
    assert all(view._atoms[0] is first for view in views)
    assert tuple(getattr(view, "_view_geometry", getattr(view, "_geometry", None))
                 for view in views) == expected["view_geometry"]
    assert views[0].constraint and views[1].constraint and views[3].wiberg_bond_order == .25
    assert views[4] is views[5].rings[0]
    assert all(any(member is atom for atom in mol.atoms) for view in views for member in view._atoms)
    assert np.isfinite(views[0].degrees) and np.isfinite(views[1].degrees)
    assert views[2].pair_distance.shape == (6,) and np.isfinite(views[3].distance)
    assert isinstance(first, Atom) and isinstance(bond, Bond)
    added = mol.create_atom(atomic_number=7)
    assert added._uid == expected["high_water"][0] > detached._uid
    fresh = Bond(first, added, bond_order=1.)
    assert fresh._uid == expected["high_water"][1] > removed._uid


def test_joint_deepcopy_keeps_cross_owner_member_view_and_detached_identity():
    roots, expected = _joint_roots()
    restored = deepcopy(roots)
    assert restored["molecule"]._owner_token is not roots["molecule"]._owner_token
    assert restored["molecule"]._mutation_lock is not roots["molecule"]._mutation_lock
    assert restored["annotation"] is not roots["annotation"]
    _assert_joint_roots(restored, expected)


def test_current_joint_pickle_restores_cold_from_view_first_roots(tmp_path):
    roots, expected = _joint_roots()
    artifact = tmp_path / "current_joint_roots.pkl"
    artifact.write_bytes(pickle.dumps((roots, expected), protocol=5))
    program = '''
from dataclasses import fields, is_dataclass
import pickle
import struct
import numpy as np
'''
    program += "\n" + inspect.getsource(immutable_snapshot) + "\n" + inspect.getsource(_assert_joint_roots)
    program += '''
roots, expected = pickle.loads(Path(sys.argv[2]).read_bytes())
_assert_joint_roots(roots, expected)
print(json.dumps({**origin_record(), "roots": len(roots), "views": len(roots["views"])}))
'''
    receipt = _run_child(tmp_path, program, artifact)
    assert receipt["roots"] == len(roots) and receipt["views"] == 6
