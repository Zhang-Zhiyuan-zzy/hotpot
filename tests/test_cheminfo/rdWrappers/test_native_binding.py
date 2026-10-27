from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import hotpot
from hotpot.cheminfo import rdWrappers
from hotpot.cheminfo._native.buffers import (
    NativeAtomTransportBuffers,
    NativeMoleculeBuffers,
    NativeMoleculeField,
    NativeUtf8Table,
)
from hotpot.cheminfo.conformers.adapters import (
    _materialized_hydrogen_projection,
)
from hotpot.cheminfo.rdWrappers import native


def _load_test_extension():
    path_value = os.environ.get("HOTPOT_RDWRAPPERS_TEST_EXTENSION")
    if path_value is None:
        pytest.skip("HOTPOT_RDWRAPPERS_TEST_EXTENSION is not set")
    path = Path(path_value)
    name = "hotpot.cheminfo.rdWrappers._rd_native"
    existing = sys.modules.get(name)
    if existing is not None and hasattr(existing, "C_API_VERSION"):
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot load native test extension: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_loader_packs_only_the_rdkit_field_set(monkeypatch):
    molecule = object()
    options = rdWrappers.ETKDGOptions(seed=7)
    sentinel = object()
    captured = {}

    def fake_pack(received_molecule, fields):
        captured["molecule"] = received_molecule
        captured["fields"] = fields
        return sentinel

    class Backend:
        @staticmethod
        def generate(buffers, received_options, constraints):
            captured["buffers"] = buffers
            captured["options"] = received_options
            captured["constraints"] = constraints
            return "result"

    monkeypatch.setattr(native, "pack_molecule", fake_pack)
    monkeypatch.setattr(native, "_backend", lambda: Backend)

    assert native.generate(molecule, options, None) == "result"
    assert captured == {
        "molecule": molecule,
        "fields": frozenset(
            {
                NativeMoleculeField.ATOMIC_NUMBERS,
                NativeMoleculeField.FORMAL_CHARGES,
                NativeMoleculeField.ATOM_AROMATIC,
                NativeMoleculeField.BOND_INDICES,
                NativeMoleculeField.BOND_KINDS,
                NativeMoleculeField.BOND_AROMATIC,
                NativeMoleculeField.STEREO_RECORDS,
                NativeMoleculeField.ATOM_TRANSPORT,
                NativeMoleculeField.BOND_TRANSPORT,
            }
        ),
        "buffers": sentinel,
        "options": options,
        "constraints": None,
    }


def test_compiled_runtime_identity_and_exports():
    extension = _load_test_extension()

    assert extension.C_API_VERSION == 1
    assert extension.RDKIT_COMMIT == "fece8caa860bdf6c9c82bdb9253f22a0cf87150c"
    assert extension.runtime_info() == rdWrappers.RuntimeInfo(
        hotpot_version="0.5.4.5",
        rdkit_version="2026.09.1",
        native=True,
    )


def test_compiled_generation_is_deterministic_and_nonmutating():
    _load_test_extension()
    molecule = hotpot.to_hotpot_mol("CCO")
    coordinates_before = molecule.coordinates.copy()
    conformer_count_before = len(molecule.conformers)
    options = rdWrappers.ETKDGOptions(conformer_count=3, seed=123)

    first = rdWrappers.generate_conformers(molecule, options=options)
    second = rdWrappers.generate_conformers(molecule, options=options)

    assert first.status is rdWrappers.GenerationStatus.SUCCESS
    assert first.coordinates.shape == (3, 3, 3)
    assert first.coordinates.dtype == np.float64
    assert first.coordinates.flags.c_contiguous
    assert np.isfinite(first.coordinates).all()
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    assert (
        first.records
        == second.records
        == (
            rdWrappers.ConformerRecord(ordinal=0, engine_conformer_id=0),
            rdWrappers.ConformerRecord(ordinal=1, engine_conformer_id=1),
            rdWrappers.ConformerRecord(ordinal=2, engine_conformer_id=2),
        )
    )
    assert first.engine == "RDKit ETKDGv3"
    assert first.engine_version == "2026.09.1"
    assert first.diagnostic is None
    np.testing.assert_array_equal(molecule.coordinates, coordinates_before)
    assert len(molecule.conformers) == conformer_count_before


def test_compiled_public_path_accepts_kekule_ordered_aromatic_bonds():
    _load_test_extension()
    molecule = hotpot.read_mol("c1ccccc1", fmt="smi")

    assert {bond.bond_order for bond in molecule.bonds} == {1.0, 2.0}
    assert all(bond.is_aromatic for bond in molecule.bonds)

    result = rdWrappers.generate_conformers(
        molecule,
        options=rdWrappers.ETKDGOptions(seed=123),
    )

    assert result.status is rdWrappers.GenerationStatus.SUCCESS
    assert result.coordinates.shape == (1, 6, 3)


def test_compiled_public_path_preserves_payload_free_aromatic_nh():
    _load_test_extension()
    molecule = hotpot.read_mol("c1cc(-c2nnn[nH]2)nc(-c2nnn[nH]2)c1", fmt="smi")
    coordinates_before = molecule.coordinates.copy()
    options = rdWrappers.ETKDGOptions(conformer_count=2, seed=123)

    first = rdWrappers.generate_conformers(molecule, options=options)
    second = rdWrappers.generate_conformers(molecule, options=options)

    assert first.status is rdWrappers.GenerationStatus.SUCCESS
    assert first.coordinates.shape == (2, len(molecule.atoms), 3)
    assert np.isfinite(first.coordinates).all()
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    np.testing.assert_array_equal(molecule.coordinates, coordinates_before)


def test_compiled_path_accepts_forcefield_materialized_graph_hydrogens():
    _load_test_extension()
    molecule = hotpot.read_mol("CO", fmt="smi")
    molecule.add_hydrogens(rm_polar_hs=False, rng=np.random.default_rng(123))

    projected = _materialized_hydrogen_projection(molecule)
    result = rdWrappers.generate_conformers(
        projected,
        options=rdWrappers.ETKDGOptions(seed=123),
    )

    assert result.status is rdWrappers.GenerationStatus.SUCCESS
    assert result.coordinates.shape == (1, len(molecule.atoms), 3)
    assert np.isfinite(result.coordinates).all()
    assert tuple(atom.implicit_hydrogens for atom in molecule.atoms[:2]) == (3, 1)


def test_compiled_path_accepts_cohort_reconstructed_pyrrolic_hydrogens():
    from tests.benchmarks.coordination_complexes.validation_cohort import (
        _graph_document,
        canonical_digest,
        semantic_case_molecule,
    )
    from tests.benchmarks.coordination_complexes.validation_providers import (
        _worker_components,
    )

    _load_test_extension()
    molecule = hotpot.read_mol(
        "c1cc(-c2nnn[nH]2)nc(-c2nnn[nH]2)c1",
        fmt="smi",
    )
    molecule.add_hydrogens()
    molecule.force_remove_polar_hydrogens()
    metal = molecule.create_atom(atomic_number=95)
    molecule.add_bond(8, metal.idx)
    molecule.refresh_atom_id()
    graph = _graph_document(molecule)
    coordination = {
        "metal": {
            "row": metal.idx,
            "atom_id": metal.id,
            "atomic_number": 95,
            "symbol": "Am",
        },
        "donor_rows": [8],
    }
    case = {
        "status": "cbond",
        "graph": graph,
        "coordination": coordination,
        "semantic_graph_sha256": canonical_digest(
            {"graph": graph, "coordination": coordination}
        ),
    }

    rebuilt = semantic_case_molecule(case)
    (component,) = _worker_components(rebuilt, add_hydrogens=True, seed=123)
    projected = _materialized_hydrogen_projection(component)
    result = rdWrappers.generate_conformers(
        projected,
        options=rdWrappers.ETKDGOptions(seed=123),
    )

    assert [
        atom.implicit_hydrogens
        for atom in rebuilt.atoms
        if atom.atomic_number == 7
    ].count(1) == 2
    pyrrolic_nitrogen_hydrogens = [
        sum(neighbour.atomic_number == 1 for neighbour in atom.neighbours)
        for atom in component.atoms
        if atom.atomic_number == 7
    ]
    assert pyrrolic_nitrogen_hydrogens.count(1) == 2
    assert result.status is rdWrappers.GenerationStatus.SUCCESS
    assert result.coordinates.shape == (1, len(component.atoms), 3)


def test_canonical_packer_fills_current_hydrogens_for_payload_free_atoms():
    molecule = hotpot.read_mol("[13CH3]CO", fmt="smi")
    buffers = native.pack_molecule(molecule, native._RDKIT_ETKDG_FIELDS)

    transport = buffers.atom_transport

    assert transport is not None
    assert tuple(transport.atoms) == tuple(range(len(molecule.atoms)))
    assert tuple(transport.current_implicit_hydrogens) == tuple(
        atom.implicit_hydrogens for atom in molecule.atoms
    )
    assert transport.source_namespaces.present.tolist() == [0, 0, 0]


def test_compiled_binding_rejects_mismatched_constraint_rows():
    extension = _load_test_extension()
    buffers = NativeMoleculeBuffers(
        schema_version=1,
        atom_count=2,
        bond_count=1,
        atomic_numbers=np.array([6, 6], dtype=np.int32),
        formal_charges=np.zeros(2, dtype=np.int32),
        atom_aromatic=np.zeros(2, dtype=np.uint8),
        bond_indices=np.array([[0, 1]], dtype=np.int32),
        bond_orders=np.ones(1, dtype=np.float64),
        bond_kinds=np.ones(1, dtype=np.uint8),
        bond_aromatic=np.zeros(1, dtype=np.uint8),
    )
    constraints = rdWrappers.CoordinationEmbeddingConstraints(
        coordination_pairs=np.array([[0, 1]], dtype=np.int32),
        metal_donor_bounds=np.array([[1.0, 2.0], [1.1, 2.1]]),
    )

    with pytest.raises(ValueError, match="metal_donor_bounds"):
        extension.generate(buffers, rdWrappers.ETKDGOptions(), constraints)


def _minimal_native_buffers() -> NativeMoleculeBuffers:
    return NativeMoleculeBuffers(
        schema_version=1,
        atom_count=2,
        bond_count=1,
        atomic_numbers=np.array([6, 6], dtype=np.int32),
        formal_charges=np.zeros(2, dtype=np.int32),
        atom_aromatic=np.zeros(2, dtype=np.uint8),
        bond_indices=np.array([[0, 1]], dtype=np.int32),
        bond_orders=np.ones(1, dtype=np.float64),
        bond_kinds=np.ones(1, dtype=np.uint8),
        bond_aromatic=np.zeros(1, dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    (
        (
            "atomic_numbers",
            np.array([6, 6], dtype=np.int64),
            TypeError,
            "atomic_numbers has an invalid dtype",
        ),
        (
            "formal_charges",
            np.zeros((2, 1), dtype=np.int32),
            ValueError,
            "formal_charges has an invalid rank",
        ),
        (
            "atomic_numbers",
            np.arange(4, dtype=np.int32)[::2],
            ValueError,
            "atomic_numbers must be C-contiguous",
        ),
        (
            "atomic_numbers",
            np.array([6, 6], dtype=">i4"),
            TypeError,
            "atomic_numbers has an invalid dtype",
        ),
    ),
)
def test_compiled_binding_validates_every_numpy_array_boundary(
    field,
    value,
    error,
    message,
):
    from dataclasses import replace

    extension = _load_test_extension()
    buffers = replace(_minimal_native_buffers(), **{field: value})

    with pytest.raises(error, match=message):
        extension.generate(buffers, rdWrappers.ETKDGOptions(), None)


def test_compiled_binding_rejects_unaligned_numpy_storage():
    from dataclasses import replace

    extension = _load_test_extension()
    storage = np.zeros(2 * np.dtype(np.int32).itemsize + 1, dtype=np.uint8)
    atomic_numbers = np.ndarray(shape=(2,), dtype=np.int32, buffer=storage, offset=1)
    assert atomic_numbers.flags.c_contiguous
    assert not atomic_numbers.flags.aligned
    buffers = replace(_minimal_native_buffers(), atomic_numbers=atomic_numbers)

    with pytest.raises(ValueError, match="atomic_numbers must be aligned"):
        extension.generate(buffers, rdWrappers.ETKDGOptions(), None)


@pytest.mark.parametrize(
    ("offset", "length", "present", "message"),
    (
        (1, 0, 0, "data for an absent value"),
        (8, 1, 1, "structurally invalid"),
        (0, 0, 2, "structurally invalid"),
    ),
)
def test_compiled_binding_validates_utf8_presence_and_offsets(
    offset,
    length,
    present,
    message,
):
    from dataclasses import replace

    extension = _load_test_extension()
    strings = NativeUtf8Table(
        data=np.array([0], dtype=np.uint8),
        offsets=np.array([offset], dtype=np.uint64),
        lengths=np.array([length], dtype=np.uint64),
        present=np.array([present], dtype=np.uint8),
    )
    atom_transport = NativeAtomTransportBuffers(
        schema_version=1,
        atoms=np.array([0], dtype=np.uint64),
        presence_masks=np.zeros(1, dtype=np.uint32),
        isotopes=np.zeros(1, dtype=np.int32),
        radical_electrons=np.zeros(1, dtype=np.int32),
        has_current_hydrogens=np.zeros(1, dtype=np.uint32),
        current_hydrogen_presence_masks=np.zeros(1, dtype=np.uint32),
        current_implicit_hydrogens=np.zeros(1, dtype=np.int32),
        current_bracket_explicit_hydrogens=np.zeros(1, dtype=np.int32),
        current_no_implicit=np.zeros(1, dtype=np.uint32),
        has_source_hydrogens=np.zeros(1, dtype=np.uint32),
        source_hydrogen_presence_masks=np.zeros(1, dtype=np.uint32),
        source_implicit_hydrogens=np.zeros(1, dtype=np.int32),
        source_bracket_explicit_hydrogens=np.zeros(1, dtype=np.int32),
        source_no_implicit=np.zeros(1, dtype=np.uint32),
        source_namespaces=strings,
    )
    buffers = replace(_minimal_native_buffers(), atom_transport=atom_transport)

    with pytest.raises(ValueError, match=message):
        extension.generate(buffers, rdWrappers.ETKDGOptions(), None)
