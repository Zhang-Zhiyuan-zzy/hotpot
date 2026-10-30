from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot.cheminfo import rdWrappers
from hotpot.cheminfo._native.buffers import (
    NativeMoleculeField,
    pack_molecule,
)
from hotpot.cheminfo.core import BondKind, Molecule


def _load_test_extension():
    path_value = os.environ.get("HOTPOT_RDWRAPPERS_TEST_EXTENSION")
    if path_value is None:
        from hotpot.cheminfo.rdWrappers import _rd_native

        return _rd_native
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


def _load_benchmark_runner():
    path = (
        Path(__file__).parents[3]
        / "plan"
        / "17_rdkit_wrapper"
        / "run_c1_c2_benchmark.py"
    )
    specification = importlib.util.spec_from_file_location(
        "plan17_run_c1_c2_benchmark", path
    )
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot load benchmark runner: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _add_dative(molecule: Molecule, donor: int, metal: int) -> None:
    molecule.add_bond(
        donor,
        metal,
        bond_order=1.0,
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
    )


def _monodentate_complex():
    molecule = Molecule()
    for atomic_number in (7, 6, 30):
        molecule.create_atom(atomic_number=atomic_number)
    molecule.add_bond(0, 1, bond_order=1.0)
    _add_dative(molecule, 0, 2)
    return molecule, ((0, 2),), ()


def _bidentate_complex():
    molecule = Molecule()
    for atomic_number in (7, 6, 6, 7, 30):
        molecule.create_atom(atomic_number=atomic_number)
    for first, second in ((0, 1), (1, 2), (2, 3)):
        molecule.add_bond(first, second, bond_order=1.0)
    _add_dative(molecule, 0, 4)
    _add_dative(molecule, 3, 4)
    return molecule, ((0, 4), (3, 4)), ((0, 3),)


def _multiple_ligand_complex():
    molecule = Molecule()
    for atomic_number in (7, 6, 8, 6, 30):
        molecule.create_atom(atomic_number=atomic_number)
    molecule.add_bond(0, 1, bond_order=1.0)
    molecule.add_bond(2, 3, bond_order=1.0)
    _add_dative(molecule, 0, 4)
    _add_dative(molecule, 2, 4)
    return molecule, ((0, 4), (2, 4)), ((0, 2),)


def _constraints(coordination_pairs, donor_pairs=()):
    coordination_pairs = np.asarray(coordination_pairs, dtype=np.int32)
    donor_pairs = np.asarray(donor_pairs, dtype=np.int32).reshape((-1, 2))
    return rdWrappers.CoordinationEmbeddingConstraints(
        coordination_pairs=coordination_pairs,
        metal_donor_bounds=np.tile(
            np.array([[1.90, 2.10]], dtype=np.float64),
            (len(coordination_pairs), 1),
        ),
        donor_pair_indices=donor_pairs if len(donor_pairs) else None,
        donor_pair_bounds=np.tile(
            np.array([[2.75, 3.25]], dtype=np.float64),
            (len(donor_pairs), 1),
        )
        if len(donor_pairs)
        else None,
    )


def _distance_rows(coordinates, pairs):
    return np.asarray(
        [
            [np.linalg.norm(frame[first] - frame[second]) for first, second in pairs]
            for frame in coordinates
        ]
    )


def test_coordination_pairs_are_directed_donor_to_metal_dative_edges():
    _load_test_extension()
    molecule, pairs, _ = _monodentate_complex()
    packed = pack_molecule(
        molecule,
        frozenset(
            {
                NativeMoleculeField.BOND_INDICES,
                NativeMoleculeField.BOND_KINDS,
            }
        ),
    )
    np.testing.assert_array_equal(packed.bond_indices[-1], [0, 2])

    reversed_pair = _constraints(((2, 0),))
    result = rdWrappers.generate_conformers(
        molecule,
        options=rdWrappers.ETKDGOptions(seed=19),
        constraints=reversed_pair,
    )

    assert result.status is rdWrappers.GenerationStatus.FAILED
    assert result.coordinates.shape == (0, 3, 3)
    assert "donor-to-metal dative bond" in result.diagnostic


@pytest.mark.parametrize(
    "factory",
    (_monodentate_complex, _bidentate_complex, _multiple_ligand_complex),
)
def test_coordination_bounds_cover_mono_bidentate_and_multiple_ligands(factory):
    _load_test_extension()
    molecule, coordination_pairs, donor_pairs = factory()
    constraints = _constraints(coordination_pairs, donor_pairs)
    options = rdWrappers.ETKDGOptions(conformer_count=8, seed=47)

    first = rdWrappers.generate_conformers(
        molecule, options=options, constraints=constraints
    )
    second = rdWrappers.generate_conformers(
        molecule, options=options, constraints=constraints
    )

    assert first.status is rdWrappers.GenerationStatus.SUCCESS
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    metal_distances = _distance_rows(first.coordinates, coordination_pairs)
    assert np.max(np.maximum(1.90 - metal_distances, 0.0)) <= 0.05
    assert np.max(np.maximum(metal_distances - 2.10, 0.0)) <= 0.05
    if donor_pairs:
        donor_distances = _distance_rows(first.coordinates, donor_pairs)
        assert np.max(np.maximum(2.75 - donor_distances, 0.0)) <= 0.05
        assert np.max(np.maximum(donor_distances - 3.25, 0.0)) <= 0.05


@pytest.mark.parametrize(
    "factory",
    (_monodentate_complex, _bidentate_complex, _multiple_ligand_complex),
)
def test_stock_whole_complex_route_embeds_the_same_complete_graph(factory):
    _load_test_extension()
    molecule, _, _ = factory()

    result = rdWrappers.generate_conformers(
        molecule,
        options=rdWrappers.ETKDGOptions(conformer_count=4, seed=43),
    )

    assert result.status is rdWrappers.GenerationStatus.SUCCESS
    assert result.coordinates.shape == (4, len(molecule.atoms), 3)
    assert np.isfinite(result.coordinates).all()


def test_triangle_inconsistent_coordination_bounds_fail_before_embedding():
    _load_test_extension()
    molecule = Molecule()
    for atomic_number in (7, 8, 30):
        molecule.create_atom(atomic_number=atomic_number)
    _add_dative(molecule, 0, 2)
    _add_dative(molecule, 1, 2)
    constraints = rdWrappers.CoordinationEmbeddingConstraints(
        coordination_pairs=np.array([[0, 2], [1, 2]], dtype=np.int32),
        metal_donor_bounds=np.array([[1.00, 1.10], [1.00, 1.10]], dtype=np.float64),
        donor_pair_indices=np.array([[0, 1]], dtype=np.int32),
        donor_pair_bounds=np.array([[3.00, 3.10]], dtype=np.float64),
    )

    result = rdWrappers.generate_conformers(
        molecule,
        options=rdWrappers.ETKDGOptions(seed=29),
        constraints=constraints,
    )

    assert result.status is rdWrappers.GenerationStatus.FAILED
    assert result.coordinates.shape == (0, 3, 3)
    assert "failed triangle smoothing" in result.diagnostic


def test_donor_pair_bounds_are_symmetric_in_the_distance_matrix():
    _load_test_extension()
    molecule, coordination_pairs, donor_pairs = _bidentate_complex()
    forward = _constraints(coordination_pairs, donor_pairs)
    reverse = _constraints(coordination_pairs, ((3, 0),))
    options = rdWrappers.ETKDGOptions(conformer_count=4, seed=53)

    forward_result = rdWrappers.generate_conformers(
        molecule, options=options, constraints=forward
    )
    reverse_result = rdWrappers.generate_conformers(
        molecule, options=options, constraints=reverse
    )

    assert forward_result.status is rdWrappers.GenerationStatus.SUCCESS
    assert reverse_result.status is rdWrappers.GenerationStatus.SUCCESS
    np.testing.assert_array_equal(
        forward_result.coordinates, reverse_result.coordinates
    )


def test_coordination_route_does_not_import_python_forcefield_backends(monkeypatch):
    _load_test_extension()
    molecule, coordination_pairs, donor_pairs = _bidentate_complex()
    constraints = _constraints(coordination_pairs, donor_pairs)

    class ForbiddenModule:
        def __getattr__(self, name):
            raise AssertionError(f"force-field backend was accessed: {name}")

    monkeypatch.setitem(sys.modules, "openbabel", ForbiddenModule())
    monkeypatch.setitem(sys.modules, "rdkit", ForbiddenModule())

    result = rdWrappers.generate_conformers(
        molecule,
        options=rdWrappers.ETKDGOptions(seed=31),
        constraints=constraints,
    )

    assert result.status is rdWrappers.GenerationStatus.SUCCESS


def test_native_sampler_has_no_business_forcefield_entrypoint_calls():
    sampler_source = (
        Path(__file__).parents[3]
        / "hotpot"
        / "cheminfo"
        / "rdWrappers"
        / "_native"
        / "src"
        / "sampler.cpp"
    ).read_text(encoding="utf-8")

    assert "UFFGetMoleculeForceField" not in sampler_source
    assert "MMFFGetMoleculeForceField" not in sampler_source
    assert "UFFOptimizeMolecule" not in sampler_source
    assert "MMFFOptimizeMolecule" not in sampler_source


def test_unknown_bond_state_is_reported_as_unsupported():
    _load_test_extension()
    molecule = Molecule()
    molecule.create_atom(atomic_number=7, implicit_hydrogens=0)
    molecule.create_atom(atomic_number=30, implicit_hydrogens=0)
    molecule.add_bond(0, 1, bond_order=1.0, bond_kind=BondKind.UNKNOWN)

    result = rdWrappers.generate_conformers(
        molecule,
        options=rdWrappers.ETKDGOptions(seed=37),
    )

    assert result.status is rdWrappers.GenerationStatus.UNSUPPORTED
    assert "UNKNOWN bond" in result.diagnostic


def test_unknown_bond_without_hydrogen_declarations_rejects_strict_projection():
    _load_test_extension()
    molecule = Molecule()
    molecule.create_atom(atomic_number=7)
    molecule.create_atom(atomic_number=30)
    molecule.add_bond(0, 1, bond_order=1.0, bond_kind=BondKind.UNKNOWN)

    with pytest.raises(ValueError, match="requires resolved implicit-H input"):
        rdWrappers.generate_conformers(molecule, options=rdWrappers.ETKDGOptions(seed=37))


def test_c1_c2_loader_preserves_supplied_representation_without_perception(monkeypatch):
    from hotpot.cheminfo.operations import chemistry

    def no_perception(*args, **kwargs):
        raise AssertionError("factual reconstruction must not run chemical preparation")

    monkeypatch.setattr(chemistry, "_prepare_hydrogens", no_perception)
    monkeypatch.setattr(chemistry, "_prepare_aromatic_representation", no_perception)
    runner = _load_benchmark_runner()
    molecule = runner._build_molecule({
        "atoms": [
            {"atomic_number": 7, "formal_charge": 0, "atom_id": 17,
             "is_aromatic": True, "implicit_hydrogens": 1},
            {"atomic_number": 30, "formal_charge": 0, "atom_id": 17,
             "is_aromatic": False, "implicit_hydrogens": 0},
        ],
        "bonds": [{"begin": 0, "end": 1, "bond_order": 1.0, "bond_kind": "dative",
                   "bond_direction": "atom1_to_atom2", "is_aromatic": False}],
    })
    record, _ = molecule._capture_factual_record()
    values = np.frombuffer(record.atom_values, dtype="<f8").reshape(2, 19)
    assert values[:, 15].tolist() == [17, 17]
    assert values[:, 14].tolist() == [1, 0]
    assert values[:, 9].tolist() == [1, 0]
    assert record.bond_kinds == ("dative",)
    assert record.bond_directions == ("atom1_to_atom2",)
    assert record.bond_aromatic == (False,)


def _write_smoke_coordination_artifacts(root: Path):
    manifest = root / "manifest.json"
    structures = root / "structures"
    structures.mkdir()
    manifest.write_text(
        json.dumps({"schema_version": 1, "cases": [{"index": 1, "seed": 73}]}),
        encoding="utf-8",
    )
    structure = structures / "case-0001.json"
    structure.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_index": 1,
                "atoms": [
                    {"atomic_number": 7},
                    {"atomic_number": 6},
                    {"atomic_number": 30},
                ],
                "bonds": [
                    {
                        "begin": 0,
                        "end": 1,
                        "bond_order": 1.0,
                        "bond_kind": "single",
                    },
                    {
                        "begin": 0,
                        "end": 2,
                        "bond_order": 1.0,
                        "bond_kind": "dative",
                        "bond_direction": "atom1_to_atom2",
                    },
                ],
                "constraints": {
                    "policy": {
                        "name": "development-explicit-bounds",
                        "distance_unit": "angstrom",
                    },
                    "coordination_pairs": [[0, 2]],
                    "metal_donor_bounds_angstrom": [[1.9, 2.1]],
                    "donor_pair_indices": [],
                    "donor_pair_bounds_angstrom": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, structures


def test_c1_c2_runner_labels_smoke_and_keeps_routes_coordinate_only(tmp_path):
    runner = _load_benchmark_runner()
    manifest, structures = _write_smoke_coordination_artifacts(tmp_path)
    calls = []

    def fake_generate(molecule, *, conformer_count, seed, constraints):
        calls.append((conformer_count, seed, constraints))
        frame = np.array(
            [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        coordinates = np.repeat(frame[None, :, :], conformer_count, axis=0)
        return SimpleNamespace(
            status=rdWrappers.GenerationStatus.SUCCESS,
            diagnostic=None,
            records=(),
            coordinates=coordinates,
        )

    report = runner.run(
        manifest,
        structures,
        smoke=True,
        generate=fake_generate,
        runtime_identity=lambda: {
            "hotpot_version": "test",
            "rdkit_version": "test",
            "rdkit_commit": "test",
            "native": True,
        },
    )

    assert report["evidence_class"] == "development-smoke"
    assert report["formal_claim_allowed"] is False
    assert report["case_count"] == 1
    assert len(report["records"]) == 6
    assert set(report["summary"]) == {
        "C1:1",
        "C1:8",
        "C1:32",
        "C2:1",
        "C2:8",
        "C2:32",
    }
    assert all(record["same_seed_deterministic"] for record in report["records"])
    assert all(record["input_graph_preserved"] for record in report["records"])
    assert all(call[1] == 73 for call in calls)
    assert sum(call[2] is None for call in calls) == 6
    assert sum(call[2] is not None for call in calls) == 6
    assert report["structure_inventory"][0]["sha256"]
    assert report["manifest_sha256"] != runner.FORMAL_MANIFEST_SHA256


def test_c1_c2_runner_refuses_to_label_unknown_manifest_as_formal(tmp_path):
    runner = _load_benchmark_runner()
    manifest, structures = _write_smoke_coordination_artifacts(tmp_path)

    with pytest.raises(ValueError, match="formal C1/C2 manifest SHA-256 mismatch"):
        runner.run(manifest, structures, smoke=False)


def test_bound_observation_separates_generation_from_strict_conformance():
    runner = _load_benchmark_runner()
    constraints = _constraints(((0, 2),))
    coordinates = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.20, 0.0, 0.0]]])

    observation = runner._bound_observation(coordinates, constraints)

    assert observation["strict_bound_satisfied_conformer_count"] == 0
    assert observation["all_conformers_strictly_bound_satisfied"] is False
    assert observation["max_bound_violation_angstrom"] == pytest.approx(0.10)
