from __future__ import annotations

import builtins
import hashlib
import importlib.util
import sys
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "plan/17_rdkit_wrapper/run_r0_benchmark.py"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "plan17_r0_benchmark", RUNNER_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return _load_runner()


def test_corpus_requires_exact_bytes_count_and_uses_physical_line_numbers(
    runner, tmp_path
):
    corpus = tmp_path / "sample.smi"
    corpus.write_bytes(b"CC first\n\n# retained in byte identity\nO second\n")
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()

    cases = runner._load_corpus(
        corpus, expected_count=2, expected_sha256=digest
    )

    assert cases == (
        {"case_index": 1, "smiles": "CC"},
        {"case_index": 4, "smiles": "O"},
    )
    with pytest.raises(ValueError, match="SHA-256 differs"):
        runner._load_corpus(
            corpus, expected_count=2, expected_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="expected 3 input ligands"):
        runner._load_corpus(
            corpus, expected_count=3, expected_sha256=digest
        )


def test_formal_cli_requires_input_and_keeps_the_frozen_defaults(runner):
    parser = runner._argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "/tmp/result"])

    options = parser.parse_args(
        ["--input", "corpus.smi", "--output-dir", "/tmp/result"]
    )
    assert options.counts == (1, 8, 32)
    assert runner.CANONICAL_INPUT_COUNT == 187
    assert (
        runner.CANONICAL_INPUT_SHA256
        == "e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7"
    )


def test_public_hotpot_parser_and_explicit_etkdg_options(runner, monkeypatch):
    import hotpot
    from hotpot.cheminfo import rdWrappers

    molecule = object()
    captured = {}

    def read_mol(value, *, fmt):
        captured["parse"] = (value, fmt)
        return molecule

    def generate(received, *, options):
        captured["generate"] = (received, options)
        return "batch"

    monkeypatch.setattr(hotpot, "read_mol", read_mol)
    monkeypatch.setattr(rdWrappers, "generate_conformers", generate)

    assert runner._parse_smiles("CC") is molecule
    assert runner._generate_conformers(
        molecule, conformer_count=8, seed=20260922
    ) == "batch"
    assert captured["parse"] == ("CC", "smi")
    assert captured["generate"] == (
        molecule,
        rdWrappers.ETKDGOptions(
            conformer_count=8,
            seed=20260922,
            prune_rms_threshold=None,
            use_random_coordinates=False,
        ),
    )


def test_detached_observer_reports_fixed_row_not_symmetry_aware_uniqueness(
    runner,
):
    first = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    duplicate = first @ rotation + np.array([5.0, -2.0, 1.0])
    distinct = first.copy()
    distinct[1, 0] = 2.0
    coordinates = np.stack((first, duplicate, distinct))
    before = coordinates.copy()

    observed = runner._detached_coordinate_observation(
        coordinates, atom_count=3
    )

    np.testing.assert_array_equal(coordinates, before)
    assert observed["finite"] is True
    assert observed["atom_row_shape_exact"] is True
    assert observed["unique_conformer_count"] == 2
    assert observed["pairwise_rmsd_min_angstrom"] == pytest.approx(0.0)
    assert observed["pairwise_rmsd_median_angstrom"] is not None
    assert len(observed["conformer_coordinate_sha256"]) == 3
    assert len(set(observed["conformer_coordinate_sha256"])) == 3
    assert observed["uniqueness_method"].endswith(
        "not_chemically_symmetry_aware"
    )


def test_summary_reports_stage_statistics_status_and_throughput(runner):
    records = (
        {
            "case_index": 1,
            "requested_conformer_count": 1,
            "status": "SUCCESS",
            "returned_conformer_count": 1,
            "parse_seconds": 1.0,
            "generate_seconds": 2.0,
            "determinism_rerun_seconds": 2.1,
            "observation_seconds": 0.5,
            "total_seconds": 5.6,
        },
        {
            "case_index": 2,
            "requested_conformer_count": 1,
            "status": "FAILED",
            "returned_conformer_count": 0,
            "parse_seconds": 3.0,
            "generate_seconds": 2.0,
            "determinism_rerun_seconds": 2.2,
            "observation_seconds": 1.5,
            "total_seconds": 8.7,
        },
    )

    summary = runner._summarize(
        records,
        corpus={"ligand_count": 2},
        runtime={"native": {"rdkit_version": "2026.09.1"}},
    )

    assert summary["status_counts"] == {"FAILED": 1, "SUCCESS": 1}
    assert summary["timings"]["parse"]["p50_seconds"] == 2.0
    assert summary["timings"]["parse"]["p95_seconds"] == pytest.approx(2.9)
    assert summary["timings"]["parse"]["max_seconds"] == 3.0
    assert summary["timings"]["parse"]["aggregate_seconds"] == 4.0
    assert summary["throughput"]["ligands_per_second"] == 0.5
    assert summary["throughput"]["conformers_per_second"] == 0.25
    assert summary["verification_counts"] == {
        "full_return_count": 0,
        "nonfinite_count": 0,
        "atom_row_shape_failure_count": 0,
        "atom_order_parity_failure_count": 0,
        "topology_parity_failure_count": 0,
        "level1_parity_failure_count": 0,
        "input_nonmutation_failure_count": 0,
        "same_seed_mismatch_count": 0,
    }


class _Status(IntEnum):
    SUCCESS = 0


def test_route_calls_only_parse_native_generation_and_detached_observation(
    runner, monkeypatch
):
    calls = []
    molecule = object()
    snapshot = {
        "atom_count": 2,
        "bond_count": 1,
        "atom_order_sha256": "atoms",
        "formal_charge_sha256": "charges",
        "aromaticity_sha256": "aromatic",
        "topology_sha256": "topology",
        "level1_sha256": "level1",
        "coordinates_sha256": "coordinates",
        "conformer_count": 0,
    }
    coordinates = np.zeros((1, 2, 3), dtype=np.float64)
    batch = SimpleNamespace(
        status=_Status.SUCCESS,
        diagnostic=None,
        records=(SimpleNamespace(ordinal=0),),
        engine="RDKit ETKDGv3",
        engine_version="2026.09.1",
        coordinates=coordinates,
    )

    def parse(smiles):
        calls.append(("parse", smiles))
        return molecule

    def take_snapshot(received):
        calls.append(("snapshot", received))
        return dict(snapshot)

    def generate(received, *, conformer_count, seed):
        calls.append(("generate", received, conformer_count, seed))
        return batch

    def observe(received, *, atom_count):
        calls.append(("observe", atom_count))
        return runner._detached_coordinate_observation(
            received, atom_count=atom_count
        )

    original_import = builtins.__import__

    def forbid_route_imports(name, *args, **kwargs):
        if any(
            forbidden in name
            for forbidden in (
                "obWrappers",
                "forcefields",
                "cheminfo.geometry",
            )
        ):
            pytest.fail(f"R0 invoked forbidden route dependency {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_route_imports)
    record = runner._run_trial(
        {"case_index": 7, "smiles": "CC"},
        1,
        parse_molecule=parse,
        generate=generate,
        snapshot_molecule=take_snapshot,
        observe=observe,
    )

    assert [call[0] for call in calls] == [
        "parse",
        "snapshot",
        "generate",
        "generate",
        "snapshot",
        "observe",
    ]
    assert record["same_seed_deterministic"] is True
    assert record["input_nonmutating"] is True
    assert record["seed"] == 20260928
    assert record["returned_conformer_count"] == 1


def test_explicit_development_extension_loader(runner, tmp_path):
    extension = tmp_path / "_rd_native.py"
    extension.write_text("MARKER = 'development-build'\n", encoding="utf-8")
    module_name = "hotpot.cheminfo.rdWrappers._rd_native"
    previous = sys.modules.get(module_name)
    try:
        loaded = runner._load_development_extension(extension)
        assert loaded.MARKER == "development-build"
        assert sys.modules[module_name] is loaded
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
