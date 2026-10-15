from pathlib import Path

from hotpot.ds.gates import BenchmarkGate, EvidenceGate, QualityGate
from hotpot.ds.gates._checker import CHECKER_PROTOCOL_DIGEST
from hotpot.ds.tools.migrate_legacy_store import load_bootstraps


ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = (
    ROOT
    / "hotpot"
    / "dataset"
    / "manifests"
    / "stage13-migration-bootstrap.json"
)


def test_owner_migration_bootstrap_is_explicit_and_current() -> None:
    datasets = load_bootstraps(BOOTSTRAP)

    assert set(datasets) == {"csd-electronic-state", "oxidation", "spin"}
    for dataset in datasets.values():
        assert tuple(type(gate) for gate in dataset.gates) == (
            EvidenceGate,
            QualityGate,
            BenchmarkGate,
        )
        assert dataset.identity_schema == {"components": []}
        assert dataset.include_incomplete is True
        assert dataset.owner == "Hotpot dataset owner"
        assert dataset.decision
        assert all(
            gate.info["checker_protocol_digest"] == CHECKER_PROTOCOL_DIGEST
            for gate in dataset.gates
        )

    assert datasets["csd-electronic-state"].core_labels == ()
    assert datasets["csd-electronic-state"].required_structure_roles == ()
    assert datasets["oxidation"].core_labels == ("oxidation_state",)
    assert datasets["oxidation"].required_structure_roles == ("*",)
    assert datasets["spin"].core_labels == ("spin_multiplicity",)
    assert datasets["spin"].required_structure_roles == ("*",)


def test_owner_migration_bootstrap_closes_each_observation_vocabulary() -> None:
    datasets = load_bootstraps(BOOTSTRAP)

    assert set(datasets["oxidation"].property_definitions) == {
        "oxidation_state"
    }
    assert set(datasets["spin"].property_definitions) == {
        "spin_multiplicity",
        "spin_energy_gap",
        "spin_transition_temperature",
    }
    assert set(datasets["csd-electronic-state"].property_definitions) == {
        "oxidation_state",
        "spin_category",
        "spin_state_text",
        "spin_transition_behavior",
    }
    for dataset in datasets.values():
        assert set(dataset.property_comparators) == set(
            dataset.property_definitions
        )
