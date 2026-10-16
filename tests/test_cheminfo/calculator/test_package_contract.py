"""Public contracts for calculator electronic-state services."""


def test_electronic_state_package_exposes_typed_services() -> None:
    from hotpot.cheminfo.calculator import electronic_state
    from hotpot.cheminfo.calculator.electronic_state import (
        ElectronicStateCandidate,
        ElectronicStateCandidateEvidence,
        ElectronicStateInferenceResult,
        NeutralAtomReference,
        SpinContributionCandidate,
        SpinContributionScope,
        infer_electronic_states,
        neutral_atom_reference,
    )

    assert electronic_state.__all__ == [
        "infer_electronic_states",
        "neutral_atom_reference",
        "ElectronicStateCandidateEvidence",
        "SpinContributionScope",
        "SpinContributionCandidate",
        "ElectronicStateCandidate",
        "ElectronicStateInferenceResult",
        "NeutralAtomReference",
    ]
    assert callable(infer_electronic_states)
    assert callable(neutral_atom_reference)
    assert ElectronicStateCandidateEvidence.__dataclass_params__.frozen
    assert SpinContributionScope.MOLECULAR.value == "molecular"
    assert SpinContributionCandidate.__dataclass_params__.frozen
    assert ElectronicStateCandidate.__dataclass_params__.frozen
    assert ElectronicStateInferenceResult.__dataclass_params__.frozen
    assert NeutralAtomReference.__dataclass_params__.frozen


def test_calculator_package_exposes_scientific_state_resolver() -> None:
    from hotpot.cheminfo.calculator import (
        MolecularFacts,
        BackendProjectionResult,
        BackendStateRequest,
        ScientificOperationRequirements,
        ScientificStateBundle,
        ScientificStateWorkflowResult,
        StatePolicy,
        StateRequest,
        extract_molecular_facts,
        project_state,
        resolve_and_project_state,
        resolve_scientific_state,
    )

    assert callable(extract_molecular_facts)
    assert callable(resolve_scientific_state)
    assert callable(project_state)
    assert callable(resolve_and_project_state)
    assert MolecularFacts.__dataclass_params__.frozen
    assert StateRequest.__dataclass_params__.frozen
    assert ScientificStateBundle.__dataclass_params__.frozen
    assert StatePolicy.STRICT.value == "strict"
    assert ScientificOperationRequirements.__dataclass_params__.frozen
    assert BackendStateRequest.__dataclass_params__.frozen
    assert BackendProjectionResult.__dataclass_params__.frozen
    assert ScientificStateWorkflowResult.__dataclass_params__.frozen

