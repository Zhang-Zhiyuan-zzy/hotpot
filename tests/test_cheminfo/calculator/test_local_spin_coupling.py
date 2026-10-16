"""Local-to-molecular spin coupling domain tests."""

import hotpot

from hotpot.cheminfo.calculator.charge import resolve_net_charge
from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    CandidateProvenance,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.electronic_state import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
    infer_electronic_states,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateDomainScope,
    CandidateProposal,
    CandidateSearchStatus,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
    compose_scientific_state_space,
)
from hotpot.cheminfo.calculator.state_composition.coupling import (
    _coupled_two_s_values,
)
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _contribution(facts, atom_index, two_s, suffix):
    return SpinContributionCandidate(
        CandidateProvenance(
            f"spin:{suffix}",
            StateValueAuthority.DOMAIN_RULE,
            "test",
        ),
        SpinContributionScope.ATOM,
        f"atom:{facts.atom_map[atom_index]}",
        (),
        two_s,
        True,
    )


def _factor(subject, *candidates):
    return ProposedCandidateFactor(
        FactorKey(ScientificQuantity.ELECTRONIC_STATE, subject),
        f"local-spin-domain:{subject}",
        tuple(ProposedCandidateValue(candidate) for candidate in candidates),
        CandidateDomainScope.DOMAIN_BOUNDED,
        MaterializationStatus.COMPLETE,
        None,
    )


def _space(facts, charge, factors, budget=CandidateBudget(16, 128, 32)):
    net_charge = resolve_net_charge(
        facts,
        StateRequest(net_charge=charge),
    )[0]
    return compose_scientific_state_space(
        facts,
        net_charge_results=(net_charge,),
        candidate_proposals=(
            CandidateProposal("local-spin:test", factors, ()),
        ),
        candidate_budget=budget,
    )


def test_spin_only_triangle_domain_is_complete_and_deduplicated() -> None:
    assert _coupled_two_s_values((1, 1)) == (0, 2)
    assert _coupled_two_s_values((2, 1)) == (1, 3)
    assert _coupled_two_s_values((1, 1, 1)) == (1, 3)


def test_two_local_doublets_form_singlet_and_triplet_paths() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[H].[H]", "smi"))
    first = _contribution(facts, 0, 1, "first-doublet")
    second = _contribution(facts, 1, 1, "second-doublet")

    space = _space(
        facts,
        0,
        (
            _factor(first.subject_identity, first),
            _factor(second.subject_identity, second),
        ),
    )

    molecular_factor = next(
        factor
        for factor in space.factors
        if factor.factor_key.quantity is ScientificQuantity.ELECTRONIC_STATE
        and factor.factor_key.subject_identity.startswith("molecule:")
    )
    assert {
        value.candidate.two_s for value in molecular_factor.values
    } == {0, 2}
    assert len(space.expanded_paths) == 2
    assert all(
        path.electronic_state_candidate_identity is not None
        for path in space.expanded_paths
    )
    assert space.search_status is CandidateSearchStatus.NO_CERTIFIED_DECISION


def test_closed_shell_component_preserves_one_open_center_spin() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Eu].N", "smi"))
    metal = _contribution(facts, 0, 6, "eu-open-shell")
    ligand_component = next(
        component
        for component in facts.covalent_components
        if 0 not in component.atom_indices
    )
    ligand = SpinContributionCandidate(
        CandidateProvenance(
            "spin:closed-ligand",
            StateValueAuthority.DOMAIN_RULE,
            "test",
        ),
        SpinContributionScope.COMPONENT,
        ligand_component.identity,
        (),
        0,
        True,
    )

    space = _space(
        facts,
        1,
        (
            _factor(metal.subject_identity, metal),
            _factor(ligand.subject_identity, ligand),
        ),
    )

    molecular_factor = next(
        factor
        for factor in space.factors
        if factor.factor_key.quantity is ScientificQuantity.ELECTRONIC_STATE
        and factor.factor_key.subject_identity.startswith("molecule:")
    )
    assert tuple(
        value.candidate.two_s for value in molecular_factor.values
    ) == (6,)
    assert len(space.expanded_paths) == 1


def test_coupling_constraints_preserve_each_local_parent_combination() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Li].[He]", "smi"))
    closed = _contribution(facts, 0, 0, "closed")
    triplet = _contribution(facts, 0, 2, "triplet")
    doublet = _contribution(facts, 1, 1, "doublet")

    space = _space(
        facts,
        0,
        (
            _factor(closed.subject_identity, closed, triplet),
            _factor(doublet.subject_identity, doublet),
        ),
    )

    molecular_candidates = {
        value.candidate.provenance.identity: value.candidate
        for factor in space.factors
        if factor.factor_key.quantity is ScientificQuantity.ELECTRONIC_STATE
        and factor.factor_key.subject_identity.startswith("molecule:")
        for value in factor.values
    }
    assert len(molecular_candidates) == 3
    assert len(space.expanded_paths) == 3
    for path in space.expanded_paths:
        coupled = next(
            candidate
            for identity, candidate in molecular_candidates.items()
            if identity in path.evidence_ids
        )
        assert set(coupled.parent_candidate_ids).issubset(path.evidence_ids)


def test_parent_constraints_are_forward_pruned_before_path_budget() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[Li].[He]", "smi"))
    closed = _contribution(facts, 0, 0, "budget-closed")
    triplet = _contribution(facts, 0, 2, "budget-triplet")
    doublet = _contribution(facts, 1, 1, "budget-doublet")

    space = _space(
        facts,
        0,
        (
            _factor(closed.subject_identity, closed, triplet),
            _factor(doublet.subject_identity, doublet),
        ),
        CandidateBudget(16, 1, 32),
    )

    assert len(space.expanded_paths) == 1
    assert space.frontier


def test_explicit_molecular_state_is_not_displaced_by_derived_coupling() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[H].[H]", "smi"))
    first = _contribution(facts, 0, 1, "explicit-first")
    second = _contribution(facts, 1, 1, "explicit-second")
    net_charge = resolve_net_charge(
        facts,
        StateRequest(net_charge=0),
    )[0]
    electronic = infer_electronic_states(
        facts,
        StateRequest(net_charge=0, two_s=0),
        net_charge_results=(net_charge,),
    )

    space = compose_scientific_state_space(
        facts,
        net_charge_results=(net_charge,),
        electronic_state_results=electronic,
        candidate_proposals=(
            CandidateProposal(
                "local-spin:explicit-test",
                (
                    _factor(first.subject_identity, first),
                    _factor(second.subject_identity, second),
                ),
                (),
            ),
        ),
        candidate_budget=CandidateBudget(2, 32, 16),
    )

    molecular_factor = next(
        factor
        for factor in space.factors
        if factor.factor_key.quantity is ScientificQuantity.ELECTRONIC_STATE
        and factor.factor_key.subject_identity.startswith("molecule:")
    )
    assert len(molecular_factor.values) == 1
    assert isinstance(
        molecular_factor.values[0].candidate,
        ElectronicStateCandidate,
    )
    assert molecular_factor.values[0].candidate.provenance.authority is (
        StateValueAuthority.VERIFIED_EXPLICIT
    )
    assert len(space.expanded_paths) == 1


def test_non_pure_local_branch_preserves_unresolved_molecular_state() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[H].[H]", "smi"))
    first = _contribution(facts, 0, 1, "non-pure-first")
    pure_second = _contribution(facts, 1, 1, "pure-second")
    non_pure_second = SpinContributionCandidate(
        CandidateProvenance(
            "spin:non-pure-second",
            StateValueAuthority.DOMAIN_RULE,
            "test",
        ),
        SpinContributionScope.ATOM,
        pure_second.subject_identity,
        (),
        1,
        False,
    )
    net_charge = resolve_net_charge(
        facts,
        StateRequest(net_charge=0),
    )[0]
    unresolved = infer_electronic_states(
        facts,
        StateRequest(),
        net_charge_results=(net_charge,),
    )

    space = compose_scientific_state_space(
        facts,
        net_charge_results=(net_charge,),
        electronic_state_results=unresolved,
        candidate_proposals=(
            CandidateProposal(
                "local-spin:non-pure-test",
                (
                    _factor(first.subject_identity, first),
                    _factor(
                        pure_second.subject_identity,
                        pure_second,
                        non_pure_second,
                    ),
                ),
                (),
            ),
        ),
    )

    assert not any(
        factor.factor_key.quantity is ScientificQuantity.ELECTRONIC_STATE
        and factor.factor_key.subject_identity.startswith("molecule:")
        for factor in space.factors
    )
    assert len(space.expanded_paths) == 2
    assert all(
        path.electronic_state_candidate_identity is None
        for path in space.expanded_paths
    )


def test_coupling_domain_reports_materialization_truncation() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[C].[C]", "smi"))
    first = _contribution(facts, 0, 6, "first-septet")
    second = _contribution(facts, 1, 6, "second-septet")

    space = _space(
        facts,
        0,
        (
            _factor(first.subject_identity, first),
            _factor(second.subject_identity, second),
        ),
        CandidateBudget(2, 128, 32),
    )

    molecular_factor = next(
        factor
        for factor in space.factors
        if factor.factor_key.quantity is ScientificQuantity.ELECTRONIC_STATE
        and factor.factor_key.subject_identity.startswith("molecule:")
    )
    assert len(molecular_factor.values) == 2
    assert (
        molecular_factor.materialization_status
        is MaterializationStatus.RESOURCE_TRUNCATED
    )
    assert molecular_factor.unexpanded_domain_summary is not None
    assert space.search_status is (
        CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
    )


def test_overlapping_atom_and_component_spin_are_not_double_counted() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("CCO", "smi"))
    atom = _contribution(facts, 0, 1, "overlapping-atom")
    component = SpinContributionCandidate(
        CandidateProvenance(
            "spin:overlapping-component",
            StateValueAuthority.DOMAIN_RULE,
            "test",
        ),
        SpinContributionScope.COMPONENT,
        facts.covalent_components[0].identity,
        (),
        0,
        True,
    )

    space = _space(
        facts,
        1,
        (
            _factor(atom.subject_identity, atom),
            _factor(component.subject_identity, component),
        ),
    )

    assert not any(
        factor.factor_key.quantity is ScientificQuantity.ELECTRONIC_STATE
        and factor.factor_key.subject_identity.startswith("molecule:")
        for factor in space.factors
    )
    assert space.expanded_paths == ()
