"""PROVIDER-001 fixed-bond main-group Lewis tests."""

from dataclasses import replace

import hotpot

from hotpot.cheminfo.calculator.charge import (
    resolve_net_charge,
    solve_main_group_lewis,
)
from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    ResolutionStatus,
    ScientificAdmissibility,
)
from hotpot.cheminfo.core import BondKind
from hotpot.cheminfo.calculator.state_resolver import (
    StateRequest,
    extract_molecular_facts,
)


def _solve(smiles: str, charge: int = 0):
    facts = extract_molecular_facts(hotpot.read_mol(smiles, "smi"))
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=charge))[0]
    return solve_main_group_lewis(facts, net_charge)[0]


def test_provider_solves_neutral_main_group_structure() -> None:
    result = _solve("CCO")

    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.resolution is ResolutionStatus.UNIQUE
    assert result.candidates[0].atom_formal_charges == (0, 0, 0)
    assert sum(result.candidates[0].atom_formal_charges) == 0


def test_provider_preserves_nitro_formal_charge_separation() -> None:
    result = _solve("C[N+](=O)[O-]")

    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.candidates[0].atom_formal_charges == (0, 1, 0, -1)
    assert result.candidates[0].radical_electrons == (0, 0, 0, 0)


def test_provider_abstains_for_metal_domain() -> None:
    result = _solve("[Fe]", charge=2)

    assert result.admissibility is ScientificAdmissibility.NOT_EVALUATED
    assert result.resolution is ResolutionStatus.OUT_OF_DOMAIN
    assert result.candidates == ()
    assert result.findings[0].code == "PROVIDER-001.UNSUPPORTED_ELEMENT"


def test_provider_does_not_change_the_net_charge_result() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("[NH4+]", "smi"))
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=1))[0]
    before = net_charge

    solve_main_group_lewis(facts, net_charge)

    assert net_charge == before


def test_inadmissible_charge_envelope_cannot_feed_lewis_solver() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("CCO", "smi"))
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=0))[0]

    result = solve_main_group_lewis(
        facts,
        replace(
            net_charge,
            admissibility=ScientificAdmissibility.INADMISSIBLE,
        ),
    )[0]

    assert result.admissibility is ScientificAdmissibility.INADMISSIBLE
    assert result.candidates == ()
    assert result.findings[0].code == (
        "PROVIDER-001.NET_CHARGE_RESULT_UNAVAILABLE"
    )


def test_declared_coordination_edge_is_excluded_from_covalent_valence() -> None:
    mol = hotpot.read_mol("B.N", "smi")
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=0))[0]

    result = solve_main_group_lewis(facts, net_charge)[0]

    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.candidates[0].bond_orders == ()
    assert result.findings == ()


def test_resource_truncation_never_claims_scientific_uniqueness() -> None:
    facts = extract_molecular_facts(hotpot.read_mol("B.B", "smi"))
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=0))[0]

    result = solve_main_group_lewis(
        facts,
        net_charge,
        candidate_budget=CandidateBudget(1, 1, 1),
    )[0]

    assert result.resolution is ResolutionStatus.AMBIGUOUS
    assert result.findings[0].code == "PROVIDER-001.RESOURCE_TRUNCATED"
