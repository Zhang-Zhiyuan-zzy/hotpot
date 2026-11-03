"""PROVIDER-001 fixed-bond main-group Lewis tests."""

from dataclasses import asdict, replace

import pytest

import hotpot

from hotpot.cheminfo.rules.chemistry.scientific_state.charge.net_charge import resolve_net_charge
from hotpot.cheminfo.rules.chemistry.scientific_state.charge.lewis import solve_main_group_lewis
from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateBudget,
    ResolutionStatus,
    ScientificAdmissibility,
)
from hotpot import Molecule
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.adapters.structure.scientific_state import _extract_molecular_facts
from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts


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


@pytest.mark.parametrize("known_zero", (False, True))
def test_undirected_nb_requires_known_site_total_hydrogens(known_zero) -> None:
    mol = Molecule()
    supplied_hydrogens = {"implicit_hydrogens": 0} if known_zero else {}
    for number in (7, 5):
        mol.create_atom(atomic_number=number, **supplied_hydrogens)
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    facts = extract_molecular_facts(mol)
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=-6))[0]
    before = asdict(facts), asdict(net_charge)

    result = solve_main_group_lewis(facts, net_charge)[0]

    assert (asdict(facts), asdict(net_charge)) == before
    assert facts.hydrogen_inventory.implicit_counts == (
        (0, 0) if known_zero else (None, None)
    )
    assert facts.nuclear_charge_sum == 12
    if known_zero:
        assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
        assert result.resolution is ResolutionStatus.UNIQUE
        candidate = result.candidates[0]
        assert candidate.atom_formal_charges == (-3, -3)
        assert candidate.nonbonding_electrons == (8, 6)
        assert candidate.bond_orders == ()
        assert result.findings == ()
    else:
        assert result.admissibility is ScientificAdmissibility.NOT_EVALUATED
        assert result.resolution is ResolutionStatus.OUT_OF_DOMAIN
        assert result.candidates == ()
        assert result.findings[0].code == (
            "PROVIDER-001.INCOMPLETE_HYDROGEN_INVENTORY"
        )
        assert result.findings[0].message.endswith("atom indices (0, 1).")
        assert result.findings[0].related_identities == (
            net_charge.candidates[0].provenance.identity,
        )


def test_directed_nb_preserves_known_positive_hydrogens() -> None:
    mol = Molecule()
    for number in (7, 5):
        mol.create_atom(atomic_number=number)
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE, bond_direction="atom1_to_atom2")
    facts = extract_molecular_facts(mol)
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=-1))[0]

    result = solve_main_group_lewis(facts, net_charge)[0]

    assert facts.hydrogen_inventory.implicit_counts == (3, 2)
    assert facts.nuclear_charge_sum == 17
    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.resolution is ResolutionStatus.UNIQUE
    assert result.candidates[0].atom_formal_charges == (0, -1)
    assert result.candidates[0].nonbonding_electrons == (2, 2)
    assert result.candidates[0].bond_orders == ()
    assert result.findings == ()


@pytest.mark.parametrize("known_site_total", (False, True))
def test_explicit_h_neighbour_does_not_resolve_unknown_site_total(known_site_total) -> None:
    mol = Molecule()
    supplied_hydrogens = {"implicit_hydrogens": 1} if known_site_total else {}
    mol.create_atom(atomic_number=7, **supplied_hydrogens)
    mol.create_atom(atomic_number=5, implicit_hydrogens=0)
    mol.create_atom(atomic_number=1)
    mol.add_bond(0, 1, bond_kind=BondKind.DATIVE)
    mol.add_bond(0, 2)
    facts = extract_molecular_facts(mol)
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=-5))[0]

    result = solve_main_group_lewis(facts, net_charge)[0]

    assert facts.hydrogen_inventory.explicit_counts == (1, 0, 0)
    assert facts.hydrogen_inventory.implicit_counts == (
        (1, 0, 0) if known_site_total else (None, 0, 0)
    )
    assert facts.nuclear_charge_sum == 13
    if known_site_total:
        assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
        assert result.resolution is ResolutionStatus.UNIQUE
        assert result.candidates[0].atom_formal_charges == (-2, -3, 0)
        assert result.candidates[0].bond_orders == (1.0,)
        assert result.findings == ()
    else:
        assert result.admissibility is ScientificAdmissibility.NOT_EVALUATED
        assert result.resolution is ResolutionStatus.OUT_OF_DOMAIN
        assert result.candidates == ()
        assert result.findings[0].code == (
            "PROVIDER-001.INCOMPLETE_HYDROGEN_INVENTORY"
        )
        assert result.findings[0].message.endswith("atom indices (0,).")


def test_explicit_h_atom_does_not_require_its_own_implicit_h_assignment() -> None:
    mol = Molecule()
    mol.create_atom(atomic_number=1)
    mol.create_atom(atomic_number=1)
    mol.add_bond(0, 1)
    # Capture actual stored H2 facts without preparing implicit-H slots on H.
    facts = _extract_molecular_facts(mol)
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=0))[0]

    result = solve_main_group_lewis(facts, net_charge)[0]

    assert facts.hydrogen_inventory.explicit_counts == (1, 1)
    assert facts.hydrogen_inventory.implicit_counts == (None, None)
    assert facts.nuclear_charge_sum == 2
    assert result.admissibility is ScientificAdmissibility.ADMISSIBLE
    assert result.resolution is ResolutionStatus.UNIQUE
    assert result.candidates[0].atom_formal_charges == (0, 0)
    assert result.candidates[0].bond_orders == (1.0,)
    assert result.findings == ()


@pytest.mark.parametrize(
    ("numbers", "bond_kind", "expected_code", "implicit_counts"),
    (
        ((26, 7), BondKind.DATIVE, "UNSUPPORTED_ELEMENT", (0, None)),
        ((6, 8), BondKind.UNKNOWN, "UNSUPPORTED_BOND_SEMANTICS", (None, None)),
    ),
)
def test_existing_domain_findings_precede_incomplete_h_inventory(
    numbers, bond_kind, expected_code, implicit_counts,
) -> None:
    mol = Molecule()
    for number in numbers:
        mol.create_atom(atomic_number=number)
    mol.add_bond(
        0, 1, 0.0 if bond_kind is BondKind.UNKNOWN else 1.0, bond_kind=bond_kind,
    )
    facts = extract_molecular_facts(mol)
    net_charge = resolve_net_charge(facts, StateRequest(net_charge=0))[0]

    result = solve_main_group_lewis(facts, net_charge)[0]

    assert facts.hydrogen_inventory.implicit_counts == implicit_counts
    assert result.admissibility is ScientificAdmissibility.NOT_EVALUATED
    assert result.resolution is ResolutionStatus.OUT_OF_DOMAIN
    assert result.candidates == ()
    assert result.findings[0].code == f"PROVIDER-001.{expected_code}"
