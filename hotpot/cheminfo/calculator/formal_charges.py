"""Formal-charge assignment from native Hotpot molecular graphs."""

from typing import Callable, Literal, Optional, Union

from ..core import Atom, Molecule
from .electronic_state.contracts import (
    AmbiguousHydrogenRepresentationError,
    ChargeInferenceError,
    ChargeInferenceResult,
    ChargeInferenceSource,
    FragmentCharge,
)

__all__ = ["formal_charge", "infer_charge"]


FormalChargeModel = Literal["valence", "valence-constrained", "preserve"]
MetalChargeResolver = Callable[[Atom, Molecule], int]
MetalChargeModel = Union[Literal["default", "preserve"], MetalChargeResolver]

_MAIN_GROUP_ELECTRON_MODELS = {
    1: (1, (2,)),
    5: (3, (6, 8)),
    6: (4, (8,)),
    7: (5, (8,)),
    8: (6, (8,)),
    9: (7, (8,)),
    14: (4, (8, 10, 12)),
    15: (5, (8, 10)),
    16: (6, (8, 10, 12)),
    17: (7, (8, 10, 12, 14)),
    33: (5, (8, 10)),
    34: (6, (8, 10, 12)),
    35: (7, (8, 10, 12, 14)),
    52: (6, (8, 10, 12)),
    53: (7, (8, 10, 12, 14)),
}


def _effective_bond_valence(atom: Atom) -> int:
    explicit_hydrogen_count = sum(
        neighbour.atomic_number == 1 for neighbour in atom.neighbours
    )
    if atom.atomic_number == 1:
        return atom.sum_covalent_orders
    if explicit_hydrogen_count == 0:
        return atom.sum_covalent_orders + atom.implicit_hydrogens
    if atom.implicit_hydrogens in {0, explicit_hydrogen_count}:
        return atom.sum_covalent_orders
    raise AmbiguousHydrogenRepresentationError(
        f"Atom {atom.idx} ({atom.symbol}) has {explicit_hydrogen_count} explicit "
        f"hydrogens but stores {atom.implicit_hydrogens} implicit hydrogens"
    )


def _formal_charge_candidates(atom: Atom) -> tuple[tuple[int, int], ...]:
    valence_electrons, preferred_shells = _MAIN_GROUP_ELECTRON_MODELS[
        atom.atomic_number
    ]
    bond_valence = _effective_bond_valence(atom)
    candidates = []

    for charge in range(-4, 5):
        nonbonding_electrons = valence_electrons - bond_valence - charge
        if nonbonding_electrons < 0:
            continue

        shell_electrons = nonbonding_electrons + 2 * bond_valence
        shell_distance, shell_rank = min(
            (abs(shell_electrons - shell), rank)
            for rank, shell in enumerate(preferred_shells)
        )
        radical_penalty = nonbonding_electrons % 2
        score = (
            100 * shell_distance
            + 20 * radical_penalty
            + 4 * abs(charge)
            + shell_rank
        )
        candidates.append((charge, score))

    return tuple(candidates)


def _check_supported_valence_atoms(atoms: tuple[Atom, ...]) -> None:
    unsupported = [
        atom.symbol
        for atom in atoms
        if atom.atomic_number not in _MAIN_GROUP_ELECTRON_MODELS
    ]
    if unsupported:
        symbols = ", ".join(sorted(set(unsupported)))
        raise ValueError(
            "The valence formal-charge models support H, B, C, N, O, F, Si, "
            "P, S, Cl, As, Se, Br, Te, and I; "
            f"unsupported nonmetal elements: {symbols}"
        )


def _valence_formal_charges(atoms: tuple[Atom, ...]) -> tuple[int, ...]:
    _check_supported_valence_atoms(atoms)
    assigned = []
    for atom in atoms:
        candidates = _formal_charge_candidates(atom)
        if not candidates:
            raise ChargeInferenceError(
                f"No classical valence assignment exists for atom {atom.idx} "
                f"({atom.symbol})"
            )
        assigned.append(min(candidates, key=lambda candidate: candidate[1])[0])
    return tuple(assigned)


def _constrained_valence_formal_charges(
    atoms: tuple[Atom, ...],
    total_charge: int,
) -> tuple[int, ...]:
    _check_supported_valence_atoms(atoms)
    states = {0: (0, ())}

    for atom in atoms:
        next_states = {}
        for accumulated_charge, (accumulated_score, assigned) in states.items():
            for atom_charge, atom_score in _formal_charge_candidates(atom):
                candidate_charge = accumulated_charge + atom_charge
                candidate_score = accumulated_score + atom_score
                incumbent = next_states.get(candidate_charge)
                if incumbent is None or candidate_score < incumbent[0]:
                    next_states[candidate_charge] = (
                        candidate_score,
                        assigned + (atom_charge,),
                    )
        states = next_states

    if total_charge not in states:
        raise ChargeInferenceError(
            f"No classical valence assignment sums to molecular charge {total_charge}"
        )
    return states[total_charge][1]


def _separate_metal_ligand_fragments(
    mol: Molecule,
) -> tuple[tuple[tuple[int, Atom], ...], tuple[int, ...]]:
    """Split a copy at metal-ligand bonds and retain original atom indices."""
    clone_mol = mol.copy()
    for atom_index, atom in enumerate(clone_mol.atoms):
        atom.id = atom_index
    clone_mol.hide_metal_ligand_bonds(clear_conformers=False)

    ligand_fragments = []
    metal_indices = []
    for component in clone_mol.components:
        indexed_atoms = tuple((atom.id, atom) for atom in component.atoms)
        if all(atom.is_metal for atom in component.atoms):
            metal_indices.extend(atom_index for atom_index, atom in indexed_atoms)
        elif all(not atom.is_metal for atom in component.atoms):
            ligand_fragments.append(indexed_atoms)
        else:
            raise ValueError(
                "Metal-ligand separation left a mixed metal/nonmetal component"
            )

    return tuple(ligand_fragments), tuple(metal_indices)


def _resolve_metal_charge(
    atom: Atom,
    mol: Molecule,
    metal_model: MetalChargeModel,
) -> tuple[int, ChargeInferenceSource, tuple[str, ...]]:
    if callable(metal_model):
        return (
            int(metal_model(atom, mol)),
            ChargeInferenceSource.METAL_RESOLVER,
            (),
        )
    if metal_model == "default":
        if atom.formal_charge:
            return atom.formal_charge, ChargeInferenceSource.PRESERVED, ()
        charge = atom.get_formal_charge()
        assumption = (
            f"Assumed default formal charge {charge:+d} for {atom.symbol} atom "
            f"{atom.idx} because its oxidation state is not graph-unique."
        )
        return charge, ChargeInferenceSource.METAL_DEFAULT, (assumption,)
    if metal_model == "preserve":
        return atom.formal_charge, ChargeInferenceSource.PRESERVED, ()
    raise ValueError(
        f"Unknown metal formal-charge model {metal_model!r}; choose 'default', "
        "'preserve', or provide a callable"
    )


def infer_charge(
    mol: Molecule,
    model: FormalChargeModel = "valence",
    *,
    metal_model: MetalChargeModel = "default",
    target_charge: Optional[int] = None,
) -> ChargeInferenceResult:
    """Infer integer atom, fragment, and molecular charges without mutation."""
    if (
        model in {"valence", "valence-constrained"}
        and not callable(metal_model)
        and metal_model not in {"default", "preserve"}
    ):
        raise ValueError(
            f"Unknown metal formal-charge model {metal_model!r}; choose 'default', "
            "'preserve', or provide a callable"
        )
    if model not in {"valence", "valence-constrained", "preserve"}:
        raise ValueError(
            f"Unknown formal-charge model {model!r}; choose 'valence', "
            "'valence-constrained', or 'preserve'"
        )

    ligand_fragments, metal_indices = _separate_metal_ligand_fragments(mol)
    assigned_by_index: dict[int, int] = {}
    fragment_sources: dict[tuple[int, ...], ChargeInferenceSource] = {}
    assumptions = []

    for atom_index in metal_indices:
        if model == "preserve":
            metal_charge = mol.atoms[atom_index].formal_charge
            metal_source = ChargeInferenceSource.PRESERVED
            metal_assumptions = ()
        else:
            metal_charge, metal_source, metal_assumptions = _resolve_metal_charge(
                mol.atoms[atom_index], mol, metal_model
            )
        assigned_by_index[atom_index] = metal_charge
        fragment_sources[(atom_index,)] = metal_source
        assumptions.extend(metal_assumptions)

    indexed_ligand_atoms = tuple(
        indexed_atom for fragment in ligand_fragments for indexed_atom in fragment
    )
    ligand_atoms = tuple(atom for _, atom in indexed_ligand_atoms)
    if model == "preserve":
        ligand_charges = tuple(
            mol.atoms[atom_index].formal_charge
            for atom_index, _ in indexed_ligand_atoms
        )
        ligand_source = ChargeInferenceSource.PRESERVED
    elif model == "valence":
        ligand_charges = _valence_formal_charges(ligand_atoms)
        ligand_source = ChargeInferenceSource.VALENCE
    else:
        constrained_total = mol.charge if target_charge is None else target_charge
        ligand_total = constrained_total - sum(assigned_by_index.values())
        ligand_charges = _constrained_valence_formal_charges(
            ligand_atoms,
            ligand_total,
        )
        ligand_source = ChargeInferenceSource.VALENCE_CONSTRAINED

    assigned_by_index.update(
        (atom_index, charge)
        for (atom_index, _), charge in zip(indexed_ligand_atoms, ligand_charges)
    )
    for fragment in ligand_fragments:
        fragment_sources[tuple(atom_index for atom_index, _ in fragment)] = (
            ligand_source
        )

    atom_formal_charges = tuple(
        assigned_by_index[atom_index] for atom_index in range(len(mol.atoms))
    )
    fragments = tuple(
        FragmentCharge(
            atom_indices=atom_indices,
            charge=sum(atom_formal_charges[index] for index in atom_indices),
            source=fragment_sources[atom_indices],
        )
        for atom_indices in sorted(fragment_sources, key=lambda indices: indices[0])
    )
    total_charge = sum(atom_formal_charges)
    if sum(fragment.charge for fragment in fragments) != total_charge:
        raise ChargeInferenceError(
            "Atom, fragment, and molecular formal-charge sums are inconsistent"
        )

    return ChargeInferenceResult(
        atom_formal_charges=atom_formal_charges,
        fragments=fragments,
        total_charge=total_charge,
        source=ligand_source,
        assumptions=tuple(assumptions),
    )


def formal_charge(
    mol: Molecule,
    model: FormalChargeModel = "valence",
    *,
    metal_model: MetalChargeModel = "default",
) -> tuple[int, ...]:
    """Assign classical integer formal charges from the native Hotpot graph.

    No RDKit or OpenBabel charge calculator is called. For molecules containing
    metals, a copy is split with :meth:`Molecule.hide_metal_ligand_bonds`; ligand
    fragments and metal atoms are then handled independently so coordination bonds
    are not mistaken for ordinary covalent bonds.

    Parameters
    ----------
    mol
        Hotpot molecule whose connectivity, bond orders, implicit hydrogens, and
        protonation state already describe the intended chemical structure.
    model
        ``"valence"`` independently selects the lowest-penalty Lewis state at
        each supported nonmetal atom and infers the molecular total. It is suited
        to ordinary closed-shell organics and common main-group ions.

        ``"valence-constrained"`` performs the same Lewis analysis globally but
        requires all assigned charges, including resolved metal charges, to sum to
        the current ``mol.charge``. Use it when the total ionic state is known,
        especially for carbocations, carbenes, radicals, and charged complexes.

        ``"preserve"`` trusts every atom formal charge already present and only
        synchronizes ``mol.charge``. Use it for authoritative imported charges,
        nonclassical bonding, or systems outside the valence model's scope.
    metal_model
        Metal-charge extension point used by the two valence models. ``"default"``
        preserves a nonzero stored metal charge and otherwise uses Hotpot's current
        ``Atom.get_formal_charge()`` default for that element. ``"preserve"`` keeps
        the stored metal charge even when it is zero. A callable may instead be
        supplied with signature ``resolver(metal_atom, original_molecule) -> int``;
        it can inspect the intact coordination environment and implement a future
        oxidation-state or metal-specific charge model.

    Returns
    -------
    tuple[int, ...]
        Assigned formal charges in original atom order. ``mol.charge`` is set to
        their sum and therefore equals ``mol.sum_atoms_charge`` after the call.

    Notes
    -----
    The valence models use bond orders, implicit hydrogens, main-group valence
    electrons, the duet/octet rule, and allowed expanded shells. Formal charge is
    representation-dependent; these models do not choose protonation states,
    calculate partial charges, or infer transition-metal oxidation states.
    """
    result = infer_charge(
        mol,
        model=model,
        metal_model=metal_model,
        target_charge=mol.charge if model == "valence-constrained" else None,
    )
    charges = result.atom_formal_charges

    for atom, charge in zip(mol.atoms, charges):
        atom.formal_charge = charge
    mol.charge = sum(charges)
    mol._obmol = None
    return charges

