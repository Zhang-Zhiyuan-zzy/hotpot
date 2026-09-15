# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : calculator
 Created   : 2025/5/19 11:07
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
  The collection of `Calculators` to determine Crystal, Molecule, Ring, Bond, Atom
 attributes
 
===========================================================
"""
import os
from functools import lru_cache
from typing import Callable, Literal, Union

from .core import Atom, Molecule


_inorg_frag_charges = {
    "C12=C3C4=C5C6=C7C3=C3C8=C1C1=C9C%10=C2C4=C2C4=C%10C%10=C%11C%12=C4C4=C%13C%14=C(C5=C24)C6=C2C4=C7C3=C3C5=C8C1=C1C(=C9%10)C6=C%11C7=C8C9=C6C1=C5C1=C9C5=C(C4=C31)C2=C%14C(=C85)C%13=C%127": 0,
    'O[Te](F)(F)(F)(F)F': -1,

}


class Calculator:
    """ The base class of calculator """
    pass


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


def _formal_charge_candidates(atom: Atom) -> tuple[tuple[int, int], ...]:
    valence_electrons, preferred_shells = _MAIN_GROUP_ELECTRON_MODELS[
        atom.atomic_number
    ]
    bond_valence = atom.sum_covalent_orders + atom.implicit_hydrogens
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
    return tuple(
        min(_formal_charge_candidates(atom), key=lambda candidate: candidate[1])[0]
        for atom in atoms
    )


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
        raise ValueError(
            f"No classical valence assignment sums to molecular charge {total_charge}"
        )
    return states[total_charge][1]


def _separate_metal_ligand_fragments(
        mol: Molecule,
) -> tuple[tuple[tuple[int, Atom], ...], tuple[int, ...]]:
    """Split a copy at metal-ligand bonds and retain original atom indices."""
    clone = mol.copy()
    for atom_index, atom in enumerate(clone.atoms):
        atom.id = atom_index
    clone.hide_metal_ligand_bonds(clear_conformers=False)

    ligand_fragments = []
    metal_indices = []
    for component in clone.components:
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
) -> int:
    if callable(metal_model):
        return int(metal_model(atom, mol))
    if metal_model == "default":
        if atom.formal_charge:
            return atom.formal_charge
        return atom.get_formal_charge()
    if metal_model == "preserve":
        return atom.formal_charge
    raise ValueError(
        f"Unknown metal formal-charge model {metal_model!r}; choose 'default', "
        "'preserve', or provide a callable"
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
    if (
            model in {"valence", "valence-constrained"}
            and not callable(metal_model)
            and metal_model not in {"default", "preserve"}
    ):
        raise ValueError(
            f"Unknown metal formal-charge model {metal_model!r}; choose 'default', "
            "'preserve', or provide a callable"
        )

    if model == "preserve":
        charges = tuple(atom.formal_charge for atom in mol.atoms)
    elif model in {"valence", "valence-constrained"}:
        if mol.metals:
            ligand_fragments, metal_indices = _separate_metal_ligand_fragments(mol)
        else:
            ligand_fragments = (tuple(enumerate(mol.atoms)),)
            metal_indices = ()

        assigned_by_index = {
            atom_index: _resolve_metal_charge(
                mol.atoms[atom_index], mol, metal_model
            )
            for atom_index in metal_indices
        }
        indexed_ligand_atoms = tuple(
            indexed_atom
            for fragment in ligand_fragments
            for indexed_atom in fragment
        )
        ligand_atoms = tuple(atom for _, atom in indexed_ligand_atoms)

        if model == "valence":
            ligand_charges = _valence_formal_charges(ligand_atoms)
        else:
            ligand_total = mol.charge - sum(assigned_by_index.values())
            ligand_charges = _constrained_valence_formal_charges(
                ligand_atoms,
                ligand_total,
            )

        assigned_by_index.update(
            (atom_index, charge)
            for (atom_index, _), charge in zip(
                indexed_ligand_atoms,
                ligand_charges,
            )
        )
        charges = tuple(
            assigned_by_index[atom_index]
            for atom_index in range(len(mol.atoms))
        )
    else:
        raise ValueError(
            f"Unknown formal-charge model {model!r}; choose 'valence', "
            "'valence-constrained', or 'preserve'"
        )

    for atom, charge in zip(mol.atoms, charges):
        atom.formal_charge = charge
    mol.charge = sum(charges)
    mol._obmol = None
    return charges


def _calc_mol_charge(mol: Molecule, pH: float = 7.4) -> int:
    if mol.is_organic or mol.is_full_halogenated:
        if len(mol.hydrogens) == 0:
            return 0
        else:
            obmol = mol.to_obmol()
            obmol.AddHydrogens(False, True, pH)
            return len(mol.atoms) - obmol.NumAtoms()

    elif len(mol.atoms) == 1:
        atom = mol.atoms[0]
        if atom.formal_charge == 0:
            return mol.atoms[0].get_formal_charge()
        else:
            return atom.formal_charge

    elif len(mol.metals) >= 1:
        clone = mol.copy()
        clone.hide_metal_ligand_bonds()
        return sum(_calc_mol_charge(c, pH=pH) for c in clone.components)

    # If the molecule is inorganic fragment
    elif mol.smiles in _inorg_frag_charges:
        return _inorg_frag_charges[mol.smiles]

    else:
        raise ValueError(f'Unknown molecule fragment: {mol.smiles}')


class MolChargeCalculator(Calculator):
    def __call__(self, mol, pH: float = 7.4) -> float:
        return _calc_mol_charge(mol, pH=pH)


@lru_cache(maxsize=4)
def _get_mca_predictor(device: str, allow_charged: bool):
    from .AImodels.mca import MCAPredictor

    return MCAPredictor(
        device=device,
        allow_charged=allow_charged,
    )


def mca(
        mol: Molecule,
        *,
        device: str = None,
        allow_charged: bool = False,
):
    """Predict every atom's MCA and identify important nucleophilic sites."""
    selected_device = device or os.environ.get("HOTPOT_MCA_DEVICE", "auto")
    prediction = _get_mca_predictor(selected_device, allow_charged).predict(mol)
    atom_values = {
        atom.atom_index: atom.mca_kj_mol
        for atom in prediction.atom_predictions
    }
    site_values = {site.atom_index: site.mca_kj_mol for site in prediction.sites}

    for atom in mol.atoms:
        object.__setattr__(atom, "_mca", atom_values[atom.idx])
    object.__setattr__(
        mol,
        "_mca_sites",
        {mol.atoms[index]: value for index, value in site_values.items()},
    )
    mol.properties["mca"] = prediction
    return prediction
