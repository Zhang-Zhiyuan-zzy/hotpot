"""High-level MCA inference API."""

from __future__ import annotations

from collections.abc import Iterable

from rdkit import Chem

from hotpot.cheminfo.convert import is_molecule_input, to_hotpot_mol
from hotpot.cheminfo.core import Molecule as HotpotMolecule

from .conformer import ensure_3d_conformer
from .featurizer import collate_site_rows, mol_to_unimolv2
from .result_types import AtomPrediction, MoleculePrediction, SitePrediction
from .runtime import MCARuntime
from .site_detection import find_nucleophilic_sites


def _is_single_input(value):
    return is_molecule_input(value) or not isinstance(value, Iterable)


def _feature_mol(value, native_mol: HotpotMolecule) -> Chem.Mol:
    if isinstance(value, Chem.Mol):
        return Chem.Mol(value)
    mol = Chem.Mol(native_mol.to_rdmol())
    Chem.SanitizeMol(mol)
    return mol


class MCAPredictor:
    def __init__(
        self,
        model_dir=None,
        device: str = "auto",
        variant: str | None = None,
        batch_size: int = 64,
        conformer_seed: int = 42,
        verify_model: bool = True,
        allow_charged: bool = False,
    ):
        self.runtime = MCARuntime(model_dir, device, variant, verify_model)
        self.batch_size = batch_size
        self.conformer_seed = conformer_seed
        self.max_atoms = int(self.runtime.manifest["max_atoms"])
        self.allow_charged = allow_charged

    def predict(self, molecules):
        single = _is_single_input(molecules)
        values = [molecules] if single else list(molecules)
        native_mols = [to_hotpot_mol(value) for value in values]
        hydrogenated = [
            index
            for index, mol in enumerate(native_mols)
            if mol.hydrogens
        ]
        if hydrogenated:
            raise ValueError(
                f"Explicit hydrogen atoms at molecule positions {hydrogenated} cannot "
                "be MCA targets; remove explicit hydrogens before prediction"
            )
        inconsistent_charges = [
            index
            for index, mol in enumerate(native_mols)
            if mol.charge != mol.sum_atoms_charge
        ]
        if inconsistent_charges:
            raise ValueError(
                "Molecular total charge does not match the sum of atom formal "
                f"charges at molecule positions {inconsistent_charges}"
            )
        if not self.allow_charged:
            charged = [
                index
                for index, mol in enumerate(native_mols)
                if mol.charge != 0
            ]
            if charged:
                raise ValueError(
                    f"Charged molecules at positions {charged} are outside the validated domain; "
                    "set allow_charged=True to opt in"
                )
        feature_mols = [
            _feature_mol(value, native_mol)
            for value, native_mol in zip(values, native_mols)
        ]
        conformers = [
            ensure_3d_conformer(mol, self.conformer_seed) for mol in feature_mols
        ]
        sites = [find_nucleophilic_sites(mol) for mol in native_mols]
        features = [mol_to_unimolv2(mol, self.max_atoms) for mol in conformers]

        molecule_indices = [
            molecule_index
            for molecule_index, mol in enumerate(native_mols)
            for _ in mol.atoms
        ]
        atom_indices = [
            atom_index
            for mol in native_mols
            for atom_index in range(len(mol.atoms))
        ]
        predictions = []
        for start in range(0, len(atom_indices), self.batch_size):
            stop = start + self.batch_size
            arrays = collate_site_rows(
                features,
                molecule_indices[start:stop],
                atom_indices[start:stop],
            )
            predictions.extend(
                self.runtime.predict(arrays, self.batch_size).tolist()
            )

        cursor = 0
        results = []
        for mol, molecule_sites in zip(native_mols, sites):
            molecule_predictions = predictions[cursor: cursor + len(mol.atoms)]
            cursor += len(mol.atoms)
            atom_results = tuple(
                AtomPrediction(
                    atom_index=atom.idx,
                    element=atom.symbol,
                    mca_kj_mol=float(molecule_predictions[atom.idx]),
                )
                for atom in mol.atoms
            )
            site_results = []
            for site in molecule_sites:
                site_results.append(
                    SitePrediction(
                        atom_index=site.atom_index,
                        element=mol.atoms[site.atom_index].symbol,
                        site_type=site.site_type,
                        mca_kj_mol=float(molecule_predictions[site.atom_index]),
                    )
                )
            results.append(
                MoleculePrediction(
                    smiles=mol.smiles,
                    formal_charge=mol.charge,
                    sites=tuple(site_results),
                    model_variant=self.runtime.variant,
                    atom_predictions=atom_results,
                )
            )
        return results[0] if single else results


def predict_mca(molecules, **kwargs):
    """Predict MCA for one molecule or an iterable such as a pandas Series."""

    return MCAPredictor(**kwargs).predict(molecules)
