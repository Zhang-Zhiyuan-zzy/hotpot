"""High-level MCA inference API."""

from __future__ import annotations

from rdkit import Chem

from .conformer import ensure_3d_conformer
from .featurizer import collate_site_rows, mol_to_unimolv2
from .graph_adapter import MoleculeGraph, to_rdkit_mol
from .result_types import AtomPrediction, MoleculePrediction, SitePrediction
from .runtime import MCARuntime
from .site_detection import find_nucleophilic_sites


def _is_single_input(value):
    return isinstance(value, (str, Chem.Mol, MoleculeGraph)) or hasattr(value, "to_rdmol")


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
        rdkit_mols = [to_rdkit_mol(value) for value in values]
        hydrogenated = [
            index
            for index, mol in enumerate(rdkit_mols)
            if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
        ]
        if hydrogenated:
            raise ValueError(
                f"Explicit hydrogen atoms at molecule positions {hydrogenated} cannot "
                "be MCA targets; remove explicit hydrogens before prediction"
            )
        if not self.allow_charged:
            charged = [index for index, mol in enumerate(rdkit_mols) if Chem.GetFormalCharge(mol) != 0]
            if charged:
                raise ValueError(
                    f"Charged molecules at positions {charged} are outside the validated domain; "
                    "set allow_charged=True to opt in"
                )
        conformers = [ensure_3d_conformer(mol, self.conformer_seed) for mol in rdkit_mols]
        sites = [find_nucleophilic_sites(mol) for mol in rdkit_mols]
        features = [mol_to_unimolv2(mol, self.max_atoms) for mol in conformers]

        molecule_indices = [
            molecule_index
            for molecule_index, mol in enumerate(rdkit_mols)
            for _ in range(mol.GetNumAtoms())
        ]
        atom_indices = [
            atom_index
            for mol in rdkit_mols
            for atom_index in range(mol.GetNumAtoms())
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
        for mol, molecule_sites in zip(rdkit_mols, sites):
            molecule_predictions = predictions[cursor: cursor + mol.GetNumAtoms()]
            cursor += mol.GetNumAtoms()
            atom_results = tuple(
                AtomPrediction(
                    atom_index=atom.GetIdx(),
                    element=atom.GetSymbol(),
                    mca_kj_mol=float(molecule_predictions[atom.GetIdx()]),
                )
                for atom in mol.GetAtoms()
            )
            site_results = []
            for site in molecule_sites:
                site_results.append(
                    SitePrediction(
                        atom_index=site.atom_index,
                        element=mol.GetAtomWithIdx(site.atom_index).GetSymbol(),
                        site_type=site.site_type,
                        mca_kj_mol=float(molecule_predictions[site.atom_index]),
                    )
                )
            results.append(
                MoleculePrediction(
                    smiles=Chem.MolToSmiles(mol),
                    formal_charge=Chem.GetFormalCharge(mol),
                    sites=tuple(site_results),
                    model_variant=self.runtime.variant,
                    atom_predictions=atom_results,
                )
            )
        return results[0] if single else results


def predict_mca(molecules, **kwargs):
    """Predict MCA for one molecule or an iterable such as a pandas Series."""

    return MCAPredictor(**kwargs).predict(molecules)
