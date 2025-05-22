# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : data_mining
 Created   : 2025/5/16 9:24
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os
import os.path as osp
import shutil

from tqdm import tqdm

import ccdc
from ccdc.molecule import Molecule as CMol, Bond

from hotpot.plugins.ccdc_api.solvents import solvents_smiles
from .custom_json import dumps_with_partial_indent


def mining_mono_metal_complexes(save_dir):
    reader = ccdc.io.EntryReader('CSD')
    for entry in tqdm(reader, 'Mining mono-metal complexes from CSD'):

        # Exclude structure without 3D coordinates
        # Exclude structure with polymeric structure
        # Exclude structure with disorder atoms
        if not entry.has_3d_structure or entry.is_polymeric or entry.has_disorder:
            continue

        mol = entry.molecule
        heaviest_component = mol.heaviest_component

        # The mono complexes should have and only have one metal
        metals = [a for a in heaviest_component.atoms if a.is_metal]
        if len(metals) != 1:
            continue

        # The ligand should be organic compounds
        if (any(a.atomic_symbol == 'C' for a in heaviest_component.atoms) and
                any(a.atomic_symbol == 'H' for a in heaviest_component.atoms)):
            with ccdc.io.MoleculeWriter(osp.join(save_dir, f'{entry.identifier}.mol2')) as writer:
                writer.write(heaviest_component)


def statistics_complexes(struct_dir: str, results_path: str):
    """
    Parameters:
        struct_dir: directory of structures
        results_path: result path, which is a json file
    """
    files = os.listdir(struct_dir)
    results = {
        'metal_counts': {},
        'coordinate_numbers': {},
    }
    for file in tqdm(files, "Statistic complexes info"):
        path_mol2 = osp.join(struct_dir, file)
        mol = ccdc.io.MoleculeReader(path_mol2)[0]

        metals = [a for a in mol.atoms if a.is_metal]
        assert len(metals) == 1
        metal = metals[0]

        count = results['metal_counts'].get(metal.atomic_symbol, 0)
        results['metal_counts'][metal.atomic_symbol] = count + 1

        # coordination atoms counts
        c_num = len(metal.neighbours)
        dict_c_num = results['coordinate_numbers'].setdefault(metal.atomic_symbol, {})
        dict_c_num[c_num] = dict_c_num.get(c_num, 0) + 1

    # Save results
    # Convert to json format
    results = dumps_with_partial_indent(results)
    with open(results_path, 'w') as f:
        f.write(results)


def filter_high_cn_complexes(struct_dir: str, des_dir: str):
    files = os.listdir(struct_dir)
    for file in tqdm(files, "Filter high CN-complexes info"):
        path_mol2 = osp.join(struct_dir, file)
        mol = ccdc.io.MoleculeReader(path_mol2)[0]

        metals = [a for a in mol.atoms if a.is_metal]
        assert len(metals) == 1
        metal = metals[0]

        c_num = len(metal.neighbours)
        if c_num > 16 or c_num <= 1:
            shutil.copy(path_mol2, osp.join(des_dir, file))

def filter_metallocene_complexes(struct_dir: str, des_dir: str):
    files = os.listdir(struct_dir)
    for file in tqdm(files, "Filter metallocene complexes"):
        path_mol2 = osp.join(struct_dir, file)
        mol = ccdc.io.MoleculeReader(path_mol2)[0]

        if is_metallocene(mol):
            shutil.move(path_mol2, osp.join(des_dir, file))


def is_metallocene(mol: CMol):
    mol.normalise_labels()
    metal = [a for a in mol.atoms if a.is_metal][0]

    m_neighbors = list(metal.neighbours)
    neigh_labels = {a.label for a in m_neighbors}
    for neigh_atom in m_neighbors:
        rings = [r for r in neigh_atom.rings if not any(a.is_metal for a in r.atoms)]
        if rings:
            for ring in rings:
                if all(a.label in neigh_labels for a in ring.atoms):
                    return True
    return False


def extract_ml_pair_from_complexes(struct_dir: str, des_dir: str, stat_dir: str):
    files = os.listdir(struct_dir)

    coord_molecule = {}
    non_solvents = set()
    for file in tqdm(files, "Extract ML-pair"):
        path_mol2 = osp.join(struct_dir, file)
        mol = ccdc.io.MoleculeReader(path_mol2)[0]

        assert isinstance(mol, CMol)

        mol.normalise_labels()
        clone = mol.copy()

        metal = [a for a in clone.atoms if a.is_metal][0]
        coord_atoms = metal.neighbours
        clone.remove_bonds(metal.bonds)
        components = sorted(clone.components, key=lambda c: 0 if len(c.atoms) == 1 else c.molecular_weight)

        # Construct Metal-ligand pair and save it
        pair = components[-1]  # get largest molecule
        coord_atoms_labels = set(a.label for a in coord_atoms) & set(a.label for a in pair.atoms)
        pair_metal = pair.add_atom(metal)
        for ca_label in coord_atoms_labels:
            pair.add_bond(Bond.BondType(1), pair_metal, pair.atom(ca_label))
            pair.bond(pair_metal.label, ca_label)

        if pair.molecular_weight / mol.molecular_weight < 0.25:
            continue

        pair.identifier = osp.splitext(file)[0]
        with ccdc.io.MoleculeWriter(osp.join(des_dir, f'{pair.identifier}.mol2')) as writer:
            writer.write(pair)

        # Statistic the solvents and other media in the complexes
        c_smiles = {c.smiles for c in components}
        non_solvents.update(set(c_smiles) - solvents_smiles)
        coord_molecule[osp.splitext(file)[0]] = list(c_smiles)

    stat_results = {
        'non_sol_mol': list(non_solvents),
        'coord_mole': coord_molecule,
    }

    json_text = dumps_with_partial_indent(stat_results)
    with open(osp.join(stat_dir, 'MLPairStat.json'), 'w') as f:
        f.write(json_text)


def stat_redundant_pairs(struct_dir: str, save_path: str):
    smiles_to_file = {}
    files = os.listdir(struct_dir)
    for file in tqdm(files, "Stat redundant pairs"):
        file_name = osp.splitext(file)[0]
        path_mol2 = osp.join(struct_dir, file)

        mol = ccdc.io.MoleculeReader(path_mol2)[0]
        smi = mol.smiles

        list_same_smi_files = smiles_to_file.setdefault(smi, [])
        list_same_smi_files.append(file_name)

    # Re-organize the results dict
    results = {}
    for smi, list_files in smiles_to_file.items():
        results[list_files[0]] = {
            'smiles': smi,
            'same_graph_pairs': list_files
        }

    json_text = dumps_with_partial_indent(results, indent=1)
    with open(save_path, 'w') as f:
        f.write(json_text)

