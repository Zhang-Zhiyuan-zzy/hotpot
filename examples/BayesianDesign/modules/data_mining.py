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
from tqdm import tqdm
import ccdc


def mining_mono_metal_complexes():
    reader = ccdc.io.MoleculeReader('CSD')
    for entry in tqdm(reader):

        # Exclude structure without 3D coordinates
        # Exclude structure with polymeric structure
        # Exclude structure with disorder atoms
        if not entry.has_3d_structure or entry.is_polymeric or entry.has_disorder:
            continue

        mol = entry.molecule
        heaviest_component = mol.heaviest_component

        metals = [a for a in heaviest_component.atoms if a.is_metal]
        if len(metals) != 1:
            continue

        
