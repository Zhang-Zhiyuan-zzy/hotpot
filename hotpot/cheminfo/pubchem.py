# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : pubchem
 Created   : 2025/6/14 19:48
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import re
from typing import Optional
from functools import wraps

import pubchempy as pcp

__all__ = [
    'get_compound',
    'get_compounds_cid',
    'smi_to_cid',
    'smi_to_cas',
    'smi_to_name',
    'cid_to_smi',
    'name_to_smi',
    'cid_to_cas'
]

@wraps(pcp.get_compounds)
def get_compound(*args, **kwargs):
    compounds = pcp.get_compounds(*args, **kwargs)
    if len(compounds) > 0:
        return compounds[0]
    else:
        return None

def get_compounds_cid(identifier: str, id_type: str = 'name'):
    try:
        compounds = pcp.get_compounds(identifier.strip(), id_type)[0]
        return compounds.cid
    except IndexError:
        print(UserWarning(f'Failed to get compounds cid for {identifier}'))
        return None

cas_regex = re.compile(r'^\d{2,7}-\d{2}-\d')
def _get_cas(compound: pcp.Compound):
    for syn in compound.synonyms:
        if cas_regex.match(syn):
            return syn

def _get_name(compound: pcp.Compound):
    return compound.synonyms[0]

def smi_to_cas(smiles: str) -> Optional[str]:
    compound = get_compound(smiles, 'smiles')
    if compound:
        return _get_cas(compound)
    return None

def cid_to_cas(cid: int) -> Optional[str]:
    compound = get_compound(cid, 'cid')
    if compound:
        return _get_cas(compound)
    return None

def smi_to_name(smiles: str) -> Optional[str]:
    compound = get_compound(smiles, 'smiles')
    if compound:
        return _get_name(compound)
    return None

def smi_to_cid(smiles: str):
    return get_compounds_cid(smiles, 'smiles')

def cid_to_smi(cid):
    compound = get_compound(cid, 'cid')
    if compound:
        return compound.canonical_smiles
    return None

def name_to_smi(name: str):
    compound = get_compound(name, 'name')
    if compound:
        return compound.synonyms[0]
    return None


if __name__ == '__main__':
    import os.path as osp
    import pandas as pd
    path_data = osp.join(osp.abspath(osp.dirname(__file__)), 'ChemData', 'SolventsProperties.xlsx')
    path_save = osp.join(osp.abspath(osp.dirname(__file__)), 'ChemData', 'Data2CAS.csv')
    data2 = pd.read_excel(path_data, sheet_name='Data2')

    cids = data2['Cid'].values.tolist()
    list_cas = [cid_to_cas(cid) for cid in cids]

    series = pd.Series(list_cas, name='cas')
    series.to_csv(path_save)
