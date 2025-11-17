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
from functools import wraps, lru_cache

import pubchempy as pcp

__all__ = [
    'get_compound',
    'get_compounds_cid',
    'smi_to_cid',
    'smi_to_cas',
    'smi_to_name',
    'cid_to_smi',
    'name_to_smi',
    'cid_to_cas',
    'pubchem_service'
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


class PubChemService:
    """
    Unified wrapper around pubchempy for identifier conversion.
    It supports CAS, SMILES, CID, and chemical names mapping.
    Caches results to minimize network traffic.
    """

    cas_regex = re.compile(r'^\d{2,7}-\d{2}-\d$')

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ------------------------------------------------------------
    # Core get wrapper
    # ------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=256)
    def get_compound(identifier: str, id_type: str = 'name') -> Optional[pcp.Compound]:
        """Fetch a compound by identifier (cached)."""
        try:
            compounds = pcp.get_compounds(identifier.strip(), id_type)
            if compounds:
                return compounds[0]
        except Exception as e:
            print(f"PubChem query failed for {identifier}: {e}")
        return None

    # ------------------------------------------------------------
    # CAS extraction helpers
    # ------------------------------------------------------------
    @classmethod
    def _extract_cas(cls, compound: pcp.Compound) -> Optional[str]:
        """Extract CAS number from compound synonyms."""
        if not compound:
            return None
        for syn in compound.synonyms or []:
            if cls.cas_regex.match(syn):
                return syn
        return None

    @staticmethod
    def _extract_name(compound: pcp.Compound) -> Optional[str]:
        """Extract a representative name (first synonym)."""
        if compound and compound.synonyms:
            return compound.synonyms[0]
        return None

    # ------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------
    def name_to_smi(self, name: str) -> Optional[str]:
        c = self.get_compound(name, 'name')
        return c.canonical_smiles if c else None

    def smi_to_cas(self, smiles: str) -> Optional[str]:
        c = self.get_compound(smiles, 'smiles')
        return self._extract_cas(c)

    def smi_to_name(self, smiles: str) -> Optional[str]:
        c = self.get_compound(smiles, 'smiles')
        return self._extract_name(c)

    def smi_to_cid(self, smiles: str) -> Optional[int]:
        c = self.get_compound(smiles, 'smiles')
        return c.cid if c else None

    def cid_to_smi(self, cid: int) -> Optional[str]:
        c = self.get_compound(str(cid), 'cid')
        return c.canonical_smiles if c else None

    def cid_to_cas(self, cid: int) -> Optional[str]:
        c = self.get_compound(str(cid), 'cid')
        return self._extract_cas(c)

    def cid_to_name(self, cid: int) -> Optional[str]:
        c = self.get_compound(str(cid), 'cid')
        return self._extract_name(c)

    def name_to_cid(self, name: str) -> Optional[int]:
        c = self.get_compound(name, 'name')
        return c.cid if c else None

    # ------------------------------------------------------------
    # Flexible conversion interface
    # ------------------------------------------------------------
    def convert(self, identifier: str, from_type: str, to_type: str) -> Optional[str]:
        """
        Generic conversion: convert(identifier, from_type, to_type)
        e.g. convert('64-17-5', 'cas', 'smiles')
        """
        from_type, to_type = from_type.lower(), to_type.lower()
        c = self.get_compound(identifier, from_type)
        if not c:
            return None

        mapping = {
            'cas': lambda x: self._extract_cas(x),
            'smiles': lambda x: x.canonical_smiles,
            'name': lambda x: self._extract_name(x),
            'cid': lambda x: x.cid
        }
        return mapping[to_type](c) if to_type in mapping else None

pubchem_service = PubChemService()

