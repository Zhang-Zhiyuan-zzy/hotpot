"""
python v3.9.0
@Project: hotpot
@File   : search
@Auther : Zhiyuan Zhang
@Data   : 2023/9/12
@Time   : 20:13

A Module for Search
"""

import openbabel.openbabel as ob

from hotpot.cheminfo import Molecule, Atom


class SubstructureSearcher:
    """ To perform substructure search """
    def __init__(self, query):
        """"""
        self._searcher = self._substructure_define(query)

    def _substructure_define(self, query):
        """"""
        return self._smarts_substructure_define(query)

    @staticmethod
    def _smarts_substructure_define(query: str):
        """ Define a substructure by SMARTS string """
        searcher = ob.OBSmartsPattern()
        searcher.Init(query)
        return searcher

    def search(self, *mols: Molecule):
        """ search the defined substructure in the given Molecule and return a matched substructure list """
        matched_mols = []
        for mol in mols:
            self._searcher.Match(mol.ob_mol)
            map_list = [ob_ids for ob_ids in self._searcher.GetMapList()]
            matched_mols.append(MatchedMol(map_list, mol.copy()))

        return matched_mols

    @property
    def atom_counts(self) -> int:
        return self._searcher.NumAtoms()

    @property
    def smarts(self) -> str:
        return self._searcher.GetSMARTS()


class SearchHit:
    """"""
    def __init__(self, ob_ids, mol):
        self.ob_ids = ob_ids
        self.mol = mol

    def matched_atoms(self) -> list[Atom]:
        ob_idx_dict = {a.idx: a for a in self.mol.atoms}
        return [ob_idx_dict[ob_id] for ob_id in self.ob_ids]


class MatchedMol:
    """"""
    def __init__(self, ob_vector, mol):
        self._ob_vector = ob_vector
        self.mol = mol

    def __repr__(self):
        return f"MatchedList({self.__len__()})"

    def __len__(self):
        return len(self._ob_vector)

    def __iter__(self):
        return iter(SearchHit(oid, self.mol) for oid in self._ob_vector)

    def __getitem__(self, item):
        return SearchHit(self._ob_vector[item], self.mol)

    @property
    def identifier(self):
        return self.mol.identifier

