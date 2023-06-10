"""
python v3.9.0
@Project: hotpot
@File   : load_chem_lib
@Auther : Zhiyuan Zhang
@Data   : 2023/6/8
@Time   : 3:27

This module is used to lazily load the chemical information database when other modules need it.
"""
from pathlib import Path

class Library:
    """ the Main class to load and save chemical information lazily """

    _lib = {}  # library for chemical books

    def __init__(self):
        self._books = {}

    @property
    def book_list(self):
        return list(self._lib.keys())

    @classmethod
    def register(cls, book_class: type):
        """ sign up the chemical books """
        cls._lib[book_class.__name__] = book_class

    def get(self, book_name: str):
        return self._books.setdefault(book_name, self._lib[book_name]())


class ChemicalBook:
    """ The base class for all chemical books """

@Library.register
class Solvents(ChemicalBook):
    """ the ChemicalBook to store common solvents """
    def __init__(self):
        dir_solvents = Path(hp.data_root).joinpath('solvents')
        self._solvents = [hp.Molecule.read_from(p) for p in dir_solvents.glob('*.mol2')]
        self._sols_smi = [m.smiles for m in self._solvents]

    def __iter__(self):
        return self._solvents

    def __getitem__(self, item):
        return self._solvents[item]

    def __repr__(self):
        return f'SolventsBook({len(self._solvents)})'

    def is_solvent(self, mol: 'hp.Molecule'):
        """ to judge whether a molecule is a solvent """
        return mol.smiles in self._sols_smi


import hotpot as hp

# initialization Chemical Library
library = Library()

# the public variable
__all__ = ['library']
