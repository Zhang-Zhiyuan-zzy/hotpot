"""
python v3.9.0
@Project: hotpot
@File   : substitute
@Auther : Zhiyuan Zhang
@Data   : 2023/9/14
@Time   : 10:43

This module to:
    1) define Substituent, a molecular fragment prepared to replace a substructure in
    a Molecule such as a hydrogen or a methyl.

    2) give functions to perform substitute a substructure in a Molecule to be Substitute.

    3) the format to save the Substitute data to disk.
"""
import os
import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Union, Generator

from hotpot.cheminfo import Molecule, Atom
from hotpot.search import SubstructureSearcher, MatchedMol


class Substituent(ABC):
    """ the base class for all Substituent class """
    _check_func_matcher = re.compile(r"_check_\w+")  # Applied in method _init_check

    def __init__(self, name: str, substituent: Molecule, plugin_atoms: list[Atom], socket_smarts: str = None):
        """
        Args:
            substituent: the substituent fragment.
            plugin_atoms: the ob_id of atoms which will work which the main or framework molecule atoms
        """
        self.name = name
        self.substituent = substituent
        self.plugin_atoms = plugin_atoms
        if socket_smarts:
            self.socket_searcher = SubstructureSearcher(socket_smarts)
        elif self.default_socket_smarts:
            self.socket_searcher = SubstructureSearcher(self.default_socket_smarts)
        else:
            self.socket_searcher = None

        self._init_check()

    def __call__(self, frame_mol: Molecule, specified_socket_atoms: list[int, str, Atom] = None):
        """"""
        # TODO: This operation may should be performing in the next loop.
        list_socket_atoms = self.socket_atoms_search(frame_mol, specified_socket_atoms)

        substituted_mols = []
        for socket_atoms in list_socket_atoms:
            clone_mol = frame_mol.copy()
            self.substitute(clone_mol, socket_atoms)

            substituted_mols.append(clone_mol)

        return substituted_mols

    def _init_check(self):
        """ Check if the defined substitute are reasonable """
        # check whether the plugin atoms are in the substituent fragments
        for atom in self.plugin_atoms:
            self.substituent.atom(atom.ob_id)

        # Performing other custom checks in the substituent
        custom_checks = [name for name in dir(str) if self._check_func_matcher.match(name)]
        for check_func_name in custom_checks:
            check_func = getattr(self, check_func_name)
            check_func()

            logging.info(f"run check func {check_func_name}")

    @property
    def default_socket_smarts(self):
        return None

    @staticmethod
    def mol_post_transform(clone_mol: Molecule):
        return clone_mol

    @staticmethod
    def mol_pre_transform(clone_mol: Molecule):
        return clone_mol

    @classmethod
    def read_from(cls, file_path: Union[str, os.PathLike]) -> Generator["Substituent", None, None]:
        """ Constructing Substituent by a json file, return a generator """
        data = json.load(open(file_path))
        for name, items in data.items():
            fragment_mol = Molecule.read_from(items['fragment_smiles'], 'smi')
            plugin_atoms = [fragment_mol.atom(ob_id) for ob_id in items["plugin_atoms_id"]]
            socket_smarts = items('socket_smarts')

            yield cls(name, fragment_mol, plugin_atoms, socket_smarts)

    def socket_atoms_search(
            self, frame_mol: Molecule, specified_socket_atoms: list[int, str, Atom] = None
    ) -> list[list[int]]:
        """ searching out all socket atoms in the frame_mol """
        # if the socket_atoms are specified
        if specified_socket_atoms:
            return [[frame_mol.atom(atom).ob_id for atom in specified_socket_atoms]]

        # search the socket atoms by socket SMARTS pattern
        else:
            matched_mol: MatchedMol = self.socket_searcher.search(frame_mol)[0]
            return [[atom.ob_id for atom in hit.matched_atoms] for hit in matched_mol]

    @abstractmethod
    def substitute(self, frame_mol: Molecule, socket_atoms: list[int] = None):
        """
        Performing the substitute for the given frame_mol, and Return the copy of the frame_mol,
        where the Substitute have been grafted into it.
        """

    def writefile(self, save_path: Union[str, os.PathLike]):
        """"""
        try:
            data = json.load(open(save_path))
        except FileExistsError:
            data = {}

        socket_smarts = self.socket_searcher.smarts if self.socket_searcher else None

        data.update({
            self.name: {
                "fragment_smiles": self.substituent.smiles,
                "plugin_atom_id": [atom.ob_id for atom in self.plugin_atoms],
                "socket_smarts": socket_smarts
            }
        })

        json.dump(data, open(save_path, 'w'), indent=True)


class NodeSubstituent(Substituent):
    """ the Substituent to perform the Node substituent """
    def substitute(self, frame_mol: Molecule, socket_atoms: list[int] = None):
        """"""


class EdgeJoinSubstituent(Substituent):
    """ The substitution is performed by joining the plugin edge (two atoms) with the socket edges in the frame mol """
    def substitute(self, clone_mol: Molecule, socket_atoms: list[int] = None):
        # Check whether the length of socket atoms is 2
        if len(socket_atoms) != 2:
            raise ValueError(f'the number of socket atoms should be 2, got {len(socket_atoms)} atom indices')

        atoms_mapping, bonds_mapping = clone_mol.add_component(self.substituent)


class _Substituent:
    """"""
    def __init__(self, mol: Molecule, plugin_atoms: list[int]):
        """
        Args:
            mol: the molecular fragment.
            plugin_atoms: the ob_id of atoms which will work which the main or framework molecule atoms
        """
        self.mol = mol
        self.action_atoms = plugin_atoms
        self._check()

    def _check(self):
        """ Check if the defined substitute are reasonable """
        for ob_id in self.action_atoms:
            self.mol.atom(ob_id)

    def action_atoms_counts(self):
        return len(self.action_atoms)

    def build_3d(self):
        self.mol.build_3d()

    @property
    def has_3d(self):
        return self.mol.has_3d

    def unset_coordinates(self):
        self.mol.unset_coordinates()


def substitute(
        frame_mol: Molecule, substituent: Union[_Substituent, Molecule],
        socket_atoms: list[Union[int, str, Atom]] = None,
        socket_smarts: str = None,
        plugin_atoms: list[Union[int, str, Atom]] = None
):
    """
    Performing the molecule substitute, link the frame molecule with the Substitute by
    replace the replaced atoms in the framework molecule to the action atoms of Substitute
    Args:
        frame_mol(Molecule): framework molecule
        substituent(Molecule|Substituent): the molecule frame to substitute the atoms in the frame_mol
        socket_atoms(list[int|str|Atom]): the atoms in the frame_mol to be substituted.
        socket_smarts(str): the SMARTS string to define the patten of
        plugin_atoms(list[int|str|Atom]):

    Returns:
        the new framework Molecule after the substitution transform.
    """
    # Check arguments
    if isinstance(substituent, Molecule) and not plugin_atoms:
        raise ValueError('When a Molecule object is used as substituent, the action_atoms must be given')

    if socket_atoms:
        list_socket_atoms = [socket_atoms]
    elif socket_smarts:
        socket_searcher = SubstructureSearcher(socket_smarts)
        list_socket_atoms = socket_searcher.search(frame_mol)

        if not list_socket_atoms:
            raise ValueError("the given socket_smarts did not matched any atoms in the frame_mol")

    else:
        raise ValueError('the replace_atoms and replace_smarts should be given one at least')

    plugin_atoms = [frame_mol.atom(pa) for pa in plugin_atoms]

    if len(list_socket_atoms[0]) != len(plugin_atoms):
        raise ValueError("the number of socket atoms and the plugin atoms!")

    # Substitute
    if frame_mol.has_3d and not substituent.has_3d:
        substituent.build_3d()
