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

import hotpot as hp
from hotpot.cheminfo import Molecule, Atom
from hotpot.search import SubstructureSearcher, MatchedMol


substituent_root = os.path.join(hp.data_root, 'substituents.json')


# class _SubstituentRegistry:
#     """ used to register the defined the Substituent classes """
#     def __init__(self):
#         self._sheet = {}
#
#     def register(self, cls: type):
#         """"""
#         self._sheet[cls.__name__] = cls
#         return cls
#
#
# register = _SubstituentRegistry()


class Substituent(ABC):
    """ the base class for all Substituent class """
    _check_func_matcher = re.compile(r"_check_\w+")  # Applied in method _init_check

    _registry_sheet = {}

    def __init__(
            self, name: str,
            substituent: Union[str, Molecule],
            plugin_atoms: list[Union[Atom, int]],
            socket_smarts: str = None,
            unique_mols: bool = True
    ):
        """
        Args:
            substituent(Molecule|str): the substituent fragment.
            plugin_atoms: the ob_id of atoms which will work which the main or framework molecule atoms
        """
        self.name = name
        self.unique_mols = unique_mols

        if isinstance(substituent, Molecule):
            self.substituent = substituent
        elif isinstance(substituent, str):
            self.substituent = Molecule.read_from(substituent, "smi")
        else:
            raise TypeError('the substituent should be a Molecule or a SMILES string')

        # Get the ob_id of plugin_atoms in the substituent Molecule.
        self.plugin_atoms = [self.substituent.atom(pa).ob_id for pa in plugin_atoms]

        if socket_smarts:
            self.socket_searcher = SubstructureSearcher(socket_smarts)
        elif self.default_socket_smarts:
            self.socket_searcher = SubstructureSearcher(self.default_socket_smarts)
        else:
            self.socket_searcher = None

        self._init_check()

    def __call__(self, frame_mol: Molecule, specified_socket_atoms: list[int, str, Atom] = None) -> list[Molecule]:
        """"""
        # TODO: This operation may should be performing in the next loop.
        list_socket_atoms = self.socket_atoms_search(frame_mol, specified_socket_atoms)

        substituted_mols = []
        for socket_atoms in list_socket_atoms:
            clone_mol = frame_mol.copy()
            self.substitute(clone_mol, socket_atoms)

            substituted_mols.append(clone_mol)

        if self.unique_mols:
            return self.make_mols_unique(*substituted_mols)
        else:
            return substituted_mols

    def _init_check(self):
        """ Check if the defined substitute are reasonable """
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
    def filter_out_unreasonable_struct(list_mols):
        """"""
        new_list_mol = []
        for mol in list_mols:
            mol.add_hydrogens()
            mol.build_3d()

            if not mol.has_nan_coordinates:
                new_list_mol.append(new_list_mol)

        return new_list_mol

    @staticmethod
    def mol_post_transform(frame_mol: Molecule):
        return frame_mol

    @staticmethod
    def mol_pre_transform(frame_mol: Molecule):
        return frame_mol

    @classmethod
    def read_from(cls, file_path: Union[str, os.PathLike] = None) -> Generator["Substituent", None, None]:
        """ Constructing Substituent by a json file, return a generator """
        if not file_path:
            file_path = substituent_root

        data = json.load(open(file_path))
        for name, items in data.items():
            type_name
            fragment_mol = Molecule.read_from(items['fragment_smiles'], 'smi')
            plugin_atoms = [fragment_mol.atom(ob_id) for ob_id in items["plugin_atoms_id"]]
            socket_smarts = items('socket_smarts')

            yield cls(name, fragment_mol, plugin_atoms, socket_smarts)

    @classmethod
    def register(cls, substituent_type: type):
        """ register the children class of Substituent, applied as a decorator """
        if not issubclass(cls, substituent_type):
            raise TypeError('the registered item must be subclass of Substituent!')

        cls._registry_sheet[substituent_type.__name__] = substituent_type

        return substituent_type

    @staticmethod
    def make_mols_unique(*mols):
        """"""
        unique_mols = []
        for mol in mols:
            if mol not in unique_mols:
                unique_mols.append(mol)

        return unique_mols

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
            return [[atom.ob_id for atom in hit.matched_atoms()] for hit in matched_mol]

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
        except (FileExistsError, json.decoder.JSONDecodeError):
            data = {}

        socket_smarts = self.socket_searcher.smarts if self.socket_searcher else None

        data.update({
            self.name: {
                "type": self.__class__.__name__,
                "fragment_smiles": self.substituent.smiles,
                "plugin_atom_id": [pa_ob_id for pa_ob_id in self.plugin_atoms],
                "socket_smarts": socket_smarts
            }
        })

        json.dump(data, open(save_path, 'w'), indent=True)


@Substituent.register
class NodeSubstituent(Substituent):
    """ the Substituent to perform the Node substituent """
    def substitute(self, frame_mol: Molecule, socket_atoms: list[int] = None):
        """"""


@Substituent.register
class EdgeJoinSubstituent(Substituent):
    """ The substitution is performed by joining the plugin edge (two atoms) with the socket edges in the frame mol """
    def substitute(self, frame_mol: Molecule, socket_atoms_oid: list[int] = None):
        # Check whether the length of socket atoms is 2
        if len(socket_atoms_oid) != 2:
            raise ValueError(f'the number of socket atoms should be 2, got {len(socket_atoms_oid)} atom indices')

        # frame_mol.remove_hydrogens()
        # self.substituent.remove_hydrogens()
        atoms_mapping, bonds_mapping = frame_mol.add_component(self.substituent)

        # Get the plugin atoms in the frame_mol after the substituent is added into frame_mol
        plugin_atoms_oid = [atoms_mapping[pa] for pa in self.plugin_atoms]
        p_atom1, p_atom2 = [frame_mol.atom(oid) for oid in plugin_atoms_oid]  # p_atom == plugin atom

        # Recording the neighbours of the s_atom (socket atom) and the type of the
        # bond between these neighbours and the s_atom.
        s_atom1, s_atom2 = frame_mol.atom(socket_atoms_oid[0]), frame_mol.atom(socket_atoms_oid[1])

        # A list composed of tuples, where the first item of each tuple is an atom
        # connected to the p_atom while not being the s_atom itself; the second item
        # is the bond type between the atom in the first item and the s_atom.
        bridge_to_s_atom1 = [
            (na, frame_mol.bond(na.ob_id, s_atom1.ob_id).type)
            for na in s_atom1.neighbours if na.ob_id != s_atom2.ob_id
        ]
        bridge_to_s_atom2 = [
            (na, frame_mol.bond(na.ob_id, s_atom2.ob_id).type)
            for na in s_atom2.neighbours if na.ob_id != s_atom1.ob_id
        ]

        logging.info(f"the socket bond is {frame_mol.bond(s_atom1, s_atom2)}")

        # switch the linkage of atoms which bond with the socket atoms, from the socket atoms to the plugin atoms
        frame_mol.remove_atoms(s_atom1, s_atom2)
        for na1, bond_type in bridge_to_s_atom1:
            frame_mol.add_bond(na1, p_atom1, bond_type)
        for na2, bond_type in bridge_to_s_atom2:
            frame_mol.add_bond(na2, p_atom2, bond_type)

        logging.info(f"the substitution in framework molecule {frame_mol} have been performed!")
