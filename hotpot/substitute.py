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

import numpy as np
import openbabel.openbabel as ob

import hotpot as hp
from hotpot.cheminfo import Molecule, Atom
from hotpot.search import SubstructureSearcher, MatchedMol

substituent_root = os.path.join(hp.data_root, 'substituents.json')

_ob_builder = ob.OBBuilder()


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
            self, name: str, subst_can_smi: str, plugin_atoms: list[int],
            socket_smarts: str = None, unique_mols: bool = True
    ):
        """
        Args:
            subst_can_smi(Molecule|str): the canonical SMILES of substituent fragment.
            plugin_atoms: the ob_id of atoms which will work which the main or framework molecule atoms
        """
        self.name = name
        self.unique_mols = unique_mols

        if isinstance(subst_can_smi, str):
            self.substituent = Molecule.read_from(subst_can_smi, "smi")
            # Check whether the given SMILES is a canonical SMILES
            if self.substituent.smiles != subst_can_smi:
                raise ValueError(f"the given SMILES {subst_can_smi} is non-canonical, "
                                 f"the correct writing should be {self.substituent.smiles}")
        else:
            raise TypeError('the substituent should be a Molecule or a SMILES string')

        # Build 3d structure
        self.substituent.build_3d()

        self.plugin_atoms = plugin_atoms

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
        list_socket_atoms_ob_id = self.socket_atoms_search(frame_mol, specified_socket_atoms)

        substituted_mols = []
        for socket_atoms_ob_id in list_socket_atoms_ob_id:
            clone_mol = frame_mol.copy()

            self.determine_generations(clone_mol, socket_atoms_ob_id)

            self.substitute(clone_mol, socket_atoms_ob_id)

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

    def determine_generations(self, frame_mol: Molecule, socket_atoms_ob_id: list[int]):
        """ determine the generations of the atom in the substituent """
        max_gens = max(frame_mol.atoms_dict[ob_id].generations for ob_id in socket_atoms_ob_id)
        logging.info(f"the max generations is {max_gens}")
        for atom in self.substituent.atoms:
            atom.generations = max_gens + 1
            logging.info(f"Generations of {atom} is {atom.generations}")

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

    @staticmethod
    def read_from(file_path: Union[str, os.PathLike] = None) -> Generator["Substituent", None, None]:
        """ Constructing Substituent by a json file, return a generator """
        if not file_path:
            file_path = substituent_root

        data = json.load(open(file_path))
        for name, items in data.items():
            cls = Substituent._registry_sheet[items['type']]
            frag_can_smi = items['fragment_smiles']
            plugin_atoms = items["plugin_atom_id"]
            socket_smarts = items['socket_smarts']

            yield cls(name, frag_can_smi, plugin_atoms, socket_smarts)

    @classmethod
    def register(cls, substituent_type: type):
        """ register the children class of Substituent, applied as a decorator """
        if not issubclass(substituent_type, cls):
            raise TypeError('the registered item must be subclass of Substituent!')

        cls._registry_sheet[substituent_type.__name__] = substituent_type

        return substituent_type

    @staticmethod
    def make_mols_unique(*mols):
        """"""
        # do job same as a dict, but the key (i.e., Molecule) are not hashable objects.
        key_mols, value_mols = [], []
        for mol in mols:
            try:
                idx = key_mols.index(mol)
            except ValueError:
                idx = len(value_mols)
                value_mols.append([])

            value_mols[idx].append(mol)

        unique_mols = []
        for j, mols in enumerate(value_mols):
            disorder_bond_counts = []
            for i, mol in enumerate(mols):
                if not mol.is_disorder:
                    unique_mols.append(mol)
                    break

                disorder_bond_counts.append(len(mol.disorder_bonds))

            if len(unique_mols) == j:
                unique_mols.append(mols[np.argmin(disorder_bond_counts)])
            elif len(unique_mols) != j + 1:
                raise RuntimeError("Some Error happen, check the above loop!!!")

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
        """
        Store the substituent info to json file
        Args:
            save_path: the path of saved json file
        """
        try:
            with open(save_path) as file:
                data = json.load(file)
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

        with open(save_path, 'w') as writer:
            json.dump(data, writer, indent=True)

    def zero_generations(self):
        """ Zero generations for all atoms in the substituent """
        for atom in self.substituent.atoms:
            atom.generations = 0


@Substituent.register
class NodeSubstituent(Substituent):
    """ the Substituent to perform the Node substituent """
    def substitute(self, frame_mol: Molecule, socket_atoms: list[int] = None):
        """"""


@Substituent.register
class EdgeSubst(Substituent):
    """ The substitution is performed by joining the plugin edge (two atoms) with the socket edges in the frame mol """
    def substitute(self, frame_mol: Molecule, socket_atoms_oid: list[int] = None):
        # Check whether the length of socket atoms is 2
        if len(socket_atoms_oid) != 2:
            raise ValueError(f'the number of socket atoms should be 2, got {len(socket_atoms_oid)} atom indices')

        # if not frame_mol.has_3d:
        frame_mol.build_3d()

        frame_mol.remove_hydrogens()
        self.substituent.remove_hydrogens()
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

        # frame_mol.remove_atoms()
        # frame_mol.add_hydrogens()
        frame_mol.localed_optimize(to_optimal=True)
