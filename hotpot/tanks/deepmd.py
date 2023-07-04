"""
python v3.9.0
@Project: hotpot
@File   : deepmd.py
@Author : Zhiyuan Zhang
@Date   : 2023/6/26
@Time   : 21:42
"""
import os
import json
from typing import *
from pathlib import Path
from copy import copy

import numpy as np

from hotpot import data_root
from hotpot.cheminfo import Molecule, periodic_table

# Manuscript training script
_script = json.load(open(os.path.join(data_root, 'deepmd_script.json')))


class DeepSystem:
    """
    a handle class to track DeepMD data
    Args:
        mol(Molecule):

    """

    required_items = ('coord', 'type')
    check_atom_num = ('coord', 'force', 'charge')
    share_same_conformers = ('type', 'coord', 'energy', 'force', 'charge', 'virial')
    need_reshape = ('coord', 'force')

    def __init__(self, mol: Molecule = None, data: dict = None):
        if mol:
            self.data = self._organize_data(mol)
            # check numpy.ndarray shape
            self._check_shape()
            # reshape array
            self._reshape_array()
        elif data:
            self.data = data
        else:
            raise ValueError('the args mol or data should be given at least one!')

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.data["coord"])})'

    def __call__(
            self,
            save_dir: Union[str, os.PathLike], mode: str = 'std',
            validate_ratio: Optional[float] = None,
            validate_dir: Union[str, os.PathLike] = None
    ):
        """
        Save the DeepMData to files
        Args:
            save_dir(str|os.PathLike|Path): the root dir for all corresponding DeepMDate system files,
             if the validate ratio is given, this the represent the training set save dir
            validate_ratio(float): the ratio of validate set, if not given, not split the dataset
            validate_dir: should be give when validate_ratio has been given, the root dir for validate data
        """
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)

        if validate_ratio:
            if not isinstance(validate_dir, (str, os.PathLike)):
                raise ValueError('the arguments validate_dir has not been given!')
            elif isinstance(validate_dir, str):
                validate_dir = Path(validate_dir)

            if not 0 < validate_ratio < 1:
                raise ValueError('the validate ratio should from 0 to 1')

            indices = np.arange(len(self))
            validate_idx = np.random.choice(indices, size=int(len(self) * validate_ratio), replace=False)
            training_idx = np.setdiff1d(indices, validate_idx)

            validate_data = self[validate_idx]
            training_data = self[training_idx]

            self._save_deep_md(training_data, save_dir, mode)
            self._save_deep_md(validate_data, validate_dir, mode)

        else:
            self._save_deep_md(self, save_dir, mode)

    def __getitem__(self, item: Union[int, slice, np.ndarray]):
        data = copy(self.data)
        if not isinstance(item, (int, slice, np.ndarray)):
            raise TypeError('the item should be int, slice or numpy.ndarray')

        for name in self.share_same_conformers:
            arrays = self.data.get(name)
            if isinstance(arrays, np.ndarray):
                data[name] = arrays[item]

        return self.__class__(data=data)

    def __getattr__(self, item: str):
        if item not in self.__dir__():
            raise AttributeError(f'the {self.__class__.__name__} not have attribute {item}')
        return self.data.get(item, None)

    def __dir__(self) -> Iterable[str]:
        return  [
            'type', 'type_map', 'nopbc', 'coord', 'box', 'energy', 'force', 'charge',
            'atom_counts','virial', 'atom_ener', 'atom_pref', 'dipole', 'atom_dipole',
            'polarizability', 'atomic_polarizability'
        ]

    def __len__(self):
        return len(self.data['coord'])

    def _check_shape(self):
        """ Check whether the shape ndarray is correct """
        conf_counts = len(self.data['coord'])
        atom_counts = self.data['atom_counts']

        for name in self.required_items:
            if self.data.get(name) is None:
                raise ValueError('the required composition to make the dpmd system is incomplete!')

        # Check whether the number of conformers are matching among data
        if any(len(self.data[n]) != conf_counts for n in self.share_same_conformers if self.data[n] is not None):
            raise ValueError('the number of conformers is not match')

        # Check whether the number of atoms in data are matching to the molecular atoms
        if any(self.data[n].shape[1] != atom_counts for n in self.check_atom_num if self.data[n] is not None):
            raise ValueError('the number of atoms is not matching the number of atom is the molecule')

    def _reshape_array(self):
        for name in self.need_reshape:
            item = self.data.get(name)
            if isinstance(item, np.ndarray):
                shape = item.shape

                assert len(shape) == 3

                self.data[name] = item.reshape((shape[0], shape[1] * shape[2]))

    @staticmethod
    def _organize_data(mol: Molecule) -> Dict[str, Any]:
        """ Organize the conformer data to a dict """
        conf_num = len(mol.all_coordinates)
        crystal = mol.crystal()
        if crystal:
            box = mol.crystal().vector  # angstrom
            is_periodic = True
        else:
            box = np.zeros((3, 3))
            for i in range(3):
                box[i, i] = 100.
            is_periodic = False
        box = box.reshape(-1, 9).repeat(conf_num, axis=0)

        return {
            'type': mol.atomic_numbers_array,  # matrix of (conformer_counts, atom_counts)
            'type_map': ['-'] + list(periodic_table.symbols),
            'nopbc': not is_periodic,
            'coord': mol.all_coordinates,  # angstrom,
            'box': box,
            'energy': mol.all_energy,  # eV
            'force': mol.all_forces,  # Hartree/Bohr,
            'charge': mol.all_atom_charges,  # q
            'atom_counts': mol.atom_counts,
            'virial': None,
            'atom_ener': None,
            'atom_pref': None,
            'dipole': None,
            'atom_dipole': None,
            'polarizability': None,
            'atomic_polarizability': None
        }

    @staticmethod
    def _save_deep_md(system: 'DeepSystem', save_dir: Path, mode: str):
        """ Save DeepMData to dir """
        if not save_dir.exists():
            save_dir.mkdir()

        # the dir of set data
        set_root = save_dir.joinpath('set.000')
        if not set_root.exists():
            set_root.mkdir()

        for name, value in system.data.items():

            # if the value is None, go to next
            if value is None:
                continue

            # Write the type raw
            if name == 'type':
                if mode == 'std':
                    type_raw = value[0]
                elif mode == 'att':
                    type_raw = np.zeros(value[0].shape, dtype=int)
                    np.save(set_root.joinpath("real_atom_types.npy"), value)
                else:
                    raise ValueError('the mode just allows to be "std" or "att"')

                with open(save_dir.joinpath('type.raw'), 'w') as writer:
                    writer.write('\n'.join([str(i) for i in type_raw]))

            elif name == 'type_map':
                with open(save_dir.joinpath('type_map.raw'), 'w') as writer:
                    writer.write('\n'.join([str(i) for i in value]))

            # Create an empty 'nopbc', when the system is not periodical
            elif name == 'nopbc' and value is True:
                with open(save_dir.joinpath('nopbc'), 'w') as writer:
                    writer.write('')

            # Save the numpy format data
            elif isinstance(value, np.ndarray):
                np.save(str(set_root.joinpath(f'{name}.npy')), value)


def make_script():
    """"""
