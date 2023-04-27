"""
python v3.7.9
@Project: hotpot
@File   : bundle.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/22
@Time   : 3:18
"""
import copy
from os import PathLike
from typing import *
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import src.cheminfo as ci
from src.tools import check_path


# Typing AnnotationsG
GraphFormatName = Literal['Pytorch', 'numpy']

feature_formats = {
    'basic': ['atomic_number', 's', 'p', 'f']
}


class MolBundle:
    """"""

    def __init__(self, mols: Union[list[ci.Molecule], Generator[ci.Molecule, None, None]] = None):
        self.mols = mols
        self._data = {}

    def __iter__(self):
        return iter(tqdm(self.mols, 'Process Molecule'))

    def __add__(self, other: Union['MolBundle', ci.Molecule]):
        """"""
        if isinstance(other, MolBundle):
            return MolBundle(self.to_list() + other.to_list())
        elif isinstance(other, ci.Molecule):
            return MolBundle(self.to_list() + [other])
        else:
            raise TypeError('the MolBundle is only allowed to add with Molecule or MolBundle object')

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, item: int):
        return self.mols[int]

    def __bundle_mol_indices_with_same_attr(self, attr_name: str):
        self.to_list()

        dict_attrs = {}
        for i, mol in enumerate(self):
            list_indices = dict_attrs.setdefault(getattr(mol, attr_name), [])
            list_indices.append(i)

        if not dict_attrs:
            return 0
        if len(dict_attrs) == 1:
            return getattr(self.mols[0], attr_name)
        else:
            return dict_attrs

    @property
    def atom_num(self) -> Dict[int, List[int]]:
        """
        Notes:
            if the Bundle is a generator, convert to a list of Molecule first.
        Returns:
            returns a dict with the key is the number of the atoms and the key is the indices of Molecules
        """
        return self.__bundle_mol_indices_with_same_attr('atom_num')

    @property
    def atomic_numbers(self):
        return self.__bundle_mol_indices_with_same_attr('atomic_numbers')

    @classmethod
    def read_from_dir(
            cls, fmt: str,
            read_dir: Union[str, PathLike],
            match_pattern: str = '*',
            generate: bool = False,
            ranges: Union[Sequence[int], range] = None,
            condition: Callable = None
    ):
        """
        Read Molecule objects from a directory.

        Args:
            fmt(str): read file with the specified-format method.
            read_dir(str|PathLike): the directory all file put
            match_pattern(str): the file name pattern
            generate(bool): read file to a generator of Molecule object
            ranges(Sequence[int]|range): A list or range of integers representing the indices of
                the input files to read. Defaults to None.
            condition(Callable): A callable object that takes two arguments (the path of the input file
                and the corresponding Molecule object) and returns a boolean value indicating whether to include the
                Molecule object in the output. Defaults to None.

        Returns:

        """
        def mol_generator():
            nonlocal read_dir

            if isinstance(read_dir, str):
                read_dir = Path(read_dir)
            elif not isinstance(read_dir, PathLike):
                raise TypeError(f'the read_dir should be a str or PathLike, instead of {type(read_dir)}')

            processes = []
            for i, path_mol in enumerate(read_dir.glob(match_pattern)):

                if not ranges or i in ranges:
                    mol = ci.Molecule.read_from(path_mol, fmt)
                else:
                    continue

                if mol and (not condition or condition(path_mol, mol)):
                    yield mol

        if generate:
            return cls(mol_generator())
        else:
            return cls([m for m in tqdm(mol_generator(), 'reading molecules')])

    def graph_represent(self, graph_fmt: GraphFormatName, *feature_type):
        """ Transform molecules to the molecule to graph representation,
        the transformed graph with 'numpy.ndarray' or 'PyTorch.Tensor' format """
        feature_matrices = []
        for mol in self.mols:
            feature_matrix = mol.feature_matrix(feature_names=feature_type)
            feature_matrices.append(feature_matrix)

        if graph_fmt == 'numpy':
            return feature_matrices
        elif graph_fmt == 'Pytorch':
            return [torch.tensor(matrix) for matrix in feature_matrices]

    def gaussian(
            self, g16root: Union[str, PathLike], dir_out: Union[str, PathLike],
            link0: Union[str, List[str]], route: Union[str, List[str]],
            dir_err: Optional[Union[str, PathLike]] = None,
            dir_chk: Optional[Union[str, PathLike]] = None,
            clean_configure: bool = True,
            perturb_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            *args, **kwargs
    ) -> None:
        """
        Run Gaussian16 calculations on the Molecule objects.

        Args:
            g16root (Union[str, PathLike]): The path to the Gaussian16 root directory.
            dir_out (Union[str, PathLike]): The path to the directory to output the log files.
            link0 (Union[str, List[str]]): The link0 information for Gaussian16 calculations.
            route (Union[str, List[str]]): The route information for Gaussian16 calculations.
            dir_err (Optional[Union[str, PathLike]], optional): The path to the directory to output the error files.
                Defaults to None.
            dir_chk (Optional[Union[str, PathLike]], optional): The path to the directory to store the .chk files.
                Defaults to None.
            clean_configure (bool, optional): A flag indicating whether to clean the configuration before perturbing
                the molecule or lattice. Defaults to True.
            perturb_kwargs (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional): The parameters for
                perturbing the molecule or lattice. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Check and process paths
        g16root: Path = check_path(g16root, file_or_dir='dir')
        dir_out: Path = check_path(dir_out, mkdir=True)
        dir_err: Optional[Path] = check_path(dir_err, mkdir=True)
        dir_chk: Optional[Path] = check_path(dir_chk, mkdir=True)

        for mol in self.mols:
            assert isinstance(mol, ci.Molecule)

            # Clean before perturb configures
            if clean_configure:
                mol.clean_configures()

            # Assign the dirs
            # assign the dir to put out.log files for the mol
            dir_out_mol = dir_out.joinpath(mol.identifier)
            if not dir_out_mol.exists():
                dir_out_mol.mkdir()

            # assign the dir to put err.log files for the mol
            if dir_err:
                dir_err_mol = dir_err.joinpath(mol.identifier)
                if not dir_err_mol.exists():
                    dir_err_mol.mkdir()
            else:
                dir_err_mol = None

            # Performing the molecule or lattice perturb
            if isinstance(perturb_kwargs, Dict):
                perturb_kwargs = [perturb_kwargs]  # Wrap it into a list

            if isinstance(perturb_kwargs, List) and all(isinstance(pk, dict) for pk in perturb_kwargs):
                for pk in perturb_kwargs:
                    pk['inplace'] = True  # Force to inplace
                    mol.perturb_mol_lattice(**pk)
            elif perturb_kwargs is not None:
                ValueError('The perturb_kwargs should be a dict or list of dict')

            # Running the gaussian16
            for config_idx in range(mol.configure_number):
                mol.configure_select(config_idx)

                # Reorganize the arguments for each configures
                path_out = dir_out_mol.joinpath(f'{config_idx}.log')
                path_err = dir_err_mol.joinpath(f'{config_idx}.err') if dir_err else None

                if dir_chk:  # for dir_chk
                    dir_chk_mol = dir_chk.joinpath(mol.identifier)
                    dir_chk_mol.mkdir(exist_ok=True)

                    if isinstance(link0, str):
                        link0_cfg = [f'chk={str(dir_chk_mol)}/{config_idx}.chk', link0]
                    elif isinstance(link0, list):
                        link0_cfg = copy.copy(link0)
                        link0_cfg.insert(0, f'chk={str(dir_chk_mol)}/{config_idx}.chk')
                    else:
                        TypeError('the link0 should be str or list of str!')
                else:
                    link0_cfg = link0

                mol.gaussian(
                    g16root=g16root,
                    link0=link0_cfg,
                    route=route,
                    path_log_file=str(path_out.absolute()),
                    path_err_file=str(path_err) if path_err else None,
                    *args, **kwargs
                )

    def merge_conformers(self):
        """
        Get the sum of conformers for all molecule in the mol bundle "self.mols"
        This method can only be successfully executed
        when all molecules in the molecular bundle can be added to each other
        Returns:
            a Molecule object with all of conformers in the self.mols
        """
        atomic_numbers = self.atomic_numbers

        if isinstance(atomic_numbers, tuple):
            return sum(self.mols[1:], start=self.mols[0])
        elif isinstance(atomic_numbers, dict):
            mol_array = np.array(self.mols)
            return MolBundle([mol_array[i].sum() for ans, i in self.atomic_numbers.items()])

    def merge_atoms_same_mols(self):
        """ Merge Molecules with same atoms to a MixSameAtomMol """
        bundle = self.to_mix_mols()
        atom_num = bundle.atom_num

        if isinstance(atom_num, tuple):
            return sum(bundle.mols[1:], start=bundle.mols[0])
        elif isinstance(atom_num, dict):
            mol_array = np.array(bundle.mols)
            return MolBundle([mol_array[i].sum() for ans, i in atom_num.items()])

    def to_list(self) -> List[ci.Molecule]:
        """ Convert the molecule container (self.mol) to list """
        if isinstance(self.mols, Generator):
            self.mols = list(self)

        return self.mols

    def to_mix_mols(self):
        """
        Return a new MolBundle, in which Molecule objects in container are converted to MixSameAtomMol
        Returns:
            MolBundle(MixSameAtomMol)
        """
        return MolBundle([m if isinstance(m, ci.Molecule) else m.to_mix_mols() for m in self])

    def to_mols(self):
        """
        Return a new MolBundle, in which MixSameAtomMol objects in container are converted to Molecule
        Returns:
            MolBundle(Molecule)
        """
        return MolBundle([m if isinstance(m, ci.MixSameAtomMol) else m.to_mols() for m in self])
