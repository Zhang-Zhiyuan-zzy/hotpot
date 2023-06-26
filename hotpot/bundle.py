"""
python v3.7.9
@Project: hotpot
@File   : bundle.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/22
@Time   : 3:18
"""
import copy
import random
from os import PathLike
from typing import *
from pathlib import Path
import numpy as np
from tqdm import tqdm
from openbabel import pybel as pb
import hotpot.cheminfo as ci
from hotpot.tools import check_path
import multiprocessing as mp

feature_formats = {
    'basic': ['atomic_number', 's', 'p', 'f']
}


class MolBundle:
    """"""

    def __init__(self, mols: Union[Sequence[ci.Molecule], Generator[ci.Molecule, None, None]] = None):
        self._data = {'mols': mols}

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(generator)" if isinstance(self.mols, Generator) else f"{class_name}({self.mols})"

    def __iter__(self):
        return iter(self.mols)

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
        return self.mols[item]

    def __get_all_unique_attrs(self, attr_name: str):
        """ Given a Molecule attr_name, get all unique values of the attributes among all Molecule in the MolBundle"""
        if self.is_generator:
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
        return self.__get_all_unique_attrs('atom_num')

    @property
    def atomic_numbers(self):
        return self.__get_all_unique_attrs('atomic_numbers')

    def choice(self, size: int = 1, replace: bool = True, p: Union[Sequence, float, Callable] = None) -> 'MolBundle':
        """
         Generate new MolBundle with a list of random Molecule objects from the current MolBundle
        Args:
            size: the size of generating MolBundle, that the number of contained Molecule objects
            replace: whether allow to choice a Molecule object multiply times
            p: If the current MolBundle is a Generator it should be a float or a Callable object
             which arg is the generated molecule. If the current MolBundle is a Sequence, the p
             should bea sequence with same size as the number of Molecule objects in current
             MolBundle, specify the probability of each Molecule to be chosen.

        Return:
            MolBundle contained specified-number molecules
        """
        def choice_from_generator():
            """ Generator molecule according to specified probability """
            if isinstance(p, float):
                for mol in self:
                    if random.choices([0, 1], [1-p, p]):
                        yield mol

            if isinstance(p, Callable):
                for mol in self:
                    if p(mol):
                        yield mol

        if self.is_generator:
            if isinstance(p, (float, Callable)):
                return MolBundle(choice_from_generator())
            else:
                raise TypeError('When the MolBundle is a generator, the p should be a float or Callable object')
        else:
            return MolBundle(np.random.choice(self.mols, size=size, replace=replace, p=p))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, dict):
            self._data = data
        else:
            raise TypeError(f'the {self.__class__.__name__}.data must be a dict')

    @classmethod
    def read_from(
            cls, fmt: str,
            dir_or_strings: Union[str, PathLike, Iterable[str]],
            match_pattern: str = '*',
            generate: bool = False,
            ranges: Union[Sequence[int], range] = None,
            condition: Callable = None,
            num_proc: int = None
    ):
        """
        Read Molecule objects from a directory.

        Args:
            fmt(str): read file with the specified-format method.
            dir_or_strings(str|PathLike): the directory all file put, or a sequence of string
            match_pattern(str): the file name pattern
            generate(bool): read file to a generator of Molecule object
            ranges(Sequence[int]|range): A list or range of integers representing the indices of
                the input files to read. Defaults to None.
            condition(Callable): A callable object that takes two arguments (the path of the input file
                and the corresponding Molecule object) and returns a boolean value indicating whether to include the
                Molecule object in the output. Defaults to None.
            num_proc: the number of process to read

        Returns:
            List(Molecule) or Generator(Molecule)
        """
        def read_mol(pm: Path, conn):

            try:
                mol = ci.Molecule.read_from(pm, fmt)

                # the OBMol, OBAtom, OBBond are C++ objects wrapped by SWIG,
                # they can't be pickle, so pop them from Molecule and convert their info to .mol2 string
                pmol = pb.Molecule(mol.ob_mol_pop())
                script = pmol.write('mol2')  # Write to the mol2 script to allow it to pickle

                # communicate with the main process
                conn.send((mol, script, pm))

            except AttributeError:
                conn.send((None, None, pm))

        def mol_generator():
            nonlocal dir_or_strings
            for i, path_mol in enumerate(generate_path_or_string()):

                if not ranges or i in ranges:
                    mol = ci.Molecule.read_from(path_mol, fmt)
                else:
                    continue

                if mol and (not condition or condition(path_mol, mol)):
                    yield mol

        def mol_mp_generator():
            mols_info = []

            parent_conn, child_conn = mp.Pipe()
            ps = []  # list of Process: Queue pairs

            for i, source in enumerate(generate_path_or_string()):

                # if the number of process more than num_proc, get the read molecule info and stop to start new process
                while len(ps) >= num_proc:
                    for p in ps:
                        if not p.is_alive():
                            mols_info.append(parent_conn.recv())
                            p.terminate()
                            ps.remove(p)

                # When received some Molecule info, reorganize these info and yield
                while mols_info:
                    mol, script, pf = mols_info.pop()  # hotpot Molecule, mol2_script, path_file of Molecule

                    # If you get a valid Molecule info, re-wrap to be hotpot Molecule
                    if mol and script:
                        pmol = pb.readstring('mol2', script)  # pybel Molecule object
                        mol.ob_mol_rewrap(pmol.OBMol)  # re-wrap OBMol by hotpot Molecule

                        # if the reorganized Molecule is expected, yield
                        if not condition or condition(mol, pf):
                            yield mol

                # Start new process to read Molecule from file
                if not ranges or i in ranges:
                    p = mp.Process(target=read_mol, args=(source, child_conn))
                    p.start()
                    ps.append(p)

            # After all path_file have pass into process to read
            while ps:
                for p in ps:
                    if not p.is_alive():
                        mols_info.append(parent_conn.recv())
                        p.terminate()
                        ps.remove(p)

            for mol, script, pf in mols_info:
                # if get a valid Molecule info, re-wrap to be hotpot Molecule
                if mol and script:
                    pmol = pb.readstring('mol2', script)  # pybel Molecule object
                    mol.ob_mol_rewrap(pmol.OBMol)  # re-wrap OBMol by hotpot Molecule

                    # if the reorganized Molecule is expected, yield
                    if not condition or condition(mol, pf):
                        yield mol

        def generate_path_or_string():
            """"""
            if isinstance(dir_or_strings, Path):
                for path in dir_or_strings.glob(match_pattern):
                    yield path

            elif isinstance(dir_or_strings, Iterable):
                for string in dir_or_strings:
                    yield string

            else:
                raise TypeError(f'the dir_or_strings is required to be a Path or str, get {type(dir_or_strings)}')

        if isinstance(dir_or_strings, str):
            dir_or_strings = Path(dir_or_strings)
        elif not isinstance(dir_or_strings, PathLike) and not isinstance(dir_or_strings, Iterable):
            raise TypeError(
                f'the read_dir should be a str, PathLike or iterable str, instead of {type(dir_or_strings)}'
            )

        generator = mol_mp_generator() if num_proc else mol_generator()

        if generate:
            return cls(generator)
        else:
            return cls([m for m in tqdm(generator, 'reading molecules')])

    def graph_representation(self, *feature_names) -> Generator[Union[str, np.ndarray, np.ndarray], None, None]:
        """ Transform molecules to their graph representation """
        for mol in self.mols:
            yield mol.graph_representation(*feature_names)

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
        Run Gaussian16 calculations for Molecule objects stored in the MolBundle.
        These Molecules are allowed to be perturbed their atoms' coordinates before submit to Gaussian 16

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
                    mol.perturb_atoms_coordinates(**pk)
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

    @property
    def is_generator(self):
        """ To judge weather the object is a Molecule generator """
        return isinstance(self.mols, Generator)

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

    @property
    def mols(self):
        return self._data.get('mols', [])

    @mols.setter
    def mols(self, mols):
        self._data['mols'] = mols

    def to_dpmd_sys(self, sys_root):
        """"""

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
        return MolBundle([m.to_mix_mol() if not isinstance(m, ci.MixSameAtomMol) else m for m in self])

    def to_mols(self):
        """
        Return a new MolBundle, in which MixSameAtomMol objects in container are converted to Molecule
        Returns:
            MolBundle(Molecule)
        """
        return MolBundle([m.to_mol() if isinstance(m, ci.MixSameAtomMol) else m for m in self])

    def unique_mols(self, mode: Literal['smiles', 'similarity'] = 'smiles'):
        """
        get a new Bundle with all unique Molecule objects
        Args:
            mode: the standard to identify whether two molecule to be regard as identical

        Returns:
            A new Bundle with all the unique Molecule objects
        """
        clone = copy.copy(self)
        clone.data = copy.copy(self.data)
        if mode == 'smiles':
            clone.mols = ({m.smiles: m for m in self.mols}.values())
            return clone
        elif mode == 'similarity':
            dict_mols = {}
            for mol in self.mols:
                mols_with_same_atom_num = dict_mols.setdefault(mol.atom_num, [])
                mols_with_same_atom_num.append(mol)

            new_mols = []
            for _, mols_with_same_atom_num in dict_mols.items():
                uni_mols = []
                for mol in mols_with_same_atom_num:
                    if mol not in uni_mols:
                        uni_mols.append(mol)

                new_mols.extend(uni_mols)

            clone.mols = new_mols

            return clone
