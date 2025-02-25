import os
import sys
import stat
import json
import random
from glob import glob, iglob
import os.path as osp
from typing import Optional, Literal, Callable
from tqdm import tqdm
import subprocess

import hotpot as hp


__xtb_dir__ = os.path.dirname(os.path.realpath(__file__))

# Examples
# >>> import hotpot as hp
# >>> from hotpot.plugins.xtb import XtbCalculator
# >>> mol = next(hp.MolReader('c1ccccc1C(=O)O[Sr]'))
# >>> calculator = XtbCalculator(
#         work_dir='path/to/an/empty/directory',
#         xtb_executable='path/to/xtb/executable'
# )
# >>> calculator.mol = mol  # add hotpot.Molecule object
#
# Set the molecule charges or number of unpair electrons
# >>> calculator.charge = 1
# >>> calculator.unpair = 0
# Or, you can assign a default charge and unpair value by calling:
# >>> calculator.set_mol_charge_unpairEs()
#
# Performing XTB calculation
# >>> res = calculator.run()
# >>> print(res.stdout)  # print results


class XtbCalculator(object):
    """
    Calculator class interfacing with the xTB executable for molecular calculations.

    This class provides methods and functionalities for running calculations using the
    xTB executable. It supports setting options, initializing molecular properties
    (charge and unpaired electrons), and executing the calculations while managing the
    required files and configurations.

    Attributes:
        work_dir (str): Directory where all calculation-related files are stored.
        xtb_executable (str): Path to the xTB executable to use for calculations.
        mol (Molecule): The molecular structure for which calculations are performed.
        charge (int): The molecular charge for the calculation.
        unpair (int): The number of unpaired electrons in the molecule.
        options (list of str): Additional options passed to the xTB executable.
    """
    def __init__(
            self,
            work_dir: str,
            xtb_executable: str = None,
    ):
        self.work_dir = work_dir
        self.xtb_executable = self._get_xtb_root(xtb_executable)
        self.mol = None
        self.charge = None
        self.unpair = None

        self.options = []

    def clear_options(self):
        self.options = []

    def set_opt(self):
        self.options.extend('--opt')

    def set_mol_charge_unpairEs(self, charge: Optional[int] = None, unpair: Optional[int] = None):
        if isinstance(charge, int):
            self.charge = charge
        elif not isinstance(self.charge, int):
            self.charge = self.mol.calc_mol_default_charge()

        electrons_num = sum([a.atomic_number for a in self.mol.atoms]) - self.charge
        if isinstance(self.unpair, int):
            self.unpair = unpair
        elif not isinstance(self.unpair, int):
            self.unpair = electrons_num % 2

        if (self.unpair % 2) ^ (electrons_num % 2):
            raise ValueError(f'Molecule with total electrons {electrons_num} and unpaired electrons {self.unpair}'
                             'is not possible !!')

    def write_charge_unpair(self):
        with open(osp.join(self.work_dir, '.CHRG'), 'w') as writer:
            writer.write(str(self.charge))
        with open(osp.join(self.work_dir, '.UHF'), 'w') as writer:
            writer.write(str(self.unpair))

    @staticmethod
    def _get_xtb_root(xtb_executable):
        with open(osp.join(__xtb_dir__, '.cache.json')) as f:
            cache = json.load(f)

        if xtb_executable is None:
            old_executable = cache.get('executable')
            if not old_executable:
                raise ValueError('xtb executable not found !!')

            path_executable = None
            for p_exe in old_executable:
                if osp.exists(p_exe) and osp.isfile(p_exe):
                    st = os.stat(p_exe)
                    if bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                        path_executable = p_exe
                        break

            if path_executable is None:
                raise ValueError('xtb executable not found !!')

            return path_executable

        else:
            if not osp.exists(xtb_executable):
                raise ValueError('Given xtb executable is not exists !!')
            if not osp.isfile(xtb_executable):
                raise ValueError('Given xtb executable is not a file !!')
            st = os.stat(xtb_executable)
            if not bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                raise ValueError('Given xtb executable is not a executable file !!')

            list_exe = cache.setdefault('executable', [])
            list_exe.append(xtb_executable)

            script = json.dumps(cache, indent=4)
            with open(osp.join(__xtb_dir__, '.cache.json'), 'w') as writer:
                writer.write(script)

            return xtb_executable

    def write_mol(self):
        mol_path = osp.join(self.work_dir, 'struct.mol')
        self.mol.write(mol_path, overwrite=True)
        return mol_path

    def run(self, perform: bool = True):
        """
        perform (bool, optional): Whether to perform xTB computations.
        If False, only prepare perform file (mol file, .CHRG, .UHF) in dir. Defaults to True.
        """
        os.chdir(self.work_dir)
        if self.mol is None:
            raise AttributeError("The XtbCalculator object need a Molecule to perform calculation.")

        mol_path = self.write_mol()
        if not isinstance(self.charge, int) or not isinstance(self.unpair, int):
            self.set_mol_charge_unpairEs()
        self.write_charge_unpair()

        if perform:
            cmd = [self.xtb_executable, mol_path] + self.options
            results = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with open('stdout.log', 'w') as writer:
                writer.write(results.stdout.decode())
            with open('stderr.log', 'w') as writer:
                writer.write(results.stderr.decode())

            return results



def xtb_batch_run(
        mol_file_dir: str,
        res_file_dir: str,
        mol_file_pattern: str = "*.mol2",
        format_: str = None,
        options: Optional[list] = None,
        xtb_executable: str = None,
        perform: bool = True,
        which: Literal['all', 'first', 'random'] = 'all',
        item_num: int = 100,
        batch_size: Optional[int] = None,
        mol_filter: Callable = None,
        **kwargs
):
    """
    Executes batch xTB calculations on molecular files.

    This function performs xTB computations on all molecular files in the provided
    directory that match the given file pattern. It creates separate working
    directories for each file, processes the molecules using the xTB executable,
    and stores the results in the specified output directory.

    Parameters:
        mol_file_dir (str): Path to the directory containing the molecular files.
        res_file_dir (str): Path to the directory where results should be stored.
        mol_file_pattern (str): File pattern to match molecular files. Defaults
            to "*.mol2".
        format_ (str): Format of the input molecular files. Defaults to None.
        options (list, optional): List of additional options for the xTB command.
            Defaults to ["--ohess"] if not provided.
        xtb_executable (str, optional): Path to the xTB executable. Defaults
            to None.
        perform (bool, optional): Whether to perform xTB computations.
        If False, only prepare perform file (mol file, .CHRG, .UHF) in dir. Defaults to True.
        which ('all', 'first', 'random'): This argument is used to test. When 'all' is specified,
            all files in the mol_file_dir are processed. When 'first' is specified, only the first
            `item_num` mol will be processed. When 'random' is specified, only the `item_num` files
            randomly chosen from mol_file_dir will be processed.
        item_num (int, optional): The number of items to process. Defaults to 100. This argument
            only works when `which` is 'random' or 'first'.
        batch_size (int, optional): If an integer is given, split result into batches dir.
        mol_filter: Passing a Callable object, for a molecule willing to perform calculations,
            return a True, otherwise False. Defaults to None.

    Returns:
        None
    """
    if not options:
        options = ["--ohess"]

    if not osp.exists(res_file_dir):
        os.mkdir(res_file_dir)

    files = list(tqdm(iglob(osp.join(mol_file_dir, mol_file_pattern)), 'Checking mol files'))
    if which == 'first':
        files = files[:item_num]
    elif which == 'random':
        files = random.sample(files, item_num)

    if perform:
        p_bar_desc = "XTB Calculations"
    else:
        p_bar_desc = "Export XTB calculations files"

    count = 0
    if isinstance(batch_size, int) and batch_size > 1:
        batch_dir = f'0-{batch_size-1}'
        if not osp.exists(osp.join(res_file_dir, batch_dir)):
            os.mkdir(osp.join(res_file_dir, batch_dir))
    else:
        batch_dir = None


    for mol_file in tqdm(files, desc=p_bar_desc):
        reader = hp.MolReader(mol_file, fmt=format_)
        stem = '.'.join(osp.basename(mol_file).split('.')[:-1])
        for i, mol in enumerate(reader):

            if isinstance(mol_filter, Callable) and not mol_filter(mol):
                continue

            if batch_dir:
                work_dir = osp.join(res_file_dir, batch_dir, f'{stem}_{i}')
            else:
                work_dir = osp.join(res_file_dir, f'{stem}_{i}')

            if not osp.exists(work_dir):
                os.mkdir(work_dir)

            calculator = XtbCalculator(work_dir, xtb_executable=xtb_executable)
            calculator.mol = mol
            calculator.charge = kwargs.get('charge', None)

            # calculator.set_mol_charge_unpairEs()
            calculator.options = calculator.options + options

            calculator.run(perform=perform)

            count += 1
            if isinstance(batch_size, int) and batch_size > 1 and count % batch_size == 0:
                batch_dir = f'{count}-{count + batch_size-1}'
                if not osp.exists(osp.join(res_file_dir, batch_dir)):
                    os.mkdir(osp.join(res_file_dir, batch_dir))

