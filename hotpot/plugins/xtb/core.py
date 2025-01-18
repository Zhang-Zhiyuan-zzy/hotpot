import os
import sys
import stat
import json
from glob import glob
import os.path as osp
from typing import Optional
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
    """"""
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

    def _write_charge_unpair(self):
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

    def _write_mol(self):
        mol_path = osp.join(self.work_dir, 'struct.mol')
        self.mol.write(mol_path, overwrite=True)
        return mol_path

    def run(self):
        """"""
        os.chdir(self.work_dir)
        if self.mol is None:
            raise AttributeError("The XtbCalculator object need a Molecule to perform calculation.")

        mol_path = self._write_mol()
        if not isinstance(self.charge, int) or not isinstance(self.unpair, int):
            self.set_mol_charge_unpairEs()
        self._write_charge_unpair()

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
):
    if not options:
        options = ["--ohess"]

    if not osp.exists(res_file_dir):
        os.mkdir(res_file_dir)

    for mol_file in tqdm(glob(osp.join(mol_file_dir, mol_file_pattern))):
        reader = hp.MolReader(mol_file, fmt=format_)
        stem = '.'.join(osp.basename(mol_file).split('.')[:-1])
        for i, mol in enumerate(reader):
            work_dir = osp.join(res_file_dir, f'{stem}_{i}')
            if not osp.exists(work_dir):
                os.mkdir(work_dir)

            calculator = XtbCalculator(work_dir, xtb_executable=xtb_executable)
            calculator.mol = mol

            calculator.set_mol_charge_unpairEs()
            calculator.options = calculator.options + options

            calculator.run()
