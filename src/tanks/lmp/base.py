"""
python v3.7.9
@Project: hotpot
@File   : base.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/24
@Time   : 3:43
"""
import os
import re
from os.path import join as ptj
import json

import numpy as np
from lammps import PyLammps
from typing import *
import src


class HpLammps:
    """
    A wrapper to run LAMMPS tasks
    """
    _dir_cmd: str = os.getcwd()

    def __init__(self, main, *args, **kwargs):
        """"""
        self.main: ci.Molecule = main
        self.pylmp = PyLammps()
        self._data = {
            'elements': main.elements
        }  # store any data
        self.args = args
        self.kwargs = kwargs

    def __call__(self, cmd: str):
        self.command(cmd)

    def __dir__(self) -> Iterable[str]:
        return self.pylmp.__dir__() + ["commands_list", "commands_string", 'read_main_data', "script"]

    def __getattr__(self, item):
        return self.pylmp.__getattr__(item)

    def data_to_type_map(self, script: str, offset: int = 0):
        """ Convert the LAMMPS data format script (string) to data map dict """
        pattern = re.compile(r"[A-Z].+s")
        data_body_headers = pattern.findall(script)
        masses_idx = data_body_headers.index('Masses')
        masses_body_contents: List[str] = pattern.split(script)[masses_idx+1].split('\n')

        type_map = self._data.setdefault('type_map', {})
        for line in masses_body_contents:
            line = line.strip()
            if line:
                type_num, _, _, type_label = line.split()
                type_num = int(type_num) + offset  # offset
                former_label = type_map.get(type_num)

                if not former_label:  # if the atom type have not been recorded
                    type_map[type_num] = type_label
                else:
                    # Never allow to change the type_map for defined atom type
                    # if the current type label is different from the former, raise error
                    # if the current type label is same with the former, keeping still.
                    if former_label != type_label:
                        raise RuntimeError(
                            f'the the type_map for atom type {type_num} is attempt to change '
                            f'from {former_label} to {type_label}, Never allowed!!'
                        )

    @property
    def box(self):
        return np.array([
            [self.eval('xlo'), self.eval('xhi')],
            [self.eval('ylo'), self.eval('yhi')],
            [self.eval('zlo'), self.eval('zhi')]
        ])

    def close(self):
        self.pylmp.close()
        del self

    def command(self, cmd: str):
        self.pylmp.command(cmd)

    def commands_list(self, list_cmd: List[str]):
        for cmd in list_cmd:
            self.command(cmd)

    def commands_string(self, multicod: str):
        cmd_list = multicod.split('\n')
        for cmd in cmd_list:
            self.command(cmd.strip())

    @property
    def computes(self):
        return self.pylmp.computes

    @property
    def cryst_matrix(self):
        xl = self.eval('xhi')-self.eval('xlo')  # length of the crystal box in the x axis
        yl = self.eval('xhi')-self.eval('xlo')  # length of the crystal box in the y axis
        zl = self.eval('xhi')-self.eval('xlo')  # length of the crystal box in the z axis
        xy = self.eval('xy')  # the project v_b to x axis
        xz = self.eval('xz')  # the project v_c to x axis
        yz = self.eval('yz')  # the project v_c to y axis
        return np.array([[xl, 0., 0.], [xy, yl, 0.], [xz, yz, zl]])

    @property
    def dumps(self):
        return self.pylmp.dumps

    def eval(self, expr):
        return self.pylmp.eval(expr)

    @property
    def fixes(self):
        return self.pylmp.fixes

    @property
    def groups(self):
        return self.pylmp.groups

    def file(self, filepath: str):
        self.pylmp.file(filepath)

    @property
    def lmp(self):
        return self.pylmp.lmp

    def read_main_data(self, atom_offset=0, **kwargs):
        path_main_data = os.path.join(self._dir_cmd, 'main.data')

        # to the main.data file
        script = self.main.writefile('lmpdat', path_main_data, retrieve_script=True)
        self.data_to_type_map(script, atom_offset)

        # read to LAMMPS
        cmd = f'read_data {path_main_data}' + ' ' + ' '.join(f'{k} {v}' for k, v in kwargs.items())
        self.command(cmd)

    def run(self, *args, **kwargs):
        self.pylmp.run(*args, **kwargs)

    @property
    def runs(self):
        return self.pylmp.runs

    def script(self):
        path_tmp_script = os.path.join(self._dir_cmd, 'script.in')
        self.pylmp.write_script(path_tmp_script)
        with open(path_tmp_script) as file:
            script = file.read()

        return script

    @property
    def type_map(self):
        return self._data.get('type_map')

    @property
    def variables(self):
        return self.pylmp.variables

    @property
    def version(self):
        return self.pylmp.version()

    def write_script(self, filepath: str):
        self.pylmp.write_script(filepath)


import src.cheminfo as ci
