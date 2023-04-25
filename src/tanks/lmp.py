"""
python v3.7.9
@Project: hotpot
@File   : lmp.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/24
@Time   : 3:43
"""
import os
import json
from lammps import PyLammps
from typing import *
import src


class HpLammps:
    """
    A wrapper to run LAMMPS tasks
    """
    _dir_tmp: str = src.pkg_root + '/tmp'

    def __init__(self, main, *args, **kwargs):
        """"""
        self.main: ci.Molecule = main
        self.pylmp = PyLammps()
        self.defaults: dict = json.load(open('../data/lmp_default.json'))
        self._data = {}  # store any data
        self.args = args
        self.kwargs = kwargs

    def __dir__(self) -> Iterable[str]:
        return self.pylmp.__dir__()

    def __getattr__(self, item):
        return self.pylmp.__getattr__(item)

    def close(self):
        self.pylmp.close()
        del self

    def command(self, cmd: str):
        self.pylmp.command(cmd)

    @property
    def computes(self):
        return self.pylmp.computes

    @property
    def dumps(self):
        return self.pylmp.dumps

    def ensemble(self, ensemble: str = 'nvt'):
        """"""

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

    def initialize(self):
        init_item = ('units', 'dimension', 'boundary', 'atom_type', 'pair_style')
        for name in init_item:
            cmd = self.kwargs.get(name)
            if cmd:
                self.command(f'{name} {cmd}')
            else:
                self.command(f'{name} {self.defaults[name]}')

    @property
    def lmp(self):
        return self.pylmp.lmp

    def read_main_data(self, **kwargs):
        path_main_data = os.path.join(self._dir_tmp, 'main.data')

        # to the main.data file
        self.main.writefile('lmpdat', path_main_data)

        # read to LAMMPS
        cmd = f'read_data {path_main_data}' + ' '.join(f'{k} {v}' for k, v in kwargs.items())
        self.command(cmd)

    def run(self, *args, **kwargs):
        self.pylmp.run(*args, **kwargs)

    @property
    def runs(self):
        return self.pylmp.runs

    def script(self):
        path_tmp_script = os.path.join(self._dir_tmp, 'script.in')
        self.pylmp.write_script(path_tmp_script)
        with open(path_tmp_script) as file:
            script = file.read()

        return script

    @property
    def variables(self):
        return self.pylmp.variables

    @property
    def version(self):
        return self.pylmp.version()

    def write_script(self, filepath: str):
        self.pylmp.write_script(filepath)


import src.cheminfo as ci
