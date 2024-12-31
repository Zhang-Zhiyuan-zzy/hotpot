"""
python v3.9.0
@Project: hotpot
@File   : _io
@Auther : Zhiyuan Zhang
@Data   : 2024/12/5
@Time   : 10:08
"""
import os
import re
import sys
from os import PathLike
from pathlib import Path
import io
from typing import Literal, Union, Optional, Callable

import cython
import numpy as np
import cclib
from openbabel import openbabel as ob, pybel as pb

from hotpot.cheminfo.obconvert import obmol2mol, set_obmol_coordinates, get_ob_conversion

# set Openbabel Message
ob_log_handler = ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
ob.obErrorLog.StopLogging()
pb.ob.obErrorLog.StopLogging()

if not sys.modules.get('hotpot.cheminfo.core', None):
    from .. import core


def _extract_force_matrix(lines, atomic_numbers):
    # Define the format of force sheet
    # the Force sheet like this:
    #  -------------------------------------------------------------------
    #  Center     Atomic                   Forces (Hartrees/Bohr)
    #  Number     Number              X              Y              Z
    #  -------------------------------------------------------------------
    #       1        8           0.039901671    0.000402574    0.014942530
    #       2        8           0.017381613    0.001609531    0.006381231
    #       3        6          -0.092853735   -0.025654844   -0.005885898
    #       4        6           0.067801154    0.024130172   -0.022794721
    #       5        8          -0.023702905    0.005486251   -0.004938175
    #       6        8          -0.006359715   -0.008543465    0.010350815
    #       7       55          -0.002168084    0.002569781    0.001944217
    #  -------------------------------------------------------------------
    force_head1 = re.compile(r'\s*Center\s+Atomic\s+Forces\s\(Hartrees/Bohr\)\s*')
    force_head2 = re.compile(r'\s*Number\s+Number\s+X\s+Y\s+Z\s*')
    sheet_line = re.compile(r'\s*----+\s*')

    HEAD_LINES_NUM = 3  # the offset line to write the header

    head_lines = [i for i, line in enumerate(lines) if force_head1.match(line)]

    all_forces = []
    i: cython.int
    for i in head_lines:
        # enhance the inspection of Force sheet head
        assert force_head2.match(lines[i + 1])
        assert sheet_line.match(lines[i + 2])

        rows = 0
        forces = []
        while True:

            if sheet_line.match(lines[i + HEAD_LINES_NUM + rows]):
                if len(forces) == len(atomic_numbers):
                    all_forces.append(forces)
                    break
                else:
                    raise ValueError('the number of force vector do not match the number of atoms')

            ac, an, x, y, z = map(
                lambda v: int(v[1]) if v[0] < 2 else float(v[1]),
                enumerate(lines[i + HEAD_LINES_NUM + rows].split())
            )

            # Enhance the inspection
            assert ac == rows + 1
            if atomic_numbers[rows] != an:
                raise ValueError('the atomic number do not match')

            forces.append([x, y, z])

            rows += 1

    return np.array(all_forces)


def _align_unit(conformers):
    _energies = ('zero_point', 'gibbs', 'enthalpy', 'entropy')
    _hartree2ev = 27.211386245988

    item: cython.char
    for item in _energies:
        try:
            setattr(conformers, f'_{item}', getattr(conformers, f"_{item}") * _hartree2ev)
        except (TypeError, AttributeError):
            continue

def _extract_g16_thermo(lines):
    # Grab thermal energy, delta capacity at volume, delta entropy
    anchor_line = 0
    title_pattern = re.compile(r'\s+E \(Thermal\)\s+CV\s+S')
    for i, line in enumerate(lines):
        if title_pattern.match(line):
            anchor_line = i
            break

    if anchor_line != 0:
        thermal_energy, capacity, _ = map(float, re.split(r'\s+', lines[anchor_line + 2].strip())[1:])
        thermal_energy = 0.043361254529175 * thermal_energy  # kcal to ev
        capacity = 0.043361254529175e-3 * capacity  # cal to ev

        return thermal_energy, capacity

    else:
        return None, None


class IoBase:
    def __init__(self, src, fmt=None, **kwargs):
        self.src = src
        self.src_type = self._src_checks(src)
        self.fmt = self._determine_fmt(src, fmt)
        self.ob_opt = kwargs.pop('ob_opt', {})
        self.kwargs = kwargs

    @staticmethod
    def _src_checks(src) -> Literal['path', 'str', 'bytes', 'StringIO', 'BytesIO', 'FileIO', 'None']:
        if src is None:
            return 'None'

        if isinstance(src, str):
            if os.path.dirname(src) and os.path.exists(os.path.dirname(src)):
                return 'path'
            else:
                return 'str'
        elif isinstance(src, PathLike):
            return 'path'
        elif isinstance(src, bytes):
            return 'bytes'
        elif isinstance(src, io.StringIO):
            return 'StringIO'
        elif isinstance(src, io.BytesIO):
            return 'BytesIO'
        elif isinstance(src, io.FileIO):
            return 'FileIO'
        else:
            raise TypeError(f'get unsupported input type {type(src)}')

    def _determine_fmt(self, src, fmt):
        if not fmt:
            if isinstance(src, (str, PathLike)):
                p_src = Path(src)
                fmt = p_src.suffix[1:]

                if not fmt:
                    fmt = "smi"

        if not fmt:
            raise ValueError('the fmt has not been known!')

        return fmt


class MolReader(IoBase):
    fmt: cython.char
    src: cython.char

    def __init__(self, src, fmt=None, **kwargs):
        """"""
        super().__init__(src, fmt, **kwargs)
        self._generator = self.get_generator()
        self._items = None

    def _cclib_read(self):
        """ IO by cclib package """
        try:
            if self.src_type == 'str':
                data = cclib.ccopen(io.StringIO(self.src)).parse()
            elif self.src_type == 'path':
                data = cclib.ccopen(self.src).parse()
            elif self.src_type == 'IOString':
                data = cclib.ccopen(self.src).parse()
            else:
                raise RuntimeError(f'the source type {type(self.src)} have not been supported in cclib')

        except (RuntimeError, AttributeError):
            data = None

        if not data:
            raise IOError(f'Cannot read any calculational data from {self.src}')

        if not hasattr(data, 'atomnos'):
            raise ValueError(f'read a incorrect MOL file!! {self.fmt}')

        mol = core.Molecule()
        mol.charge = data.charge
        mol.spin_mult = data.mult

        _create_atom = getattr(mol, '_create_atom')
        for atomic_number in data.atomnos:
            _create_atom(atomic_number=atomic_number)

        # get information about the coordination collections
        conformers = mol.conformers
        if hasattr(data, 'atomcoords'):
            setattr(conformers, '_coordinates', getattr(data, 'atomcoords'))

        # get information about the energy (SCF energies) vector
        if hasattr(data, 'scfenergies'):
            setattr(conformers, '_energy', getattr(data, 'scfenergies'))

        if hasattr(data, "atomcharges"):
            if data.atomcharges['mulliken'].ndim == 1:
                partial_charges = np.zeros(data.atomcoords.shape[:-1])
                partial_charges[-1] = data.atomcharges['mulliken']
            else:
                partial_charges = data.atomcharges['mulliken']
            setattr(conformers, '_partial_charge', partial_charges)

        setattr(conformers, '_gibbs', getattr(data, 'freeenergy', None))
        setattr(conformers, '_zero_point', getattr(data, 'zpve', None))
        setattr(conformers, '_spin_mult', getattr(data, 'mult', None))


        if self.src_type == 'str':
            lines = self.src.splitlines()
        else:
            with open(self.src) as file:
                lines = file.readlines()

        thermo, capacity = _extract_g16_thermo(lines)
        setattr(conformers, '_thermo', thermo)
        setattr(conformers, '_capacity', capacity)

        setattr(conformers, '_force', _extract_force_matrix(lines, data.atomnos))

        _align_unit(conformers)
        mol.conformer_load(-1)  # load conformer

        mol.link_atoms(assign_bond_order=False)
        mol.determine_rings_aromatic()
        mol.assign_bond_order()
        mol.calc_atom_valence()

        def _generator():
            _reader = [mol]
            for m in _reader:
                yield m

        return _generator()

    def _openbabel_read(self):
        if self.src_type == 'IOString':
            src = self.src.read()
        else:
            src = self.src

        if self.src_type == 'path':
            reader = pb.readfile(self.fmt, str(src), self.ob_opt)
        else:
            reader = pb.readstring(self.fmt, src, self.ob_opt)

        def _generator():
            nonlocal reader
            if isinstance(reader, pb.Molecule):
                reader = [reader]

            for pmol in reader:
                yield obmol2mol(pmol.OBMol, core.Molecule())

        return _generator()

    def get_generator(self):
        if self.fmt in ['g16', 'g16log', 'log']:
            return self._cclib_read()
        else:
            return self._openbabel_read()

    def refresh(self):
        self._generator = self.get_generator()

    def __next__(self):
        return next(self._generator)

    def __getitem__(self, item: int):
        if not self._items:
            self._items = list(self.get_generator())

        return self._items[item]

    def __iter__(self):
        return self.get_generator()


class MolWriter(IoBase):
    _plugins = {}

    def __init__(self, fp: Union[str, Path], fmt: str = None, overwrite: bool = False, **kwargs) -> None:
        if isinstance(fp, (str, Path)) and os.path.exists(fp) and not overwrite:
            raise FileExistsError(f"{fp} has exists! Pass overwrite=True to overwrite.")

        super().__init__(fp, fmt, **kwargs)
        self.ob_conv = None  # Initialize conversion
        self.fp = self.src  # Just a name

    def obmol_write_string(self, obmol: ob.OBMol):
        return self.ob_conv.WriteString(obmol)

    def _ob_write(self, mol: "core.Molecule", write_single: bool = False, *args, **kwargs) -> Optional[str]:
        self.ob_conv = get_ob_conversion(self.fmt, **self.ob_opt)
        obmol: ob.OBMol = mol.to_obmol()
        _script = ""
        if mol.conformers_number < 1 or write_single:
            _script += self.ob_conv.WriteString(obmol)
        else:
            for coords in mol.conformers:
                set_obmol_coordinates(obmol, coords)
                _script += self.ob_conv.WriteString(obmol)

        return _script

    def _pre(self, mol, *args, **kwargs):
        if preprocessor := self._plugins.get(self.fmt, {}).get('pre', None):
            assert isinstance(preprocessor, Callable)
            preprocessor(self, mol, *args, **kwargs)

    def _write(self, mol: "core.Molecule", *args, **kwargs) -> Optional[str]:
        if writer := self._plugins.get(self.fmt, {}).get('write', None):
            assert isinstance(writer, Callable)
            return writer(self, mol, *args, **kwargs)
        else:
            return self._ob_write(mol, *args, **kwargs)

    def _post(self, script, mol, *args, **kwargs):
        if postprocessor := self._plugins.get(self.fmt, {}).get('post', None):
            assert isinstance(postprocessor, Callable)
            script = postprocessor(self, script, mol, *args, **kwargs)
        return script

    def to_write(self, script: str):
        if not self.fp:
            return script
        else:
            with open(self.fp, "w") as f:
                f.write(script)

    def write(self, mol: "core.Molecule", *args, **kwargs) -> Optional[str]:
        kwargs.update(self.kwargs)
        self._pre(mol, *args, **kwargs)
        script = self._write(mol, *args, **kwargs)
        return self.to_write(self._post(script, mol, *args, **kwargs))

    @staticmethod
    def add_plugin(fmt: str, where: Literal['pre', 'write', 'post'] = 'post'):
        def decorator(func):
            funcs = MolWriter._plugins.setdefault(fmt, {})
            funcs[where] = func
        return decorator
