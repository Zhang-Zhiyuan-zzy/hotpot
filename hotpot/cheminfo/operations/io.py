"""One reader/writer implementation owning allocation, installation and plugins."""

import os
from pathlib import Path
from typing import Callable, Literal, Optional, Union
import cython

from ..adapters.io.base import IoBase
from ..adapters.io.formats import g16 as _g16, openbabel as _openbabel, pubchem as _pubchem
from ..adapters.openbabel import conversion as _ob_conversion
from ..structure._commit import _FactTransaction
from ..structure._fields import ATOM_SOURCE_CODES
from . import conversion as _conversion

__all__ = ["MolReader", "MolWriter", "read_mol"]


class MolReader(IoBase):
    """
    Represents a molecular file reader capable of parsing various file formats.

    This class is used for reading and parsing molecular data from different file
    sources, supporting multiple formats. It processes molecular structures,
    properties, and associated calculations into an accessible internal representation.

    Attributes:
        fmt (cython.char): The format of the molecular file.
        src (cython.char): The source content to be read, which could be a path,
            string, or IO object.

    Methods:
        __init__(src, fmt=None, **kwargs):
            Initializes the molecular file reader and sets up initial state.

        get_generator():
            Determines the appropriate parser based on the file format and
            creates a generator for molecular data.

        refresh():
            Refreshes the generator to allow re-iteration.

        __next__():
            Retrieves the next molecule from the generator.

        __getitem__(item):
            Retrieves the molecule at the specified index.

        __iter__():
            Returns the molecular data generator for iteration.
    """
    fmt: cython.char
    src: cython.char

    def __init__(self, src, fmt=None, **kwargs):
        """"""
        super().__init__(src, fmt, **kwargs)
        self._generator = self.get_generator()
        self._items = None

    def _cclib_read(self):
        """IO by cclib, followed by the existing chemistry preparations."""
        data = _g16._read_cclib(self.src, self.src_type, self.fmt)
        mol = _conversion._molecule_type()()
        initial = mol._capture_factual_record()[0]
        record = _g16._cclib_record(data, self.src, self.src_type, initial)
        mol._install_molecule(record)
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
        reader = _openbabel._read_source(self.src, self.src_type, self.fmt, self.ob_opt)

        def _generator():
            for obmol in reader:
                yield _conversion.obmol2mol(obmol, _conversion._molecule_type()())
        return _generator()

    def _pubchem_read(self):
        reader = _pubchem._read_source(self.src)

        def _generator():
            for obmol in reader:
                yield _conversion.obmol2mol(obmol, _conversion._molecule_type()())
        return _generator()

    def get_generator(self):
        if self.fmt in ['g16', 'g16log', 'log']:
            return self._cclib_read()
        elif self.fmt == 'name':
            return self._pubchem_read()
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
    """
    Handles molecular file writing.

    The MolWriter class provides a mechanism to handle the writing of molecular data
    using various formats. It allows for the customization of the writing process
    through pre-processing, writing, and post-processing plugins. It supports file
    handling with optional overwriting and uses an Open Babel (OB) interface for
    conversion and writing tasks.

    Attributes:
        ob_conv: An Open Babel conversion object used for writing molecular data.
        fp: The file path or name for the output file.

    Methods:
        __init__: Initializes the MolWriter object and sets up file handling.
        obmol_write_string: Writes a molecular object to a string using Open Babel.
        _ob_write: Handles the Open Babel-based writing process for molecules.
        _pre: Executes pre-processing plugins for the specified format.
        _write: Handles the molecular writing process, utilizing plugins when available.
        _post: Executes post-processing plugins on the generated script for the format.
        to_write: Writes the generated molecular script to a file or returns the string.
        write: Combines the complete writing process including pre, writing, and post steps.
        add_plugin: Adds a custom plugin for pre-processing, writing, or post-processing.
    """
    _plugins = {}

    def __init__(self, fp: Union[str, Path], fmt: str = None, overwrite: bool = False, **kwargs) -> None:
        if isinstance(fp, (str, Path)) and os.path.exists(fp) and not overwrite:
            raise FileExistsError(f"{fp} has exists! Pass overwrite=True to overwrite.")

        super().__init__(fp, fmt, **kwargs)
        self.ob_conv = None  # Initialize conversion
        self.fp = self.src  # Just a name

    def obmol_write_string(self, obmol: "ob.OBMol"):
        return _openbabel._write_string(self.ob_conv, obmol)

    def _ob_write(self, mol: "core.Molecule", write_single: bool = False, *args, **kwargs) -> Optional[str]:
        self.ob_conv = _openbabel.get_ob_conversion(self.fmt, **self.ob_opt)
        obmol: "ob.OBMol" = mol.to_obmol()
        _script = ""
        if mol.conformers_number < 1 or write_single:
            _script += _openbabel._write_string(self.ob_conv, obmol)
        else:
            for coords in mol.conformers:
                _ob_conversion.set_obmol_coordinates(obmol, coords)
                _script += _openbabel._write_string(self.ob_conv, obmol)

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


def read_mol(src, fmt=None, **kwargs):
    return next(MolReader(src, fmt, **kwargs))


@MolWriter.add_plugin('gjf', 'pre')
def to_gjf_preprocess(writer, mol, *args, **kwargs):

    if not kwargs.get('miss_charge_calc', False) or not writer.kwargs.get('miss_charge_calc', False):
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]):
            mol.charge = mol.calc_mol_default_charge()

    writer.ob_opt.update({'b': None})
    print(f"mol charge: {mol.charge}")


@MolWriter.add_plugin('gjf', 'post')
def to_gjf(writer, script, mol, *args, **kwargs) -> str:
    """ Given a molecule, return a Gaussian16 gjf string """
    lines = script.splitlines()
    lines[0] = _g16._gjf_link0(kwargs.get('link0') or f'%nproc={min(16, os.cpu_count() // 2)}\n%mem=8GB')
    lines[1] = _g16._gjf_route(kwargs.get('route') or '#p opt/freq b3lyp/3-21g')
    lines[3] = f"Hotpot Molecule: {mol.formula}"
    lines[5] = f"{mol.charge}  {mol.spin_mult or mol.default_spin_mult}"

    script = "\n".join(lines) + '\n\n'

    return script
