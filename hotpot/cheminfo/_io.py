"""
python v3.9.0
@Project: hotpot
@File   : io
@Auther : Zhiyuan Zhang
@Data   : 2023/10/29
@Time   : 20:59

Note: defining processes for reading and writing chemical information from Input and Output stream
"""
import os
import io
import re
from os import PathLike
from abc import ABC, abstractmethod
from typing import *


from openbabel import pybel


from . import _chem as ci


class _IO(ABC):
    """ the base class for IO classes """
    _formats = {'parser': {}, 'dumper': {}}

    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, src, *args, **kwargs):
        self.pre(src, *args, **kwargs)
        tgt = self.io(src, *args, **kwargs)
        return self.post(tgt, src, *args, **kwargs)

    @staticmethod
    def pre(src, *args, **kwargs):
        return src

    @abstractmethod
    def io(self, src, *args, **kwargs):
        raise NotImplemented

    @staticmethod
    def post(tgt, src, *args, **kwargs):
        return tgt

    @classmethod
    def get_io(cls, fmt: str) -> "_IO":
        if cls is Parser:
            return cls._formats['parser'].get(fmt, Parser)(fmt)
        elif cls is Dumper:
            return cls._formats['dumper'].get(fmt, Dumper)(fmt)

        raise ValueError(f'the format {fmt} is not be found in {cls}')

    @classmethod
    def register(cls, fmt: str):
        def decorator(parser_or_dumper: type):
            if issubclass(parser_or_dumper, Parser):
                cls._formats['parser'][fmt] = parser_or_dumper
            elif issubclass(parser_or_dumper, Dumper):
                cls._formats['dumper'][fmt] = parser_or_dumper
            else:
                raise ValueError('the _IO.register just be allow to decorate the subclass of Parser or Dumper')

            return parser_or_dumper

        return decorator


class Parser(_IO, ABC):
    """ the basis Parser to parse text object to hotpot objects """

    @staticmethod
    def _src_checks(src) -> Literal['path', 'str', 'bytes', 'StringIO', 'BytesIO', 'FileIO']:
        if isinstance(src, str):
            if os.path.exists(src):
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

    def io(self, src, *args, **kwargs):
        """ Parse by openbabel.pybel """
        src_type = self._src_checks(src)  # Get the source type name
        if src_type == 'str':
            pybel_mol = pybel.readstring(self.fmt, src)
        elif src_type == 'path':
            pybel_mol = next(pybel.readfile(self.fmt, str(src)))
        elif src_type == 'IOString':
            pybel_mol = pybel.readstring(self.fmt, src.read())
        else:
            raise RuntimeError(f'the source type {type(src)} have not been supported')

        return ci.Molecule(pybel_mol.OBMol)


class Dumper(_IO, ABC):
    """ the basis Dumper to convert the hotpot objects to string or bytes """
    def io(self, src, *args, **kwargs):
        """ Dump by openbabel.pybel """
        # Try to dump by openbabel.pybel
        type_err_pattern = re.compile(r"write\(\) got an unexpected keyword argument '\w+'")
        pb_mol = pybel.Molecule(src.ob_mol)

        if not kwargs:
            return pb_mol.write(self.fmt)

        while kwargs:
            try:
                return pb_mol.write(self.fmt, **kwargs)

            except TypeError as error:
                if type_err_pattern.match(str(error)):
                    pop_kwargs = str(error).split()[-1].strip("'")
                    kwargs.pop(pop_kwargs)
                else:
                    raise error

            except ValueError:
                print(IOError(f'the cheminfo.Molecule obj cannot dump to Literal'))
                return None


@_IO.register('lmpdat')
class LammpsData(Dumper):
    @staticmethod
    def _parse_lmp_data_script(script: str):
        """ Parse the LAMMPS data script to two dict, header and body """
        # Define_body_title
        bt_name = (
            # atom-property sections
            'Atoms', 'Velocities', 'Masses', 'Ellipsoids', 'Lines', 'Triangles', 'Bodies',
            # molecular topology sections
            'Bonds', 'Angles', 'Dihedrals', 'Impropers',
            # force field sections
            'Pair Coeffs', 'PairIJ Coeffs', 'Bond Coeffs', 'Angle Coeffs', 'Dihedral Coeffs', 'Improper Coeffs',
            # class 2 force field sections
            'BondBond Coeffs', 'BondAngle Coeffs', 'MiddleBondTorsion Coeffs', 'EndBondTorsion Coeffs',
            'AngleTorsion Coeffs', 'AngleAngleTorsion Coeffs', 'BondBond13 Coeffs', 'AngleAngle Coeffs'
        )

        # Compile the body and header pattern
        header_title = re.compile(r'[a-z]+')
        header_int = re.compile(r'[0-9]+')
        header_float = re.compile(r'-?[0-9]+\.[0-9]*')

        lines = script.split('\n')
        body_split_point = [i for i, lin in enumerate(lines) if lin in bt_name] + [len(lines)]

        # Extract header info
        headers = {}
        for line in lines[1:body_split_point[0]]:
            line = line.strip()
            if line:
                ht = ' '.join(header_title.findall(line))
                hvs = line[:line.find(ht)].strip()  # header values
                header_values = []
                for hv in re.split(r'\s+', hvs):
                    if header_int.fullmatch(hv):
                        header_values.append(int(hv))
                    elif header_float.fullmatch(hv):
                        header_values.append(float(hv))
                    else:
                        raise ValueError('the header line not match well')

                headers[ht] = header_values

        # Extract bodies info
        bodies = {}
        for sl_idx, el_idx in zip(body_split_point[:-1], body_split_point[1:]):
            bt = lines[sl_idx].strip()  # body title
            bc = [line.strip() for line in lines[sl_idx + 1: el_idx] if line.strip()]  # body content
            bodies[bt] = bc

        return lines[0], headers, bodies

    @staticmethod
    def _process_lmpdat_bonds(lmp_bonds_types: list[int], bond_contents: list[str]):
        """"""
        assert len(lmp_bonds_types) == len(bond_contents)
        sep = re.compile(r'\s+')
        for i, (lmp_bt, bc) in enumerate(zip(lmp_bonds_types, bond_contents)):
            split_bc = sep.split(bc)
            split_bc[1] = str(lmp_bt)
            bond_contents[i] = '  '.join(split_bc)

        return bond_contents

    @staticmethod
    def _unique_bonds(src) -> list:
        uni_keys = []
        lammps_bonds_types = []

        for bond in src.bonds:
            if bond.atom1.atomic_number < bond.atom2.atomic_number:
                bond_key = (bond.atom1.atomic_number, bond.type, bond.atom2.atomic_number)
            else:
                bond_key = (bond.atom2.atomic_number, bond.type, bond.atom1.atomic_number)

            try:
                key = uni_keys.index(bond_key) + 1
            except ValueError:
                uni_keys.append(bond_key)
                key = len(uni_keys)

            lammps_bonds_types.append(key)

        return lammps_bonds_types

    def post(self, tgt: str, src, *args, **kwargs):
        """ post-process for LAMMPS data file """
        title, headers, bodies = self._parse_lmp_data_script(tgt)
        script = title + '\n'

        # process the headers contents
        list_bonds_types = self._unique_bonds(src)
        for ht, hvs in headers.items():
            if ht == 'bond types':  # header title, header values
                hvs[0] = len(list_bonds_types)

            script += ' '.join(map(str, hvs)) + ' ' + ht + '\n'

        script += '\n' * 3

        # process the bodies contents
        for body_title, body_info in bodies.items():
            if body_title == 'Bonds':
                # modify the body info in the Bonds contexts
                body_info = self._process_lmpdat_bonds(list_bonds_types, body_info)

            if body_info:  # if the body contents exist
                script += body_title + '\n' * 2
                script += '\n'.join(body_info)
                script += '\n' * 3

        return script

