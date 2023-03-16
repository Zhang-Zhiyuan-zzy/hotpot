"""
python v3.7.9
@Project: hotpot
@File   : _io.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/14
@Time   : 4:18
"""
from abc import ABCMeta, ABC, abstractmethod
import re
import json
from typing import *
from openbabel import openbabel as ob, pybel as pb
from cheminfo import Molecule

# get list of elemental periodic table
_elements = json.load(open('../data/periodic_table.json', encoding='utf-8'))['elements']
_elements = {d['symbol']: d for d in _elements}


class FormatDefinedError(Exception):
    """ Raise when a format name have been defined """


class FormatNotDefineError(Exception):
    """ Raise when a Process not define the format method """


class _MetaProcess(ABCMeta):
    """
    Abstract Meta class to record the `format`: `Process` pair.
    Make the defined `Process` could be called by `format`
    """
    __defined_process = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        """"""
        # Get the format name
        fmt = namespace.get('format')()

        # If the format name have not been defined
        if fmt not in mcs.__defined_process:
            cls = ABCMeta(name, bases, namespace, **kwargs)
            mcs.__defined_process[fmt] = cls

        else:  # If the format name have been defined
            raise FormatDefinedError

        return ABCMeta(name, bases, namespace, **kwargs)


class Process(metaclass=_MetaProcess):
    """"""
    def __new__(cls, *args, **kwargs):
        return super(Process, cls).__new__(cls)

    @abstractmethod
    def format(self) -> str:
        """    Abstract method to defined the format to call this class    """
        raise FormatNotDefineError(f"{self.__class__.__name__} not define the 'format' method")


class MolProcess(Process, ABC):
    """    Base class to process between the Molecule and specific Format    """
    def __init__(self, mol: Molecule = None):
        self._mol = mol

    def make_mol(self, **kwargs):
        """"""
        mol = ob.OBMol()

    @property
    def _set_atom(self):
        return {
            'number': self._set_atomic_number,
            'coords': self._set_coordinates
        }

    @staticmethod
    def _set_atomic_number(atom: ob.OBAtom, number: int):
        atom.SetAtomicNum(number)

    @staticmethod
    def _set_coordinates(atom: ob.OBAtom, coordinates):
        atom.SetVector(*coordinates)



class MolGJFProcess(MolProcess, ABC):
    """"""
    def format(self) -> str:
        return "gaussian/gjf"

    def parse(self, info):
        """"""
        partition = [p for p in info.split("\n\n") if p]

        # Parse the link0 and route lines
        link0, route = [], []
        for line in partition[0].split('\n'):
            if line[0] == '%':
                link0.append(line)
            elif line[0] == '#':
                route.append(line)
            else:
                raise IOError("the format of gjf file error")

        # Parse the title line
        title = partition[1]

        # molecule specification
        # regular expression for elemental symbols
        regx_ele_sym = re.compile(r'[A-Z][a-z]?')

        # TODO: Now, the method parses only the first-four required atomic properties,
        # TODO: the more optional properties might be specified ,such as charge, spin,
        # TODO: more subtle process should be designed to parse the optional properties.
        mol_spec_lines = partition[2].split('\n')
        mol_charge, spin_multiplicity = map(int, mol_spec_lines[0].split())

        labels, atomic_number, coords, charge = [], [], [], []
        for line in mol_spec_lines[1:]:
            atom_info = line.split()
            atomic_label = atom_info[0]
            x, y, z = map(float, atom_info[1:4])
            atomic_symbol = regx_ele_sym.findall(atomic_label)[0]

            labels.append(atomic_label)
            atomic_number.append(_elements[atomic_symbol]['number'])
            coords.append([x, y, z])
