"""
python v3.7.9
@Project: hotpot
@File   : _io.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/14
@Time   : 4:18
"""
from os import PathLike
from typing import *
from abc import ABC, ABCMeta, abstractmethod
from copy import copy
import re
from functools import wraps
from openbabel import pybel


class Register:
    # serve as a handle to store the custom formats of dumpers
    custom_dumpers = {}
    postprocessing = {}

    def __call__(self, fmt: str, types: str = "dumper"):
        """
        To register any function as a dumper or a postprocess to convert mol to formats
        Args:
            fmt:
            types:

        Returns:

        """

        def decorator(func: Callable):

            if types == 'dumper':
                self.custom_dumpers[fmt] = func
            elif types == 'postprocess':
                self.postprocessing[fmt] = func
            else:
                raise TypeError('the type of register is not supported')

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator


# Retrieve the IO class by it's format name
def retrieve_format(fmt: str = None):
    return _MoleculeIO.registered_format().get(fmt)


# Get all registered format name
def registered_format_name():
    return tuple(_MoleculeIO.registered_format().keys())


class _MoleculeIO(ABCMeta):
    """    Metaclass for registration of IO class format """
    _registered_format = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Get the format keywords
        fmt = namespace.get('format')(mcs)

        if not fmt:
            return super(_MoleculeIO, mcs).__new__(mcs, name, bases, namespace, **kwargs)
        elif not isinstance(fmt, str):
            raise TypeError('the defined format should be a string')
        elif fmt in mcs._registered_format:
            raise ValueError(f'the format {fmt} have been defined before')
        else:
            cls = super(_MoleculeIO, mcs).__new__(mcs, name, bases, namespace, **kwargs)
            mcs._registered_format[fmt] = cls
            return cls

    @classmethod
    def registered_format(mcs):
        return copy(mcs._registered_format)


class MoleculeIO(metaclass=_MoleculeIO):
    """ The abstract base class for all IO class """

    @abstractmethod
    def format(self) -> str:
        return None

    @staticmethod
    @abstractmethod
    def dump(mol, *args, **kwargs) -> Union[str, bytes]:
        """"""

    @staticmethod
    @abstractmethod
    def parse(info) -> Dict:
        """"""

    def write(self, mol, path_file: Union[str, PathLike], *args, **kwargs):
        """"""
        script = self.dump(mol, *args, **kwargs)

        if isinstance(script, str):
            mode = 'w'
        elif isinstance(script, bytes):
            mode = 'b'
        else:
            raise TypeError('the type of dumping valve is not supported')

        with open(path_file, mode) as writer:
            writer.write(script)

    def read(self, path_file: Union[str, PathLike], *args, **kwargs) -> Dict:
        """"""
        with open(path_file) as file:
            data = self.parse(file.read())
        return data


class GaussianGJF(MoleculeIO, ABC):

    @staticmethod
    def dump(mol, *args, **kwargs) -> Union[str, bytes]:

        # separate keyword arguments:
        link0 = kwargs['link0']
        route = kwargs['route']
        custom_charge = kwargs.get('charge')
        custom_spin = kwargs.get('spin')

        pybal_mol = pybel.Molecule(mol._OBMol)

        script = pybal_mol.write('gjf')
        assert isinstance(script, str)

        lines = script.splitlines()

        lines[0] = f'%{link0}'
        lines[1] = f'#{route}'

        charge, spin = lines[5].split()
        if custom_charge:
            charge = str(custom_charge)
        if custom_spin:
            spin = str(custom_spin)

        script = '\n'.join(lines)

        return script

    @staticmethod
    def parse(info) -> Dict:
        """
        Returns:
            {
                'identifier': ...,
                'charge': ...,
                'spin': ...,
                atoms: [
                    {'symbol': .., 'label': .., 'coordinates': ..},
                    {'symbol': .., 'label': .., 'coordinates': ..},
                    ...,
                }
            }
        """
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
        charge, spin = map(int, mol_spec_lines[0].split())

        atoms = []
        for line in mol_spec_lines[1:]:
            atom_info = line.split()
            atomic_label = atom_info[0]
            x, y, z = map(float, atom_info[1:4])
            atomic_symbol = regx_ele_sym.findall(atomic_label)[0]

            atoms.append({'symbol': atomic_symbol, 'label': atomic_label, 'coordinates': (x, y, z)})

        return {
            'identifier': title,
            'charge': charge,
            'spin': spin,
            'atoms': atoms
        }

    def format(self) -> str:
        return 'gjf'


class Dumper:
    """
    Dump the Molecule information into specific format.
    The output in general is the string or bytes
    """
    # Initialize the register
    register = Register()

    def __init__(self, fmt: str, mol, *args, **kwargs):
        self.fmt = fmt
        self.mol = mol

        self.args = args
        self.kwargs = kwargs

    def dump(self) -> Union[str, bytes]:
        """
        Try, in turn, to dump the Molecule to the specified format by various method:
            1) the 'openbabel.pybal' module
            2) 'cclib' library
            3) coutom dumper
        """

        # Trying dump by pybel
        script = None

        try:
            pb_mol = pybel.Molecule(self.mol._OBMol)
            script = pb_mol.write(self.fmt)

            success = True

        except ValueError:
            success = False

        if not success:
            # TODO: try to dump by cclib
            pass

        if not success:
            custom_dumper = self.register.custom_dumpers.get(self.fmt)

            if custom_dumper:
                script = custom_dumper(self.mol)
            else:
                raise ValueError(f'the format {self.fmt} cannot support!!')

        assert isinstance(script, str)

        # Try to perform the postprocessing
        processor = self.register.postprocessing.get(self.fmt)
        if processor:
            script = processor(self, script)

        return script

    @staticmethod
    @register(fmt='gjf', types='postprocess')
    def _gjf_post_processor(self: 'Dumper', script: str):
        """"""
        # To count the insert lines
        inserted_lines = 0

        # separate keyword arguments:
        link0 = self.kwargs['link0']
        route = self.kwargs['route']
        custom_charge = self.kwargs.get('charge')
        custom_spin = self.kwargs.get('spin')

        lines = script.splitlines()

        # Write link0 command
        if isinstance(link0, str):
            lines[0] = f'%{link0}'
        elif isinstance(link0, list):
            for i, stc in enumerate(link0):  # stc=sentence
                assert isinstance(stc, str)
                if not i:  # For the first line of link0, replace the the original line in raw script
                    lines[0] = f'%{stc}'
                else:  # For the other lines, insert into after the 1st line
                    inserted_lines += 1
                    lines.insert(inserted_lines, f'%{stc}')
        else:
            raise TypeError('the link0 should be string or list of string')

        # Write route command
        if isinstance(route, str):
            lines[1+inserted_lines] = f'# {route}'
        elif isinstance(route, list):
            for i, stc in enumerate(route):
                assert isinstance(stc, str)
                if not i:  # For the first line of link0, replace the the original line in raw script
                    lines[1+inserted_lines] = f'#{stc}'
                else:  # For the other lines, insert into after the original route line.
                    inserted_lines += 1
                    lines.insert(inserted_lines+1, f'%{stc}')
        else:
            raise TypeError('the route should be string or list of string')

        charge, spin = lines[5+inserted_lines].split()
        if custom_charge:
            charge = str(custom_charge)
        if custom_spin:
            spin = str(custom_spin)

        lines[5+inserted_lines] = f'{charge} {spin}'

        script = '\n'.join(lines)

        # End black line
        script += '\n\n'

        return script
