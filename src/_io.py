"""
python v3.7.9
@Project: hotpot
@File   : _io.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/14
@Time   : 4:18
"""
import os
from os import PathLike
from typing import *
from abc import ABCMeta, abstractmethod
from copy import copy
import io
from io import IOBase
from openbabel import pybel
import cclib
import numpy as np
import src.cheminfo as ci

"""
Notes:
This module contain Classes to IO between the Molecule object and various file formats.
the main classes are:
    - Reader: from Formatted files to Molecule, called by the Molecule.read() method.
    - Writer: from Molecule to formatted files, called by the Molecule.write() method.
    - Dumper: from Molecule to formatted Literals(str or bytes), called by the Molecule.dump() method.
    - Parser: from formatted Literals to the Molecule obj, called by the Molecule.parse() method.
all of them are inherit from the _IOBase class.

For the implementation of dump and parse:
    - some third-part package are used, like openbabel, cclib and so on.
    - there are some self own IO function be defined too.
    - the user also custom and register to the Dumper and Parser too.

When implementing the IO: 
    1) the IO class will firstly check whether a custom IO function is defined and register, if the IO function 
        are defined and registered, the IO are implemented by the registered.
    2) else, the IO class will try to call, in turn, try to call some third-part packages,
    3) finally, if all third-part packages fail to complement IO, Raise the IOError !!!

Following the steps to customise your IO function:
    1) determine which IO operation(read, write, dump or parse) you want to define， import the relevant IO class
        into your own python modules.
    2) define the IO function which customises your IO implementation, the custom function should meet the base
        requirements demand by the corresponding IO class. When the IO functions are defined, applying the
        `IOClass`.register decorator to register the IO function into the `IOClass`, the `IOClass`.register should
        pass some args. This is a example:
        --------------------------------- Example ------------------------------------
        Examples:
        # importing the relevant IO classes
        from hotpot.io import Dumper, Reader
        
        # define and register a read function
        # the fmt defined the format key to handle the custom IO function
        # the types defined where the IO function will be applied, pre: preprocess, io: main io, post: postprocess
        @Reader.register(fmt='the/format/key', types='pre|io|post')
        def my_read_func(*arg, **kwargs)  # the args should meet the Reader demand
            ...
            
        # define and register a dump function
        def my_dump_func(*arg, **kwargs) # the args should meet the Dumper demand
            ...
        ---------------------------------- END ---------------------------------------
"""

# Define the IO function types
IOFuncPrefix = Literal['pre', 'io', 'post']
IOStream = Union[IOBase, str, bytes]


class Register:
    """
    Register the IO function for Dumper, Parser or so on
    """
    # these dicts are container to store the custom io functions
    # the keys of the dict are serve as the have to get the mapped io functions(the values)
    pre_methods = {}
    io_methods = {}
    post_methods = {}

    def __call__(self, io_cls: type, fmt: str, prefix: IOFuncPrefix):
        """
        To register any function as a dumper or a postprocess to convert mol to formats
        Args:
            fmt:
            prefix:

        Returns:

        """

        def decorator(func: Callable):

            if prefix == 'pre':
                self.pre_methods[fmt] = func
            elif prefix == 'io':
                self.io_methods[fmt] = func
            elif prefix == 'post':
                self.post_methods[fmt] = func
            else:
                raise TypeError('the type of register is not supported')

            return func

        return decorator

    def pre(self, fmt: str):
        return self.pre_methods.get(fmt)

    def io(self, fmt: str):
        return self.io_methods.get(fmt)

    def post(self, fmt: str):
        return self.post_methods.get(fmt)


# Retrieve the IO class by it's format name
def retrieve_format(fmt: str = None):
    return _MoleculeIO.registered_format().get(fmt)


# Get all registered format name
def registered_format_name():
    return tuple(_MoleculeIO.registered_format().keys())


# TODO: deprecated in the later version
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


class MetaIO(type):
    """
    Meta class to specify how to construct the IO class
    This Meta class is defined to register IO function conveniently.

    The IO functions are divided into three categories:
        - preprocess: do something before performing any of IO operation, with prefix '_pre'
        - io: performing the IO operation, with prefix '_io'
        - postprocess: do something after preforming IO operation, with prefix '_post'

    This Meta class offer two approach to defined and register the IO functions:
        - Define inside the IO class (IOClass)
        - Define outside the IO class and decorate the defined function by IOClass.register function

    To define inside the IOClass, one should name the IO function with the pattern:
        def _prefix_keys():
            ...
    where, the prefix is one of 'pre', 'io' or 'post'; the keys is the handle name to retrieve the
    IO functions.

    To define outside the IOClass, one should applied the class method register as the decorator of the
    IO functions, specified the prefix and the handle name as the decorator arguments, like:
        @IOClass.register(fmt='keys', types='prefix')
        def outside_io_func(*args, **kwargs):
            ...
    where the IOClass is one of Reader, Writer, Dumper, Parser or other custom IOClass, the 'key' and 'prefix'
    should be replace to the handle name and prefix you specified.
    """

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        """ If the subclasses contain methods with the prefix of '_pre', '_io' or '_post'
        they are seen as the IO function, that the preprocess, io or postprocess functions, respectively
        """
        _register = Register()

        for attr_name, attr in namespace.items():

            # Make sure the io function is a Callable obj
            if not isinstance(attr, Callable):
                continue

            split_names = attr_name.split('_')

            # the custom IO function should with prefix: '_pre', '_io' and '_post'
            # the handle keys of these function are follow the above prefix and separate be '_'
            # for example:
            #     def _pre_gjf(*args, **kwargs):
            #         ...
            # this is a preprocess IO function with a handle key: 'gjf' to retrieve the function.
            if len(split_names) <= 2:
                continue

            io_type = split_names[1]

            # Register the io functions:
            # if a define methods with the prefix '_pre', '_io' or '_post'
            # these methods are seen as preprocess, io or postprocess functions, respectively
            if io_type == 'pre':
                _register.pre_methods['_'.join(split_names[2:])] = attr
            if io_type == 'io':
                _register.io_methods['_'.join(split_names[2:])] = attr
            if io_type == 'post':
                _register.post_methods['_'.join(split_names[2:])] = attr

        namespace['_register'] = _register

        return type(name, bases, namespace, **kwargs)


class IOBase:
    """ The base IO class """
    # Initialize the register function, which is a callable obj embed in IO classes
    # When to register new IO function, apply the register function as decorator

    _register = None

    def __init__(self, fmt: str, source: Union['ci.Molecule', IOStream], *args, **kwargs):
        """"""
        self.fmt = fmt
        self.src = source

        self.args = args
        self.kwargs = kwargs

        # override this methods to check the
        self.result = self._checks()

    def __call__(self):
        """ Call for the performing of IO """
        self._pre()
        # For dumper, the obj is Literal str or bytes obj
        # For parser, the obj is Molecule obj
        io_func = self._get_io()
        if io_func:  # If a custom io function have been defined, run custom functions
            obj = io_func(self)
        else:  # else get the general io function define in class
            obj = self._io()

        return self._post(obj)

    @abstractmethod
    def _checks(self) -> Dict[str, Any]:
        """
        This method should be override when definition of new IO class
        The purpose of this class is to check the regulation of initialized arguments.
        If not any arguments should be check, return None directly.
        """
        raise NotImplemented()

    def _get_pre(self) -> Callable:
        return self.register.pre(self.fmt)

    def _get_io(self) -> Callable:
        return self.register.io(self.fmt)

    def _get_post(self) -> Callable:
        return self.register.post(self.fmt)

    def _pre(self, *args, **kwargs):
        """ Regulate the method of preprocess """
        pre_func = self._get_pre()
        if pre_func:
            self.src = pre_func(self)

    @abstractmethod
    def _io(self, *args, **kwargs):
        """ Regulate the main io method """
        raise NotImplemented

    def _post(self, obj, *args, **kwargs):
        """ Regulate the method of postprocess """
        post_func = self._get_post()
        if post_func:
            return post_func(self, obj)
        else:
            return obj

    @property
    def register(self) -> Register:
        return self._register


class Dumper(IOBase, metaclass=MetaIO):
    """
    Dump the Molecule information into specific format.
    The output in general is the string or bytes
    """

    # def __call__(self) -> Union[str, bytes]:
    #     """"""
    #     self._pre()
    #     script = self._io()
    #     return self._post(script)

    # def _pre(self):
    #     """ Preprocess the Molecule obj before performing the dumping """
    #     pre_func = self._get_pre()
    #     if pre_func:
    #         self.src = pre_func(self)

    def _io(self):
        """ Performing the IO operation, convert the Molecule obj to Literal obj """
        # Try to dump by openbabel.pybel
        try:
            pb_mol = pybel.Molecule(self.src.ob_mol)
            return pb_mol.write(self.fmt)

        except ValueError:
            print(IOError(f'the cheminfo.Molecule obj cannot dump to Literal'))
            return None

    # def _post(self, script: Union[str, bytes]):
    #     post_func = self._get_post()
    #     if post_func:
    #         return post_func(self, script)
    #     else:
    #         raise script

    # Define the dumper functions
    # The dumper functions should have two passing args
    # the first is the Dumper self obj and the second is the strings

    def _checks(self) -> Dict[str, Any]:
        if self.src.__class__.__name__ != 'Molecule':
            raise TypeError(f'the dumped object should be hotpot.cheminfo.Molecule, instead of {type(self.src)}')

        return {}

    def _post_gjf(self, script):
        """ postprocess the dumped Gaussian 16 .gjf script to add the link0 and route context """
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


class Parser(IOBase, metaclass=MetaIO):
    """ Parse the str or bytes obj to Molecule obj """
    _pybel_fmt_convert = {
        'g16log': 'g16'
    }

    def _checks(self) -> Dict[str, Any]:
        if not isinstance(self.src, (IOBase, str, bytes, PathLike)):
            raise TypeError(f'the parsed object should be IOBase, str or bytes, instead of {type(self.src)}')

        if isinstance(self.src, str):
            if os.path.exists(self.src):
                return {'src_type': 'path'}
            else:
                return {'src_type': 'str'}

        if isinstance(self.src, PathLike):
            return {'src_type': 'path'}

        if isinstance(self.src, bytes):
            return {'src_type': 'bytes'}
        if isinstance(self.src, io.StringIO):
            return {'src_type': 'StringIO'}
        if isinstance(self.src, io.BytesIO):
            return {'src_type': 'BytesIO'}
        if isinstance(self.src, io.FileIO):
            return {'src_type': 'FileIO'}
        print(f'the get source type is {type(self.src)}')
        return {'src_type': type(self.src)}

    def _io(self, *args, **kwargs):
        # Get the source type name
        src_type = self.result.get('src_type')

        # Try parse the log file by openbabel.pybel file firstly
        try:
            if src_type == 'str':
                pybel_mol = pybel.readstring(self._pybel_fmt_convert.get(self.fmt, self.fmt), self.src)
            elif src_type == 'path':
                pybel_mol = next(pybel.readfile(self._pybel_fmt_convert.get(self.fmt, self.fmt), self.src))
            elif src_type == 'IOString':
                pybel_mol = pybel.readstring(self._pybel_fmt_convert.get(self.fmt, self.fmt), self.src.read())
            else:
                raise RuntimeError(f'the source type {type(self.src)} have not been supported')

            obj = ci.Molecule(pybel_mol.OBMol)

        except RuntimeError:
            obj = None

        # Try to supplementary Molecule data by cclib
        try:
            if src_type == 'str':
                data = cclib.ccopen(io.FileIO(self.src)).parse()
            elif src_type == 'path':
                data = cclib.ccopen(self.src).parse()
            elif src_type == 'IOString':
                data = cclib.ccopen(self.src).parse()
            else:
                raise RuntimeError(f'the source type {type(self.src)} have not been supported in cclib')

        except (RuntimeError, AttributeError):
            data = None

        if data:
            if not obj:
                # if get information about the atoms species
                if hasattr(data, 'atomnos'):
                    atoms_attrs = [{'atomic_number': an} for an in getattr(data, 'atomnos')]
                    obj = ci.Molecule(atoms=atoms_attrs)
                else:
                    print(IOError(f'the parsing of {self.src} is not successful!'))
                    return obj  # Return None

            # if get information about the coordination collections
            if hasattr(data, 'atomcoords'):
                obj.set(coord_collect=getattr(data, 'atomcoords'))

            # if get information about the energy (SCF energies) vector
            if hasattr(data, 'scfenergies'):
                obj.set(energies=getattr(data, 'scfenergies'))

            # assign the first configure for the molecule
            obj.configure_select(0)

        return obj

    # Start to the prefix IO functions

    # postprocess for g16log file
    def _post_g16log(self, obj: 'ci.Molecule'):
        """
        post process for g16log format, to extract:
            1) Mulliken charge
            2) Spin densities
        """
        src_type = self.result.get('src_type')

        if src_type == 'str':
            lines = self.src.split('\n')

        elif src_type == 'path':
            with open(self.src) as file:
                lines = file.readlines()

        elif src_type == 'IOString':
            lines = self.src.readlines()

        else:
            raise RuntimeError('the source type {type(self.src)} have not been supported')

        # Get the line index of Mulliken charges
        head_lines = [i for i, line in enumerate(lines) if line.strip() == 'Mulliken charges and spin densities:']

        # Extract the Mulliken charge and spin densities
        charges, spin_densities = [], []
        for i in head_lines:
            # Enhance inspection
            col1, col2 = lines[i+1].strip().split()
            assert col1 == '1' and col2 == '2'

            sheet_idx = 2
            charge, spin_density = [], []

            while True:
                split_line = lines[i+sheet_idx].strip().split()
                if len(split_line) != 4:
                    break
                else:
                    row, syb, c, s = split_line  # row number, symbol, charges, spin density

                try:
                    row, c, s = int(row), float(c), float(s)
                    # check the sheet row number
                    if row != sheet_idx-1:
                        break

                # Inspect the types of values
                except ValueError:
                    break

                # record the charge and spin density
                charge.append(c)
                spin_density.append(s)
                sheet_idx += 1

            # store the extracted
            if charge and spin_density:
                if len(charge) == len(spin_density) == len(obj.atoms):
                    charges.append(charge)
                    spin_densities.append(spin_density)
                else:
                    raise ValueError('the number of charges do not match to the number of atoms')
            else:
                raise ValueError('get a empty charge and spin list, check the input!!')

        obj.set(atom_charges=np.array(charges))
        obj.set(atom_spin_densities=np.array(spin_densities))

        # assign the first configure for the molecule
        obj.configure_select(0)

        return obj
