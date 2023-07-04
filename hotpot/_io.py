"""
python v3.7.9
@Project: hotpot
@File   : _io.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/14
@Time   : 4:18
"""
import os
import re
from pathlib import Path
from os import PathLike
from typing import *
from abc import ABCMeta, abstractmethod
from copy import copy
import io
from io import IOBase
from openbabel import pybel
import cclib
import numpy as np

import hotpot.cheminfo as ci
from hotpot.tanks.deepmd import DeepSystem

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


# Define custom Exceptions
class IOEarlyStop(BaseException):
    """ monitor the situation that the IO should early stop and return None and the IO result """


# Define the IO function types
IOFuncPrefix = Literal['pre', 'io', 'post']
IOStream = Union[IOBase, str, bytes]


def _parse_lmp_data_script(script: str):
    """ Parse the LAMMPS data script to two dict, header and body"""
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
        bc = [line.strip() for line in lines[sl_idx+1: el_idx] if line.strip()]  # body content
        bodies[bt] = bc

    return lines[0], headers, bodies


class Register:
    """
    Register the IO function for Dumper, Parser or so on
    """
    def __init__(self):
        # these dicts are container to store the custom io functions
        # the keys of the dict are serve as the handle to get the mapped io functions(the values)
        self.pre_methods = {}
        self.io_methods = {}
        self.post_methods = {}

    def __repr__(self):
        return f"Register:\n" + \
               f"pre_method:\n" + \
               f"\n\t".join([n for n in self.pre_methods]) + "\n\n" + \
               f"io methods:\n" + \
               f"\n\t".join([n for n in self.io_methods]) + '\n\n' + \
               f"post methods:\n" + \
               f"\n\t".join([n for n in self.post_methods])

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


# Retrieve the IO class by its format name
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
    The Meta class to specify how to construct the IO class
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

        namespace[f'_register'] = _register

        return type(name, bases, namespace, **kwargs)


class IOBase:
    """ The base IO class """
    # Initialize the register function, which is a callable obj embed in IO classes
    # When to register new IO function, apply the register function as decorator

    # _register = None

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
        try:
            self._pre()
            # For dumper, the obj is Literal str or bytes obj
            # For parser, the obj is Molecule obj
            io_func = self._get_io()
            if io_func:  # If a custom io function have been defined, run custom functions
                obj = io_func(self)
            else:  # else get the general io function define in class
                obj = self._io()

            return self._post(obj)

        except IOEarlyStop:
            return None

    @abstractmethod
    def _checks(self) -> Dict[str, Any]:
        """
        This method should be overriden when definition of new IO class
        The purpose of this class is to check the regulation of initialized arguments.
        If not any arguments should be checked, return None directly.
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
            pre_func(self)

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
        return getattr(self, f'_register')


class Dumper(IOBase, metaclass=MetaIO):
    """
    Dump the Molecule information into specific format.
    The output in general is the string or bytes
    """

    _pybel_fmt_convert = {
    }

    def _process_lmpdat_bonds(self, bond_contents: list):
        """"""
        uni_bonds = tuple(self.src.unique_bonds)
        bonds = self.src.bonds
        sep = re.compile(r'\s+')
        for i, bc in enumerate(bond_contents):
            split_bc = sep.split(bc)
            split_bc[1] = str(uni_bonds.index(bonds[i]) + 1)
            bond_contents[i] = '  '.join(split_bc)

        return bond_contents

    def _io(self):
        """ Performing the IO operation, convert the Molecule obj to Literal obj """
        # Try to dump by openbabel.pybel
        type_err_pattern = re.compile(
            r"write\(\) got an unexpected keyword argument '\w+'"
        )
        pb_mol = pybel.Molecule(self.src.ob_mol)
        kwargs = copy(self.kwargs)

        while kwargs:
            try:
                return pb_mol.write(self._pybel_fmt_convert.get(self.fmt, self.fmt), **kwargs)

            except TypeError as error:
                if type_err_pattern.match(str(error)):
                    pop_kwargs = str(error).split()[-1].strip("'")
                    kwargs.pop(pop_kwargs)
                else:
                    raise error

            except ValueError:
                print(IOError(f'the cheminfo.Molecule obj cannot dump to Literal'))
                return None

        return pb_mol.write(self._pybel_fmt_convert.get(self.fmt, self.fmt))

    def _checks(self) -> Dict[str, Any]:
        if not isinstance(self.src, ci.Molecule):
            raise TypeError(f'the dumped object should be hotpot.cheminfo.Molecule, instead of {type(self.src)}')

        return {}

    def _pre_cif(self):
        """
        pre-process for Molecule object to convert to cif file.
        if the hotpot object do not place in a Crystal, create a P1 compact Crystal for it
        """
        crystal = self.src.crystal()
        if not isinstance(crystal, ci.Crystal) or (
                np.logical_not(crystal.vector >= 0.).any() and np.logical_not(crystal.vector < 0.).any()
        ):
            self.src.compact_crystal(inplace=True)

        if self.src.crystal().space_group:
            self.src.crystal().space_group = 'P1'

    def _pre_gjf(self):
        """ Assign the Molecule charge before to dump to gjf file """
        if not self.src.has_3d:
            self.src.build_3d()

        self.src.assign_atoms_formal_charge()
        self.src.identifier = self.src.formula

    def _io_dpmd_sys(self):
        """ convert molecule information to numpy arrays """
        return DeepSystem(self.src)

    def _io_lmpmol(self):
        """
        write a molecule script
        default values: coordinates, velocities, atom IDs and types
        additional attributes for atomic: Bonds
        additional attributes for full: Bonds + molecular + charge
        """

        def bonds(m):
            """ Add bond body """
            bond_str = 'Bonds' + '\n\n'  # bond body title

            # the formula of bond_type key: atom1[bond_type]atom2
            uni_bonds = tuple(m.unique_bonds)  # store bonds type
            for j, bond in enumerate(m.bonds, 1):

                bt_id = uni_bonds.index(bond) + 1
                bond_str += f'{j} {bt_id} {bond.ob_atom1_id + 1} {bond.ob_atom2_id + 1}\n'

            bond_str += '\n'

            return bond_str

        def charge():
            """ Retrieve atom charge information """
            charge_str = '\n' + 'Charges' + '\n\n'

            for ic, a in enumerate(atoms_list, 1):  # ID of charge, atom
                if isinstance(a, ci.Atom):
                    charge_str += f'{ic} {a.partial_charge}\n'
                else:
                    assert isinstance(a, ci.PseudoAtom)
                    charge_str += f'{ic} {a.charge}\n'

            charge_str += '\n'

            return charge_str

        mol = self.src
        kwargs = self.kwargs  # keywords arguments

        # default values: coordinates, velocities, atom IDs and types;
        # additional attributes for atomic: None;
        # additional attributes for full: molecular + charge
        atom_style = kwargs.get('atom_style', 'atomic')   # default atom_style is atomic
        mol_name = kwargs.get('mol_name', mol.smiles)

        # combine real atoms with pseudo atoms in a list
        atoms_list = []
        for m_a in mol.atoms:
            atoms_list.append(m_a)

        if mol.pseudo_atoms:   # determine if there are pseudo_atoms
            for pse_a in mol.pseudo_atoms:
                atoms_list.append(pse_a)

        # title information
        title = f"Create by hotpot package, convert from {mol_name}"
        script = title + '\n\n'   # write the molecular script for lammps

        # TODO: some header information missing
        # Header partition
        # add atom header
        num_atoms = len(atoms_list)
        num_atoms_str = f'{num_atoms}  atoms'
        script += num_atoms_str + '\n'

        # add bond header
        num_bonds = len(mol.bonds)
        num_bonds_str = f'{num_bonds}  bonds'
        script += num_bonds_str + '\n'

        # Add new blank line to end the header partition
        script += '\n'

        # Body partition
        # Coords body
        script += 'Coords' + '\n\n'
        for i, atom in enumerate(atoms_list, 1):
            script += f'{i}' + '  ' + '  '.join(map(str, atom.coordinates)) + '\n'
        script += '\n'

        # Types body
        script += 'Types' + '\n\n'

        dict_types = {}
        for i, atom in enumerate(atoms_list, 1):
            atom_type = dict_types.setdefault(atom.symbol, len(dict_types)+1)
            script += f'{i} {atom_type}  # {atom.symbol}\n'

        script += '\n'

        # additional attributes
        # to atomic style, only basis information (ID，Coords, types, velocitier)
        if atom_style == 'atomic':
            if num_bonds:
                script += bonds(mol)

        # to full style, basis information + molecular + charge
        elif atom_style == 'full':
            if num_bonds:
                script += bonds(mol)
            script += charge()

        return script

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
                if not i:  # For the first line of link0, replace the original line in raw script
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

    def _post_lmpdat(self, script: str):
        """ post-process for LAMMPS data file """

        title, headers, bodies = _parse_lmp_data_script(script)
        script = title + '\n'

        for ht, hvs in headers.items():
            if ht == 'bond types':  # header title, header values
                hvs[0] = len(self.src.unique_bonds)

            script += ' '.join(map(str, hvs)) + ' ' + ht + '\n'

        script += '\n' * 3

        for bt, bcs in bodies.items():  # body title, body contents
            if bt == 'Bonds':
                bcs = self._process_lmpdat_bonds(bcs)

            if bcs:  # if the body contents exist
                script += bt + '\n' * 2
                script += '\n'.join(bcs)
                script += '\n' * 3

        return script


class Parser(IOBase, metaclass=MetaIO):
    """ Parse the str or bytes obj to Molecule obj """
    _pybel_fmt_convert = {
        'g16log': 'g16'
    }

    def _open_source_to_string_lines(self, *which_allowed: str, output_type: Literal['lines', 'script'] = 'lines'):
        """
        Open the source file to string lines
        Args:
            which_allowed: which types of source are allowed to process to string lines

        Returns:
            (List of string|string)
        """
        src_type = self.result.get('src_type')
        if src_type not in which_allowed:
            raise RuntimeError(f'the source type {type(self.src)} have not been supported')
        else:
            if src_type == 'str':
                script = self.src

            elif src_type == 'path':
                with open(self.src) as file:
                    try:
                        script = file.read()
                    # If the file pointed by the path is not a text file
                    # such as a bytes file
                    except UnicodeDecodeError:
                        raise IOEarlyStop()

            elif src_type == 'IOString':
                script = self.src.read()

            else:
                raise RuntimeError(f'the source type {type(self.src)} have not been supported')

            if output_type == 'lines':
                return script.split('\n')
            elif output_type == 'script':
                return script
            else:
                raise ValueError('the arg output_type given a wrong values, lines or script allow only')

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

    def _ob_io(self):
        """ IO by openbabel.pybel """
        # Get the source type name
        src_type = self.result.get('src_type')
        try:
            if src_type == 'str':
                pybel_mol = pybel.readstring(self._pybel_fmt_convert.get(self.fmt, self.fmt), self.src)
            elif src_type == 'path':
                pybel_mol = next(pybel.readfile(self._pybel_fmt_convert.get(self.fmt, self.fmt), str(self.src)))
            elif src_type == 'IOString':
                pybel_mol = pybel.readstring(self._pybel_fmt_convert.get(self.fmt, self.fmt), self.src.read())
            else:
                raise RuntimeError(f'the source type {type(self.src)} have not been supported')

            obj = ci.Molecule(pybel_mol.OBMol)

        except RuntimeError:
            obj = None

        return obj

    def _cclib_io(self, obj):
        """ IO by cclib package """
        src_type = self.result.get('src_type')

        try:
            if src_type == 'str':
                data = cclib.ccopen(io.StringIO(self.src)).parse()
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
                obj.set(all_coordinates=getattr(data, 'atomcoords'))

            # if get information about the energy (SCF energies) vector
            if hasattr(data, 'scfenergies'):
                obj.set(all_energy=getattr(data, 'scfenergies'))

        return obj

    def _io(self, *args, **kwargs):
        """ Standard IO process """
        # Try parse the log file by openbabel.pybel file firstly
        return self._ob_io()

    # Start to the prefix IO functions

    # preprocess for g16log file
    # This preprocess is used to judge whether a Error happened when perform g16 calculate
    def _pre_g16log(self):
        """ g16log preprocess to judge whether some Error happened """
        def is_convergence_failure():
            if 'Convergence failure -- run terminated.' in script:
                return True
            return False

        def is_hessian_no_longer_linear_valid():
            march_pattern = re.compile(r'Error termination via Lnk1e in (/.+)*/l103\.exe')

            if any (march_pattern.match(line.strip()) for line in script.splitlines()[-5:]):
                return True
            return False

        script = self._open_source_to_string_lines('str', 'path', "IOString", output_type='script')

        # Check whether a failure have happened when calculation.
        if is_convergence_failure():
            raise IOEarlyStop('Gaussian16 SCF cannot convergence!')
        if is_hessian_no_longer_linear_valid():
            raise IOEarlyStop('Gaussian16 Hessian no longer linear valid')

    # Parse the XYZ file
    def _io_xyz(self):
        """ Parse the XYZ file """
        src_type = self.result['src_type']
        if src_type != 'path':
            return self._io()
        else:
            from ase import io
            from openbabel import openbabel as ob
            data_generator = io.iread(self.src)

            # atomic_numbers, coordinates, cell_params
            atomic_numbers, all_coordinates, cell_matrix = [], [], None
            for data in data_generator:
                atomic_numbers.append(data.numbers)
                all_coordinates.append(data.positions)
                if cell_matrix is None:
                    cell_matrix = data.cell.array

            atomic_numbers = np.stack(atomic_numbers)
            all_coordinates = np.stack(all_coordinates)

            number_min = atomic_numbers.min(axis=0)
            number_max = atomic_numbers.max(axis=0)

            # the values in same columns should be same.
            assert all(number_max == number_min)

            obj = ci.Molecule()
            obj.quick_build_atoms(number_min)

            obj.set(all_coordinates=all_coordinates)
            obj.set(crystal=cell_matrix)
            obj.conformer_select(0)

            return obj

    # postprocess for g16log file
    def _post_g16log(self, obj: 'ci.Molecule'):
        """
        post process for g16log format, to extract:
            1) Mulliken charge
            2) Spin densities
        """
        def extract_charges_spin():
            """ Extract charges and spin information from g16.log file """
            # Get the line index of Mulliken charges
            head_lines = [i for i, line in enumerate(lines) if line.strip() == 'Mulliken charges and spin densities:']
            if not head_lines:
                head_lines = [i for i, line in enumerate(lines) if line.strip() == 'Mulliken charges:']
                charge_only = True
            else:
                charge_only = False

            # Skip the first charge&spin sheet, it can't find corresponding coordinates
            if not head_lines:
                raise IOEarlyStop
            elif len(head_lines) == obj.conformer_counts + 1:
                head_lines = head_lines[1:]

            # Extract the Mulliken charge and spin densities
            charges, spin_densities = [], []  # changes(cgs) spin_densities(sds)
            for i in head_lines:
                # Enhance inspection
                col_heads = lines[i + 1].strip().split()
                if charge_only:
                    assert len(col_heads) == 1 and col_heads[0] == '1'
                else:
                    assert len(col_heads) == 2 and col_heads[0] == '1' and col_heads

                HEAD_LINES_NUM = 2
                cg, sd = [], []  # change, spin_density

                while True:
                    split_line = lines[i + HEAD_LINES_NUM].strip().split()
                    if charge_only and len(split_line) == 3:
                        row, syb, c = split_line  # row number, symbol, charges
                        s = 0.0  # spin density
                    elif not charge_only and len(split_line) == 4:
                        row, syb, c, s = split_line  # row number, symbol, charges, spin density
                    else:
                        break

                    try:
                        row, c, s = int(row), float(c), float(s)
                        # check the sheet row number
                        if row != HEAD_LINES_NUM - 1:
                            break

                    # Inspect the types of values
                    except ValueError:
                        break

                    # record the charge and spin density
                    cg.append(c)
                    sd.append(s)
                    HEAD_LINES_NUM += 1

                # store the extracted
                if cg and sd:
                    if len(cg) == len(sd) == len(obj.atoms):
                        charges.append(cg)
                        spin_densities.append(sd)
                    else:
                        raise ValueError('the number of charges do not match to the number of atoms')
                else:
                    raise ValueError('get a empty charge and spin list, check the input!!')

            obj.set(all_atom_charges=np.array(charges))
            obj.set(all_atom_spin_densities=np.array(spin_densities))
                
        def extract_force_matrix():
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
            for i in head_lines:
                # enhance the inspection of Force sheet head
                assert force_head2.match(lines[i + 1])
                assert sheet_line.match(lines[i + 2])

                rows = 0
                forces = []
                while True:

                    if sheet_line.match(lines[i + HEAD_LINES_NUM + rows]):
                        if len(forces) == obj.atom_counts:
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
                    if obj.atoms[rows].atomic_number != an:
                        raise ValueError('the atomic number do not match')

                    forces.append([x, y, z])

                    rows += 1

            try:
                obj.set(all_forces=np.array(all_forces))
            except ValueError:
                return

        obj = self._cclib_io(obj)  # Try to supplementary Molecule data by cclib

        lines = self._open_source_to_string_lines('str', 'path', 'IOString')

        try:  # TODO: For now, this is the case, the spin densities may lost in some case  # the units is Hartree/Bohr
            extract_charges_spin()
            extract_force_matrix()
        except IndexError:
            raise IOEarlyStop

        # assign the first conformer for the molecule
        obj.conformer_select(0)

        return obj


