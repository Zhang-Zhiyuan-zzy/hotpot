"""
python v3.7.9
@Project: hotpot
@File   : quantum.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/20
@Time   : 2:44
"""
import os
import resource
import subprocess
import io
from typing import *
from abc import ABC, abstractmethod

import cclib


class GaussianRunError(BaseException):
    """ Raise when the encounter error in run gaussian """


class Gaussian:
    """
    A class for setting up and running Gaussian 16 calculations.

    Attributes:
        g16root (str): The path to the Gaussian 16 root directory.

    """
    def __init__(
            self,
            g16root: Union[str, os.PathLike],
            report_set_resource_error: bool = False,
            error_handle_methods: Callable = None
    ):
        """
        This method sets up the required environment variables and resource limits for Gaussian 16.
        Args:
            g16root (Union[str, os.PathLike]): The path to the Gaussian 16 root directory.
            report_set_resource_error: Whether to report the errors when set the environments and resource
            error_handle_methods: the method to handle the release from g16

        Raises:
            TypeError: If `g16root` is not a string or a path-like object.
        """
        if isinstance(g16root, str):
            self.g16root = g16root
        elif isinstance(g16root, os.PathLike):
            self.g16root = str(g16root)
        else:
            raise TypeError('the g16root should be str or os.PathLike type!')

        self.envs = self._set_environs()
        self._set_resource_limits(report_set_resource_error)

        self.error_handle_methods = error_handle_methods

        # reverse for storing running data
        self.chk_path = None
        self.parsed_input = None
        self.g16process = None  # to link to the g16 subprocess

        self.stdout = None
        self.stderr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'{exc_type}, {exc_val}, {exc_tb}, {self.g16process.poll()}')
        if self.g16process.poll():
            self.g16process.kill()

    def _set_environs(self):
        """Sets up the environment variables required for running Gaussian 16.

        This method sets the environment variables required for Gaussian 16 to function correctly. If the
        `g16root` attribute is not set, the method sets it to the user's home directory.

        Returns:
            Dict[str, str]: A dictionary of the updated environment variables."""

        if self.g16root:
            g16root = self.g16root
        else:
            g16root = os.path.expanduser("~")

        # Setting environment for gaussian 16
        gr = g16root
        env_vars = {
            'g16root': gr,
            'GAUSS_EXEDIR': f"{gr}/g16/bsd:{gr}/g16",
            'GAUSS_LEXEDIR': f"{gr}/g16/linda-exe",
            'GAUSS_ARCHDIR': f"{gr}/g16/arch",
            'GAUSS_BSDDIR': f"{gr}/g16/bsd",
            'GV_DIR': f"{gr}/gv",
            'PATH': f"{os.environ['PATH']}:{gr}/gauopen:{gr}/g16/bsd:{gr}/g16",
            'PERLLIB': f"{os.environ['PERLLIB']}:{gr}/gauopen:{gr}/g16/bsd:{gr}/g16" if 'PERLLIB' in os.environ else f"{gr}/gauopen:{gr}/g16/bsd:{gr}/g16",
            'PYTHONPATH': f"{os.environ['PYTHONPATH']}:{gr}/gauopen:{gr}/g16/bsd:{gr}/g16" if 'PYTHONPATH' in os.environ else f"{gr}/gauopen:{gr}/g16/bsd:{gr}/g16",
            '_DSM_BARRIER': 'SHM',
            'LD_LIBRARY64_PATH': f"{gr}/g16/bsd:{gr}/gv/lib:{os.environ['LD_LIBRARY64_PATH']}" if 'LD_LIBRARY64_PATH' in os.environ else "",
            'LD_LIBRARY_PATH': f"{gr}/g16/bsd:{os.environ['LD_LIBRARY_PATH']}:{gr}/gv/lib" if 'LD_LIBRARY_PATH' in os.environ else f"{gr}/g16/bsd:{gr}/gv/lib",
            'G16BASIS': f"{gr}/g16/basis",
            'PGI_TERM': 'trace,abort'
        }

        # Merge the environment variables with the current environment
        updated_env = os.environ.copy()
        updated_env.update(env_vars)

        return updated_env

    @staticmethod
    def _parse_input_script(script: str) -> Dict[str, list[str]]:
        """ Parse the input script to dict """
        lines = script.splitlines()
        c = 0  # count of current line

        info = {}

        # Extract link0
        while lines[c][0] == '%':
            link0 = info.setdefault('link0', [])
            link0.append(lines[c])

            c += 1

        # Check link0
        if not info['link0']:
            raise ValueError('the provided input script is incorrect, not found link0 lines')

        # Extract route
        while lines[c] and lines[c][0] == '#':
            route = info.setdefault('route', [])
            route.append(lines[c])
            c += 1

        if not info['route']:
            raise ValueError('the provided input script is incorrect, not found route lines')

        # Extract the title line
        c += 1  # skip the blank line
        if lines[c]:
            info['title'] = lines[c]
        else:
            raise ValueError('the provided input script is incorrect, not found title lines')
        c += 2  # skip the blank line

        # Extract the molecular specification
        charge, spin = map(int, lines[c].strip().split())
        info['charge'], info['spin'] = charge, spin
        while lines[c].strip():
            mol_spec = info.setdefault('mol_spec', [])
            mol_spec.append(lines[c])
            c += 1

        # Extract other info
        i = 0
        while c < len(lines):
            other = info.setdefault(f'other_{i}', [])
            if lines[c].strip():
                other.append(lines[c])
            elif other:
                i += 1

            c += 1

        return info

    @staticmethod
    def _set_resource_limits(report_error: bool):
        """Sets resource limits for the Gaussian 16 process to avoid system crashes.

        This method sets resource limits for the Gaussian 16 process to avoid system crashes. Specifically,
        it sets the limits for the following resources: core dump size, data segment size, file size,
        locked-in-memory address space, resident set size, number of open files, stack size, CPU time,
        and number of processes.
        """
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_CORE limit.'))

        try:
            resource.setrlimit(resource.RLIMIT_DATA, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_DATA limit.'))

        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_FSIZE limit.'))

        try:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_MEMLOCK limit.'))

        try:
            resource.setrlimit(resource.RLIMIT_RSS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_RSS limit.'))

        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_NOFILE limit.'))

        try:
            resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_STACK limit.'))

        try:
            resource.setrlimit(resource.RLIMIT_CPU, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_CPU limit.'))

        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except ValueError:
            if report_error:
                print(RuntimeWarning('Unable to raise the RLIMIT_NPROC limit.'))

    def error_handle(self, stdout: str, stderr: str) -> bool:
        """
        Handle the error message release information.
        Args:
            stdout: the standard output message
            stderr: the standard error message

        Returns:
            whether to raise the error (bool), error massage
        """
        return isinstance(self.error_handle_methods, Callable) and self.error_handle_methods(self, stdout, stderr)

    def molecule_setter_dict(self, stdout: str) -> dict:
        """ Prepare the property dict for Molecule setters """
        data = self.parse_log(stdout)
        return {
            'atoms.partial_charge': data.atomcharges['mulliken'],
            'energy': data.scfenergies[-1],
            'spin': data.mult,
            'charge': data.charge,
            'mol_orbital_energies': data.moenergies,  # eV,
            'coordinates': data.atomcoords[-1]
        }

    @staticmethod
    def parse_log(stdout: str):
        """ Parse the gaussian log file and save them into self """
        string_buffer = io.StringIO(stdout)
        return cclib.ccopen(string_buffer).parse()

    def run(self, script: str):
        """Runs the Gaussian 16 process with the given script and additional arguments.

        This method sets up the required environment variables and resource limits for Gaussian 16 before
        running the process using `subprocess.Popen`. It takes an input script and any additional arguments
        to pass to `Popen`, and returns a tuple of the standard output and standard error of the process.

        Args:
            script (str): The input script for the Gaussian 16 process.
        Returns
            Tuple[str, str]: A tuple of the standard output and standard error of the process
        """
        self.parsed_input = self._parse_input_script(script)  # parse input data
        with open('input.gjf', 'w') as writer:
            writer.write(script)

        # Run Gaussian using subprocess
        self.g16process = subprocess.Popen(
            ['g16', 'input.gjf', 'output.log'], bufsize=-1, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.envs, universal_newlines=True
        )
        self.stdout, self.stderr = self.g16process.communicate()

        if not self.stdout:
            with open('output.log') as file:
                self.stdout = file.read()

        if self.stderr:
            # If the error_handle_methods have been configured and the error is handled correctly, return normally.
            if self.error_handle_methods and self.error_handle_methods(self):
                return self.stdout, self.stderr
            else:
                raise GaussianRunError(self.stderr)

        return self.stdout, self.stderr


# class Gauss:
#     """ Bata version of Gaussian """
#     def __init__(
#             self,
#             g16root: Union[str, os.PathLike] = None,
#             path_input: Union[str, os.PathLike] = None,
#             path_output: Union[str, os.PathLike] = None,
#             path_chk: Union[str, os.PathLike] = None,
#             path_rwf: Union[str, os.PathLike] = None,
#     ):
#         """"""
#         # Check or specify the g16root
#         if not g16root:
#             g16root = os.environ.get('g16root')
#
#         if not g16root:
#             raise EnvironmentError("do not find the environmental variable 'g16root', please specify manually")
#         else:
#             g16root = Path(g16root)
#
#         try:
#             assert g16root.joinpath('g16', 'g16').is_file()
#             assert g16root.joinpath('g16', 'bsd', 'g16.profile').is_file()
#         except AssertionError:
#             raise FileNotFoundError('the specified g16root is error, do not find executable and profile files')
#
#         os.system(f"{g16root.joinpath('g16', 'bsd', 'g16.profile')}")
#         self.g16root = g16root
#
#         self.path_input = Path(path_input) if path_input else Path.cwd().joinpath('input.gjf')
#         self.path_output = Path(path_output) if path_input else  Path.cwd().joinpath('output.log')
#         self.path_chk = Path(path_chk)
#         self.path_rwf = Path(path_rwf)
#
#         # reserve variables for g16 running
#         self.input_info = None
#         self.g16process = None
#         self.stdout = None
#         self.stderr = None
#
#     @property
#     def _envs(self):
#         return
#
#     @staticmethod
#     def _parse_input_script(script: str) -> Dict[str, list[str]]:
#         """ Parse the input script to dict """
#         lines = script.splitlines()
#         c = 0  # count of current line
#
#         info = {}
#
#         # Extract link0
#         while lines[c][0] == '%':
#             link0 = info.setdefault('link0', [])
#             link0.append(lines[c])
#
#             c += 1
#
#         # Check link0
#         if not info['link0']:
#             raise ValueError('the provided input script is incorrect, not found link0 lines')
#
#         # Extract route
#         while lines[c][0] == '#':
#             route = info.setdefault('route', [])
#             route.append(lines[c])
#
#         if not info['route']:
#             raise ValueError('the provided input script is incorrect, not found route lines')
#
#         # Extract the title line
#         c += 1  # skip the blank line
#         if lines[c]:
#             info['title'] = lines[c]
#         else:
#             raise ValueError('the provided input script is incorrect, not found title lines')
#         c += 1  # skip the blank line
#
#         # Extract the molecular specification
#         charge, spin = map(int, lines[c].strip().split())
#         info['charge'], info['spin'] = charge, spin
#         while lines[c].strip():
#             mol_spec = info.setdefault('mol_spec', [])
#             mol_spec.append(lines[c])
#             c += 1
#
#         # Extract other info
#         i = 0
#         while c < len(lines):
#             other = info.setdefault(f'other_{i}', [])
#             if lines[c].strip():
#                 other.append(lines[c])
#             elif other:
#                 i += 1
#
#             c += 1
#
#         return info
#
#     def _set_resource_limits(self):
#         pass
#
#     def run(
#             self, *,
#             script: str = None,
#             mol: ci.Molecule = None,
#             link0: Union[List[str], str] = "nproc=4",
#             route: Union[List[str], str] = "SP B3LYP/6-311++G**",
#             script_supply: str = ''
#     ):
#         """"""
#         if script:
#             with (open(self.path_input, 'w')) as writer:
#                 writer.write(script)
#
#         elif isinstance(mol, ci.Molecule):
#
#             # Organize the link0
#             if isinstance(link0, str):
#                 link0 = [link0]
#             link0.extend([f'RWF={str(self.path_rwf)}', 'NoSave', f'chk={str(self.path_chk)}'])
#
#             # Organize the route
#             if isinstance(route, str):
#                 route = [route]
#
#             script = mol.dump('gjf', link0=link0, route=route)
#
#             script += script_supply
#
#             with (open(self.path_input, 'w')) as writer:
#                 writer.write(script)
#
#         else:
#             raise ValueError('the script or mol arg should give at least one')
#
#         self.input_info = self._parse_input_script(script)
#
#         # Run Gaussian using subprocess
#         self.g16process = subprocess.Popen(
#             ['g16', 'input.gjf', 'output.log'], bufsize=-1, stdin=subprocess.PIPE,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#             env=self._envs, universal_newlines=True
#         )
#         self.stdout, self.stderr = self.g16process.communicate()


class GaussErrorHandle(ABC):
    """ Basic class to handle the error release from gaussian16 """
    def __call__(self, g16proc: Gaussian, stdout: str, stderr: str) -> bool:
        """ Call for handle the g16 errors """
        return self.error_handle(g16proc, stdout, stderr)

    @abstractmethod
    def error_handle(self, g16proc: Gaussian, stdout: str, stderr: str) -> bool:
        """ Specified by the children classes """
