"""
python v3.7.9
@Project: hotpot
@File   : qm.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/20chgrp
@Time   : 2:44
"""
import os
import re
import io
import json
import resource
import subprocess
from pathlib import Path
from typing import *

import cclib

from hotpot import settings


_dir_root = os.path.dirname(os.path.abspath(__file__))


class GaussianRunError(BaseException):
    """ Raise when the encounter error in run gaussian """


class FailToHandle(Warning):
    """ Report this Warning when GaussErrorHandle Fail to handle an error """


class ECPNotFound(BaseException):
    """ Raise when failing to retrieve an Effective Core Potential"""


class GaussOut:
    """
    This class is used to store Gaussian output and error message from g16 process.
    In addition, this class will extract and organize critical information.
    """

    # Compile the error notice sentence
    _head = re.compile('Error termination via Lnk1e in')
    _link = re.compile(r'l\d+[.]exe')
    _path = re.compile(r'([/|\\]\S+)*[/|\\]' + _link.pattern)
    _week = re.compile('(Mon|Tue|Wed|Thu|Fri|Sat|Sun)')
    _month = re.compile('(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)')
    _date = re.compile(_week.pattern + ' ' + _month.pattern + r' +[0-2]?\d')
    _time = re.compile(r'\d{2}:\d{2}:\d{2} 20\d{2}\.')

    _error_link = re.compile(_head.pattern + ' ' + _path.pattern + ' at ' + _date.pattern + ' ' + _time.pattern)

    def __init__(self, stdout: str, stderr: str = ''):
        self.stdout = stdout
        self.stderr = stderr

    @property
    def is_error(self) -> bool:
        return True if self.stderr or self._error_link.search(self.stdout) else False

    @property
    def error_link(self) -> str:
        if self.is_error:
            match = self._error_link.search(self.stdout)
            if match:
                matched_line = self.stdout[match.start():match.end()]

                link = self._link.search(matched_line)
                return matched_line[link.start(): link.end()][:-4]

    @property
    def is_hangup_error(self):
        if self.is_error and self.stderr.find('Error: hangup') > 0:
            return True
        return False

    @property
    def is_opti_convergence_error(self):
        """ The gaussian error is caused by the non-convergence of the optimizing conformer """
        if self.is_error and self.error_link == 'l9999' and self.stdout.find('-- Number of steps exceeded,'):
            return True
        return False

    @property
    def is_normal_terminate(self):
        """ Judge whether the finished gaussian work is complete with normal terminate """
        if self.stdout.splitlines()[-1].strip().startswith('Normal termination of Gaussian 16 at'):
            return True
        return False

    @property
    def is_scf_convergence_error(self):
        """ Get True when the Output show the SCF non-convergence """
        if self.error_link == 'l502' and self.stdout.find("Convergence failure -- run terminated."):
            return True
        return False

    @property
    def is_scrf_Vdw_cage_error(self):
        """ Error caused by the Vdw surface is not suitable to estimate the accessible surface inside molecular cage """
        if self.error_link == 'l502' and self.stdout.find("Inv3 failed in PCMMkU."):
            return True
        return False

    @property
    def is_ZMatrix_error(self):
        if self.error_link == 'l103' and \
                self.stdout.find('FormBX had a problem.') and \
                self.stdout.find('Berny optimization.'):
            return True

        return False

    def report(self, show_screen=False) -> list[str]:
        """ Report all error messages """
        error_judge = re.compile(r'is_.+_error')

        print("Meet Gaussian Error:")
        errors = []
        for name in self.__dir__():
            if hasattr(self, name) and error_judge.fullmatch(name) and getattr(self, name):
                errors.append(name)

                if show_screen:
                    print(f'\t--{name[3:]};')

        return errors


class Gaussian:
    """
    A class for setting up and running Gaussian 16 calculations.

    Attributes:
        g16root (str): The path to the Gaussian 16 root directory.

    """
    def __init__(
            self,
            g16root: Union[str, os.PathLike] = None,
            path_gjf: Union[str, os.PathLike] = None,
            path_log: Union[str, os.PathLike] = None,
            path_err: Union[str, os.PathLike] = None,
            report_set_resource_error: bool = False,
            output_in_running: bool = True
    ):
        """
        This method sets up the required environment variables and resource limits for Gaussian 16.
        Args:
            g16root (Union[str, os.PathLike]): The path to the Gaussian 16 root directory.
            path_gjf: the path of input script to be written and read
            path_log: the path of output result to be written and read
            path_err: the path of  error message to be written
            report_set_resource_error: Whether to report the errors when set the environments and resource
            options: the Option object
            output_in_running: If true, the gaussian program will write the output.log file when running,
             else get the stdout after the program terminal

        Keyword Args:
            this could give any arguments for GaussErrorHandle

        Raises:
            TypeError: If `g16root` is not a string or a path-like object.
        """
        if g16root:
            self.g16root = Path(g16root)
        elif settings.get("paths", {}).get("g16root", {}):
            self.g16root = Path(settings.get("paths", {}).get("g16root", {}))
        else:
            raise ValueError('the argument g16root is not given!')

        # Configure running environments and resources
        self.envs = self._set_environs()
        self._set_resource_limits(report_set_resource_error)

        # the default running input and output file
        self.gjf_path = Path('input.gjf') if not path_gjf else Path(path_gjf)
        self.logout_path = Path('output.log') if not path_log else Path(path_log)
        self.error_path = Path(path_err) if path_err else None

        # preserve for storing running data
        self.path_chk = None
        self.path_rwf = None

        self.parsed_input = {}

        self.g16process = None  # to link to the g16 subprocess
        self.output_in_running = output_in_running

        self.stdin = None
        self.output = None
        self.stdout = None
        self.stderr = None

    def _set_environs(self):
        """
        Sets up the environment variables required for running Gaussian 16.

        This method sets the environment variables required for Gaussian 16 to function correctly. If the
        `g16root` attribute is not set, the method sets it to the user's home directory.

        Returns:
            Dict[str, str]: A dictionary of the updated environment variables.
        """

        if self.g16root:
            g16root = str(self.g16root)
        else:
            g16root = os.path.expanduser("~")

        GAUOPEN = f'{g16root}:gauopen'
        GAUSS_EXEDIR = f'{g16root}/g16/bsd:{g16root}/g16'
        GAUSS_LEXEDIR = f"{g16root}/g16/linda-exe"
        GAUSS_ARCHDIR = f"{g16root}/g16/arch"
        GAUSS_BSDDIR = f"{g16root}/g16/bsd"
        GV_DIR = f"{g16root}/gv"

        PATH = os.environ.get('PATH')
        if PATH:
            PATH = f'{PATH}:{GAUOPEN}:{GAUSS_EXEDIR}'
        else:
            PATH = f'{GAUOPEN}:{GAUSS_EXEDIR}'

        PERLLIB = os.environ.get('PERLLIB')
        if PERLLIB:
            PERLLIB = f'{PERLLIB}:{GAUOPEN}:{GAUSS_EXEDIR}'
        else:
            PERLLIB = f'{GAUOPEN}:{GAUSS_EXEDIR}'

        PYTHONPATH = os.environ.get('PYTHONPATH')
        if PYTHONPATH:
            PYTHONPATH = f'{PYTHONPATH}:{GAUOPEN}:{GAUSS_EXEDIR}'
        else:
            PYTHONPATH = f'{PYTHONPATH}:{GAUSS_EXEDIR}'

        _DSM_BARRIER = "SHM"
        LD_LIBRARY64_PATH = None
        LD_LIBRARY_PATH = None
        if os.environ.get('LD_LIBRARY64_PATH'):
            LD_LIBRARY64_PATH = f"{GAUSS_EXEDIR}:{GV_DIR}/lib:{os.environ['LD_LIBRARY64_PATH']}"
        elif os.environ.get('LD_LIBRARY64_PATH'):
            LD_LIBRARY_PATH = f"{GAUSS_EXEDIR}:{os.environ['LD_LIBRARY_PATH']}:{GV_DIR}/lib"
        else:
            LD_LIBRARY_PATH = f"{GAUSS_EXEDIR}:{GV_DIR}/lib"

        G16BASIS = f'{g16root}/g16/basis'
        PGI_TEAM = f'trace,abort'

        env_vars = {
            'g16root': g16root,
            'GAUSS_EXEDIR': GAUSS_EXEDIR,
            'GAUSS_LEXEDIR': GAUSS_LEXEDIR,
            'GAUSS_ARCHDIR': GAUSS_ARCHDIR,
            'GAUSS_BSDDIR': GAUSS_BSDDIR,
            'GV_DIR': GV_DIR,
            'PATH': PATH,
            'PERLLIB': PERLLIB,
            'PYTHONPATH': PYTHONPATH,
            '_DSM_BARRIER': _DSM_BARRIER,
            'LD_LIBRARY64_PATH': LD_LIBRARY64_PATH,
            'LD_LIBRARY_PATH': LD_LIBRARY_PATH,
            'G16BASIS': G16BASIS,
            'PGI_TERM': PGI_TEAM
        }
        env_vars = {n: v for n, v in env_vars.items() if v is not None}

        # Merge the environment variables with the current environment
        updated_env = os.environ.copy()
        updated_env.update(env_vars)

        return updated_env

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

    @staticmethod
    def auto_ECP_select(
            element: str, size: Literal['S', 'L'] = 'L', charge: int = None, allowHF: bool = False
    ) -> str:
        """
        Given the element, retrieving the ECP name according to the Core size
        Args:
            element:
            size: Core size, selecting for small(S), middle(M) and large(L)
            allowHF: whether to allow HF level core to be used

        Returns:
            the ECPs name
        Raises:
            ECPNotFound: there is not any recommend ECP;
            ValueError: give an error size
        """
        if not charge:
            from hotpot.cheminfo import Atom
            charge = Atom(symbol=element).stable_charge

        if charge < 3:
            recommend_XY = ['SDF', 'MWB', 'MDF'] if allowHF else ['SDF', 'MWB', 'MDF', 'SHF', 'MHF']
        else:
            recommend_XY = ['MWB', 'MDF', 'SDF'] if allowHF else ['MWB', 'MDF', 'SDF', 'MHF', 'SHF']

        with open(os.path.join(_dir_root, 'ECPs.json')) as file:
            ecp = json.load(file)
            assert isinstance(ecp, dict)

        for XY in recommend_XY:
            if electron_number_list := ecp.get(element, {}).get(XY):
                if len(electron_number_list) == 1 or size == 'S':
                    return f"{XY}{electron_number_list[0]}"
                elif size != 'L':
                    raise ValueError("the value of size just allows 'S' and 'L'")

                if XY == "MWB":
                    if len(electron_number_list) == 2:
                        return f"{XY}{electron_number_list[-1]}"

                    # For Lanthanide
                    elif len(electron_number_list) == 3:
                        if charge == 2:
                            return f"{XY}{electron_number_list[-1]}"
                        elif charge == 3:
                            return f"{XY}{electron_number_list[-2]}"

                elif XY == "MHF":
                    assert len(electron_number_list) == 2
                    if electron_number_list[1] - electron_number_list[0] == 1:
                        return f"{XY}{electron_number_list[0]}" if charge == 3 else f"{XY}{electron_number_list[1]}"
                    else:
                        return f"{XY}{electron_number_list[0]}" if size == 'S' else f"{XY}{electron_number_list[1]}"

                # TODO: other XY only with one electron_number_list in current Gaussian16

        raise ECPNotFound('fail to found a recommended ECP')

    @staticmethod
    def get_ECPs(
            element: str = None, XY: Literal['MWB', 'SDF', 'SHF', 'MDF', 'MDF'] = None, n: int = None
    ) -> Union[str, None]:
        """ Retrieve Effective Core Potentials (ECPs), if the given ECP is not found return None  """
        if not element or not XY:
            return 'SDD'

        with open(_dir_root, 'ECPs.json') as file:
            ecp = json.load(file)
            assert isinstance(ecp, dict)

        if electron_number_list := ecp.get(element, {}).get(XY):
            return f'{XY}{n}' if n else (f"{XY}{electron_number_list[0]}" if n in electron_number_list else None)

    def molecule_setter_dict(self) -> dict:
        """ Prepare the property dict for Molecule setters """
        data = self.parse_log()
        return {
            'atoms.partial_charge': data.atomcharges['mulliken'],
            'energy': data.scfenergies[-1],
            'spin': data.mult,
            'charge': data.charge,
            'mol_orbital_energies': data.moenergies,  # eV,
            'coordinates': data.atomcoords[-1]
        }

    def set_molecule_attrs(self, mol):
        """ set Molecule attributes according to calculated results """
        from hotpot import Molecule
        assert isinstance(mol, Molecule)

        data = self.parse_log()
        assert all(atom.atomic_number == atomic_number for atom, atomic_number in zip(mol.atoms, data.atomnos))

        for atom, partial_charge, coordinate in zip(mol.atoms, data.atomcharges['mulliken'], data.atomcoords[-1]):
            atom.partial_charge = partial_charge
            atom.coordinate = coordinate

    def parse_log(self):
        """ Parse the gaussian log file and save them into self """
        return cclib.ccopen(io.StringIO(self.stdout)).parse()

    def run(self, script: str = None, test: bool = False):
        """Runs the Gaussian 16 process with the given script and additional arguments.

        This method sets up the required environment variables and resource limits for Gaussian 16 before
        running the process using `subprocess.Popen`. It takes an input script and any additional arguments
        to pass to `Popen`, and returns a tuple of the standard output and standard error of the process.

        Args:
            script (str): The input script for the Gaussian 16 process.
            test: if tree, running with the test model, at the time the running of Gaussian program will be skipped.
        Returns
            Tuple[str, str]: A tuple of the standard output and standard error of the process
        """
        with open(self.gjf_path, 'w') as writer:
            writer.write(script)

        # Configure the input and output mode
        if self.output_in_running:
            cmd = ['g16', str(self.gjf_path), str(self.logout_path)]
            stdin = None
        else:
            cmd = ['g16']
            stdin = self.stdin

        # Run Gaussian using subprocess
        if not test:
            self.g16process = subprocess.Popen(
                cmd, bufsize=-1, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=self.envs, universal_newlines=True
            )
            self.stdout, self.stderr = self.g16process.communicate(stdin)

        if self.output_in_running and not self.stdout:
            with open(self.logout_path) as file:
                self.stdout = file.read()

        self.output = GaussOut(self.stdout, self.stderr)
