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
import cclib
from typing import *


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
            report_set_resource_error: bool = False
    ):
        """
        This method sets up the required environment variables and resource limits for Gaussian 16.
        Args:
            g16root (Union[str, os.PathLike]): The path to the Gaussian 16 root directory.
            report_set_resource_error: Whether to report the errors when set the environments and resource

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
        self.che_path = None

        self.data = None  # to receive the data from the cclib parser
        self.g16process = None  # to link to the g16 subprocess

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

    def error_handle(self, stdout: str, stderr: str):
        """
        Handle the error message release information.
        Args:
            stdout: the standard output message
            stderr: the standard error message

        Returns:
            whether to raise the error (bool), error massage
        """

    @property
    def molecule_setter_dict(self):
        """ Prepare the property dict for Molecule setters """
        return {
            'atoms.partial_charge': self.data.atomcharges['mulliken'],
            'energy': self.data.scfenergies[-1],
            'spin': self.data.mult,
            'charge': self.data.charge,
            'mol_orbital_energies': self.data.moenergies,  # eV,
            'coordinates': self.data.atomcoords[-1]
        }

    def parse_log(self, stdout: str):
        """ Parse the gaussian log file and save them into self """
        string_buffer = io.StringIO(stdout)
        self.data: cclib.parser.data.ccData_optdone_bool = cclib.ccopen(string_buffer).parse()

    def run(self, script: str):
        """Runs the Gaussian 16 process with the given script and additional arguments.

        This method sets up the required environment variables and resource limits for Gaussian 16 before
        running the process using `subprocess.Popen`. It takes an input script and any additional arguments
        to pass to `Popen`, and returns a tuple of the standard output and standard error of the process.

        Args:
            script (str): The input script for the Gaussian 16 process.
            *args: Additional arguments to pass to subprocess.Popen.
            **kwargs: Additional keyword arguments to pass to subprocess.Popen.

        Returns:
            Tuple[str, str]: A tuple of the standard output and standard error of the process.
        """
        with open('input.gjf', 'w') as writer:
            writer.write(script)

        # Run Gaussian using subprocess
        self.g16process = subprocess.Popen(
            ['g16', 'input.gjf', 'output.log'], bufsize=-1, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.envs, universal_newlines=True
        )
        _, stderr = self.g16process.communicate()

        with open('output.log') as file:
            stdout = file.read()

        if stderr:
            return stdout, stderr

        self.parse_log(stdout)

        return stdout, stderr

