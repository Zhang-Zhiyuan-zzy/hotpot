"""
python v3.7.9
@Project: hotpot
@File   : quantum.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/20
@Time   : 2:44
"""
import copy
import os
import re
from pathlib import Path
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
        self.g16root = Path(g16root)

        self.envs = self._set_environs()
        self._set_resource_limits(report_set_resource_error)

        self.error_handle_methods = error_handle_methods

        # reverse for storing running data
        self.path_chk = None
        self.path_rwf = None

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
            g16root = str(self.g16root)
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
    def _parse_route(route: str) -> Dict:
        """ Parse the route of gjf file """
        # compile regular expressions
        parenthesis = re.compile(r'\(.+\)')

        # Normalize the input route
        route = re.sub(r'\s*=\s*', r'=', route)  # Omit the whitespace surround with the equal signal
        route = re.sub(r'=\(', r'\(', route)  # Omit the equal signal before the opening parenthesis
        route = re.sub(r'\s+', ' ', route)  # Reduce the multiply whitespace to one

        # Replace the delimiter outside the parenthesis to whitespace, inside to common
        in_parenthesis = {m.start(): m.end() for m in parenthesis.finditer(route)}
        list_route = []
        for i, char in enumerate(route):
            if char in [',', '\t', '/', ' ']:
                if any(si < i < ei for si, ei in in_parenthesis.items()):
                    list_route.append(',')
                else:
                    list_route.append(' ')
            else:
                list_route.append(char)

        route = ''.join(list_route)

        # Separate route to items
        items = route.split()

        parsed_route = {}
        for item in items:
            opening_parenthesis = re.findall(r'\(', item)
            closing_parenthesis = re.findall(r'\)', item)

            if opening_parenthesis:
                assert len(opening_parenthesis) == 1 and len(closing_parenthesis) == 1 and item[-1] == ')'
                kword = item[:item.index('(')]
                options = item[item.index('(') + 1:-1]

                opt_dict = parsed_route.setdefault(kword, {})
                for option in options.split(','):
                    opt_value = option.split('=')
                    if len(opt_value) == 1:
                        opt_dict[opt_value[0]] = None
                    elif len(opt_value) == 2:
                        opt_dict[opt_value[0]] = opt_value[1]
                    else:
                        raise ValueError('the given route string is wrong!!')

            else:
                kword_opt_value = item.split('=')
                if len(kword_opt_value) == 1:
                    parsed_route[kword_opt_value[0]] = None
                elif len(kword_opt_value) == 2:
                    parsed_route[kword_opt_value[0]] = kword_opt_value[1]
                elif len(kword_opt_value) == 3:
                    parsed_route[kword_opt_value[0]] = {kword_opt_value[1]: kword_opt_value[2]}
                else:
                    raise ValueError('the given route string is wrong!!')

        return parsed_route

    def _parse_input_script(self, script: str) -> list[dict]:
        """ Parse the input script to structured data """
        info = []
        for link_script in re.split(r'--link\d+--\n', script):
            info.append(self._parse_single_task(link_script))

        return info

    def _parse_single_task(self, script: str) -> dict:
        """ Parse the input script to dict """
        lines = script.splitlines()
        c = 0  # count of current line

        info = {}

        # Extract link0
        link0 = []
        while lines[c][0] == '%':
            link0.append(lines[c])
            c += 1

        # Check link0
        if not link0:
            raise ValueError('the provided input script is incorrect, not found link0 lines')

        # Parse link0
        link0 = ' '.join(link0)
        parsed_link0 = info.setdefault('link0', {})
        for l0 in link0.split():
            assert l0[0] == '%'
            cmd_value = l0[1:].split('=')
            if len(cmd_value) == 1:
                parsed_link0[cmd_value[0]] = None
            elif len(cmd_value) == 2:
                parsed_link0[cmd_value[0]] = cmd_value[1]
            else:
                raise ValueError("can't parse the link0, the give input script might wrong!!")

        # Extract route
        route = []
        while lines[c] and lines[c][0] == '#':
            route.append(lines[c][2:])
            c += 1

        if not route:
            raise ValueError('the provided input script is incorrect, not found route lines')

        # Parse the route
        route = ' '.join(route)
        info['route'] = self._parse_route(route)

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

    def _rewrite_input_script(self):
        """"""
        # Whether the input info have been given。
        if not self.parsed_input:
            raise AttributeError(
                "Can't find the structured input data, the input script should be given by string script or parsed dict"
            )

        script = ""
        for i, link_script in enumerate(self.parsed_input):
            script += self._rewrite_single_task(link_script, i)  # rewrite the gjf script to standard style

        return script

    @staticmethod
    def _rewrite_single_task(info: dict, link_num: int):
        """"""
        script = ""

        if link_num:
            script += f'--Link{link_num}--\n'

        link0: dict = info['link0']
        for cmd, value in link0.items():
            if value:
                script += f'%{cmd}={value}\n'
            else:
                script += f'%{cmd}\n'

        script += '#'
        route: dict = info['route']
        for kw, opt in route.items():
            if not opt:
                script += f' {kw}'
            elif isinstance(opt, str):
                script += f' {kw}={opt}'
            elif isinstance(opt, dict):
                list_opt = []
                for k, v in opt.items():
                    if v:
                        list_opt.append(f'{k}={v}')
                    else:
                        list_opt.append(k)
                script += f' {kw}(' + ','.join(list_opt) + ')'
            else:
                ValueError('the give gjf input info is wrong')

        script += '\n\n'

        script += info['title']
        script += '\n\n'

        script += '\n'.join(info['mol_spec'])
        script += '\n'

        i = 0
        while True:
            other = info.get(f'other_{i}')
            if other:
                script += '\n'.join(other)
            else:
                break

        script += '\n\n'

        return script

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

    def run(self, script: str = None):
        """Runs the Gaussian 16 process with the given script and additional arguments.

        This method sets up the required environment variables and resource limits for Gaussian 16 before
        running the process using `subprocess.Popen`. It takes an input script and any additional arguments
        to pass to `Popen`, and returns a tuple of the standard output and standard error of the process.

        Args:
            script (str): The input script for the Gaussian 16 process.
        Returns
            Tuple[str, str]: A tuple of the standard output and standard error of the process
        """
        if script:
            self.parsed_input = self._parse_input_script(script)  # parse input data

        script = self._rewrite_input_script()

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


class GaussErrorHandle(ABC):
    """ Basic class to handle the error release from gaussian16 """
    def __init__(self, gauss: Gaussian, stdout: str, stderr: str):
        self.gauss = gauss
        self.stdout = stdout
        self.stderr = stderr

        self.has_handled = False  # Whether the error has handled by this Handle

    def __call__(self) -> bool:
        """ Call for handle the g16 errors """
        return not self.has_handled and self.could_handle() and self.error_handle()

    @abstractmethod
    def could_handle(self) -> bool:
        """ Could the ErrorHandle is suitable for this error """

    @abstractmethod
    def error_handle(self) -> bool:
        """ Specified by the children classes """


class L103ZMatrixHandle(GaussErrorHandle, ABC):
    """
    Handle the Gaussian16 Error raise from the Z-matrix unsuitable for optimizing the system,
    in which some angle or dihedral be 0 or 180
    """
    def could_handle(self) -> bool:
        error_proc = str(self.gauss.g16root.joinpath('g16', 'l103.exe'))
        error_signal = re.compile(rf"Error termination via Lnk1e in {error_proc} at .+")

        final_output_lines = self.stdout.splitlines()[-10:]

        for i, line in enumerate(final_output_lines):
            if error_signal.match(line) and final_output_lines[i-1] == 'FormBX had a problem.':
                return True

        return False

    def error_handle(self) -> bool:
        parsed_input = self.gauss.parsed_input

        # This handle just solve the problem in single task
        if len(parsed_input) > 1:
            return False

        task0 = parsed_input[0]
        route = task0.get('route')

        # Retrieve the options or optimization
        opt_kw = 'optimization'
        kw_cut = 2
        opt_options = None
        while not opt_options or kw_cut >= len(opt_kw):
            kw_cut += 1
            opt_options = route.get(opt_kw[:kw_cut])

        if not opt_options:
            new_options = {}
        elif isinstance(opt_options, str):
            new_options = {opt_options: None}
        elif isinstance(opt_options, dict):
            new_options = opt_options
        else:
            raise AttributeError('the structured route dict is wrong!')

        restart_items = {'Restart': None, 'Cartesian': None}
        new_options.update(restart_items)
        route[opt_kw[:kw_cut]] = new_options

        # Restart the work
        self.has_handled = True
        self.gauss.run()

        return True
