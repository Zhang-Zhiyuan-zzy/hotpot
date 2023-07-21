"""
python v3.9.0
@Project: hotpot
@File   : gaussian
@Auther : Zhiyuan Zhang
@Data   : 2023/7/19
@Time   : 2:28

TODO: New implementation for running Gaussian
TODO: How to set options to be both convenient for input and easy to debug
"""
import os
import json
from typing import *


class OptionError(BaseException):
    """ Raise when try to access a non-existent option """


class GaussianRunError(BaseException):
    """ Raise when the encounter error in run gaussian """


class FailToHandle(Warning):
    """ Report this Warning when GaussErrorHandle Fail to handle an error """


class _OptionValues:
    def __init__(self, values: list):
        self.values = values

    def __repr__(self):
        return f"{self.values}"

    def __dir__(self) -> Iterable[str]:
        return self.values

    def __getattr__(self, item):
        if item not in self.values:
            raise ValueError(f'The option {self.option} does not have the value {item}')


class _Option:
    """ The representation of certain Gaussian option """
    def __init__(
            self,
            raw_options: dict,
            title: Literal['link0', 'route'],
            keyword: str,
            option: str = None,
            value: Any = None
    ):
        self._options = raw_options

        self.title = title
        self.keyword = keyword
        self.option = option
        self.value = value

    def __dir__(self) -> Iterable[str]:
        if isinstance(self.value, _OptionValues):
            return self.value.values
        else:
            return []

    def __getattr__(self, item):
        """
        The values of some options are fixed, and they can be obtained through attribute selection to
        avoid conflicts when writing options.
        For example, the value of route->opt->Algorithm is choose from: GEDIIS, RFO and EF, one may want
        to apply the RFO as the optimization algorithm. To do so, the one just to choose the option by:

            gauss = Gaussian(...)
            gauss.options.route.opt.Algorithm.RFO  # select RFO as the optimizing algorithm

        the Gaussian instance will record the options immediately.

        Sometimes, you could need to handle the GaussError by change the RFO to other method, say GEDIIS.
        You can do this change just by:

            gauss.options.route.opt.Algorithm.GEDIIS

        the Gaussian instance will change the optimizing algorithm to GEDIIS
        """
        if not isinstance(self.value, _OptionValues) or item not in self.value.values:
            raise ValueError(f'The option {self.option} does not have the value {item}')

        key = self.title + f'.{self.keyword}' + (f".{self.option}" if self.option else "")
        self._options[key] = self

    def __repr__(self):
        return f"{self.keyword}" \
               + (f"={self.option}" if self.option else "") \
               + (f"({self.value})" if self.value else "")

    def __hash__(self):
        key = f"{self.title}" + f".{self.keyword}" \
               + (f".{self.option}" if self.option else "") \

        return hash(key)

    def __eq__(self, other):
        return self.title == other.title and self.keyword == other.keyword and self.option == other.option

    def __call__(self, value=None):
        if self.value is None and value is not None:
            raise ValueError(f"the option {self.keyword}.{self.option} doesn't have any value")

        elif self.value == 'float' or isinstance(self.value, float):
            if isinstance(value, float):
                self.value = value
            else:
                raise TypeError('the input value should be a float')

        elif self.value == "int" or isinstance(self.value, int):
            if isinstance(value, int):
                self.value = value
            else:
                raise TypeError('the input value should be an int')

        elif self.value == 'str' or isinstance(self.value, str):
            if isinstance(value, str):
                self.value = value
            else:
                raise TypeError('the input value should be a string')

        else:
            raise NotImplemented

        key = self.title + f'.{self.keyword}' + (f".{self.option}" if self.option else "")
        self._options[key] = self


class _Options:
    """
    A handle to the link0 and route options.
    This class should be initialized by `Gaussian.options` attributes, initializing directly is not recommended.
    """
    from hotpot import data_root
    _path_option_json = os.path.join(data_root, 'goptions.json')

    _tree: dict = json.load(open(_path_option_json))

    def __init__(self, raw_options: dict, path: str = ""):

        self._paths = path
        self._options = raw_options

    def __repr__(self):
        return self._paths if self._paths else 'RootOptions'

    def __dir__(self) -> Iterable[str]:
        if not self._paths:
            return list(self._tree.keys())
        else:
            paths = self._paths.split('.')

            tree = self._tree
            for option in paths:
                tree = tree.get(option)

            if isinstance(tree, list):
                return tree
            elif isinstance(tree, dict):
                return list(tree.keys())
            else:
                return []

    def __getattr__(self, item: str):
        tree = self._tree
        if self._paths:
            for option in self._paths.split('.'):
                tree = tree.get(option)

        if isinstance(tree, dict):
            option = tree.get(item)
        else:
            assert isinstance(tree, list)
            if item in tree:
                option = item
            else:
                raise AttributeError(f'{option} not have option: {item}')

        if isinstance(option, (dict, list)):
            if self._paths:
                return _Options(self._options, f"{self._paths}.{item}")
            else:
                return _Options(self._options, f"{item}")
        else:
            paths = self._paths.split('.')
            if len(paths) == 1:
                return _Option(self, paths[0], option)
            elif len(paths) == 2:
                return _Option(self, paths[0], paths[1], option)
            elif len(paths) == 3:
                return _Option(self, paths[0], paths[1], paths[3])
            else:
                raise AttributeError

    def __call__(self):
        """
        Sometimes, the keywords could be without the options, so call the _Options instance to specify it

        """
        paths = self._paths.split('.')
        if len(paths) == 2:
            option = _Option(self._options, paths[0], paths[1])
            option()

def _cook_raw_options_to_struct_input_dict(raw_options: dict) -> dict:
    """"""
    for op in self._options:
        title = raw_options.setdefault(op.title, {})
        ops = title.setdefault(op.keyword, [])
        ops.append(op)

    _input = {}
    for title, item in raw_options.items():

        keywords = _input.setdefault(title, {})

        for kwd, ops in item.items():
            ops = [op for op in ops if op.option is not None]

            if not ops:
                keywords[kwd] = None

            elif len(ops) == 1:
                if ops[0].value is None:
                    keywords[kwd] = ops[0].option
                else:
                    keywords[kwd] = {ops[0].option: ops[0].value}

            else:
                options = keywords.setdefault(kwd, {})
                for op in ops:
                    if op.value is None:
                        options[op.option] = None
                    else:
                        options[op.option] = op.value

    return _input


class Gaussian:
    """ The control panel of Gaussian program """
    def __init__(
            self, mol,
            g16root: Union[str, os.PathLike],
            path_gjf: Union[str, os.PathLike] = None,
            path_log: Union[str, os.PathLike] = None,
            path_err: Union[str, os.PathLike] = None,
            report_set_resource_error: bool = False
    ):
        """"""
        self.mol = mol
        self.g16root = Path(g16root)

        # the default running input and output file
        self.p_input = Path('input.gjf') if not path_gjf else Path(path_gjf)
        self.p_output = Path('output.log') if not path_log else Path(path_log)
        self.p_err = Path(path_err) if path_err else None

        # preserve for storing running data
        self.path_chk = None
        self.path_rwf = None

        self._raw_options = {}
        self.options = _Options(self._raw_options)

        self.parsed_input = None
        self.g16process = None  # to link to the g16 subprocess

        self.outputs = []
        self.stdout = None
        self.stderr = None

        # Configure running environments and resources
        self.envs = self._set_environs()
        self._set_resource_limits(report_set_resource_error)

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

    def _parse_input_script(self, script: str) -> dict:
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
        # Extract the charge and spin
        charge, spin = map(int, lines[c].strip().split())
        info['charge'], info['spin'] = charge, spin
        c += 1

        # Extract the atoms information
        atoms, coordinates = [], []
        while lines[c].strip():
            atom_line: list[str] = lines[c].strip().split()
            atom = atom_line[0]
            xyz = list(map(float, atom_line[1:4]))

            atoms.append(atom)
            coordinates.append(xyz)

            c += 1

        info['atoms'] = atoms
        info['coordinates'] = np.array(coordinates)

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


        info = self.parsed_input
        script = ""

        # Link0 commands
        link0: dict = info['link0']
        for cmd, value in link0.items():
            if value:
                script += f'%{cmd}={value}\n'
            else:
                script += f'%{cmd}\n'

        # Route keywords
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

        # Title
        script += info['title']
        script += '\n\n'

        # Charge and spin
        script += f"{info['charge']} {info['spin']}\n"

        # Atoms specification
        assert len(info['atoms']) == len(info['coordinates'])
        for atom, xyz in zip(info['atoms'], info['coordinates']):
            x, y, z = xyz
            script += f'{atom} {x} {y} {z}\n'

        script += '\n'

        # Other contents
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


if __name__ == "__main__":
    ro = {}
    _ops = _Options(ro)
    route = _ops.route
    route.opt.ONIOM.Micro()
