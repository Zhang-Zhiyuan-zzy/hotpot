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
import copy
import json
import resource
import subprocess
from pathlib import Path
from typing import *
from abc import ABC, abstractmethod

import numpy as np
import cclib

from hotpot import data_root


class GaussianRunError(BaseException):
    """ Raise when the encounter error in run gaussian """


class FailToHandle(Warning):
    """ Report this Warning when GaussErrorHandle Fail to handle an error """
    

_tree: dict[str, Any] = json.load(open(Path(data_root).joinpath('goptions.json')))

    
class OptionPath:
    """ Represent a path from the root to a option """
    def __init__(self, path: str):
        """"""
        self.path = path

    def __repr__(self):
        return f"OptionPath({self.path})"

    def __hash__(self):
        return hash(self.pre_selection_path)

    def __eq__(self, other):
        return self.pre_selection_path == other.pre_selection_path

    def __lt__(self, other: "OptionPath"):
        return other.path.startswith(self.path)

    def __gt__(self, other: "OptionPath"):
        return self.path.startswith(other.path)

    def __len__(self):
        return len(self.brief_nodes)

    def _make_child(self, child_node: str):
        """ Make the child path of this path """
        if not self.path:
            return self.__class__(child_node)
        else:
            return self.__class__(f"{self.path}.{child_node}")

    @property
    def brief_nodes(self):
        return self.brief_path.split('.')

    @property
    def brief_path(self):
        """ the brief path is the path that exclude the SELECTION nodes """
        if not self.path:
            return ''

        brief_nodes = []
        tree = copy.copy(_tree)
        for p in self.nodes:
            tree = tree[p]
            if not isinstance(tree, dict) or 'SELECTION' not in tree:
                brief_nodes.append(p)

        return '.'.join(brief_nodes)

    @property
    def children(self):
        """ Get the children nodes in this path """
        if isinstance(self.subtree, dict):
            return [self._make_child(c) for c in self.subtree if c != "SELECTION"]
        else:
            return []

    @classmethod
    def create_from_brief_path(cls, path: str) -> "OptionPath":
        """
        create OptionPath object from (suspected) brief path
        Args:
            path: the actual or brief path

        Returns:
            the OptionPath object with complete path
        """
        # nodes = path.split('.')
        #
        # pre_path = cls('.'.join(nodes[:-1]))
        #
        # paths = [
        #     p for p in pre_path.descendants
        #     if re.fullmatch(p.path, path, re.IGNORECASE) or re.fullmatch(p.brief_path, path, re.IGNORECASE)
        # ]
        #
        # try:
        #     assert len(paths) == 1
        # except AssertionError as err:
        #     print(path, paths)
        #     raise err
        #
        # return paths[0]

        return cls(cls.get_normalize_path(path))

    @property
    def descendants(self):
        """"""
        desc = []
        parents = self.children
        while parents:
            children = []
            for p in parents:
                if not p.is_selection:
                    desc.append(p)
                children.extend(p.children)

            parents = children  # the children grow up to parents

        return desc

    @property
    def end_name(self):
        if not self.nodes:
            return ""
        else:
            return self.nodes[-1]

    def get_child(self, child_name: str):
        if self.is_leaf:
            raise AttributeError('This is a leaf OptionPath, do not have any child path!')

        _child_path = child_name if self.is_root else self.path + f".{child_name}"
        child_path = self.get_normalize_path(_child_path)

        return self.__class__(child_path)

        # if child_name in self.subtree:
        #     if self.is_root:
        #         return self.__class__(child_name)
        #     else:
        #         return self.__class__(f"{self.path}.{child_name}")
        # else:
        #     raise KeyError(f'the {child_name} not the child of {self.path}')

    @staticmethod
    def get_normalize_path(path: Union[str, "OptionPath"]) -> str:
        """
        Find the regularized option path based on any valid Gaussian option path. An effective
        path refers to a path where the name of each node in the given path can be matched
        uniquely to a Gaussian keyword or option. For example, the regularized option path for
        Gaussian's optimization using Cartesian coordinates is "Optimization.Coordinate.Cartesian",
        which corresponds to the Gaussian route input of "Optimization(Cartesian)."

        It should be noted that the `Coordinate` is an implicit node, which does not be written
        in to the Gaussian input script but a mark for a collection of actual nodes. The options
        under these implicit nodes can't be selected simultaneously. If one of them under a same
        implicit node is being selected, the previous one selected will be removed from the
        option set.

        Using this method, the following are valid names for correctly finding the above
        regularized path:

            1) Optimization.Coordinate.Cartesian. (the regular path itself)
            2) optimization.coorDiNate.carTesian. (ignoring case sensitivity)
            3) opt.Coordi.cartes. (any abbreviation that can find a unique match)
            4) opt.cartes. (the implicit nodes might be omitted)

        If the abbreviation is too short and matches multiple options, a ValueError will be raised.
        If an incorrect node name is provided or the connection relationship of nodes in the option
        tree is incorrect, a ValueError will also be raised.

        Args:
            path: the origin path (may valid or not)

        Returns:
            The regularized path

        Raises:
            ValueError: when an invalid path is given
        """
        def match_dict_key(given_node: str, search_tree: dict):
            """"""
            match_dict = {}
            for search_key, sv in search_tree.items():

                if search_key == "SELECTION":
                    continue

                if re.match(given_node, search_key, re.IGNORECASE):
                    match_dict[search_key] = sv

            return match_dict

        def update_current_tree():
            """"""
            tr = _tree
            for n in norm_nodes:
                tr = tr[n]

            return tr

        nodes = path.nodes if isinstance(path, OptionPath) else path.split('.')
        norm_nodes = []

        tree = update_current_tree()
        for node in nodes:
            # suppose the give path is a full path
            sub_tree = match_dict_key(node, tree)

            # the given node name has a unique matched
            if len(sub_tree) == 1:
                norm_nodes.append(list(sub_tree.keys())[0])

                # update the current tree
                tree = update_current_tree()

            # the given node name does not any matched
            # search the node with the assumption that the given path is brief path
            elif not sub_tree:

                # find this node in all SELECTION subtree with a new tree
                select_key = None
                for key, value in tree.items():
                    if isinstance(value, dict) and "SELECTION" in value:
                        sub_tree.update(match_dict_key(node, value))

                        if len(sub_tree) == 1 and not select_key:
                            select_key = key

                # when find a match node,
                # add the previous SELECTION node and the found node to the norm_nodes, update the subtree
                if len(sub_tree) == 1:
                    assert isinstance(select_key, str)

                    norm_nodes.extend([select_key, list(sub_tree)[0]])

                    # update the current tree
                    tree = update_current_tree()

                elif not sub_tree:
                    raise ValueError(
                        f"the give node {node} in path {path} does not match any option or keywords"
                    )

                else:  # finally, do not found a unique next node
                    raise ValueError(
                        f"the given keyword or option {node} in path {path} matched {len(sub_tree)}:"
                        f"\t {', '.join(sub_tree)}"
                        f"these keyword or option from a set of SELECTION sub options"
                        f"you might give a more length string to choose from them"
                )

            else:  # multiply next nodes are matched
                raise ValueError(
                    f"the given keyword or option {node} in path {path} matched {len(sub_tree)}:\n"
                    f"\t {', '.join(sub_tree)}\n"
                    f"you might give a more length string to choose from them"
                )

        return '.'.join(norm_nodes)

    @property
    def is_leaf(self):
        return not isinstance(self.subtree, dict)

    @property
    def is_root(self):
        return not self.path

    @property
    def is_selection(self):
        return isinstance(self.subtree, dict) and "SELECTION" in self.subtree

    @property
    def nodes(self) -> list[str]:
        return self.path.split('.')

    @property
    def parent(self):
        if not self.path:
            raise AttributeError('the root path not have parent')

        return self.__class__(".".join(self.path.split(".")[:-1]))

    @property
    def pre_selection_path(self):
        pre_sel_nodes = []
        tree = copy.copy(_tree)
        for node in self.nodes:
            pre_sel_nodes.append(node)
            tree = tree[node]
            if isinstance(tree, dict) and "SELECTION" in tree:
                break

        return '.'.join(pre_sel_nodes)

    @property
    def subtree(self):
        """ Get the subtree derive from this path """
        if not self.path:
            return copy.copy(_tree)

        subtree = _tree
        for n in self.nodes:
            subtree = subtree[n]

        return subtree


class Options:
    def __init__(self, path: Union[str, OptionPath], root=None):
        if isinstance(path, OptionPath):
            self.path = path
        elif isinstance(path,str):
            self.path = OptionPath(path)
        else:
            raise TypeError('the path should be str or OptionPath')

        self.root = root if root is not None else self

        self.value = None
        self._ops = set()

    def __bool__(self):
        return self.path.is_root and len(self._ops) > 0

    def __repr__(self):
        return f"Option({self.path.path})"

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        return self.path == other.path

    def __dir__(self) -> Iterable[str]:
        names = []
        for c in self.path.children:
            name = c.end_name
            if re.match(r"[0-9]", name[0]):
                name = f"_{name}"
            name = name.replace('-', '_')

            names.append(name)

        return names

    def __getattr__(self, item):
        """"""
        if len(item) > 1 and item[:1] == "_":
            item = item[1:]
        item = item.replace('_', '-')

        return self.__class__(self.path.get_child(item), self.root)

    def __len__(self):
        return len(self.path)

    def __call__(self, value=None, *args, **kwargs):
        if self.path.is_leaf:
            if self.path.subtree is None:
                if value is not None:
                    raise ValueError(f'the option {self.path.path} not allowed a value')

            elif self.path.subtree != value.__class__.__name__:
                if isinstance(value, str):
                    if self.path.subtree == "int":
                        value = int(value)
                    elif self.path.subtree == 'float':
                        value = float(value)

                else:
                    raise TypeError(f'the option expect a {self.path.subtree}, instead of {value.__class__.__name__}')

        else:
            if value is not None:
                raise ValueError('the non-leaf option cannot accept any values')

        self.value = value

        if not self.path.is_root:
            self.root.add(self)
        else:
            raise AttributeError('the root option cannot add self into self')

    def add(self, op: "Options"):
        if not self.path.is_root:
            ValueError('only the option in the root path could add other option into')

        if len(op) < 2:
            raise AttributeError('only the path length equal or more than 2 could be select to be option')

        # find all options with the paths which are the children or parents of this option's path
        # and then remove them from the current options collection
        rm_ops = set()
        for op_in in self._ops:
            if op.path > op_in.path or op.path < op_in.path or op == op_in:
                rm_ops.add(op_in)

        # Remove children and parents options
        self._ops.difference_update(rm_ops)

        self._ops.add(op)

    def clear(self):
        """ Clear all options save in root options, only the root option could call this method """
        if not self.path.is_root:
            raise AttributeError('only the root option could perform the clear')

        self._ops = set()

    def get_option_dict(self):
        """"""
        op_dict = {}
        for op in self._ops:
            title = op.path.brief_nodes[0]
            keyword = op_dict.setdefault(title, {})

            if title == 'link0':
                assert len(op) == 2
                keyword[op.path.brief_nodes[1]] = op.value

            elif title == "route":
                # When the length of brief_path less than 3, it means the given keywords not have an option
                # In the case, the Options must not give any value, an empty dict represent a keyword without option
                options = keyword.setdefault(op.path.brief_nodes[1], {})

                if len(op) == 3:
                    options[op.path.end_name] = op.value

        return op_dict

    def parsed_input_to_options(self, parsed_input: dict):
        """ Convert the parsed input to options, only the root option could perform this method """
        if not self.path.is_root:
            raise AttributeError('only the root option could convert the parsed input to options')

        link0 = parsed_input['link0']
        route = parsed_input['route']

        for cmd, value in link0.items():
            path = OptionPath.create_from_brief_path(f"link0.{cmd}")
            self.__class__(path, self)(value)

        for kwd, ops in route.items():
            if not ops:
                path = OptionPath.create_from_brief_path(f'route.{kwd}')
                self.__class__(path, self)()
            elif isinstance(ops, dict):
                for op, value in ops.items():
                    path = OptionPath.create_from_brief_path(f'route.{kwd}.{op}')
                    self.__class__(path, self)(value)

    def update_parsed_input(self, parsed_input: dict):
        """"""
        parsed_input['link0'] = {}
        parsed_input['route'] = {}

        parsed_input.update(self.get_option_dict())


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
    _date = re.compile(_week.pattern + ' ' + _month.pattern + r' [0-2]?\d')
    _time = re.compile(r'\d{2}:\d{2}:\d{2} 20\d{2}\.')

    _error_link = re.compile(_head.pattern + ' ' + _path.pattern + ' at ' + _date.pattern + ' ' + _time.pattern)

    def __init__(self, stdout: str, stderr: str):
        self.stdout = stdout
        self.stderr = stderr

    @property
    def is_error(self) -> bool:
        return True if self.stderr else False

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
            g16root: Union[str, os.PathLike],
            path_gjf: Union[str, os.PathLike] = None,
            path_log: Union[str, os.PathLike] = None,
            path_err: Union[str, os.PathLike] = None,
            report_set_resource_error: bool = False,
            options: Options = None,
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
        self.g16root = Path(g16root)

        # Configure running environments and resources
        self.envs = self._set_environs()
        self._set_resource_limits(report_set_resource_error)

        # the default running input and output file
        self.p_input = Path('input.gjf') if not path_gjf else Path(path_gjf)
        self.p_output = Path('output.log') if not path_log else Path(path_log)
        self.p_err = Path(path_err) if path_err else None

        # preserve for storing running data
        self.path_chk = None
        self.path_rwf = None

        self.parsed_input = {}

        # Set options
        if options:
            if options.path.is_root:
                self.op = options
            else:
                raise ValueError('the option pass into Gaussian must be a root option')
        else:
            self.op = Options('')  # Create a new option

        self.g16process = None  # to link to the g16 subprocess
        self.output_in_running = output_in_running

        self.stdin = None
        self.output = None
        self.stdout = None
        self.stderr = None

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

            # When the keyword have multiply options
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

            else:  # When the keyword only a single option
                kword_opt_value = item.split('=')
                if len(kword_opt_value) == 1:
                    parsed_route[kword_opt_value[0]] = {}
                elif len(kword_opt_value) == 2:
                    parsed_route[kword_opt_value[0]] = {kword_opt_value[1]: None}
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
        for kw, ops in route.items():

            # If the keyword doesn't have any option
            if not ops:
                script += f' {kw}'

            # if the keyword have multiply options
            elif isinstance(ops, dict):
                list_opt = []
                for op, value in ops.items():
                    if value:
                        list_opt.append(f'{op}={value}')
                    else:
                        list_opt.append(op)
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

    def full_option_values(self, title: str, kwd: str = None, op: Any = None, value: Optional = None):
        """
        Full the value to gauss parsed_input dict
        Args:
            title: the first level of parsed input, like: 'link0', 'route', 'title', 'charge', 'spin', ...
            kwd:
            op:
            value:

        Returns:

        """
        option_values = self.parsed_input[title].get(kwd)

        if option_values is None:
            if value is None:
                self.parsed_input[title][kwd] = op  # Note: the op could be None, it's allowed
            else:
                self.parsed_input[title][kwd] = {op: value}

        elif isinstance(option_values, dict):
            self.parsed_input[title][kwd].update({op: value})

        else:
            self.parsed_input[title][kwd] = {option_values: None, op: value}

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

    def parse_log(self):
        """ Parse the gaussian log file and save them into self """
        string_buffer = io.StringIO(self.stdout)
        return cclib.ccopen(string_buffer).parse()

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
        if script:
            self.parsed_input = self._parse_input_script(script)  # parse input data
            self.op.parsed_input_to_options(self.parsed_input)
        elif self.op:  # If some option have been assigned by Gaussian.Options
            self.op.update_parsed_input(self.parsed_input)

        self.stdin = self._rewrite_input_script()

        with open(self.p_input, 'w') as writer:
            writer.write(self.stdin)

        # Configure the input and output mode
        if self.output_in_running:
            cmd = ['g16', str(self.p_input), str(self.p_output)]
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
            with open(self.p_output) as file:
                self.stdout = file.read()

        self.output = GaussOut(self.stdout, self.stderr)

    def to_conformer(self, cfm_idx=-1, fail_raise=False):
        """
        read the conformers from stdout, and convert the initial conformer in the input.gjf file to be specific one
        Args:
            cfm_idx: which conformer in the stdout is converted to the input file, default: last conformer(-1)
            fail_raise: whether to raise error, if fail to convert to the specific conformer.
             the error might be caused by the format of stdout is unreadable for Molecule class
        """
        if not self.stdout:
            raise AttributeError('the stdout do not save any conformers')

        # If success, optimize with ses surface from the last conformer
        from hotpot.cheminfo import Molecule
        mol = Molecule.read_from(self.stdout, 'g16log', force=True)

        try:
            self.parsed_input['coordinates'] = mol.all_coordinates[-cfm_idx]
            self.parsed_input['atoms'] = [a.label for a in mol.atoms]

        except AttributeError:
            if fail_raise:
                raise RuntimeError('Fail to convert the conformers, the stdout might be unreadable!')
            else:
                print(RuntimeWarning('Fail to convert the conformers, the stdout might be unreadable!'))

        except IndexError as err:
            if fail_raise:
                raise err
            else:
                print(err)


class GaussRun:
    """ Run the Gaussian program """
    def __init__(self, gauss: Gaussian, debugger: Optional[Union[str, "Debugger"]] = 'auto', **kwargs):
        self.gauss = gauss

        self.stdout = []
        self.stderr = []

        # Configure Error Handle
        if debugger is None:
            self.debugger = None

        elif isinstance(debugger, str):
            kwargs['max_try'] = kwargs.get('max_try', 2)

            if debugger == 'auto':
                self.debugger = AutoDebug()
            elif debugger == 'restart':
                self.debugger = Restart(**kwargs)
            elif debugger == 'last_conformer':
                self.debugger = RerunFromLastConformer(**kwargs)
            else:
                raise ValueError(f"can't find error handle name {debugger}")

        elif isinstance(debugger, Debugger):
            self.debugger = debugger

        else:
            raise TypeError('the given error handle should be str or a GaussErrorHandle type')

    def __call__(self, script=None, max_debug=5):
        """"""
        self.gauss.run(script)

        run_time = 0
        while self.gauss.stderr and run_time < max_debug:
            if self.debugger and self.debugger(self.gauss):

                self.stdout.append(self.gauss.stdout)
                self.stderr.append(self.gauss.stderr)

                self.gauss.run()

                if self.gauss.stderr:
                    print(f"Fail to debug with error: {self.gauss.output.error_link}!!!")
                else:
                    print("Debug Successful!!!")

                run_time += 1

            else:
                break

        # Report the results
        if self.gauss.stderr:
            print(f"Terminate {self.gauss.parsed_input['title']} Gaussian with {self.gauss.output.error_link} Error!!!")
        else:
            print(f"Normalize Complete {self.gauss.parsed_input['title']} Gaussian Calculation !!!")

        return self.gauss


# Error Handles
class Debugger(ABC):
    """ Basic class to handle the error release from gaussian16 """

    def __call__(self, gauss: Gaussian) -> bool:
        """ Call for handle the g16 errors """
        if self.trigger(gauss):
            self.notice(gauss)
            return self.handle(gauss)

        return False

    @staticmethod
    def _find_keyword_name(target: dict, shortest: str) -> Union[None, str]:
        """
        Find the input keyword by their allowed shortest name.
        For example:
            the shortest name `optimization` in route is `opt`, so any keyword from the `opt` to `optimization`
            are allowed. In the input script, the user might give any one of the allowed keyword, such as `opt`,
            `optimiz` or `optimizati`, this method will find the real user-given keyword by `opt`.
        Args:
            target(dict): the subitem of parsed_input, like: link0, route.
            shortest: the shortest name, say `opt`

        Returns:
            the actual user-given keyword
        """
        searcher = re.compile(f"{shortest}.*", re.IGNORECASE)

        kwd_name = None
        for name in target:
            match = searcher.fullmatch(name)
            if match:
                kwd_name = match.string[match.start():match.end()]
                break

        return kwd_name

    @abstractmethod
    def trigger(self, gauss: Gaussian) -> bool:
        """ Could the ErrorHandle is suitable for this error """

    @abstractmethod
    def handle(self, gauss: Gaussian) -> bool:
        """ Specified by the children classes """

    def notice(self, gauss: Gaussian):
        errors = gauss.output.report()
        if not errors:
            print(f"Gauss meet {gauss.output.error_link} error -> debug by {self.__class__.__name__}")
        else:
            print(f"Gauss meet {', '.join(errors)} error -> debug by {self.__class__.__name__}")


class AutoDebug(Debugger, ABC):
    """ AutoHandle Gaussian Error """
    _handles = {}

    def __init__(self, *selected_method: str, invert=False):
        """
        Args:
            *selected_method: if given, only the selected methods will be applied to debug
            invert: if is true, the selected_method will be excluded from the debuggers set
        """
        if not selected_method:
            self.handles = {name: handle() for name, handle in self._handles.items()}
        else:
            if invert:
                self.handles = {name: handle() for name, handle in self._handles.items() if name not in selected_method}
            else:
                self.handles = {name: handle() for name, handle in self._handles.items() if name in selected_method}

        self.applied_handle_name = None

    @classmethod
    def register(cls, handle_type: type):
        cls._handles[handle_type.__name__] = handle_type
        return handle_type

    def trigger(self, gauss: Gaussian) -> bool:
        for name, handle in self.handles.items():
            if handle.trigger(gauss):
                self.applied_handle_name = name
                return True

        return False

    def handle(self, gauss: Gaussian) -> bool:
        handle = self.handles[self.applied_handle_name]
        return handle(gauss)


@AutoDebug.register
class Ignore(Debugger, ABC):
    """
    To handle the optimization can't fall into the optimal point which a tiny imaginary frequency.

    The optimization tasks might be hard to converge for certain reasons.
    In especial, the Minnesota functional(like M062X) might oscillate near the optimal point with imaginary frequency.

    There are some complicated method to solve this problem. However, If someone discovers that during the final stage
    of optimization, the molecules are only making slight vibrations around a certain equilibrium point, it would be
    more direct to ignore this error because the final configuration is likely not far from the optimal configuration.

    Trigger:
        task: optimization
        method: unspecific
        basis: unspecific
        keywords: unspecific
        error_link: l9999
        message: -- Number of steps exceeded,
        other:
            Among last 10% opti steps, Max(|E(N)-E(N-1)|) < 0.05 eV and Min(|E(N)-E(N-1)|) / Max(|E(N)-E(N-1)|) > 0.9
            where, the E(N) is the electron energy at the N opti step.

    Handle:
        ignore this error and continue the next work.
    """
    def trigger(self, gauss: Gaussian) -> bool:
        if gauss.output.is_opti_convergence_error:
            from hotpot.cheminfo import Molecule
            mol = Molecule.read_from(gauss.stdout, fmt='g16log')

            if not mol:
                raise IOError('the stdout cannot parse to Molecule object')

            try:
                last_energies = mol.all_energy[-int(len(mol.all_energy)):]
                diff_energies = np.abs(last_energies[1:] - last_energies[:-1])

            except TypeError:  # if it does not get any energy value
                return False

            if diff_energies.max() < 0.05 and diff_energies.min() / diff_energies.max() > 0.9:
                return True

            print(
                f"the max energy diff: {diff_energies.max()}, "
                f"min / max energy: {diff_energies.min() / diff_energies.max()}"
            )
            return False

    def handle(self, gauss: Gaussian) -> bool:
        """ Continue the next work and save the calculation data """
        gauss.stderr = ""
        gauss.output.stderr = ""
        return False  # more operate is no longer needed


# @AutoDebug.register
# class PreOptiB3LYP(Debugger, ABC):
#     """
#     This Debugger to handle the l9999 error which is can't be ignored by the Ignored debugger.
#
#     To handle the error:
#         1) the debugger will first read the last conformer in the last work.
#         2) replace the DFT method to `B3LYP` and perform optimization, the method `B3LYP` method
#          is a robust method and more easily to convergence.
#         3) lastly, optimizing conformer from the last conformer by original method
#     """
#     def trigger(self, gauss: Gaussian) -> bool:
#         """"""
#         if gauss.output.is_opti_convergence_error:
#             return True
#         return False
#
#     def handle(self, gauss: Gaussian) -> bool:
#         """"""
#         return False


@AutoDebug.register
class ReOptiWithSASSurfaceSCRF(Debugger, ABC):
    """
    This Handle for the optimization task in solvent, when some tiny molecular cages are formed.
    In this case, the default vdW surface overestimate the accessible surface inside the cages,
    and cause inverse PCM matrix the non-convergent.

    To handle the error, the Gaussian will re-optimize the molecule with SAS surface first and
    optimize with SES surface finally.

    Trigger:
        task: optimization
        methods: unspecified
        basis: unspecified
        keywords: SCRF
        error_link: l502, l508
        message: Inv3 failed in PCMMkU.

    Handle:
        rerun the optimization with SAS and SES surfaces serially
    """
    def trigger(self, gauss: Gaussian) -> bool:
        if gauss.output.is_scrf_Vdw_cage_error and \
          all(self._find_keyword_name(gauss.parsed_input['route'], kwd) for kwd in ('opt', 'scrf')):
            return True

        return False

    def handle(self, gauss: Gaussian) -> bool:
        route = gauss.parsed_input['route']

        scrf_name = self._find_keyword_name(route, 'scrf')

        gauss.full_option_values('route', scrf_name, 'smd')
        gauss.full_option_values('route', scrf_name, 'read')

        # convert the last conformer to the input script
        gauss.to_conformer()

        # the other items in the end of input script, add sas surface
        max_other = max(map(int, [t.split('_')[1] for t in gauss.parsed_input if 'other_' in t]))
        new_other = f"other_{max_other+1}"
        gauss.parsed_input[new_other] = 'surface=sas'

        # optimize with sas surface first
        gauss.run()
        # if the optimization is unsuccessful, terminate
        if gauss.stderr:
            print('Fail to optimize in SAS surface!!!')
            return False

        gauss.to_conformer()
        gauss.parsed_input[new_other] = 'surface=ses AddSph'

        return True


@AutoDebug.register
class ReOptiByCartesian(Debugger, ABC):
    """ """
    def trigger(self, gauss: Gaussian) -> bool:
        # If the error is ZMatrix trouble and the original task is optimization
        return gauss.output.is_ZMatrix_error and self._find_keyword_name(gauss.parsed_input['route'], 'opt')

    def handle(self, gauss: Gaussian) -> bool:
        route = gauss.parsed_input['route']
        opt_name = self._find_keyword_name(route, 'opt')

        gauss.full_option_values('route', opt_name, "Cartesian")

        return True


@AutoDebug.register
class Restart(Debugger, ABC):
    """ Handle Gaussian error by Restart """
    def trigger(self, gauss: Gaussian) -> bool:
        if gauss.output.is_hangup_error:
            return True

        return False

    def handle(self, gauss: Gaussian) -> bool:
        route = gauss.parsed_input['route']
        opt_name = self._find_keyword_name(route, 'opt')  # Get the actual user-give keyword for optimization

        gauss.full_option_values('route', opt_name, 'Restart')

        return True


@AutoDebug.register
class RerunFromLastConformer(Debugger, ABC):
    """ Handle the error by rerun the Gaussian from the last conformer """
    def trigger(self, gauss: Gaussian) -> bool:
        return True

    def handle(self, gauss: Gaussian) -> bool:
        gauss.to_conformer()  # convert to the last conformer in the stdout
        return True


# Retrieve data from the result log file:
def retrieve_log_data(
        root: Union[str, Path],
        file_pattern: str = "**.*.log",
):
    """"""
    root = Path(root)


