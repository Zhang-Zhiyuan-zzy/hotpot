"""
python v3.9.0
@Project: hotpot
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2024/12/24
@Time   : 18:51
"""
import os
import re
import io
import copy
import json
import resource
import subprocess
from os.path import join as opj
from pathlib import Path
from typing import Union, Optional, Any, Iterable, Literal

import cclib
import numpy as np
import pandas as pd

from hotpot.cheminfo._io._io import _extract_g16_thermo, _extract_force_matrix, MolReader

root_dir = os.path.dirname(__file__)
settings: dict = json.load(open(opj(root_dir, 'settings.json')))
_tree: dict[str, Any] = json.load(open(opj(root_dir, 'settings.json')))


def _time_to_seconds(time_str):
    """
    Converts a time string in the format 'X days Y hours Z minutes W seconds' to total seconds.

    Parameters:
    time_str (str): The time string to convert.

    Returns:
    float: The total time in seconds.
    """
    days = hours = minutes = seconds = 0.0

    # Split the time string into components
    parts = time_str.split()
    for i in range(len(parts)):
        if parts[i] == "days":
            days = float(parts[i - 1])
        elif parts[i] == "hours":
            hours = float(parts[i - 1])
        elif parts[i] == "minutes":
            minutes = float(parts[i - 1])
        elif parts[i] == "seconds":
            seconds = float(parts[i - 1])

    # Calculate total seconds
    total_seconds = (days * 86400) + (hours * 3600) + (minutes * 60) + seconds
    return total_seconds


def extract_times_from_gaussian_log(log_content):
    """
    Extracts running times and elapsed time from a Gaussian log file and converts them to seconds.

    Parameters:
    log_file_path (str): The path to the Gaussian log file.

    Returns:
    dict: A dictionary containing the job CPU time and elapsed time in seconds.
    """
    job_cpu_time = None
    elapsed_time = None

    for line in log_content.splitlines()[-30:]:
        # Look for the line that contains the job CPU time
        if "Job cpu time:" in line:
            job_cpu_time = line.split(":")[-1].strip().rstrip('.')
        # Look for the line that contains the elapsed time
        elif "Elapsed time:" in line:
            elapsed_time = line.split(":")[-1].strip().rstrip('.')

    # Convert times to seconds
    return {
        "cup times": _time_to_seconds(job_cpu_time) if job_cpu_time else None,
        "elapsed time": _time_to_seconds(elapsed_time) if elapsed_time else None
    }


class OptionPath:
    """ Represent a path from the root to an option """
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
            options: Options = None,
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
        else:
            self.g16root = settings.get("g16root", None) or os.environ.get("g16root") or self.extract_g16root_from_bashrc()

        if not self.g16root:
            raise ValueError('the argument g16root is not given!')

        # Configure running environments and resources
        self.envs = self._set_environs()
        self._set_resource_limits(report_set_resource_error)

        self.path_gjf = path_gjf
        self.path_log = path_log
        self.path_err = path_err

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

        self.stdin = None
        self.output = None
        self.stdout = None
        self.stderr = None

    # v0.4.0
    # def run(self, script: str = None, test: bool = False):
    #     """Runs the Gaussian 16 process with the given script and additional arguments.
    #
    #     This method sets up the required environment variables and resource limits for Gaussian 16 before
    #     running the process using `subprocess.Popen`. It takes an input script and any additional arguments
    #     to pass to `Popen`, and returns a tuple of the standard output and standard error of the process.
    #
    #     Args:
    #         script (str): The input script for the Gaussian 16 process.
    #         test: if tree, running with the test model, at the time the running of Gaussian program will be skipped.
    #     Returns
    #         Tuple[str, str]: A tuple of the standard output and standard error of the process
    #     """
    #     with open(self.gjf_path, 'w') as writer:
    #         writer.write(script)
    #
    #     # Configure the input and output mode
    #     if self.output_in_running:
    #         cmd = ['g16', str(self.gjf_path), str(self.logout_path)]
    #         stdin = None
    #     else:
    #         cmd = ['g16']
    #         stdin = self.stdin
    #
    #     # Run Gaussian using subprocess
    #     if not test:
    #         self.g16process = subprocess.Popen(
    #             cmd, bufsize=-1, stdin=subprocess.PIPE,
    #             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #             env=self.envs, universal_newlines=True
    #         )
    #         self.stdout, self.stderr = self.g16process.communicate(stdin)
    #
    #     if self.output_in_running and not self.stdout:
    #         with open(self.logout_path) as file:
    #             self.stdout = file.read()
    #
    #     self.output = GaussOut(self.stdout, self.stderr)

    # V0.3.0
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
            # self.op.parsed_input_to_options(self.parsed_input)
        elif self.op:  # If some option have been assigned by Gaussian.Options
            # self.op.update_parsed_input(self.parsed_input)
            raise NotImplementedError("the Gaussian Option is not implemented")

        else:
            raise ValueError('the script is not given')

        self.stdin = self._rewrite_input_script()

        if self.path_gjf:
            with open(self.path_gjf, 'w') as writer:
                writer.write(self.stdin)

        # Configure the input and output mode
        cmd = ['g16']
        stdin = self.stdin

        # Run Gaussian using subprocess
        self.g16process = subprocess.Popen(
            cmd, bufsize=-1, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=self.envs, universal_newlines=True
        )
        self.stdout, self.stderr = self.g16process.communicate(stdin)

        self.output = GaussOut(self.stdout, self.stderr)

    @staticmethod
    def extract_g16root_from_bashrc(bashrc_path='~/.bashrc'):
        # Expand the user's home directory
        bashrc_path = os.path.expanduser(bashrc_path)

        try:
            with open(bashrc_path, 'r') as file:
                for line in file:
                    # Check if the line contains 'g16root'
                    if 'g16root' in line:
                        # Split the line to extract the variable value
                        # Assuming the format is 'export g16root=/path/to/g16'
                        parts = line.split('=')
                        if len(parts) > 1:
                            g16root_value = parts[1].strip().rstrip('\'"')  # Remove any trailing quotes
                            return g16root_value
        except FileNotFoundError:
            print(f"The file {bashrc_path} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")

        return None

    @staticmethod
    def _parse_route(route: str) -> dict:
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

        g16root = str(self.g16root)

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

    def __init__(self, stdout: str, stderr: str = None):
        self.stdout = stdout
        self.stderr = stderr

    @classmethod
    def read_file(cls, filename: Union[Path, str]) -> "GaussOut":
        with open(filename, 'r') as f:
            return cls(f.read())

    def get_times(self, unit: Literal['day', 'hour', 'minute', 'second'] = 'second'):
        second_times = extract_times_from_gaussian_log(self.stdout)
        if unit == 'day':
            return {name: time / (24 * 60 * 60) for name, time in second_times.items()}
        elif unit == 'hour':
            return {name: time / (60 * 60) for name, time in second_times.items()}
        elif unit == 'minute':
            return {name: time / 60 for name, time in second_times.items()}
        elif unit == 'second':
            return second_times
        else:
            raise ValueError(f'The unit {unit} is not supported.')

    def parse(self):
        string_buffer = io.StringIO(self.stdout)
        data = cclib.ccopen(string_buffer).parse()

        thermo, capacity = _extract_g16_thermo(self.stdout.splitlines())
        return {
            'atomic_number': data.atomnos,
            'partial_charge': data.atomcharges['mulliken'],
            'energy': data.scfenergies[-1],
            'spin': data.mult,
            'charge': data.charge,
            'mol_orbital_energies': np.array(data.moenergies),  # eV,
            'coordinates': data.atomcoords[-1],
            'gibbs': getattr(data, 'freeenergy', 0) * 27.211386245988,
            'zero_point': getattr(data, 'zpve', 0) * 27.211386245988,
            'spin_mult': getattr(data, 'mult', None),
            'thermo': thermo,
            'capacity': capacity,
            'force': _extract_force_matrix(self.stdout.splitlines(), data.atomnos)
        }

    def update_mol(self, mol):
        attrs = self.parse()
        atomic_numbers = attrs.pop('atomic_number')
        if any(a.atomic_number != a_nos for a, a_nos in zip(mol.atoms, atomic_numbers)):
            raise AttributeError("The molecule calculated by Gaussian is inconsistent with what needs to be updated.")

        mol.setattr(**attrs)

    def export_mol(self):
        return next(MolReader(self.stdout, 'g16log'))

    def export_pandas_series(self, name='mol'):
        attrs = {name: attr for name, attr in self.parse().items() if not isinstance(attr, np.ndarray)}
        return pd.Series(attrs, name=name)

    @property
    def is_error(self) -> bool:
        return bool(self.stderr) or bool(self.error_link)

    @property
    def error_link(self) -> str:
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
