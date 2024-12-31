"""
python v3.9.0
@Project: hotpot
@File   : _works
@Auther : Zhiyuan Zhang
@Data   : 2023/11/29
@Time   : 13:30

Notes:
    defining some convenient workflow to run Gaussian
"""
import os
import re
import logging
from os import PathLike
from pathlib import Path
from typing import Union
from copy import copy
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

from hotpot.cheminfo.core import Molecule
from .gauss import Gaussian, Options, GaussOut
from hotpot.utils.mp import mp_run


def run_gaussian(
        mol: Molecule,
        link0: str = None,
        route: str = None,
        g16root: Union[str, PathLike] = None,
        gjf_save_path: Union[str, PathLike] = None,
        log_save_path: Union[str, PathLike] = None,
        err_save_path: Union[str, PathLike] = None,
        report_set_resource_error: bool = False,
        options: Options = None,
        test: bool = True,
        **kwargs
):
    _script = mol.write(fmt='gjf', link0=link0, route=route, **kwargs)
    gaussian = Gaussian(
        g16root=g16root,
        path_gjf=gjf_save_path,
        path_log=log_save_path,
        path_err=err_save_path,
        report_set_resource_error=report_set_resource_error,
        options=options,
    )
    gaussian.run(_script, test=test)

    return gaussian.output


def _read_log(fp):
    return GaussOut.read_file(fp)


def export_results(
        *log_file_path: Union[str, PathLike],
        skip_errors: bool = True,
        retrieve_mol: bool = True,
        nproc: int = None,
        timeout: float = None,
):

    def _target(p):
        o = GaussOut.read_file(p)
        n = os.path.basename(p).split('.')[0]

        if o.is_error:
            return None, n, o.error_link

        r = o.export_pandas_series(n)

        if retrieve_mol:
            m = o.export_mol()
        else:
            m = None

        return r, n, m

    lst_res = mp_run(
        _target,
        map(lambda x: (x,), log_file_path),
        nproc=nproc,
        timeout=timeout,
        desc='Exporting Gaussian results...'
    )

    results = []
    errors = []
    mols = {}
    for res, name, mol_or_elink in tqdm(lst_res, 'Sum Results...'):

        if res is None:
            if skip_errors:
                print(RuntimeWarning(f"{name} with error Link {mol_or_elink}, Skip !!"))
            else:
                raise RuntimeError(f"{name} with error Link {mol_or_elink}!!")

            errors.append(name)
            continue

        results.append(res)
        mols[name] = mol_or_elink

    return pd.concat(results, axis=1).T, mols


def parse_gjf(path_gjf: Union[str, Path]):
    """"""
    def _parse_link0():
        nonlocal state
        if line.startswith('%'):
            parsed_info['link0'].append(line.strip())
        else:
            state += 1  # into route

    def _parse_route():
        nonlocal state
        if line.startswith('#'):
            parsed_info['route'].append(line.strip())
        else:
            state += 1  # into title

    def _parse_title():
        nonlocal state
        if line.strip():
            parsed_info['title'] = line.strip()
            state += 1  # into charge_spin

    def _parse_charge_spin():
        nonlocal state
        if line.strip():
            parsed_info['charge'], parsed_info['spin'] = map(int, line.strip().split())
            state += 1  # into coordination

    def _parse_coordination():
        nonlocal state
        if line.strip():
            parsed_info['coordinates'].append(line.strip())
        else:
            state += 1  # into addition

    def _parse_addition():
        parsed_info['addition'].append(line)

    with open(path_gjf) as file:
        lines = file.readlines()

    parsed_info = {
        'link0': [],
        'route': [],
        'title': '',
        'charge': None,
        'spin': None,
        'coordinates': [],
        'addition': []
    }

    state = 0
    for line in lines:

        if state == 0:
            _parse_link0()
        if state == 1:
            _parse_route()
        if state == 2:
            _parse_title()
        elif state == 3:
            _parse_charge_spin()
        elif state == 4:
            _parse_coordination()
        elif state == 5:
            _parse_addition()

    return parsed_info


def reorganize_gjf(parsed_gjf: dict):
    """ Reorganize the parsed gjf dict to gjf script """
    script = ""
    for line in parsed_gjf['link0']:
        script += line + '\n'
    for line in parsed_gjf['route']:
        script += line + '\n'
    script += f'\n{parsed_gjf["title"]}\n\n'
    script += f'{parsed_gjf["charge"]} {parsed_gjf["spin"]}\n'
    for line in parsed_gjf['coordinates']:
        script += line + '\n'
    script += '\n'
    for line in parsed_gjf['addition']:
        script += line

    return script


def ladder_opti(mol: Molecule, ladder: list[str], *args, **kwargs):
    """"""


def update_gjf_coordinates(old_gjf_file: Union[str, PathLike], log_file: Union[str, PathLike]):
    mol = Molecule.read_from(log_file, 'g16log')
    return update_gjf(
        old_gjf_file, {'coordinates': [
            f'{atom.symbol:18}{"   ".join(map(lambda x: f"{x:f}", atom.coordinate))}' for atom in mol.atoms
        ]}
    )


def update_gjf(old_gjf_file: Union[str, PathLike], update_dict: dict):
    data = parse_gjf(old_gjf_file)
    data.update(update_dict)
    return data


def parse_route(route: str) -> dict:
    """ Parse the route of gjf file """
    # Find structure morphology: method1/basis1//method2/basis2
    found = re.findall(r'\w+/\w+//\w+/\w+', route)
    if len(found) == 1:
        route = route.replace(found[0], '')
    elif len(found) > 1:
        raise ValueError('A route line only allow one method1/basis1//method2/basis2 handle morphology')

    # compile regular expressions
    parenthesis = re.compile(r'\([^()]+\)')

    # Normalize the input route
    route = re.sub(r'\s*=\s*', r'=', route)  # Omit the whitespace surround with the equal signal
    route = re.sub(r'=\(', r'(', route)  # Omit the equal signal before the opening parenthesis
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

    parsed_route = {'head': items[0]}
    for item in items[1:]:
        opening_parenthesis = re.findall(r'\(', item)
        closing_parenthesis = re.findall(r'\)', item)

        # When the keyword have multiply options
        if opening_parenthesis:
            if not (len(opening_parenthesis) == 1 and len(closing_parenthesis) == 1 and item[-1] == ')'):
                raise ValueError(f"Error route: {route}")

            keyword = item[:item.index('(')]
            options = item[item.index('(') + 1:-1]

            opt_dict = parsed_route.setdefault(keyword, {})
            for option in options.split(','):
                opt_value = option.split('=')
                if len(opt_value) == 1:
                    opt_dict[opt_value[0]] = None
                elif len(opt_value) == 2:
                    opt_dict[opt_value[0]] = opt_value[1]
                else:
                    raise ValueError('the given route string is wrong!!')

        else:  # When the keyword only a single option
            keyword_opt_value = item.split('=')
            if len(keyword_opt_value) == 1:
                parsed_route[keyword_opt_value[0]] = {}
            elif len(keyword_opt_value) == 2:
                parsed_route[keyword_opt_value[0]] = {keyword_opt_value[1]: None}
            elif len(keyword_opt_value) == 3:
                parsed_route[keyword_opt_value[0]] = {keyword_opt_value[1]: keyword_opt_value[2]}
            else:
                raise ValueError('the given route string is wrong!!')

    if found:  # Add method1/basis1//method2/basis2 handle
        parsed_route[found[0]] = {}

    return parsed_route


class ResultsExtract:
    """
    Defining an ensemble workflow to extract information from Gaussian16 log file, or convert to other file format:
        1) calculating results to pandas.DataFrame
        2) the corresponding gjf script
        ...
    """
    def __init__(
            self,
            dir_log: Union[str, PathLike],
            pass_error: bool = False
    ):
        self.dir_log = Path(dir_log)

        self.attrs = ['energy', 'zero_point', 'free_energy', 'enthalpy', 'entropy', 'thermal_energy', 'capacity']
        self.pass_error = pass_error

    def extract(self) -> (pd.DataFrame, Molecule):
        """ Extract the calculating results to DataFrame """
        list_series, list_mols = [], []
        for log_path in tqdm(self.dir_log.glob("*.log"), 'Extract log files', total=len(os.listdir(self.dir_log))):
            logging.info(log_path.stem)

            try:
                mol = Molecule.read_by_cclib(log_path, 'g16')
                value = [getattr(mol, n) for n in self.attrs]

            except (AttributeError, ValueError):
                if self.pass_error:
                    continue
                else:
                    raise AttributeError(f'{log_path} is an error log file!')

            list_series.append(pd.Series(value, index=self.attrs, name=log_path.stem))
            list_mols.append(mol)

        return pd.DataFrame(list_series), list_mols

    @staticmethod
    def rewrite_route(parsed_route: dict):
        """ Rewrite the parsed route to the route str line """
        parsed_route = copy(parsed_route)
        script = parsed_route.pop('head')

        for kw, ops in parsed_route.items():

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

        return script

    @staticmethod
    def extract_log_info(log_path) -> dict:
        """ extract log file information to parsed dict """
        mol = Molecule.read_from(log_path, 'g16log')
        info = {
            'link0': [],
            'route': [],
            'title': log_path.stem,
            'charge': mol.charge,
            'spin': mol.spin_multiplicity,
            'coordinates': ['    '.join(map(str, c)) for c in mol.coordinates.tolist()],
            'addition': []
        }

        # Markers to identify the start and end of the input section
        link0_start = '%'
        route_start = '#'
        route_end = '-------'

        # Open the log file and the output file
        route = False
        with open(log_path, 'r') as file:
            for line in file.readlines():
                # Check for the start of the input section
                if line.strip().startswith(link0_start):
                    info['link0'].append(line.strip())

                elif line.strip().startswith(route_start):
                    route = True

                elif route and line.strip().startswith(route_end):
                    break

                if route:
                    info['route'].append(line.strip())

        return info

    def to_gjf(self, addition: Union[str, list[str]] = '\n') -> list[str]:
        """
        Convert log files to the corresponding gjf file.

        The got gjf file will copy the link0, routes of resulted log and the molecular specification
        is specified by the last frame in the resulted log.

        addition information, like the custom ECP, basis set, may be needed for this conversion.

        Args:
            addition: addition information specified by user.

        Returns:
            (str) list of gjf scripts with same link0 and route as those gjf to make the given log files,
            the molecular specification, like charge, spin multiplicity and the atom coordination is same
            with the last frame in the given log file, the addition information attaches to the tail
        """
        list_parsed_info = [self.extract_log_info(p) for p in tqdm(self.dir_log.glob('*.log'), 'To gjf')]

        if isinstance(addition, str):
            for info in list_parsed_info:
                info['addition'].append(addition)

        elif isinstance(addition, list):
            if len(addition) == len(list_parsed_info):
                for info, addi in zip(list_parsed_info, addition):
                    info['addition'].append(addi)
            else:
                raise ValueError('the length of given addition list should same as the number of log file')

        else:
            raise TypeError(f'the addition should be a str or a list of str, not{type(addition)}')

        return [reorganize_gjf(info) for info in list_parsed_info]