"""
python v3.9.0
@Project: hotpot
@File   : optimize
@Auther : Zhiyuan Zhang
@Data   : 2024/8/27
@Time   : 10:55
"""
import os
import argparse
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, MDS

from hotpot.plugins.opti.beyes.opti import draw_comics_map, next_params
from hotpot.plugins.opti.ev.core import EvoluteOptimizer


def _parse_params(params, columns: list, by_index) -> list[int]:
    indices = []
    for item in params.split(','):
        item = item.split('-')
        if len(item) == 1:
            if by_index:
                indices.append(int(item[0]))
            else:
                indices.append(columns.index(item[0]))

        elif len(item) == 2:
            if by_index:
                indices.extend(list(range(int(item[0]), int(item[1])+1)))
            else:
                indices.extend(list(range(columns.index(item[0]), columns.index(item[1])+1)))

        else:
            raise ValueError(f'the input params are not match the required format')

    return np.unique(indices).tolist()


def read_excel(excel_file, args):
    """
    Read Excel file and parse to two DataFrame sheet, i.e., inputs and target
    """
    data = pd.read_excel(excel_file)

    inputs_names = getattr(args, 'params', args.features)
    exclude_inputs = getattr(args, 'exclude_params', args.exclude_features)

    # Determine the indices of included params
    if isinstance(inputs_names, str):
        params_indices = _parse_params(inputs_names, list(data.columns), args.by_index)
    else:
        params_indices = list(range(data.shape[1]-1))

    # Determine the indices of excluded params
    if isinstance(exclude_inputs, str):
        exclude_params_indices = _parse_params(exclude_inputs, list(data.columns), args.by_index)
    else:
        exclude_params_indices = []

    # Drop excluded params
    if isinstance(params_indices, list) and isinstance(exclude_params_indices, list):
        params_indices = list(set(params_indices) - set(exclude_params_indices))

    # Assign the target columns
    if args.target is None:
        target_name = data.columns[-1]
    elif isinstance(args.target, str):
        target_name = args.target
    else:
        raise TypeError(f'Meeting a unrecognized target type {args.target} --> {type(args.target)}')

    # Make sure the target not in the params
    if target_name in data.columns[params_indices]:
        raise ValueError(f'The target name {args.target} cannot exist in the params')

    params = data.iloc[:, params_indices]
    target = data[target_name]
    return params, target


def _get_params_range(
        params: pd.DataFrame,
        params_space: str,
        by_index: bool
) -> (list[str], np.ndarray):
    """"""
    if not params_space:
        params_min = params.values.min(axis=0)
        params_max = params.values.max(axis=0)
        params_range = params_max - params_min

        return params.columns.tolist(), np.vstack((params_min-0.25*params_range, params_max+0.25*params_range)).T

    elif isinstance(params_space, str):
        params_space = params_space.split(',')

        param_names, params_range = [], []
        for ps in params_space:
            name, low, high = ps.split(':')

            if by_index:
                param_names.append(params.columns[int(name)])
            else:
                param_names.append(name)
            params_range.append([float(low), float(high)])

        return param_names, np.array(params_range)


def _get_log_indices(params: pd.DataFrame, log_params: str, by_index: bool) -> Union[None, list[int]]:
    """"""
    if not log_params:
        return None

    log_indices = []
    if by_index:
        for pi in log_params.split(','):
            log_indices.append(int(pi))
    else:
        param_names = params.columns.tolist()
        for pn in log_params.split(','):
            log_indices.append(param_names.index(pn))


def _get_embedding_method(emb_method: str):
    """"""
    if emb_method == 'TSNE':
        return TSNE()
    elif emb_method == 'MDS':
        return MDS()

    return TSNE()

def _get_init_index(init_index: Union[int, None], batch_size: int, params: pd.DataFrame) -> int:
    if isinstance(init_index, int):
        if 0 <= init_index < params.shape[0]:
            return init_index
        elif -params.shape[0] < init_index < 0:
            return params.shape[0] + init_index
        else:
            raise ValueError(f'Invalid init_index, should be in range ({-params.shape[0]}, {params.shape[0]})')
    else:
        return params.shape[0] - batch_size

def add_arguments(optimize_parser: argparse.ArgumentParser):
    """ Add arguments to the parser """
    optimize_parser.add_argument('excel_file', type=str, help='the path of excel file')
    optimize_parser.add_argument('result_dir', type=str, help='the directory to save the result files')

    # Option groups
    opti_base = optimize_parser.add_argument_group('Base configurations')
    opti_bayes = optimize_parser.add_argument_group('Bayesian Options')
    opti_plot = optimize_parser.add_argument_group('Options to control result plotting')

    opti_base.add_argument(
        '-m', '--method',
        action='store',
        default='Bayesian',
        help="The method to perform parameters optimizer, choose from ['Bayesian(B)', 'Evolution(E)']"
    )
    opti_base.add_argument(
        '-p', '--params',
        help='Which items is the parameters you want to optimize, the default is all columns in the input sheet,'
             'excluding of the last column. The items can be specified by a range of item using a connection "-", or '
             'individual items separated by a comma ",", for example:\n'
             '    python hotpot optimize excel_file result_dir -p item1-item2\n'
             '    python hotpot optimize excel_file result_dir -p item0,item1,item2,item3\n'
             '    python hotpot optimize excel_file result_dir -p item0,item1-item2,item3'
    )
    opti_base.add_argument(
        '-ep', '--exclude-params',
        help='Exclude params in the input sheet columns from the optimization'
    )
    opti_base.add_argument(
        '-i', '--by-index',
        help='Specify the params by their index in the input excel',
        action='store_true'
    )

    opti_base.add_argument(
        '-t', '--target',
        help="Which are your optimizing target or indicator in the input sheet, the default is the last column."
    )

    opti_base.add_argument(
        '--plot',
        action="store_true",
        help="Plot the 2D manifold comics to show the mu and sigma charges",
    )

    opti_base.add_argument(
        '--mesh',
        default=10,
        help="The design parameter space will be weaved, This argument specify the mesh number for each parameter, "
             "the default is the 20",
        type=int
    )

    opti_base.add_argument(
        '-b', '--batch-size',
        default=5,
        help="The batch size in the each iteration",
        type=int
    )

    opti_bayes.add_argument(
        '-s', '--params-space',
        help="Specify the low:high bounds of the each parameter, separated by comma ','\n"
             "Following are example:\n"
             "    python hotpot optimize excel_file result_dir -s [name0|index0]:low0:high0,[name1|index1]:"
             "low1:high1,[name2|index2]:low2:high2 ...\n"
             "The counts of specified space must be equal to the number of specified parameters"
    )

    opti_bayes.add_argument(
        '--log-params',
        help="Which parameters are optimized in log scale, specified parameters by their names or indices (with -i flog)"
             " and each parameters is separated by comma ','"
    )

    opti_plot.add_argument(
        '--init-index',
        help="This arguments is only used when '--plot' flag is set to true.\n"
             "The start sample index to reproduce the optimizing procedure\n"
             "The specified value is range [-sample_num, sample_num]",
        type=int
    )

    opti_plot.add_argument(
        '-e', '--emb-method',
        default='TSNE',
        help="The embedding method to project parameters to 2D manifold map"
    )

    opti_plot.add_argument(
        '--examples',
        choices=['COF'],
        # default='COF',
        help="Reproduce the optimization procedure of COF experiment"
    )


def optimize(excel_file: Union[Path, str], out_file: Union[Path, str], args):
    """"""
    excel_file = Path(excel_file)
    out_file = Path(out_file)

    if args.examples == 'COF':
        example_cof_params(out_file)

    elif args.method in ['Bayesian', 'B', 'Bayesian(B)']:

        if not out_file.is_dir():
            raise IOError(f'The output directory {out_file} does not exist or is not a directory')

        params, target = read_excel(excel_file, args)

        _log_param_indices = _get_log_indices(params, args.log_params, args.by_index)
        specified_names, specified_range = _get_params_range(params, args.params_space, args.by_index)

        # Configure the parameters space
        param_names, param_ranges = [], []
        for i, param_name in enumerate(params.columns):
            try:
                index = specified_names.index(param_name)
                param_range = specified_range[index]
            except IndexError:
                if args.background:
                    raise ValueError(f'The space of param {param_name} has not been specified in the command')
                else:
                    param_range = input(f"Specify the lower and upper bounds for {param_name} (lower:upper):")
                    param_range = [float(v) for v in param_range.split(':')]

            param_names.append(param_name)
            param_ranges.append(param_range)

        # Assign linear of logarithmic for parameters
        if args.background:
            log_param_indices = _log_param_indices
        else:
            log_param_indices = []
            for i, param_name in enumerate(params.columns):
                if _log_param_indices and i in _log_param_indices:
                    to_linear = input(f'The {param_name} is logarithmic? (Enter to Yes|Any other key to Linear)')
                    if not to_linear:
                        log_param_indices.append(i)
                else:
                    to_logarithmic = input(f'The {param_name} is Linear? (Enter to Yes|Any other key to Logarithmic)')
                    if to_logarithmic:
                        log_param_indices.append(i)

        log_param_indices = log_param_indices if log_param_indices else None
        init_index = _get_init_index(args.init_index, args.batch_size, params)

        # Report arguments
        print("The Bayesian optimization will be performed, Make sure the arguments are correct")
        print("params shape:", params.shape)
        print("target shape:", target.shape)
        print(f"target: {target.name}")
        print(f"params mesh: {args.mesh}")
        print(f"initial index: {init_index}")
        print(f"batch size: {args.batch_size}")
        print(f"map embedding method: {args.emb_method}")
        print(f"Output directory: {out_file}")

        print(f'Make sure the design space:')
        for i, (name, param_range) in enumerate(zip(param_names, param_ranges)):
            print(
                f'\t{name}: {param_range}',
                f"{'logarithmic' if log_param_indices and i in log_param_indices else 'linear'}")

        if not args.background:
            make_sure = None
            while not make_sure:
                make_sure = input('Are you sure about these arguments? (Yes|Exit)')
                if make_sure == 'Yes':
                    break
                elif make_sure == 'Exit':
                    raise KeyboardInterrupt('Exit requested!')
                else:
                    make_sure = None

        if args.plot:
            draw_comics_map(
                params.values, target.values.flatten(),
                init_index=init_index,
                batch_size=args.batch_size,
                param_names=list(params.columns),
                param_range=np.array(param_ranges),
                mesh_counts=args.mesh,
                log_indices=log_param_indices,
                figpath_dir=out_file,
                emb_method=_get_embedding_method(args.emb_method)
            )

        else:
            next_params(
                params.values, target.values.flatten(),
                param_names=list(params.columns),
                param_range=np.array(param_ranges),
                next_param_path=out_file.joinpath('next_params.xlsx'),
                figpath=out_file.joinpath('params_mapping.png'),
                mesh_counts=args.mesh,
                log_indices=log_param_indices,
            )

    else:
        raise NotImplementedError('the method {} is not implemented'.format(args.method))


def example_cof_params(out_file):
    """
    Reproduce the parameters optimization in COF experiment,
    See ref: ...
    """
    print("Reproduce the parameters optimization in COF experiment,See ref: ...")

    excel_path = Path(__file__).parents[1].joinpath('dataset', 'data', 'examples', 'COF.xlsx')
    excel = pd.read_excel(excel_path, index_col=0, engine='openpyxl')

    param_names = ['T', 'C', 'V']
    params = excel[param_names].values
    param_ranges = np.array([[0., 100.], [0., 17.5], [1., 1000.]])

    a_peak = excel['a'].values
    b_peak = excel['b'].values
    target = a_peak / b_peak

    mesh_counts = 20
    log_param_indices = None
    emb_method = 'TSNE'

    # Run examples
    draw_comics_map(
        params, target,
        init_index=64,
        batch_size=5,
        param_names=param_names,
        param_range=param_ranges,
        mesh_counts=mesh_counts,
        log_indices=log_param_indices,
        figpath_dir=out_file,
        emb_method=_get_embedding_method(emb_method)
    )


if __name__ == '__main__':
    example_cof_params('/home/zz1/cof')
