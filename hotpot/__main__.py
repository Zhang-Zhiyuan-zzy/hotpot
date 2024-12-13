#!/home/pub/conda3/envs/hp/bin/python
"""
python v3.9.0
@Project: hotpot
@File   : __main__
@Auther : Zhiyuan Zhang
@Data   : 2024/8/22
@Time   : 17:24
"""
import os
import sys
import psutil

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
from tqdm import tqdm
import argparse
import logging

from .cheminfo.core import Molecule
from .main import optimize, ml_train
from . import version


def is_running_in_foreground():
    """
    Check if the script is running in foreground based on input/output capabilities.
    This rudimentary method checks the standard input stream.
    """
    try:
        # Try checking if stdin is attached to a terminal.
        # If it's not, it may be redirected or the program may be running in background.
        return os.isatty(sys.stdin.fileno())
    except OSError:
        # Handling error cases where fileno() could not be accessed
        return False


def _to_smiles(reader, output_file, mode='w'):
    with open(output_file, mode) as writer:
        for mol in tqdm(reader):
            writer.write(f'{mol.canonical_smiles}\n')


def convert(input_file, output_file, out_fmt: str, in_fmt=None):
    """"""
    input_file = Path(input_file)
    output_file = Path(output_file)

    list_file = []
    if os.path.isfile(input_file):
        list_reader = [Molecule.read(input_file, in_fmt)]
        list_file.append(list_file)
    elif os.path.isdir(input_file):
        if not isinstance(in_fmt, str):
            raise IOError('the input format must be given, when the input file is a directory')
        list_reader = []
        for in_fp in input_file.glob(f'*.{in_fmt}'):
            logging.debug(f"adding new input file: {in_fp}")
            list_reader.append(Molecule.read(in_fp, in_fmt))
            list_file.append(in_fp)
    else:
        raise IOError('the given input file does not exist!!!')


    for i, reader in enumerate(list_reader):

        print(f'read {i}th file ...')

        if output_file.is_dir():
            out_fp = output_file.joinpath(f'{i}.{out_fmt}')
            mode = 'w'
        else:
            out_fp = output_file
            mode = 'w' if not i else 'a'

        if out_fmt == 'smi':
            _to_smiles(reader, out_fp, mode)


def show_version():
    print(f"Hotpot version: {version()}")
    print("A C++/python package designed to communicate among various chemical and materials calculational tools")


def main():
    # if is_running_in_foreground():
    #     print('running in foreground')
    # else:
    #     print('running in background')

    parser = argparse.ArgumentParser(
        prog='hotpot',
        description="Process molecule file by 'hotpot' command"
    )

    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    parser.add_argument('-b', '--background', action='store_true',
                        help='run command in background, this flag should cowork with `&` or `nohup`')
    parser.add_argument('-v', '--version', action='store_true')

    works = parser.add_subparsers(title='works', help='', dest='works')

    # Convert job arguments
    convert_parser = works.add_parser('convert', help='Convert molecule file from one format to another')
    convert_parser.add_argument('infile', type=str, help='input file')
    convert_parser.add_argument('outfile', type=str, help='output file')
    convert_parser.add_argument('-f', '--format', type=str, help="the input and output format, split by ','")

    # Optimize job arguments
    optimize_parser = works.add_parser('optimize', help='Perform parameters optimization')
    optimize.add_arguments(optimize_parser)

    # ML_train job arguments
    ml_parser = works.add_parser('ml_train', help='A standard workflow to train Machine learning models')
    ml_train.add_arguments(ml_parser)

    # Parse arguments
    args = parser.parse_args()

    if args.version:
        show_version()
        return

    # convert work
    if args.works == 'convert':
        infile = args.infile
        outfile = args.outfile

        formats = args.format
        if isinstance(formats, str):
            formats = formats.split(',')
            if len(formats) == 1:
                in_fmt = formats[0]
                out_fmt = None
            elif len(formats) == 2:
                in_fmt, out_fmt = formats
            else:
                raise ValueError("-f flag only accepts one or two value(s)")

            convert(infile, outfile, out_fmt, in_fmt)

    elif args.works == 'optimize':
        optimize.optimize(args.excel_file, args.result_dir, args)

    elif args.works == 'ml_train':
        ml_train.train(args)

    else:
        parser.print_help()

    print("Done !!!")


if __name__ == '__main__':
    # print(sys.argv )
    # logging.basicConfig(level=logging.DEBUG)
    main()
