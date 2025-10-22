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
import os.path as osp
import sys
from argparse import ArgumentError

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import argparse
from .main import optimize, ml_train, conversion
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


def show_version():
    print(f"Hotpot version: {version()}")
    print("A C++/python package designed to communicate among various chemical and materials calculational tools")


def build_parser():
    parser = argparse.ArgumentParser(
        prog='hotpot',
        description="A C++/python package designed to communicate among various chemical and materials calculational tools"
    )

    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    parser.add_argument('-b', '--background', action='store_true',
                        help='run command in background, this flag should cowork with `&` or `nohup`')
    parser.add_argument('-v', '--version', action='store_true')

    works = parser.add_subparsers(title='works', help='', dest='works')

    # Convert job arguments
    convert_parser = works.add_parser('convert', help='Convert molecule file from one format to another')
    convert_parser.add_argument('infile', type=str, help='input file')
    convert_parser.add_argument('-f', '--output_file', type=str, help='output file or directory')
    convert_parser.add_argument('-i', '--inputs-format', type=str, help="the inputs format")
    convert_parser.add_argument('-o', '--output-format', type=str, help="the output format")

    # Optimize job arguments
    optimize_parser = works.add_parser('optimize', help='Perform parameters optimization')
    optimize.add_arguments(optimize_parser)

    # ML_train job arguments
    ml_parser = works.add_parser('ml_train', help='A standard workflow to train Machine learning models')
    ml_train.add_arguments(ml_parser)
    return parser


def run(args):
    """ Run the command line """
    if args.version:
        show_version()
        return 0

    # convert work
    if args.works == 'convert':
        infile = args.infile
        outfile = args.output_file

        if args.inputs_format:
            in_fmt = args.inputs_format
        else:
            if osp.isfile(infile):
                in_fmt = osp.splitext(osp.basename(infile))[-1][1:]  # Get suffix
            else:
                in_fmt = 'smi'

        if args.outputs_format:
            out_fmt = args.outputs_format
        elif in_fmt != 'smi':
            out_fmt = 'smi'
        else:
            raise ArgumentError('The output format is not specified')

        conversion.convert(infile, outfile, out_fmt, in_fmt)

    elif args.works == 'optimize':
        optimize.optimize(args.excel_file, args.result_dir, args)

    elif args.works == 'ml_train':
        ml_train.train(args)

    else:
        return -2  # indicate the work type not be specified

    print("Done !!!")
    return 0  # Normal termination


def main(argv: list[str] = None):
    parser = build_parser()

    # Parse arguments
    if argv is None:
        args = parser.parse_args()
    else:
        # Allow pass the args from the main() interface in test
        args = parser.parse_args(argv)


    try:
        return_code = run(args)
        if return_code == -2:
            parser.print_help()
            return 1
        return return_code
    except Exception as exc:
        raise exc
        # print(f"[ERROR] {exc}", file=sys.stderr)
        # if os.environ.get("HOTPOT_DEBUG"):
        #     raise exc
        # return 2


if __name__ == '__main__':
    # print(sys.argv )
    # logging.basicConfig(level=logging.DEBUG)
    main()
