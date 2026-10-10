#!/usr/bin/env python3
"""Top-level Hotpot command dispatcher."""

from __future__ import annotations

import argparse
import importlib
import os
import os.path as osp
import sys
from argparse import ArgumentError
from dataclasses import dataclass
from typing import Optional, Sequence

from . import version
from hotpot.utils.configs.logging_config import setup_logging


PROJECT_DESCRIPTION = "A cheminformatics harness for coordination chemistry"


@dataclass(frozen=True)
class _CommandSpec:
    help: str
    module: Optional[str] = None


_COMMANDS = {
    "convert": _CommandSpec("Convert molecule file from one format to another"),
    "optimize": _CommandSpec(
        "Perform parameters optimization",
        "hotpot.main.optimize",
    ),
    "ml_train": _CommandSpec(
        "A standard workflow to train Machine learning models",
        "hotpot.main.ml_train",
    ),
    "mca": _CommandSpec(
        "Predict atom-resolved methyl cation affinity (MCA)",
        "hotpot.cheminfo.AImodels.mca.cli",
    ),
    "cbond": _CommandSpec(
        "Build metal-ligand coordination bonds with the CBond model",
        "hotpot.cheminfo.AImodels.cbond.cli",
    ),
    "models": _CommandSpec(
        "Install and verify external inference models",
        "hotpot.cheminfo.AImodels.artifacts.cli",
    ),
    "ff": _CommandSpec(
        "Build and optimize molecular geometries with force fields",
        "hotpot.cheminfo.forcefields.cli",
    ),
    "xtb": _CommandSpec(
        "Run an independent GFN-FF or GFN-xTB calculation",
        "hotpot.plugins.xtb.cli",
    ),
    "run": _CommandSpec(
        "Run a controlled sequence of molecular calculation stages",
        "hotpot.pipeline.cli",
    ),
}


def is_running_in_foreground() -> bool:
    """Return whether standard input is attached to a terminal."""
    try:
        return os.isatty(sys.stdin.fileno())
    except OSError:
        return False


def show_version() -> None:
    print(f"Hotpot version: {version()}")
    print(PROJECT_DESCRIPTION)


def _selected_command(arguments: Sequence[str]) -> Optional[str]:
    return next(
        (argument for argument in arguments if argument in _COMMANDS),
        None,
    )


def _add_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("infile", type=str, help="input file")
    parser.add_argument(
        "-f",
        "--output_file",
        type=str,
        help="output file or directory",
    )
    parser.add_argument("-i", "--inputs-format", type=str, help="input format")
    parser.add_argument("-o", "--output-format", type=str, help="output format")


def build_parser(
    selected_command: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Build the root parser while importing only the selected command module."""
    parser = argparse.ArgumentParser(
        prog="hotpot",
        description=PROJECT_DESCRIPTION,
    )
    parser.add_argument("-d", "--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "-b",
        "--background",
        action="store_true",
        help="run command in background; use together with '&' or nohup",
    )
    parser.add_argument("-v", "--version", action="store_true")

    works = parser.add_subparsers(title="works", dest="works")
    for command, spec in _COMMANDS.items():
        command_parser = works.add_parser(command, help=spec.help)
        if command == "convert":
            _add_convert_arguments(command_parser)
        elif command == selected_command and spec.module is not None:
            command_module = importlib.import_module(spec.module)
            command_module.add_arguments(command_parser)
    return parser


def _run_convert(args: argparse.Namespace) -> int:
    from .main import conversion

    infile = args.infile
    outfile = args.output_file
    if args.inputs_format:
        input_format = args.inputs_format
    elif osp.isfile(infile):
        input_format = osp.splitext(osp.basename(infile))[-1][1:]
    else:
        input_format = "smi"

    if args.output_format:
        output_format = args.output_format
    elif input_format != "smi":
        output_format = "smi"
    else:
        raise ArgumentError(None, "The output format is not specified")
    conversion.convert(infile, outfile, output_format, input_format)
    print("Done !!!")
    return 0


def run(args: argparse.Namespace) -> int:
    """Run one parsed Hotpot command."""
    if args.version:
        show_version()
        return 0
    if args.works == "convert":
        return _run_convert(args)
    if args.works is None:
        return -2

    spec = _COMMANDS[args.works]
    if spec.module is None:
        return -2
    command_module = importlib.import_module(spec.module)
    if args.works == "optimize":
        command_module.optimize(args.excel_file, args.result_dir, args)
        print("Done !!!")
        return 0
    if args.works == "ml_train":
        command_module.train(args)
        print("Done !!!")
        return 0
    return command_module.run(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging(to_stdout=False)
    raw_args = tuple(sys.argv[1:] if argv is None else argv)
    command = _selected_command(raw_args)
    pipeline_tail = ()
    if command == "run" and "--" in raw_args:
        separator_index = raw_args.index("--")
        pipeline_tail = raw_args[separator_index + 1 :]
        raw_args = raw_args[:separator_index]
    parser = build_parser(selected_command=command)
    args = parser.parse_args(raw_args)
    if pipeline_tail:
        args.pipeline_tokens.extend(pipeline_tail)
    return_code = run(args)
    if return_code == -2:
        parser.print_help()
        return 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
