# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : conversion
 Created   : 2025/5/19 10:56
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os
import logging
from pathlib import Path

from tqdm import tqdm

from hotpot import Molecule


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
