"""
python v3.7.9
@Project: hotpot
@File   : __init__.py.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/14
@Time   : 4:06
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

src_root = os.path.abspath(os.path.dirname(__file__))
data_root = os.path.abspath(os.path.join(src_root, 'data'))

from hotpot.cheminfo import Molecule
from hotpot.bundle import MolBundle

__version__ = '0.3.0.8'
