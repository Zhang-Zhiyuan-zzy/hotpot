"""
python v3.7.9
@Project: hotpot
@File   : __init__.py.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/26
@Time   : 1:47
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from base import HpLammps
from materials import AmorphousMaker
