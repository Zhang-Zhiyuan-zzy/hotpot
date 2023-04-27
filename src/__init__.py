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
pkg_root = os.path.abspath(f'{src_root}/..')
data_root = os.path.abspath(os.path.join(pkg_root, 'data'))
tmp_root = os.path.abspath(os.path.join(pkg_root, 'tmp'))
