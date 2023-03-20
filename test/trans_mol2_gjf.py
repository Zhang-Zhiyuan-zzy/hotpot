"""
python v3.7.9
@Project: hotpot
@File   : trans_mol2_gjf.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/19
@Time   : 21:25
"""
import sys
from pathlib import Path

dir_root = Path.cwd()
dir_code = dir_root.joinpath("code")
sys.path.extend([str(dir_root), str(dir_code)])
