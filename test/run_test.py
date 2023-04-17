"""
python v3.7.9
@Project: hotpot
@File   : run_test.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/2
@Time   : 8:44
"""
import sys
from pathlib import Path

dir_root = Path.cwd().parents[2]
dir_code = dir_root.joinpath("code")
sys.path.extend([str(dir_root), str(dir_code)])
