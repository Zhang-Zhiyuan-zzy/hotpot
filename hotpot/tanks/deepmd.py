"""
python v3.9.0
@Project: hotpot
@File   : deepmd.py
@Author : Zhiyuan Zhang
@Date   : 2023/6/26
@Time   : 21:42
"""
import os
import json

from hotpot import data_root

# Manuscript training script
_script = json.load(open(os.path.join(data_root, 'deepmd_script.json')))


def make_script():
    """"""
