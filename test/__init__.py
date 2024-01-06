"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2023/7/16
@Time   : 22:19
"""
from pathlib import Path
from hotpot import hp_root

test_root = Path(hp_root).joinpath('..', 'test')
