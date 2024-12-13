"""
python v3.9.0
@Project: hotpot
@File   : tools
@Auther : Zhiyuan Zhang
@Data   : 2024/12/9
@Time   : 8:58
"""
from time import time


def show_time(func, *args, **kwargs):
    start = time()
    func(*args, **kwargs)
    stop = time()
    print(stop - start)