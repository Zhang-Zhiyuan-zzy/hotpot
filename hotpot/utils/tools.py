"""
python v3.9.0
@Project: hotpot
@File   : tools
@Auther : Zhiyuan Zhang
@Data   : 2024/12/9
@Time   : 8:58
"""
from time import time
from typing import Callable


def show_time(func, *args, **kwargs):
    start = time()
    func(*args, **kwargs)
    stop = time()
    print(stop - start)


def show_func_time(func: Callable):
    def wrapper(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        stop = time()

        print(f"{func.__name__} running time: {stop - start}")
        return res

    return wrapper
