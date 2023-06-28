"""
python v3.9.0
@Project: hotpot
@File   : manage_machine.py
@Author : Zhiyuan Zhang
@Date   : 2023/6/26
@Time   : 17:20

Notes:
    Manage the machine, link memory, processors, GPUs
"""
from typing import *
import psutil


class Machine:
    """ The special class to retrieve the machine information and adjust the system parameters """

    @property
    def memory_info(self):
        return psutil.virtual_memory()

    @property
    def available_memory(self) -> int:
        """ Available memory with GB units """
        return self.memory_info.available / 1024 ** 3

    def take_memory(self, ratio: float = 0.5, integer: bool = True) -> Union[int, float]:
        """
        take a partial memory from machine available memory
        Args:
            ratio: How many ratio of taken memory take from the available
            integer: whether to force to take an integer values

        Returns:

        """
        if integer:
            return int(self.available_memory * ratio)
        else:
            return self.available_memory * ratio


machine = Machine()