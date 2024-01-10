"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2023/11/16
@Time   : 21:58

Notes:
    Building the predictive model and analyzing the structure-performance relationship by descriptor-based
    machine learning algorithm.
"""
from typing import *
from pathlib import Path

import pandas as pd
import sklearn
import shap
import matplotlib.pyplot as plt


class DesML:
    """ an easy interface to perform most common workflow of descriptor-based machine learning """
    def __init__(
            self, root: Union[str, Path], data: pd.DataFrame, target: str,
            descriptors: list[str] = None, model_sets: list[str] = None,
    ):
        self.root = root
        self.data = data
        self.target = target
        self.descriptor = descriptors

    def work(self):
        """"""
