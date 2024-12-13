"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/3/12
@Time   : 15:59
"""
import os
from typing import Union
import numpy as np
import pandas as pd

_solvent_features = pd.read_excel(os.path.join(os.path.dirname(__file__), 'solvents.xlsx'), index_col=0)


def get_all_solvents_names() -> list[str]:
    return _solvent_features.index.tolist()


def to_solvent_features(lst_sols: list, fill_miss: Union[list, bool] = False) -> pd.DataFrame:
    """
    convert a list of solvent_names into embedding descriptors
    Args:
        lst_sols(list of str): list of solvent names
        fill_miss(list|bool, optional): whether to allow to give a missing value, not defined in solvents set.
            if False, raise ValueError when meet a missing value
            if True, allow any values but print a warning message when meet a missing values
            if list of str, raise a ValueError when meet a missing value excluding in fill_miss list.
        all missing values will be filled with a zero vector.
    """
    sol_features = []
    for sol in lst_sols:
        try:
            sol_features.append(_solvent_features.loc[sol, :])
        except KeyError:
            if fill_miss is False:
                raise ValueError(f"the {sol} is not defined in solvent set, seeing the defined list"
                                 f"by calling hotpot.plugin.features.get_all_solvents_names()")
            elif fill_miss is True:
                fill_value = pd.Series(
                    np.zeros(_solvent_features.shape[1], dtype=float),
                    index=_solvent_features.columns, name=sol
                )
                sol_features.append(fill_value)
                print(UserWarning(f'meeting a miss feature name: {sol}, filling with {fill_value}'))
            elif isinstance(fill_miss, list):
                if sol in fill_miss:
                    fill_value = pd.Series(
                        np.zeros(_solvent_features.shape[1], dtype=float),
                        index=_solvent_features.columns, name=sol
                    )
                    sol_features.append(fill_value)
                else:
                    raise ValueError(
                        f"the {sol} is outside the defined feature list and the passing miss names"
                        f"seeing the defined feature list by calling hotpot.plugin.features.get_all_solvents_names()")
            else:
                raise TypeError(f"the given fill_miss should be either a list or boolean, not {type(fill_miss)}")

    return pd.DataFrame(sol_features)


__all__ = ['get_all_solvents_names', 'to_solvent_features']

