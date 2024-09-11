"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/3/12
@Time   : 17:15
"""
import os
from typing import Union

import pandas as pd
import numpy as np

_media_feature = pd.read_excel(os.path.join(os.path.dirname(__file__), 'sol_media.xlsx'), index_col=0)


def get_all_media_names() -> list[str]:
    return _media_feature.index.tolist()


def to_media_features(lst_media, fill_miss: Union[list[str], bool] = True) -> pd.DataFrame:
    """
    Converts a list of media features into embedding features.
    Args:
        lst_media(list of str): list of solvent names.
        fill_miss(list|bool, optional): whether to allow to give a missing value, not defined in media set.
            if False, raise ValueError when meet a missing value
            if True, allow any values but print a warning message when meet a missing values
            if list of str, raise a ValueError when meet a missing value excluding in fill_miss list.
        all missing values will be filled with a zero vector.
    """
    media_features = []
    for sol in lst_media:
        try:
            media_features.append(_media_feature.loc[sol, :])
        except KeyError:
            if fill_miss is False:
                raise ValueError(f"the {sol} is not defined in solvent set, seeing the defined list"
                                 f"by calling hotpot.plugin.features.get_all_solvents_names()")
            elif fill_miss is True:
                fill_value = pd.Series(
                    np.zeros(_media_feature.shape[1], dtype=float),
                    index=_media_feature.columns, name=sol
                )
                media_features.append(fill_value)
                print(UserWarning(f'meeting a miss feature name: {sol}, filling with {fill_value}'))
            elif isinstance(fill_miss, list):
                if sol in fill_miss:
                    fill_value = pd.Series(
                        np.zeros(_media_feature.shape[1], dtype=float),
                        index=_media_feature.columns, name=sol
                    )
                    media_features.append(fill_value)
                else:
                    raise ValueError(
                        f"the {sol} is outside the defined feature list and the passing miss names,"
                        f"seeing the defined feature list by calling hotpot.plugin.features.get_all_solvents_names()")
            else:
                raise TypeError(f"the given fill_miss should be either a list or boolean, not {type(fill_miss)}")

    return pd.DataFrame(media_features)


__all__ = ['get_all_media_names', 'to_media_features']
