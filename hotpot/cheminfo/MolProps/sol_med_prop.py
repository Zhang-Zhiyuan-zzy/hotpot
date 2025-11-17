"""
@File Name:        sol_med_prop
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/14 14:53
@Project:          Hotpot
"""
from typing import Literal, Union
import os.path as osp
import pandas as pd

__all__ = [
    'get_solvent_dataframe',
    'get_medium_dataframe',
    'query_solvent',
    'query_medium'
]

_root_dir = osp.dirname(osp.abspath(__file__))
_type_node = {'CASs': 'str', 'Cid': 'int'}

def _query_prop_by_cas(df, list_id, id_type: Literal['CASs', 'Cid']= 'CASs'):
    """
    Generic CAS lookup function with real PubChem enrichment.

    Args:
        df: DataFrame containing property data (e.g., MedProp or SolProp).
        list_id: list of ID numbers (duplicates are allowed).
        id_type: which id to be used ad the index to search, ['Cids', 'CASs'] are supported.

    Returns:
        DataFrame ordered according to the input CAS list.
        Missing entries are NaN; PubChem data added if requested.
    """
    # Normalize CAS numbers
    df = df.set_index(id_type)

    missing_id = set(list_id) - set(df.index.tolist())
    if missing_id:
        raise ValueError(f'Undefined {id_type}[{_type_node[id_type]}] is found: {missing_id}')

    return df.loc[list_id, :]


def get_medium_dataframe(sub_sheet: Literal['clean', 'fill'] = 'clean'):
    return pd.read_excel(osp.join(_root_dir, 'data', 'MedProp.xlsx'), sheet_name=sub_sheet)

def get_solvent_dataframe(sub_sheet: Literal['clean', 'fill'] = 'clean'):
    return pd.read_excel(osp.join(_root_dir, 'data', 'SolProp.xlsx'), sheet_name=sub_sheet)

def query_medium(
        list_id: list[Union[str, int]],
        id_type: Literal['CASs', 'Cid']= 'CASs',
        sub_sheet: Literal['clean', 'fill'] = 'clean'
) -> pd.DataFrame:
    """
    Query from Media Properties by CAS or Cid.

    Args:
        list_id: List of identifiers.
        id_type: 'CASs' or 'Cid'.
        sub_sheet: Which sheet to use ('clean' or 'fill').

    Returns:
        Filtered DataFrame.
    """
    return _query_prop_by_cas(get_medium_dataframe(sub_sheet), list_id, id_type=id_type)


def query_solvent(
        list_id: list[Union[str, int]],
        id_type: Literal['CASs', 'Cid']= 'CASs',
        sub_sheet: Literal['clean', 'fill'] = 'clean'
) -> pd.DataFrame:
    """
    Query from Solent Properties dataset by CAS or Cid.

    Args:
        list_id: List of identifiers.
        id_type: 'CASs' or 'Cid'.
        sub_sheet: Which sheet to use ('clean' or 'fill').

    Returns:
        Filtered DataFrame.
    """
    return _query_prop_by_cas(get_solvent_dataframe(sub_sheet), list_id, id_type=id_type)
