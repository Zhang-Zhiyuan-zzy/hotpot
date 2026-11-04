"""Local queries for packaged solvent and medium property reference tables."""
from typing import Literal, Union
from importlib.resources import as_file, files
import pandas as pd

__all__ = [
    'get_solvent_dataframe',
    'get_medium_dataframe',
    'query_solvent',
    'query_medium'
]

_type_node = {'CASs': 'str', 'Cid': 'int'}

def _query_prop_by_cas(df, list_id, id_type: Literal['CASs', 'Cid']= 'CASs'):
    """
    Look up existing local identifiers in their requested order.

    Args:
        df: DataFrame containing property data (e.g., MedProp or SolProp).
        list_id: list of ID numbers (duplicates are allowed).
        id_type: identifier column, either 'Cid' or 'CASs'.

    Returns:
        DataFrame ordered according to the input identifiers, including duplicates.
        Missing identifiers raise ValueError; no network enrichment is performed.
    """
    # Normalize CAS numbers
    df = df.set_index(id_type)

    missing_id = set(list_id) - set(df.index.tolist())
    if missing_id:
        raise ValueError(f'Undefined {id_type}[{_type_node[id_type]}] is found: {missing_id}')

    return df.loc[list_id, :]


def get_medium_dataframe(sub_sheet: Literal['clean', 'fill'] = 'clean'):
    resource = files('hotpot.dataset').joinpath('resources', 'properties', 'MedProp.xlsx')
    with as_file(resource) as path:
        return pd.read_excel(path, sheet_name=sub_sheet)

def get_solvent_dataframe(sub_sheet: Literal['clean', 'total'] = 'clean'):
    resource = files('hotpot.dataset').joinpath('resources', 'properties', 'SolProp.xlsx')
    with as_file(resource) as path:
        return pd.read_excel(path, sheet_name=sub_sheet)

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
        sub_sheet: Literal['clean', 'total'] = 'clean'
) -> pd.DataFrame:
    """
    Query from Solvent Properties dataset by CAS or Cid.

    Args:
        list_id: List of identifiers.
        id_type: 'CASs' or 'Cid'.
        sub_sheet: Which sheet to use ('clean' or 'total').

    Returns:
        Filtered DataFrame.
    """
    return _query_prop_by_cas(get_solvent_dataframe(sub_sheet), list_id, id_type=id_type)
