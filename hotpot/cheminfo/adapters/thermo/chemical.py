"""External Chemical construction, RD identifiers and DataFrame output."""


def _chemical_type():
    from thermo.chemical import Chemical

    return Chemical


def _construct_chemical(chemical_type, identifier, **conditions):
    return chemical_type(identifier, **conditions)


def _smiles(rdmol):
    from rdkit import Chem

    return Chem.MolToSmiles(rdmol, kekuleSmiles=True)


def _smiles_identifier(rdmol):
    return f'SMILES={_smiles(rdmol)}'


def _inchi_identifier(rdmol):
    from rdkit import Chem

    return Chem.MolToInchi(rdmol)


def _dataframe(rows, *, index):
    import pandas as pd

    return pd.DataFrame(rows, index=index)
