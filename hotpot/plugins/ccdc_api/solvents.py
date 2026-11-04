"""Load packaged solvent molecules on their first public attribute access."""

from collections import OrderedDict
from functools import cache
from importlib.resources import as_file, files


__all__ = ['solvents', 'solvents_smiles']


@cache
def _solvent_data():
    from ccdc.io import MoleculeReader

    directory = files("hotpot.cheminfo").joinpath("ChemData", "solvent")
    solvents = OrderedDict()
    for resource in directory.iterdir():
        with as_file(resource) as path:
            with MoleculeReader(str(path)) as reader:
                solvents[resource.name] = reader[0]
    return solvents, {solvent.smiles for solvent in solvents.values()}


def __getattr__(name):
    if name == "solvents":
        return _solvent_data()[0]
    if name == "solvents_smiles":
        return _solvent_data()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == '__main__':
    for sol_name, solvent in _solvent_data()[0].items():
        print(f'{sol_name}: {solvent.name.smiles}')
