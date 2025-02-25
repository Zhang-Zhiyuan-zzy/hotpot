import os
from ._io import MolWriter



@MolWriter.add_plugin('xtb', 'write')
def xtb_io(writer, mol, *args, **kwargs):
    """"""

