"""
python v3.9.0
@Project: hotpot
@File   : ob2netwx
@Auther : Zhiyuan Zhang
@Data   : 2023/7/16
@Time   : 21:17

Notes:
    This module to perform the conversion from openbabel objects or networkx to each other.
"""
from openbabel import openbabel as ob
import networkx as nx


def ob_mol2nxg(ob_mol: ob.OBMol) -> np.Graph:
    """"""


def nxg2ob_mol(nxg: nx.Graph) -> ob.OBMol:
    """ Convert the networkx Graph class to an openbabel OBMol class """

