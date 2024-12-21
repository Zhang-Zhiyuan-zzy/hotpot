"""
python v3.9.0
@Project: hotpot
@File   : substitute
@Auther : Zhiyuan Zhang
@Data   : 2024/12/13
@Time   : 15:47
"""
from typing import Literal

from .core import Molecule



def generate_mol(frame: Molecule, branches: list["Branch"]):
    """"""


class Branch:
    def __init__(
            self,
            mol,
            points: list[int],
            edges: list[tuple[int, int]],
            _type: Literal['Frame', 'Branch'] = None
    ):
        self.mol = mol
        self.points = points
        self.edges = edges
        self.type = _type



