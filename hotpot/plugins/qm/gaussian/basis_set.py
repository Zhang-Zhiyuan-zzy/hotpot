"""
python v3.9.0
@Project: hotpot
@File   : basis_set
@Auther : Zhiyuan Zhang
@Data   : 2024/1/15
@Time   : 20:54
Examples:
    #
"""
import os
from typing import Union, Sequence
from pathlib import Path


class Stuttgart:
    def __init__(self):
        self.basis_root = Path(__file__).parent.joinpath('basis_set', 'Stuttgart')

    def get(self, element, ecp, basis_set):
        with open(self.basis_root.joinpath(element, ecp, basis_set)) as file:
            basis = file.read()
        with open(self.basis_root.joinpath(element, ecp, "ECP")) as file:
            pseudo = file.read()

        return f'{basis}\n****\n\n{pseudo}'

    def get_basis(self, element, ecp, basis_set):
        with open(self.basis_root.joinpath(element, ecp, basis_set)) as file:
            basis = file.read()

        return f'{basis}\n****'

    def get_ecp(self, element, ecp):
        with open(self.basis_root.joinpath(element, ecp, "ECP")) as file:
            pseudo = file.read()

        return pseudo

    def search(self, elements: Union[str, Sequence] = None):
        """ Search all accessible Stuttgart ECPs and corresponding basis """
        if not elements:
            elements = os.listdir(self.basis_root)
        elif isinstance(elements, str):
            elements = [elements]
        elif not isinstance(elements, Sequence):
            raise TypeError("The type of arg `elements` should be either Sequence or str!")

        for element in elements:
            if not self.basis_root.joinpath(element).exists():
                raise ValueError(f'The elements {element} is not accessible for Stuttgart Effective Core Potential!')

        info = "Accessible Effective Core Potentials for:\n"
        info += "-------------------------------\n"
        info += "| Element |   ECP   |  Basis  |\n"
        for element in elements:
            for ecp in os.listdir(self.basis_root.joinpath(element)):
                for basis in os.listdir(self.basis_root.joinpath(element, ecp)):
                    if basis != 'ECP':
                        info += f"|   {element:6}|  {ecp:7}|  {basis:7}|\n"
        info += "-------------------------------"
        print(info)


if __name__ == "__main__":
    stuttgart = Stuttgart()
    stuttgart.search()

    print(stuttgart.get_basis('Am', 'MWB82', 'AVDZ'))
    print(stuttgart.get_ecp('Am', 'MWB82'))
    print(stuttgart.get('Am', 'MWB82', 'AVDZ'))
