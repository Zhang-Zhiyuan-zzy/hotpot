"""
python v3.7.9
@Project: hotpot
@File   : test_amorphous_maker.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/26
@Time   : 5:09
"""
from src.cheminfo import Molecule as Mol

mol = Mol.create_aCryst_by_mq(
    elements={'C': 1.0}, force_field='aMaterials/SiC.tersoff',
    ff_args=('C',), density=0.8, path_dump_to='/home/zz1/qyq/mq.xyz'
)

mol.writefile('cif', '/home/zz1/qyq/mq.cif')
