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
    ff_args=('C',), path_dump_to='/home/zz1/qyq/melt-quench.xyz',
    fmt='xyz', density=0.8, path_writefile='/home/zz1/qyq/write_mq.xyz'
)
mol.writefile('cif', 'path/to/save')
