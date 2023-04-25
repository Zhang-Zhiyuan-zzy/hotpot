"""
python v3.7.9
@Project: hotpot
@File   : test_lmp.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/24
@Time   : 21:42
"""
import extxyz
import src.cheminfo as ci


if __name__ == '__main__':
    mol = ci.Molecule.readfile('examples/struct/aCarbon.xyz')

    mol.lmp_setup()

    mol.lmp.command('units metal')
    mol.lmp.command('dimension 3')
    mol.lmp.command('atom_style full')
    mol.lmp.command('\n')
    mol.lmp.command('# ----------------- Atom Definition Section -----------------')
    mol.lmp.command('\n')
    mol.lmp.read_main_data()

    mol.lmp.command('pair_style tersoff')
    mol.lmp.command('pair_coeff * * /home/zz1/qyq/SiC.tersoff C')

    mol.lmp.command('fix melt all nvt temp 300 10000 100')
    mol.lmp.command('fix melt all nvt temp 10000 10000 1000.0')
    mol.lmp.command('timestep 0.00007')

    mol.lmp.command('thermo_style    custom step temp pe etotal press vol density')
    mol.lmp.command('thermo 1000')

    mol.lmp.command('dump dmelt all xyz 1000 /home/zz1/qyq/dmelt.xyz')
    mol.lmp.command('dump_modify dmelt element C')
    mol.lmp.command('restart 10000 /home/zz1/qyq/dmelt.xyz')

    # mol.lmp.command('run 1400000')



