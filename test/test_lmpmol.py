import sys

sys.path.append('/home/qyq/hotpot')

from src.cheminfo import Molecule as Mol
from src._io import Dumper

if __name__ == '__main__':
    # mol = Mol.read_from('[Br:1][CH2:2][CH2:3][CH2:4][O:5][S:7]([CH3:6])(=[O:8])=[O:9]', 'smi')
    # mol.build_conformer()
    # mol.add_pseudo_atom('T', 15.0, mol.center_of_masses, charge=0.5)
    # mol.add_pseudo_atom('D', 25.0, mol.center_of_shape, charge=0.4)
    # mol.add_pseudo_atom('D', 25.0, (0.5, 0.7, 1.4), charge=-1.2)
    # script = mol.dump('lmpmol', atom_style='full', mol_name='C4H9Br1O3S1')
    # mol.writefile('lmpmol', path_file='/home/qyq/proj/lammps/mol_test/C4H9Br1O3S1.txt', atom_style='full')


    #write a carbon dioxide molecule
    mol = Mol.read_from('[O:1]=[C:2]=[O:3]', 'smi')
    mol.build_conformer()
    script = mol.dump('lmpmol', atom_style='atomic', mol_name='CO2')
    # mol.writefile('lmpmol', path_file='/home/qyq/proj/lammps/mol_test/CO2.txt', atom_style='atomic')
    #
    # #run lmp
    # mol.lmp_setup()
    # mol.lmp('units real')
    # mol.lmp('atom_style full')
    #
    # mol.lmp('boundary p p p')
    # mol.lmp('timestep    1')
    # mol.lmp('region      box block 0 60 0 10 0 30 units box')
    # mol.lmp('create_box  2 box bond/types 1 angle/types 1 extra/bond/per/atom 2 extra/angle/per/atom 1 extra/special/per/atom 2')
    # mol.lmp('molecule    CO2  /home/qyq/proj/lammps/mol_test/CO2.txt')
    # mol.lmp('create_atoms    0 random 400 9090 box mol CO2 9567 units box')
    # mol.lmp('mass        1 15.9994')
    # mol.lmp('mass        2 12')
    # # mol.lmp('pair_style lj/cut/coul/cut 12 10')
    # # mol.lmp('pair_coeff 1 1 0.157 3.05')
    # # mol.lmp('pair_coeff 2 1 0.092 2.92')
    # # mol.lmp('pair_coeff 2 2 0.0539 2.8')
    # mol.lmp('bond_style harmonic')
    # mol.lmp('bond_coeff 1 1000 1.16')
    # mol.lmp('angle_style harmonic')
    # mol.lmp('angle_coeff 1 200 180')
    # mol.lmp('minimize 1.0e-4 1.0e-6 1000 10000')
    # mol.lmp('velocity all create 300 90866')
    # mol.lmp('compute 1 all msd com yes')
    # mol.lmp('variable msdx equal c_1[1]')
    # mol.lmp('variable msdy equal c_1[2]')
    # mol.lmp('variable msdz equal c_1[3]')
    # mol.lmp('variable msd equal c_1[4]')
    # mol.lmp('variable istep equal step')
    # mol.lmp('fix msd all print 1 "${istep} ${msdx} ${msdy} ${msdz} ${msd}" screen no file /home/qyq/proj/lammps/mol_test/msd.txt')
    # mol.lmp('dump 1 all xyz 100 /home/qyq/proj/lammps/mol_test/dump.xyz')
    # mol.lmp('dump_modify 1 element O C')
    # mol.lmp('fix 1 all nvt temp 300 300 100')
    # mol.lmp('run 10000')
