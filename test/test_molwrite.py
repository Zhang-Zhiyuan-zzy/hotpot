import src.cheminfo as ci


def random_mol():
    mol.lmp_setup()
    mol.lmp('units real')
    mol.lmp('atom_style full')
    mol.lmp('boundary p p p')
    mol.lmp('region      box block 0 60 0 10 0 30 units box')
    mol.lmp('create_box  2 box bond/types 1 angle/types 1 extra/bond/per/atom 2 extra/angle/per/atom 1 extra/special/per/atom 2')
    mol.lmp('molecule    CO2  /home/qyq/proj/lammps/mol_test/Mol4.txt')
    mol.lmp('create_atoms    0 random 400 9090 box mol CO2 9567 units box')
    mol.lmp('mass        1 15.9994')
    mol.lmp('mass        2 12')
    mol.lmp('dump mydmp all atom 1000 dump.lammpstrj')
    mol.lmp('thermo 10')
    mol.lmp('minimize 1.0e-4 1.0e-6 1000 10000')
    mol.lmp('write_data /home/qyq/proj/lammps/mol_test/Mol2_all.data')
