# import os
# import src.cheminfo as ci
# from openbabel import pybel as pb
# path = '/home/qyq/proj/lammps/a_C/perturb/aC_6.xyz'
# mol = ci.Molecule.readfile(path)
# moll = pb.readfile('xyz', mol)
# mol = pb.readfile(path)
# script = moll.dump('lmpdat')

#delect special lines of datafile
with open('/home/qyq/proj/lammps/a_C/3perturb_test/trans_xyz_lmpdat/aC_B20.data', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if 'Bonds' in line:
        bi = i
        print(bi)
        break

with open('/home/qyq/proj/lammps/a_C/3perturb_test/trans_xyz_lmpdat/aC_B20_2.data', 'w') as f:
    for i, line in enumerate(lines):
        if i < 2 or 5 < i < 7 or 10 < i < bi:
            f.write(line)

#delect special lines automatically
