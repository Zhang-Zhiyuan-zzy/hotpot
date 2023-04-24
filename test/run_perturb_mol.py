from src.cheminfo import Molecule as M

Mol = M.readfile('/home/qyq/proj/lammps/a_C/5perturb_test/initial/aC_B2.xyz', 'xyz')
generator_mol = Mol.perturb_mol_lattice(random_style='uniform', mol_distance=3, max_generate_num=10)
for i, mol in enumerate(generator_mol):
    mol.writefile(fmt='xyz', path_file=f'/home/qyq/proj/lammps/a_C/5perturb_test/perturb_xyz/aC_B2_{i}.xyz')
