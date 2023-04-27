#single of xyz2cif
# import src.cheminfo as ci
# m1 = ci.Molecule.readfile('/home/qyq/proj/lammps/a_C/252_a-C/9/9_xyz/H2C/7512000_2.xyz')
# m2 = m1.writefile('cif', '/home/qyq/proj/lammps/a_C/252_a-C/9/9_xyz/cif/7512000_2.cif')

#bundle of xyz2cif
import src.cheminfo as ci
import os
import tqdm
xyz_path = '/home/qyq/proj/lammps/a_C/252_a-C/9/9_xyz/H2C'
cif_path = '/home/qyq/proj/lammps/a_C/252_a-C/9/9_xyz/cif'
xyz_files = [f for f in os.listdir(xyz_path) if f.endswith('.xyz')]
for xyz_file in tqdm(xyz_files):
    path_read = os.path.join(xyz_path, xyz_file)
    cif_file = os.path.join(cif_path, xyz_file.replace('.xyz', '.cif'))
    m1 = ci.Molecule.readfile(path_read)
    m2 = m1.writefile('cif', cif_file)