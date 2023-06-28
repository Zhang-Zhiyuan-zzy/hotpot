"""
python v3.7.9
@Project: hotpot
@File   : test_bundle.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/19
@Time   : 21:25
"""
# from hotpot import MolBundle
# import json
# import hotpot.cheminfo as ci
import hotpot as hp
from pathlib import Path

# if __name__ == '__main__':
#     dir_log = '/home/zz1/proj/gauss/new/log'
#     md = MolBundle.read_from('g16log', dir_log, match_pattern='*/*.log', num_proc=2)
#     mmd = md.merge_conformers()
#     mamd = mmd.merge_atoms_same_mols()
#
#     count = 0
#     list_error = []
#     for m in md:
#         try:
#             data = m.dump('dpmd_sys')
#         except ValueError:
#             count += 1
#             list_error.append(m.identifier)
#             print(m.configure_number, len(m.all_energy), len(m.all_forces), len(m.all_atom_spin_densities))
#
#     # list_error = json.load(open('/home/zz1/error_file.json'))
#     # for p_e in list_error:
#     #     m = ci.Molecule.read_from(p_e, 'g16log')
#     #     if m:
#     #         print(m.configure_number, len(m.all_energy), len(m.all_forces), len(m.all_atom_spin_densities))
#
#     dm = mmd[0].to_dpmd_sys('/home/zz1/proj/dpmd/std')
#     dma = mamd[0].to_dpmd_sys('/home/zz1/proj/dpmd/att', 'att')

#acarbon1 = hp.Molecule.read_from('/home/qyq/proj/aC_database/cif_48954/cif_10_test/mq_0.6_3000_9662_4132.cif')
#acarbon2 = hp.Molecule.read_from('/home/qyq/proj/aC_database/cif_48954/cif_10_test/mq_1.0_4098_9724_3312.cif')
frames_dir = Path('/home/qyq/proj/aC_database/cif_48954/cif_10_test')
bundle = hp.MolBundle.read_from(
        'cif', frames_dir, generate=True, num_proc=10
    )

# print('bundle.mols')
# for mol in bundle.mols:
#     pass
# print(mol)
# print(mol.identifier)
# print(type(mol))
print('bundle')
for mol in bundle:
    print(mol)
    print(mol.identifier)
print(type(mol))