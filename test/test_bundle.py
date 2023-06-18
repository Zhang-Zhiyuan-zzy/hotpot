"""
python v3.7.9
@Project: hotpot
@File   : test_bundle.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/19
@Time   : 21:25
"""
from hotpot import MolBundle
import json
import hotpot.cheminfo as ci


if __name__ == '__main__':
    dir_log = '/home/zz1/proj/gauss/new/log'
    md = MolBundle.read_from('g16log', dir_log, match_pattern='*/*.log', num_proc=2)
    mmd = md.merge_conformers()
    mamd = mmd.merge_atoms_same_mols()

    count = 0
    list_error = []
    for m in md:
        try:
            data = m.dump('dpmd_sys')
        except ValueError:
            count += 1
            list_error.append(m.identifier)
            print(m.configure_number, len(m.all_energy), len(m.all_forces), len(m.all_atom_spin_densities))

    # list_error = json.load(open('/home/zz1/error_file.json'))
    # for p_e in list_error:
    #     m = ci.Molecule.read_from(p_e, 'g16log')
    #     if m:
    #         print(m.configure_number, len(m.all_energy), len(m.all_forces), len(m.all_atom_spin_densities))

    dm = mmd[0].to_dpmd_sys('/home/zz1/proj/dpmd/std')
    dma = mamd[0].to_dpmd_sys('/home/zz1/proj/dpmd/att', 'att')
