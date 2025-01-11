import sys
sys.path.append('/home/qyq/raspa_hp_ht')    # for 3090, install raspa by source file
import hotpot as hp
from hotpot.tasks.raspa import RASPA

import os
from tqdm import tqdm
import concurrent.futures
import time
from multiprocessing import Pool
import time
import pandas as pd

# compute henry coeff
def mof_adsorb_hcn_ht(file_path):
    save_folder = '/home/qyq/proj/raspa/compute_henry_for_coremof_adsorb_hcn/part1_3000_charged_ht/work_for_l'  # /home/qyq/proj/raspa/coremof_adsorb_hcn/work_selected_coremof_3000_part_m2z/work_o
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    frame_idt = file_path.split('/')[-1].replace('.cif', '')
    frame = hp.Molecule.read_from(file_path)
    # 简陋地debug，给每个原子重新定义偏电荷，atom.partial_charge = qeq?
    cif_script = pd.read_csv(file_path, skiprows=28, header=None, sep='\s+')     # 提取原子电荷信息
    charges_list = cif_script.iloc[:, -1].tolist()
    # 给frame加电荷
    for i, atom in enumerate(frame.atoms):
        atom.partial_charge = charges_list[i]
    # 开始运行raspa模拟
    raspa = RASPA(forcefield='UFF', raspa_root='/home/qyq/sw/RASPA/simulations')
    script = raspa.run_hk(frame, 'HCN', temperature=298.15, unit_cells=(1,1,1), cycles=20000)
    save_output_path = os.path.join(save_folder, f'hk_{frame_idt}_pure_HCN_UFF_1_1_1_298.15K_101325Pa.output')
    with open(save_output_path, 'w') as f:
        f.write(script.output)
    return script.output

coremof_5_folder = '/home/qyq/proj/raspa/compute_charge_by_qeq/work_for_hcn/selected_coremof_7346/charged_struct_3000/part_l'
frame_path_list = []
for file in os.listdir(coremof_5_folder):
    if file.endswith('.cif'):
        idt = file.replace('.cif', '')
        file_path = os.path.join(coremof_5_folder, file)
        frame_path_list.append(file_path)
print(f'待运行的分子有{len(frame_path_list)}个，如下所示：\n')
print(frame_path_list)
with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
    results = list(executor.map(mof_adsorb_hcn_ht, frame_path_list))