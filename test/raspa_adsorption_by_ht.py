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

# # compute helium void fraction
# def mof_adsorb_hcn_ht(file_path):
#     save_folder = '/home/qyq/proj/raspa/coremof_adsorb_hcn/work_else_coremof_4023_part_a2z/work3'  # /home/qyq/proj/raspa/coremof_adsorb_hcn/work_selected_coremof_3000_part_m2z/work_o
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     frame_idt = file_path.split('/')[-1].replace('.cif', '')
#     frame = hp.Molecule.read_from(file_path)
#     # 简陋地debug，给每个原子重新定义偏电荷，atom.partial_charge = qeq?
#     cif_script = pd.read_csv(file_path, skiprows=28, header=None, sep='\s+')     # 提取原子电荷信息
#     charges_list = cif_script.iloc[:, -1].tolist()
#     # 给frame加电荷
#     for i, atom in enumerate(frame.atoms):
#         atom.partial_charge = charges_list[i]
#     # 开始运行raspa模拟
#     raspa = RASPA(forcefield='UFF', raspa_root='/home/qyq/sw/RASPA/simulations')
#     script = raspa.run2(frame)
#     save_output_path = os.path.join(save_folder, f'output_{frame_idt}_pure_HCN_UFF_1_1_1_298.15K_101325Pa.output')
#     with open(save_output_path, 'w') as f:
#         f.write(script.output)
#     return script.output
#
# coremof_5_folder = '/home/qyq/proj/raspa/compute_charge_by_qeq/work_for_hcn/selected_coremof_7346/charged_struct_3000/part_u/test3'
# frame_path_list = []
# for file in os.listdir(coremof_5_folder):
#     if file.endswith('.cif'):
#         idt = file.replace('.cif', '')
#         file_path = os.path.join(coremof_5_folder, file)
#         frame_path_list.append(file_path)
# print(f'待运行的分子有{len(frame_path_list)}个，如下所示：\n')
# print(frame_path_list)
# with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
#     results = list(executor.map(mof_adsorb_hcn_ht, frame_path_list))


# # # COREMOF adsorb pure HCN
# # 高通量
# def mof_adsorb_hcn_ht(file_path):
#     save_folder = '/home/qyq/proj/raspa/coremof_adsorb_hcn/work_else_coremof_4023_part_a2z/work_Z'  # /home/qyq/proj/raspa/coremof_adsorb_hcn/work_selected_coremof_3000_part_m2z/work_o
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     frame_idt = file_path.split('/')[-1].replace('.cif', '')
#     frame = hp.Molecule.read_from(file_path)
#     # 简陋地debug，给每个原子重新定义偏电荷，atom.partial_charge = qeq?
#     cif_script = pd.read_csv(file_path, skiprows=28, header=None, sep='\s+')     # 提取原子电荷信息
#     charges_list = cif_script.iloc[:, -1].tolist()
#     # 给frame加电荷
#     for i, atom in enumerate(frame.atoms):
#         atom.partial_charge = charges_list[i]
#     # 开始运行raspa模拟
#     raspa = RASPA(forcefield='UFF', raspa_root='/home/qyq/sw/RASPA/simulations')
#     script = raspa.run(frame, 'HCN', mol_fractions=(1.0,), pressure=101325, temperature=298.15, unit_cells=(1,1,1), cycles=20000)
#     save_output_path = os.path.join(save_folder, f'output_{frame_idt}_pure_HCN_UFF_1_1_1_298.15K_101325Pa.output')
#     with open(save_output_path, 'w') as f:
#         f.write(script.output)
#     return script.output
#
# coremof_5_folder = '/home/qyq/proj/raspa/compute_charge_by_qeq/work_for_hcn/selected_coremof_7346/struct_4023_else/work_Z/charged_struct'
# frame_path_list = []
# for file in os.listdir(coremof_5_folder):
#     if file.endswith('.cif'):
#         idt = file.replace('.cif', '')
#         file_path = os.path.join(coremof_5_folder, file)
#         frame_path_list.append(file_path)
# print(f'待运行的分子有{len(frame_path_list)}个，如下所示：\n')
# print(frame_path_list)
# with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
#     results = list(executor.map(mof_adsorb_hcn_ht, frame_path_list))



# # 单个分子呢
# save_folder = '/home/qyq/proj/raspa/coremof_adsorb_hcn/test_raspa_hp'
# # if not os.path.exists(save_folder):
# #     os.makedirs(save_folder)
# file_path = '/home/qyq/proj/raspa/coremof_adsorb_hcn/test_raspa_hp/AGARUW_clean.cif' # /home/qyq/proj/raspa/coremof_adsorb_hcn/test_raspa_hp_5mofs/SOZSAD_clean_qeq.cif
# frame_idt = file_path.split('/')[-1].replace('.cif', '')
# frame = hp.Molecule.read_from(file_path)
# # 开始运行raspa模拟
# raspa = RASPA(forcefield='UFF', raspa_root='/home/qyq/sw/RASPA/simulations')   # for 4090, /home/qyq/miniconda3/envs/hp
# script = raspa.run(frame, 'HCN', mol_fractions=(1.0,), pressure=101325, temperature=298.15, unit_cells=(1,1,1), cycles=20000)
# save_output_path = os.path.join(save_folder, f'output_{frame_idt}_pure_HCN_UFF_1_1_1_298.15K_101325Pa.output')
# with open(save_output_path, 'w') as f:
#     f.write(script.output)




# def mof_adsorb_hcn_ht(file_path):
#     save_folder = '/home/qyq/proj/raspa/coremof_adsorb_hcn/test_raspa_hp'
#     # if not os.path.exists(save_folder):
#     #     os.makedirs(save_folder)
#     frame_idt = file_path.split('/')[-1].replace('.cif', '')
#     frame = hp.Molecule.read_from(file_path)
#     # # 简陋地debug，给每个原子重新定义偏电荷，atom.partial_charge = qeq?
#     # cif_script = pd.read_csv(file_path, skiprows=28, header=None, sep='\s+')     # 提取原子电荷信息
#     # charges_list = cif_script.iloc[:, -1].tolist()
#     # # 给frame加电荷
#     # for i, atom in enumerate(frame.atoms):
#     #     atom.partial_charge = charges_list[i]
#     # 开始运行raspa模拟
#     raspa = RASPA(forcefield='UFF', raspa_root='/home/qyq/miniconda3/envs/hp')
#     script = raspa.run(frame, 'HCN', mol_fractions=(1.0,), pressure=101325, temperature=298.15, unit_cells=(1,1,1), cycles=20000)
#     save_output_path = os.path.join(save_folder, f'output_{frame_idt}_pure_HCN_UFF_1_1_1_298.15K_101325Pa.output')
#     with open(save_output_path, 'w') as f:
#         f.write(script.output)
#     return script.output
#
# coremof_5_folder = '/home/qyq/proj/raspa/coremof_adsorb_hcn/test_raspa_hp_5mofs'
# frame_path_list = []
# for file in os.listdir(coremof_5_folder):
#     if file.endswith('.cif'):
#         idt = file.replace('.cif', '')
#         file_path = os.path.join(coremof_5_folder, file)
#         frame_path_list.append(file_path)
# print(f'待运行的分子有{len(frame_path_list)}个，如下所示：\n')
# print(frame_path_list)
# with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
#     results = list(executor.map(mof_adsorb_hcn_ht, frame_path_list))


# 多进程运行吸附模拟
# def mof_adsorb_co2(file_path):
#     """
#     permit adsorption by RASPA, 通过读取晶胞参数来判断模拟的unit_cells参数
#     Args:
#         file_path: the path of framework
#     """
#     with open(file_path, 'r') as f:
#         script = f.readlines()
#     len_a = float(script[3].split()[1])
#     len_b = float(script[4].split()[1])
#     len_c = float(script[5].split()[1])
#     if len_a >= 25.6:
#         unit_a = 1
#     else:
#         unit_a = 1
#         while len_a*unit_a < 25.6:
#             unit_a += 1
#     if len_b >= 25.6:
#         unit_b = 1
#     else:
#         unit_b = 1
#         while len_b*unit_b < 25.6:
#             unit_b += 1
#     if len_c >= 25.6:
#         unit_c = 1
#     else:
#         unit_c = 1
#         while len_c*unit_c < 25.6:
#             unit_c += 1
#     print(unit_a, unit_b, unit_c, file_path)
#
#     frame = hp.Molecule.read_from(file_path)
#     save_folder = '/home/qyq/proj/raspa/work_cof_CO2_uff/test_all_suitable_unit_cell'
#     raspa = RASPA(forcefield='UFF')
#     script = raspa.run(frame, "CO2", "N2", mol_fractions=(0.0004, 0.9996), pressure=101325, temperature=298.15, unit_cells=(unit_a, unit_b, unit_c), cycles=20000)
#     save_path = os.path.join(save_folder, f'raspa_{frame.identifier}_mix.output')
#     with open(save_path, 'w') as f:
#         f.write(script.output)
#     return script.output


# cif_folder = '/home/qyq/learn/AIMOF_WORK/dac/curated_cofs_database_modify'
# file_path_list = []
# for file in os.listdir(cif_folder):
#     file_path = os.path.join(cif_folder, file)
#     file_path_list.append(file_path)
# with concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
#     futures = [executor.submit(mof_adsorb_co2, f_p) for f_p in file_path_list]
#     results = [future.result() for future in concurrent.futures.as_completed(futures)]
# # print("Results:", results)




# # Mofs adsorb CO2
# # 多组分气体，多个分子，同一压力
# mof_dir = '/home/qyq/proj/carbon/doping_carbon/add_N_COOH_CO_OH_variegated_2700/init_cif_selected_1403'    # /home/qyq/proj/raspa/inputs/struct/coremofs/structure_11660
# save_dir = '/home/qyq/proj/raspa/init_aC_adsorb_I2/pyraspa/test'                 # /home/qyq/proj/raspa/work_mof_CO2_uff
# for i, file in enumerate(os.listdir(mof_dir)):
#     mof_path = os.path.join(mof_dir, file)
#     mof_name = file.replace('.cif', '')
#     mof = hp.Molecule.read_from(mof_path)
#     raspa = RASPA(forcefield='ND_AC_FF_UFF', raspa_root='/home/qyq/miniconda3/envs/hp')
#     script = raspa.run(mof, "CO2", "N2", mol_fractions=(0.0004, 0.9996), pressure=101325, temperature=298.15, unit_cells=(1,1,1), cycles=2000)
#     save_path = os.path.join(save_dir, f'raspa_{mof_name}_mix_1.output')
#     with open(save_path, 'w') as f:
#             f.write(script.output)

# # 单组份气体，单个分子，多个压力，吸附等温线
# mof_path = '/home/qyq/learn/AIMOF_WORK/dac/curated_cofs_database_modify/07001N2.cif'
# mof = hp.Molecule.read_from(mof_path)
# idt = mof.identifier
# save_folder = '/home/qyq/proj/raspa/work_cof_CO2_uff/test_9_isotherm'
# press_list = [151987.5, 202650, 253312.5, 303975, 405300, 506625]    # 1.01325, 3.03975, 10.1325, 30.3975, 101.325, 303.975, 1013.25, 3039.75, 10132.5, 30397.5, 101325
# for press in press_list:
#     raspa = RASPA(forcefield='UFF')
#     script = raspa.run(mof, "CO2", pressure=press, temperature=298.15, unit_cells=(2,2,2), cycles=20000)
#     save_path = os.path.join(save_folder, f'raspa_{idt}_{press}_pure.output')
#     with open(save_path, 'w') as f:
#         f.write(script.output)


# mof_name = 'IRMOF-1'
# mof_path = os.path.join(mof_dir, f'{mof_name}.cif')
# mof = hp.Molecule.read_from(mof_path)
#
# raspa = RASPA(forcefield='UFF')
# script = raspa.run(mof, "CO2", temperature=298.15, pressure=101325, unit_cells=(1,1,1), cycles=50000)
# save_path = '/home/qyq/proj/raspa/work_mof_CO2_uff/test_adsorption_1.output'
# with open(save_path, 'w') as f:
#     f.write(script.output)





# Amorphous carbon adsorb iodine


# 多个分子在407ppm的碘分压条件下运行混合吸附模拟，高通量
def aC_adsorb_I2_ht(file_path):
    # 修改区1
    iodine_mol_fraction = 0.00000407  # 0.000406908, 0.00001221, 0.00000407，0.00000122，0.00000041, 0.00000012, 0.000000041， 0.000000012， 0.0000000041
    # total_num_frames = 1500
    save_folder = '/home/qyq/proj/raspa/test_raspa_hp2'   # /home/qyq/proj/raspa/init_aCs_adsorb_iodine/work_1403_part804_4.07ppm_348K/work_5
    # save_folder = f'/public3/home/m6s000927/proj/raspa_aCs_adsorb_iodine/supply_for_348K/supply_ht_{total_num_frames}frames_{iodine_mol_fraction*1e6}ppm_348K_2'     # /public3/home/m6s000927/proj/raspa_aCs_adsorb_iodine
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    frame_idt = file_path.split('/')[-1].replace('.cif', '')
    frame = hp.Molecule.read_from(file_path)
    # frame_idt = frame.identifier
    raspa = RASPA(forcefield='ND_AC_FF_UFF', raspa_root='/home/qyq/miniconda3/envs/hp')        # ND_AC_FF_UFF, raspa_root='/public3/home/m6s000927/.conda/envs/raspa2_env'
    script = raspa.run(frame, "I2", "N2", mol_fractions=(iodine_mol_fraction, 1-iodine_mol_fraction), pressure=101325, temperature=348.15, unit_cells=(1,1,1), cycles=20000)
    save_output_path = os.path.join(save_folder, f'output_{frame_idt}_ND_AC_FF_UFF_mix_I2_N2_1_1_1_{iodine_mol_fraction*1e6}ppm_348K.output')           # f'output_{frame_idt}_ND_AC_FF_UFF_mix_I2_N2_1_1_1_400ppm_348K.output'
    with open(save_output_path, 'w') as f:
        f.write(script.output)
    print(frame_idt)
    return script.output

# # 修订区2
# total_num_frames_2 = 1500
# done_part1_66_path = '/public3/home/m6s000927/proj/raspa_aCs_adsorb_iodine/ht_1500frames_12.21ppm_348K_uptake_mg_per_g_637mols_2.xlsx'

dop_cif_folder = '/home/qyq/proj/acarbon/test_acarbon_2' # /home/qyq/proj/aC_database/init_aCs_cif_1403_selected_part804/test_5
# dop_cif_folder = f'/public3/home/m6s000927/proj/acarbon/init_aCs_cif_{total_num_frames_2}'     # /home/qyq/proj/carbon/doping_carbon/add_N_COOH_CO_OH_variegated_2700/init_cif_selected_1403
# # 排除已经运行过的75个分子
# done_part1_df = pd.read_excel(done_part1_66_path)
# done_part1_idt_list = done_part1_df['identifier'].tolist()
# # done_part2_path = '/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/work_1403frames_12.21ppm_ht_new_ND_AC_FF_UFF_part2/part2_256mols_new_ND_AC_FF_UFF_Adsorbed_I2_12.21ppm_mg_per_g.xlsx'
# # done_part2_df = pd.read_excel(done_part2_path)
# # done_part2_idt_list = done_part2_df['identifier'].tolist()
# frame_path_list = []
# for file in os.listdir(dop_cif_folder):
#     if file.endswith('.cif'):
#         idt = file.replace('.cif', '')
#         if idt not in done_part1_idt_list:                             #  and idt not in done_part2_idt_list
#             file_path = os.path.join(dop_cif_folder, file)
#             frame_path_list.append(file_path)

frame_path_list = []
for file in os.listdir(dop_cif_folder):
    if file.endswith('.cif'):
        file_path = os.path.join(dop_cif_folder, file)
        frame_path_list.append(file_path)

# print('剩余需要运行的分子个数为：\n')
# print(len(frame_path_list))

# # 记录cif的路径，排除已经运行的77个分子
# frame_path_list = []
# done_idt_path = '/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/test9_2697frames_407ppm_ht_new_ND_AC_FF_UFF_part1/first_77mols_new_ND_AC_FF_UFF_Adsorbed_I2_isotherm_mg_per_g.xlsx'
# done_result_df = pd.read_excel(done_idt_path)
# done_idt_list = done_result_df['identifier'].tolist()
# for file in os.listdir(dop_cif_folder):
#     if file.endswith('.cif'):
#         idt = file.replace('.cif', '')
#         if idt not in done_idt_list:
#             file_path = os.path.join(dop_cif_folder, file)
#             frame_path_list.append(file_path)
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(aC_adsorb_I2_ht, frame_path_list))



# # 噪声分析：一个分子多次同样条件的I2吸附模拟，查看结果的RMSE
# def single_aC_adsorb_I2(i):
#     """
#     Args:
#         i: 代表第i次重复模拟
#
#     Returns:
#
#     """
#     save_folder = '/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/test11_single_frame_407ppm_new_ND_AC_FF_UFF/selected_single_mols/work_for_dop_mq_1.0_608_3661_9843_N_10_9_6_COOH_7_OH_3_CO_4_minimize'     # /home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/work_1403frames_12.21ppm_ht_new_ND_AC_FF_UFF
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     file_path = '/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/test11_single_frame_407ppm_new_ND_AC_FF_UFF/selected_single_mols/dop_mq_1.0_608_3661_9843_N_10_9_6_COOH_7_OH_3_CO_4_minimize.cif'
#     iodine_mol_fraction = 0.000406908      # 0.000406908, 1.221e-05
#     frame_idt = file_path.split('/')[-1].replace('.cif', '')
#     frame = hp.Molecule.read_from(file_path)
#     raspa = RASPA(forcefield='ND_AC_FF_UFF')
#     script = raspa.run(frame, "I2", "N2", mol_fractions=(iodine_mol_fraction, 1-iodine_mol_fraction), pressure=101325, temperature=298.15, unit_cells=(1,1,1), cycles=20000)
#     save_output_path = os.path.join(save_folder, f'output_{i+1}_ND_AC_FF_UFF_mix_I2_N2_1_1_1_407ppm_298K.output')
#     with open(save_output_path, 'w') as f:
#         f.write(script.output)
#     return script.output
#
# with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
#     results = list(executor.map(single_aC_adsorb_I2, [0,1,2,3,4,5,6,7,8,9]))










# # 一个框架，单原子碘模型，I2/N2混合吸附，吸附等温线
# # iodine_ppm_list = [0.407, 1.221, 4.069, 12.207, 40.691, 122.073, 203.454, 406.909]
# iodine_ppm_list = [0.407, 1.221, 4.069, 12.207, 40.691, 122.073, 203.454, 406.909]
# frame_path = '/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/dop_mq_0.5_2189_4683_9239_N_2_2_1_id_1_COOH_5_OH_1_CO_5_minimize.cif'
# frame = hp.Molecule.read_from(frame_path)
# # print(frame.crystal().data)
# raspa = RASPA(forcefield='ND_AC_FF')
# for iodine_ppm in iodine_ppm_list:
#     iodine_mol_fraction = iodine_ppm * (1e-6)
#     script = raspa.run(frame, "I2", "N2", mol_fractions=(iodine_mol_fraction, 1-iodine_mol_fraction), pressure=101325, temperature=298.15, unit_cells=(1,1,1), cycles=20000)
#     save_path = os.path.join('/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/test2_isotherm', f'output_{iodine_ppm}ppm_dop_mq_0.5_2189_4683_9239_N_2_2_1_id_1_COOH_5_OH_1_CO_5_minimize_1_1_1_298.data')
#     with open(save_path, 'w') as f:
#         f.write(script.output)


# # 一个框架，单原子碘模型，I2/N2混合吸附，并行运行
# def aC_adsorb_I2_ht(iodine_concentration, file_path, save_folder):
#     iodine_mol_fraction = iodine_concentration * (1e-6)
#     frame = hp.Molecule.read_from(file_path)
#     raspa = RASPA(forcefield='ND_AC_FF')
#     script = raspa.run(frame, "I2", "N2", mol_fractions=(iodine_mol_fraction, 1-iodine_mol_fraction), pressure=101325, temperature=298.15, unit_cells=(1,1,1), cycles=20000)
#     save_output_path = os.path.join(save_folder, f'output_{iodine_concentration}ppm_mix_I2_N2_dop_mq_0.5_2189_4683_9239_N_2_2_1_id_1_COOH_5_OH_1_CO_5_minimize_1_1_1_298.data')
#     with open(save_output_path, 'w') as f:
#         f.write(script.output)
#     return script.output
#
# if __name__ == '__main__':
#     iodine_concentration_list = [12.207, 40.691]
#     file_path = '/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/dop_mq_0.5_2189_4683_9239_N_2_2_1_id_1_COOH_5_OH_1_CO_5_minimize.cif'
#     save_folder = '/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/test5_isotherm_ht'
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#
#     try:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#             futures = [executor.submit(aC_adsorb_I2_ht, iodine_concentration, file_path, save_folder) for iodine_concentration in iodine_concentration_list]
#             results = [future.result() for future in concurrent.futures.as_completed(futures)]
#         print("Results:", results)
#
#     except Exception as e:
#         print(f"Error in main execution: {e}")


# # 提取吸附量
# # 1 单原子碘分子模型
# # directory = '/home/qyq/proj/raspa/outputs/single/den_0.68'
# # uptake_dict = {}
# # for fold in os.listdir(directory):
# #     folder = os.path.join(directory, fold)
# #     for file in os.listdir(folder):
# #         file_path = os.path.join(folder, file)
# #         file_name = file.replace('.output', '')
# #         uptake_dict[file_name] = uptake ??????
#
#
# # # 单个框架模拟时
# # frame_dir = '/home/qyq/proj/raspa/inputs/struct/den_1.16'
# # frame_name = "mq_nd12_1.16_3828_9728_1133"
# # path_frame = os.path.join(frame_dir, f"{frame_name}.cif")
# # frame = hp.Molecule.read_from(path_frame)
# # # # 自定义N的lj参数，对应力场为UFF_ND
# # # suffix = 'w'
# # # for atom in frame.atoms:
# # #     if atom.symbol == 'N':
# # #         atom.label = f'{atom.symbol}{suffix}'
# # raspa = RASPA(forcefield="UFF")
# # guest_name = "I2"
# # for i in range(1, 2, 1):
# #     script = raspa.run(frame, guest_name, temperature=298.15, pressure=101325, unit_cells=(1, 1, 1), cycles=20000)
# #     save_path = f'/home/qyq/proj/raspa/outputs/single/code_update/mini_pores_carbon/{frame_name}_adsorb_{guest_name}_abc1_{i}.output'
# #     with open(save_path, 'w') as file:
# #         file.write(script.output)
#
#
# # 扫描纯碳和掺氮模型，用raspa进行GCMC模拟，对比两者的吸附量变化
# init_mol_path = '/home/qyq/proj/raspa/inputs/struct/den_0.68/init_mols/try1/mq_0.68_1000_4336_9970.cif'
# nd7_mol_path = '/home/qyq/proj/raspa/inputs/struct/den_0.68/init_mols/try1/mq_nd7_0.68_1000_4336_9970.cif'
# output_dir = '/home/qyq/proj/raspa/outputs/single/code_update/lj_potential_change/UFF_ND3/crazy_try_101325'
# # 记录候选力场参数
# sum_lj_list = [[52.83, 3.43, 34.72, 30.26], [52.83, 3.43, 34.72, 60.26], [52.83, 3.43, 34.72, 90.26], [52.83, 3.43, 34.72, 120.26]]
# lj_dict = {}
# eliminated_lj_dict = {}
# # 设定初始力场参数
#
# # 修改C和N原子的力场参数，共尝试2次
# ffmr_p = '/home/qyq/sw/RASPA/simulations/share/raspa/forcefield/UFF/force_field_mixing_rules.def'
# i = 0
# for epsilon_C, sigma_C, epsilon_N, sigma_N in sum_lj_list:
#     i += 1
#     new_ffmr_p = '/home/qyq/sw/RASPA/simulations/share/raspa/forcefield/UFF_ND3/force_field_mixing_rules.def'
#     with open(ffmr_p, 'r') as init_file:
#         script = init_file.readlines()
#     # # 先设定一个N原子的修改后的L-J参数
#     # epsilon_N = 40.72  # 34.72
#     # sigma_N = 3.26    # 3.26
#     # epsilon_C = 45    # 52.83
#     # sigma_C = 3.43     # 3.43
#     for index, value in enumerate(script):
#         if 'N_\t' in value:
#             script[index] = f'N_	    lennard-jones 	{epsilon_N}	{sigma_N}\n'
#         elif 'C_\t' in value:
#             script[index] = f'C_	    lennard-jones 	{epsilon_C}	{sigma_C}\n'
#     with open(new_ffmr_p, 'w') as writer:
#         writer.writelines(script)
#
#     # 修改了力场参数后，再运行吸附模拟
#     guest_name = 'I2'
#     raspa = RASPA(forcefield="UFF_ND3")
#     # 1纯碳模型的吸附模拟
#     init_file = os.path.basename(init_mol_path)
#     init_frame_name = init_file.replace('.cif', '')
#     init_frame = hp.Molecule.read_from(init_mol_path)
#     init_script = raspa.run(init_frame, guest_name, temperature=298.15, pressure=41, unit_cells=(1, 1, 1), cycles=20000)
#     init_save_path = os.path.join(output_dir, f'{init_frame_name}_adsorb_{guest_name}_abc1_εC{epsilon_C}_σC{sigma_C}_εN{epsilon_N}_σN{sigma_N}.output')
#     with open(init_save_path, 'w') as writer:
#         writer.write(init_script.output)
#     # 提取吸附量
#     init_output = init_script.output.split('\n')
#     init_line_num = init_output.index('Current cycle: 18000 out of 20000')
#     init_absolute_str = init_output[init_line_num+14]
#     init_uptake_str = init_absolute_str.split()[-2].replace(')', '')
#     init_uptake = float(init_uptake_str)
#     # 2掺7%的氮的模型的吸附模拟
#     nd7_file = os.path.basename(nd7_mol_path)
#     nd7_frame_name = nd7_file.replace('.cif', '')
#     nd7_frame = hp.Molecule.read_from(nd7_mol_path)
#     nd7_script = raspa.run(nd7_frame, guest_name, temperature=298.15, pressure=41, unit_cells=(1, 1, 1), cycles=20000)
#     nd7_save_path = os.path.join(output_dir, f'{nd7_frame_name}_adsorb_{guest_name}_abc1_εC{epsilon_C}_σC{sigma_C}_εN{epsilon_N}_σN{sigma_N}.output')
#     with open(nd7_save_path, 'w') as writer:
#         writer.write(nd7_script.output)
#     # 提取吸附量
#     nd7_output = nd7_script.output.split('\n')
#     nd7_line_num = nd7_output.index('Current cycle: 18000 out of 20000')
#     nd7_absolute_str = nd7_output[nd7_line_num+14]
#     nd7_uptake_str = nd7_absolute_str.split()[-2].replace(')', '')
#     nd7_uptake = float(nd7_uptake_str)
#
#     # 查看掺氮后吸附量变化，调整C和N的力场参数，目的是保持大部分情况下吸附量变大
#     if nd7_uptake > init_uptake:
#         lj_dict[i] = [epsilon_C, sigma_C, epsilon_N, sigma_N]
#     else:
#         eliminated_lj_dict[i] = [epsilon_C, sigma_C, epsilon_N, sigma_N]
#     # 保存L-J参数信息
#     lj_dict_path = os.path.join(output_dir, 'candidate_lj.csv')
#     eliminated_lj_dict_path = os.path.join(output_dir, 'eliminated_lj.csv')
#     # 还差一个功能，记录每个uptake的值
#
#
# # # 多个框架扫描时
# # frame_dir = '/home/qyq/proj/raspa/inputs/struct/den_1.16'
# # output_dir = '/home/qyq/proj/raspa/outputs/single/code_update/mini_pores_carbon/six_gcmc'
# # # 双原子碘分子模型
# # guest_name = 'I2'
# # for fold in tqdm(os.listdir(frame_dir)):
# #     folder = os.path.join(frame_dir, fold)
# #     for file in os.listdir(folder):
# #         frame_name = file.replace('.cif', '')
# #         path_frame = os.path.join(folder, file)
# #         frame = hp.Molecule.read_from(path_frame)
# #         raspa = RASPA(forcefield="UFF")
# #         script = raspa.run(frame, guest_name, temperature=298.15, unit_cells=(1, 1, 1), cycles=20000)
# #         save_dir = os.path.join(output_dir, fold)
# #         if not os.path.exists(save_dir):
# #             os.makedirs(save_dir)
# #         save_path = os.path.join(save_dir, f'{frame_name}_adsorb_{guest_name}_abc1.output')
# #         with open(save_path, 'w') as file:
# #             file.write(script.output)
#
#
# # frame_name = "mq_0.68_1000_4336_9970"
# # path_frame = os.path.join(frame_dir, f"{frame_name}.cif")
# # frame = hp.Molecule.read_from(path_frame)
# # raspa = RASPA(in_test=True)
# # guest_name = "diatomic"
# # script = raspa.run(frame, guest_name, temperature=298.15, cycles=20000)
# # save_path = f'/home/qyq/proj/raspa/outputs/double/{frame_name}_adsorb_{guest_name}_1.output'
# # with open(save_path, 'w') as file:
# #     file.write(script.output)