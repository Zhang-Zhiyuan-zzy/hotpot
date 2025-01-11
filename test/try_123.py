import shutil

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os

# 修改mq为lq，即重命名
init_folder = '/home/qyq/proj/acarbon/new_database/rand_dens_19599_cifs'
save_folder = '/home/qyq/proj/acarbon/new_database/rand_dens_19599_cifs_lq'
for file in os.listdir(init_folder):
    if file.endswith('.cif'):
        idt = file.replace('.cif', '')
        new_idt = idt.replace('mq_', 'lq_')
        f_p = os.path.join(init_folder, file)
        new_f_p = os.path.join(save_folder, f'{new_idt}.cif')
        shutil.copyfile(f_p, new_f_p)


# # 元素计数数据
# element_counts = {'Mn': 19, 'Co': 36, 'U': 7, 'Cu': 63, 'La': 12, 'In': 10, 'Ce': 7, 'Pr': 4, 'Nd': 4, 'Sm': 3, 'Eu': 9,
#  'Gd': 5, 'Mo': 3, 'Be': 1, 'Zn': 46, 'Tb': 9, 'Cd': 27, 'Na': 6, 'Ag': 15, 'V': 8, 'Ni': 15, 'Tm': 2, 'Li': 1, 'Ho': 4, 'Er': 2,
#  'Bi': 2, 'K': 2, 'Cs': 2, 'Nb': 1, 'W': 1, 'Ba': 1, 'Fe': 6, 'Pt': 1, 'Ca': 3, 'Si': 1, 'Yb': 1, 'Lu': 2, 'Ir': 1, 'Hf': 1, 'Mg': 3,
#  'Sb': 1, 'Au': 4}
# # 周期表布局
# periodic_table = [
#     ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
#     ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
#     ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
#     ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
#     ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
#     ['Cs', 'Ba', 'La-Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
#     ['Fr', 'Ra', 'Ac-Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'],
# ]
#
# lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
# actinides = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
#
# # 设置绘图
# fig, ax = plt.subplots(figsize=(16, 10))
# cmap = cm.cool
# norm = Normalize(vmin=0, vmax=max(element_counts.values()))
#
# # 绘制主周期表区域
# for row_idx, row in enumerate(periodic_table):
#     for col_idx, symbol in enumerate(row):
#         if symbol:
#             count = element_counts.get(symbol, 0)
#             color = cmap(norm(count)) if count > 0 else 'white'
#             rect = plt.Rectangle([col_idx, -row_idx], 1, 1, facecolor=color, edgecolor='black', linewidth=1)
#             ax.add_patch(rect)
#             ax.text(col_idx + 0.5, -row_idx + 0.5, symbol, ha='center', va='center', fontsize=20)
#
#
# # 偏移量设置
# lanthanide_offset = 2
# actinide_offset = 3
# table_length = len(periodic_table)
#
# # 绘制镧系
# for idx, symbol in enumerate(lanthanides):
#     count = element_counts.get(symbol, 0)
#     color = cmap(norm(count)) if count > 0 else 'white'
#     rect = plt.Rectangle([idx, -(table_length + lanthanide_offset)], 1, 1, facecolor=color, edgecolor='black', linewidth=1)
#     ax.add_patch(rect)
#     ax.text(idx + 0.5, -(table_length + lanthanide_offset) + 0.5, symbol, ha='center', va='center', fontsize=20)
#
# # 绘制锕系
# for idx, symbol in enumerate(actinides):
#     count = element_counts.get(symbol, 0)
#     color = cmap(norm(count)) if count > 0 else 'white'
#     rect = plt.Rectangle([idx, -(table_length + actinide_offset)], 1, 1, facecolor=color, edgecolor='black', linewidth=1)
#     ax.add_patch(rect)
#     ax.text(idx + 0.5, -(table_length + actinide_offset) + 0.5, symbol, ha='center', va='center', fontsize=20)
#
# # 添加颜色条
# cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
# cbar.set_label('Element Count', fontsize=12)
#
# # 设置范围和标题
# ax.set_xlim(0, 18)
# ax.set_ylim(-(table_length + 4), 0)
# ax.set_aspect('equal')
# ax.axis('off')
# # plt.title('Periodic Table with Element Counts', fontsize=16)
#
# plt.tight_layout()
# plt.show()


















# import os
#
# # import hotpot as hp
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import shutil
#
# from mendeleev import element
# from matplotlib.colors import Normalize
# from matplotlib import cm
#
# # 画一个元素周期表
# # 示例的元素和出现次数
# element_counts = {
#     'H': 10, 'He': 5, 'Li': 15, 'Be': 3, 'B': 20, 'C': 25, 'N': 18, 'O': 22, 'F': 8, 'Ne': 7,
#     'Na': 12, 'Mg': 14, 'Al': 6, 'Si': 19, 'P': 9, 'S': 21, 'Cl': 4, 'Ar': 10, 'K': 11, 'Ca': 13,
#     'Sc': 5, 'Ti': 7, 'V': 3, 'Cr': 6, 'Mn': 8, 'Fe': 20, 'Co': 4, 'Ni': 9, 'Cu': 7, 'Zn': 5,
#     'Ga': 3, 'Ge': 6, 'As': 4, 'Se': 7, 'Br': 3, 'Kr': 6, 'Rb': 8, 'Sr': 5, 'Y': 4, 'Zr': 7,
#     'Nb': 3, 'Mo': 6, 'Tc': 2, 'Ru': 5, 'Rh': 3, 'Pd': 4, 'Ag': 6, 'Cd': 5, 'In': 3, 'Sn': 7,
#     'Sb': 2, 'Te': 4, 'I': 5, 'Xe': 7, 'Cs': 3, 'Ba': 6, 'La': 4, 'Ce': 5, 'Pr': 3, 'Nd': 6,
#     'Pm': 2, 'Sm': 5, 'Eu': 3, 'Gd': 7, 'Tb': 2, 'Dy': 4, 'Ho': 3, 'Er': 6, 'Tm': 2, 'Yb': 4,
#     'Lu': 3, 'Ac': 5, 'Th': 2, 'Pa': 3, 'U': 6, 'Np': 2, 'Pu': 5, 'Am': 3, 'Cm': 6, 'Bk': 2,
#     'Cf': 3, 'Es': 5, 'Fm': 2, 'Md': 3, 'No': 4, 'Lr': 3
# }
#
# # 初始化周期表布局（包括镧系和锕系）
# periodic_table = [
#     ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
#     ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
#     ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
#     ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
#     ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
#     ['Cs', 'Ba', 'La-Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
#     ['Fr', 'Ra', 'Ac-Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'],
# ]
#
# lanthanides = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
# actinides = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
#
# # 准备绘图
# fig, ax = plt.subplots(figsize=(16, 10))
# cmap = cm.BuGn
# norm = Normalize(vmin=0, vmax=max(element_counts.values()))
#
# # 绘制周期表主区域
# for row_idx, row in enumerate(periodic_table):
#     for col_idx, symbol in enumerate(row):
#         if symbol:
#             count = element_counts.get(symbol, 0)
#             color = cmap(norm(count)) if count > 0 else 'white'
#             rect = plt.Rectangle([col_idx, -row_idx], 1, 1, facecolor=color, edgecolor='black', linewidth=1)
#             ax.add_patch(rect)
#             ax.text(col_idx + 0.5, -row_idx + 0.5, symbol, ha='center', va='center', fontsize=20)
#
# # 增加间距：镧系和锕系分别向下偏移2行
# lanthanide_offset = 2
# actinide_offset = 3
#
# # 绘制镧系
# for idx, symbol in enumerate(lanthanides):
#     count = element_counts.get(symbol, 0)
#     color = cmap(norm(count)) if count > 0 else 'white'
#     rect = plt.Rectangle([idx, -(len(periodic_table) + lanthanide_offset)], 1, 1, facecolor=color, edgecolor='black')
#     ax.add_patch(rect)
#     ax.text(idx + 0.5, -(len(periodic_table) + lanthanide_offset) + 0.5, symbol, ha='center', va='center', fontsize=10)
#
# # 绘制锕系
# for idx, symbol in enumerate(actinides):
#     count = element_counts.get(symbol, 0)
#     color = cmap(norm(count)) if count > 0 else 'white'
#     rect = plt.Rectangle([idx, -(len(periodic_table) + actinide_offset)], 1, 1, facecolor=color, edgecolor='black')
#     ax.add_patch(rect)
#     ax.text(idx + 0.5, -(len(periodic_table) + actinide_offset) + 0.5, symbol, ha='center', va='center', fontsize=10)
#
# # 添加颜色条
# cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation='vertical', fraction=0.02, pad=0.1)
# cbar.set_label('Element Count', fontsize=12)
#
# # 设置图形范围和标题
# ax.set_xlim(0, 18)
# ax.set_ylim(-9, 0)
# ax.set_aspect('equal')
# ax.axis('off')
# plt.title('Periodic Table with Element Counts', fontsize=16)
#
# plt.tight_layout()
# plt.show()





# # 根据相对压力，计算348K对应的碘分压( relative_pressure = p_I2/p0, p0=1600 , c_I2=p_I2/p外， p外=101325)
# p_relative_list = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 0.5, 1.0]
# p_iodine_list = []
# c_iodine_list = []
# c_iodine_ppm_list = []
# for pr in p_relative_list:
#     p_i2 = 1600*pr
#     p_iodine_list.append(p_i2)
#     c_iodine = p_i2/101325
#     c_iodine_list.append(c_iodine)
#     c_iodine_ppm = c_iodine*1e6
#     c_iodine_ppm_list.append(c_iodine_ppm)





# 生成多个分子的等温吸附线， 不同温度 298K & 348K，no no no, 348K时碘的饱和蒸汽压从41.23Pa(289K时)变成1600Pa（348K）
# # 编写文件
# work_dir = '/home/qyq/proj_temporary_substitute/raspa/init_aC_adsorb_I2'
# # 先生成348K的等温吸附线
# iodine_ppm_list = [0.012, 0.12, 0.41, 1.22, 4.07, 12.21, 40.7, 122, 203, 407]
# input_path = '/home/qyq/proj_temporary_substitute/raspa/init_aC_adsorb_I2/simulation.input'
# run_path = '/home/qyq/proj_temporary_substitute/raspa/init_aC_adsorb_I2/run'
# with open(input_path, 'r') as f:
#     input_script = f.readlines()
# work_folder = '/home/qyq/proj_temporary_substitute/raspa/init_aC_adsorb_I2/mq_1.25_2247_3420_9263_isotherm_298K'
# work_mol_idt = 'mq_1.25_2247_3420_9263'
# for press in iodine_ppm_list:
#     press_folder = os.path.join(work_folder, f'test_{press}ppm_298K')
#     if not os.path.exists(press_folder):
#         os.makedirs(press_folder)
#     # 复制run
#     new_run_path = os.path.join(press_folder, 'run')
#     shutil.copy(run_path, new_run_path)
#     # 生成simulation.input
#     new_input_path = os.path.join(press_folder, 'simulation.input')
#     input_script[4] = 'ExternalTemperature                     298.15\n'
#     input_script[14] = f'FrameworkName                           {work_mol_idt}\n'
#     input_script[23] = f'            MolFraction                 {press*1e-06}\n'
#     input_script[33] = f'            MolFraction                 {1-press*1e-06}\n'
#     with open(new_input_path, 'w') as f:
#         f.writelines(input_script)


# # 提取目标cif
# data_folder = '/home/qyq/proj/carbon/doping_carbon/add_N_COOH_CO_OH_variegated_2700/minimize_2697/final_dop_cif'
# save_folder = '/home/qyq/proj/carbon/doping_carbon/add_N_COOH_CO_OH_variegated_2700/minimize_2697/final_dop_cif_part1_1403_raspa'
# seleted_idt_path = '/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/work_1403frames_12.21ppm_ht_new_ND_AC_FF_UFF/sum_part1_part2_1403mols_idt.xlsx'
# selected_idt_df = pd.read_excel(seleted_idt_path)
# selected_idt_list = selected_idt_df['identifier'].tolist()
#
# for file in os.listdir(data_folder):
#     if file.endswith('.cif'):
#         idt = file.replace('.cif', '')
#         if idt in selected_idt_list:
#             file_path = os.path.join(data_folder, file)
#             new_file_path = os.path.join(save_folder, file)
#             shutil.copy(file_path, new_file_path)


# # 查看cof的晶胞参数
# curated_cofs_folder = '/home/qyq/learn/AIMOF_WORK/dac/curated_cofs_database_modify'
# crystal_params_list = []
# for file in tqdm(os.listdir(curated_cofs_folder)):
#     if file.endswith('.cif'):
#         file_name = file.replace('.cif', '')
#         file_path = os.path.join(curated_cofs_folder, file)
#         with open(file_path, 'r') as f:
#             script = f.readlines()
#         len_a = float(script[3].split()[1])
#         len_b = float(script[4].split()[1])
#         len_c = float(script[5].split()[1])
#         crystal_params_list.append([file_name, len_a, len_b, len_c])
# crystal_params_df = pd.DataFrame(crystal_params_list, columns=['identifier', 'len_a', 'len_b', 'len_c'])
# print(min(crystal_params_df['len_a'].tolist()), max(crystal_params_df['len_a'].tolist()))
# # 提取晶胞参数存在小于等于25.6的晶胞
# small_crystal_params_list = []
# for val in crystal_params_df.values:
#     len_a = val[1]
#     len_b = val[2]
#     len_c = val[3]
#     if len_a <= 25.6 or len_b <= 25.6 or len_c <= 25.6:
#         small_crystal_params_list.append(val.tolist())

# # 绘制吸附等温线
# relative_press_list = [1e-05, 3e-05, 1e-04, 3e-04, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0,
#                        1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
# amount_list  = [0.0, 0.0000652394, 0.0000489295, 0.0001304788, 0.0005545348, 0.0015005060, 0.0053007005, 0.0144831449, 0.0522893721, 0.2011493532, 0.9254207656,
#                 1.4087957256, 1.3085717107, 1.9920358838, 2.1243902989, 2.7261584443, 2.9490325148]     # mol/kg
# plt.plot(relative_press_list, amount_list)
# # plt.xlim(0,2)
# # plt.ylim(0,2)
# plt.show()

# folder = '/home/qyq/learn/AIMOF_WORK/dac/curated_cofs_database'
# save_folder = '/home/qyq/learn/AIMOF_WORK/dac/curated_cofs_database_modify'
# for file in os.listdir(folder):
#     file_name = file.replace('.cif', '')
#     file_path = os.path.join(folder, file)
#     frame = hp.Molecule.read_from(file_path)
#     print(frame.identifier)
#     frame.identifier = file_name
#     save_path = os.path.join(save_folder, file)
#     frame.writefile('cif', save_path)
#
#     new_frame = hp.Molecule.read_from(save_path)
#     print(new_frame.identifier)