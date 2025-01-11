from tqdm import tqdm
import os
import re
import pandas as pd

def extract_idt_from_file(file_path):
    """
    由于每次提取的情况不同，所以统一在这里修改，
    目标是从文件地址中提取出需要的分子的identifier
    """
    file_name = file_path.split('/')[-1]
    pattern = f'output_(.*?)_ND_AC_FF_UFF_mix_I2_N2_1_1_1_1.22ppm_348K.output'
    match = re.search(pattern, file_name)
    if match:
        content = match.group(1)
        return content
    else:
        raise TypeError(f"Sorry, Failed to extract the identifier of {file_path}.")


# 检查超算中未完成吸附模拟的分子
work_folder = '/mnt/sda1/qyq/proj/raspa/init_aC_adsorb_I2/work_in_ht/other_lower_perssure_348K_uptake_part2/supply_ht_1500frames_1.22ppm_348K'
simulation_condition = work_folder.split('_')[-2]+'_'+work_folder.split('_')[-1]
avg_uptake_list = []
failed_idt_list = []
for file in tqdm(os.listdir(work_folder)):
    if file.endswith('.output'):
        file_path = os.path.join(work_folder, file)
        idt = extract_idt_from_file(file_path)
        with open(file_path, 'r') as f:
            output_script = f.readlines()
        # 先判断文件是否运行完整
        if 'Average adsorption energy <U_gh>_1-<U_h>_0 obtained from Widom-insertion:\n' in output_script:
            final_cycle_index = output_script.index('Current cycle: 18000 out of 20000\n')
            target_line = output_script[final_cycle_index+14]
            pattern = r"[-+]?\d*\.\d+"
            matches = re.findall(pattern, target_line)
            avg_uptake_mg_per_g = float(matches[-1])
            avg_uptake_list.append([idt, avg_uptake_mg_per_g])
        else:
            failed_idt_list.append(idt)
save_avg_uptake_path = work_folder + f'_uptake_mg_per_g_{len(avg_uptake_list)}mols_part2.xlsx'
failed_idt_path = work_folder.replace('348K', f'348K_failed_{len(failed_idt_list)}_mols_identifier_part2.xlsx')
avg_uptake_df = pd.DataFrame(avg_uptake_list, columns=['identifier', f'avg_uptake_mg_per_g_at_{simulation_condition}'])
avg_uptake_df.to_excel(save_avg_uptake_path, index=False)
if failed_idt_list:
    print(f'失败的分子有{failed_idt_list}。\n')
    failed_idt_df = pd.DataFrame(failed_idt_list, columns=['identifier'])
    failed_idt_df.to_excel(failed_idt_path, index=False)
else:
    print('所有分子均运行完整！\n')

# # 高通量地读取混合吸附中component 0 的模拟最终步的绝对吸附量（单位为mg/g）
# output_folder = '/home/qyq/proj_temporary_substitute/raspa/init_aC_adsorb_I2/pyraspa/work_for_1403frames_400ppm_348K_ND_AC_FF_UFF_ht'
# avg_absolute_adsorb_co2_mg_g_list = []
# for file in tqdm(os.listdir(output_folder)):
#     if file.endswith('.output'):
#         output_path = os.path.join(output_folder, file)
#         frame_idt = extract_idt_from_file(output_path)
#         # output_path = '/home/qyq/proj/raspa/work_cof_CO2_uff/test_all_suitable_unit_cell/raspa_05000N2_mix.output'
#         with open(output_path, 'r') as f:
#             script = f.readlines()
#         pattern = r"[-+]?\d*\.\d+"   # 这个正则表达式用于匹配字符串中的浮点数
#         # 找到最后一步的关键词
#         final_cycle_index = script.index("Current cycle: 18000 out of 20000\n")
#         absolute_adsorb_co2_line = script[final_cycle_index+14]
#         matches = re.findall(pattern, absolute_adsorb_co2_line)
#         avg_absolute_adsorb_co2_mg_g = float(matches[-1])                          # ** -1对应的单位是mg/g
#         avg_absolute_adsorb_co2_mg_g_list.append([frame_idt, avg_absolute_adsorb_co2_mg_g])
# # 设置提取目标名
# target_col = 'Adsorbed_I2_400ppm_348K_mg_per_g'
# avg_absolute_adsorb_co2_mg_g_df = pd.DataFrame(avg_absolute_adsorb_co2_mg_g_list, columns=['identifier', target_col])
# avg_absolute_adsorb_co2_mg_g_values_list = avg_absolute_adsorb_co2_mg_g_df[target_col].tolist()
# # 保存吸附量列表
# # save_dir = output_folder.replace('/proj/', '/proj_temporary_substitute/')
# save_dir = '/home/qyq/proj_temporary_substitute/raspa/init_aC_adsorb_I2/pyraspa'
# # if not os.path.exists(save_dir):
# #     os.makedirs(save_dir)
# save_result_path = os.path.join(save_dir, f'part3_init_{len(avg_absolute_adsorb_co2_mg_g_values_list)}mols_new_ND_AC_FF_UFF_{target_col}.xlsx')
# # save_result_path = f'/home/qyq/proj/raspa/nd_aC_adsorb_I2/test_python_raspa/outputs/test11_single_frame_407ppm_new_ND_AC_FF_UFF/selected_single_mols/work_for_dop_mq_1.0_608_3661_9843_N_10_9_6_COOH_7_OH_3_CO_4_minimize/repeat_10mols_new_ND_AC_FF_UFF_{target_col}.xlsx'
# avg_absolute_adsorb_co2_mg_g_df.to_excel(save_result_path, index=False)
