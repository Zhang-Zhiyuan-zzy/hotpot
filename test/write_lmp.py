# #single
# from src.tanks.lmp import write_lmp_script
# path_script = '/home/qyq/proj/lammps/a_C/perturb/test/in.lmp'
# path_struct = '/home/qyq/proj/lammps/a_C/perturb/aC_6_2.data'
# write_lmp_script(
#     path_script=path_script,
#     path_struct=path_struct,
#     units='metal',
#     dim=3,
#     atom_style='full',
#     pair_style='tersoff',
#     pair_coeff='* * SiC.tersoff C',
#     min_style='cg',
#     min_args=['1.0e-4', '1.0e-6', '1000', '10000'],
#     run=1000
# )

#batching generate in.lmp

#file path dictionary
import os


def get_filepath_info(file, dir_file):
    fbn = os.path.splitext(file)[0]
    fp = os.path.join(dir_file, file)
    return {'file_base_name': fbn, 'file_path': fp}


from src.tanks.lmp import write_lmp_script
dir_script = '/home/qyq/proj/lammps/a_C/3perturb_test/in_lmp'
dir_struct = '/home/qyq/proj/lammps/a_C/3perturb_test/lmpdat'
list_filepath = []

for file in os.listdir(dir_struct):
    if file.endswith(".data"):
        filepath_info = get_filepath_info(file, dir_struct)
        list_filepath.append(filepath_info)


for data_path in list_filepath:
    write_lmp_script(
        path_script=os.path.join(dir_script, data_path['file_base_name'] + "_in.lmp"),
        path_struct=data_path['file_path'],
        units='metal',
        dim=3,
        atom_style='full',
        pair_style='tersoff',
        pair_coeff_arges=['*', '*', '/home/qyq/sw/lammps/lammps-23Jun2022/potentials/SiC.tersoff', 'C'],
        min_style='cg',
        min_args=['1e-4', '1e-6', '100', '1000'],
        run=1000
    )


