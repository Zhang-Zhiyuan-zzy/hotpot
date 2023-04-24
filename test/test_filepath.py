import os
def get_filepath_info(file, dir_file):
    file_base_name = os.path.splitext(file)[0]
    file_path = os.path.join(dir_file, file)
    return {file_base_name, file_path}

dir_struct = '/home/qyq/proj/lammps/a_C/perturb/data'
filepath_dict = {}
for file in os.listdir(dir_struct):
    if file.endswith(".data"):
        filepath_info = get_filepath_info(file, dir_struct)
        filepath_dict.update(filepath_info)

        dict_rm_sol = {}
        set_sol_smiles = dict_rm_sol.setdefault(mof_mol.identifier, set())
        set_sol_smiles.add(comp.smiles)