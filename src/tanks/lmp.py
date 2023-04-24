import os
import glob
#
# def write_lmp_script(
#         path_script, path_struct,
#         units='metal', dim=3,
#         atom_style='full',
#         pair_style='tersoff',
#         pair_coeff_arges=None,
#         min_style='cg',
#         min_args=None,
#         dump_args=None,
#         run=1000,
# ):
#     with open(path_script, 'w') as writer:
#         writer.write('# ----------------- Init Section -----------------\n')
#         writer.write(f'units {units}\n')
#         writer.write(f'dimension {dim}\n')
#         writer.write(f'atom_style {atom_style}\n')
#         writer.write('# ----------------- Atom Definition Section -----------------\n')
#         writer.write(f'read_data {path_struct}\n')
#         writer.write('# ----------------- Settings Section-----------------\n')
#         writer.write(f'pair_style {pair_style}\n')
#         if pair_coeff_arges:
#             if not isinstance(pair_coeff_arges, list) or len(pair_coeff_arges) != 4:
#                 raise ValueError('the given pair_coeff_args is not the expected values')
#             writer.write(f'pair_coeff {pair_coeff_arges[0]} {pair_coeff_arges[1]} {pair_coeff_arges[2]} {pair_coeff_arges[3]}\n')
#         else:
#             writer.write('pair_coeff * * /home/qyq/sw/lammps/lammps-23Jun2022/potentials/SiC.tersoff C\n')
#         writer.write('#-----------------minimize-----------------\n')
#         writer.write(f'min_style {min_style}\n')
#         if min_args:
#             if not isinstance(min_args, list) or len(min_args) != 4:
#                 raise ValueError('the given min_args is not the expected values')
#
#             writer.write(f'minimize {min_args[0]} {min_args[1]} {min_args[2]} {min_args[3]}\n')
#         else:
#             writer.write('minimize 1.0e-4 1.0e-6 100 1000\n')
#         if dump_args:
#             if not isinstance(dump_args. list, list) or len(dump_args) != 6:
#                 raise ValueError('the given dump_args is not the expected values\n')
#             writer.write(f'dump xyz1 all xyz {dump_args[0]} {dump_args[1]}_*.xyz\n')
#         else:
#             writer.write('dump xyz1 all xyz 1000 *.xyz\n')
#         writer.write(f'run {run}\n')

#2023.4.19
class Lammps(object):
    """ A wrapper to write the LAMMPS scripts and run LAMMPS """
    def __init__(
            self, script_dir, result_dir, struct_dir
    ):
        self.script_dir = script_dir
        self.result_dir = result_dir
        self.struct_dir = struct_dir

    def _make_scripts(self):
        for path_struct in glob.glob(self.struct_dir):
            struct_name = path_struct.split('/')[-1]
            struct_stem, suffix = struct_name.split('.')

            path_script = os.path.join(self.script_dir, f"{struct_name}.??")
            path_result = ...


    def _make_script(self):


    def _minimize(self):
        pass

    def run(self):
        pass
