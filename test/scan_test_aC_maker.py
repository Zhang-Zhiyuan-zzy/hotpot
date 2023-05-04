# #1控制变量法
# from src.cheminfo import Molecule as Mol
#
# for m_temp in range(3000, 4000, 100):
#     mol = Mol.create_aCryst_by_mq(
#         elements={'C': 1.0}, force_field='aMaterials/SiC.tersoff',
#         ff_args=('C',), path_dump_to=f'/home/qyq/proj/lammps/mq_test_high9000/test4000_5000/mq_melt-{m_temp}.xyz',
#         fmt='xyz', density=0.8, melt_temp=m_temp, highest_temp=9000
#     )
#     mol.crystal().space_group = 'P1'
#     mol.writefile('cif', f'/home/qyq/proj/lammps/mq_test_high9000/test4000_5000/mq_melt-{m_temp}.cif')
#     m_temp = m_temp + 100

# #2什么都不变
# from src.cheminfo import Molecule as Mol
#
#
# for i in range(1, 3, 1):
#     mol = Mol.create_aCryst_by_mq(
#             elements={'C': 1.0}, force_field='aMaterials/SiC.tersoff',
#             ff_args=('C',), path_dump_to=f'/home/qyq/proj/lammps/mq_test_high/high8000/mq-{i}.xyz',
#             fmt='xyz', density=0.9, melt_temp=2000, highest_temp=8000
#         )
#     mol.crystal().space_group = 'P1'
#     mol.writefile('cif', f'/home/qyq/proj/lammps/mq_test_high/high8000/mq-{i}.cif')

#3主变量为密度，变量melt和highest随机
from src.cheminfo import Molecule as Mol
import random
for d in range(6, 11, 2):  #0.8, 2.5, 0.1; 100
    dens = d / 10
    for i in range(3):
        rand_melt = random.randint(3000, 5000)
        rand_highest = random.randint(9000, 10000)
        mol = Mol.create_aCryst_by_mq(
            elements={'C': 1.0}, force_field='aMaterials/SiC.tersoff',
            ff_args=('C',), path_dump_to=f'/home/qyq/proj/lammps/mq_test_random/try5/mq_{dens}_{rand_melt}_{rand_highest}_{i}.xyz',
            fmt='xyz', density=dens, melt_temp=rand_melt, highest_temp=rand_highest
        )
        mol.crystal().space_group = 'P1'
        mol.writefile('cif', f'/home/qyq/proj/lammps/mq_test_random/try5/mq_{dens}_{rand_melt}_{rand_highest}.cif')