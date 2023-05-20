"""
python v3.7.9
@Project: hotpot
@File   : test_lmp.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/24
@Time   : 21:42
"""
import random

import numpy as np

import hotpot.cheminfo as ci


def melt():
    mol = ci.Molecule.read_from('examples/struct/aCarbon.xyz')

    mol.lmp_setup()

    mol.lmp.command('units metal')
    mol.lmp.command('dimension 3')
    mol.lmp.command('atom_style full')
    mol.lmp.command('\n')
    mol.lmp.command('# ----------------- Atom Definition Section -----------------')
    mol.lmp.command('\n')
    mol.lmp.read_main_data()

    mol.lmp.command('pair_style tersoff')
    mol.lmp.command('pair_coeff * * /home/zz1/qyq/SiC.tersoff C')

    mol.lmp.command('fix melt all nvt temp 300 10000 100')
    mol.lmp.command('fix melt all nvt temp 10000 10000 1000.0')
    mol.lmp.command('timestep 0.00007')

    mol.lmp.command('thermo_style    custom step temp pe etotal press vol density')
    mol.lmp.command('thermo 1000')

    mol.lmp.command('dump dmelt all xyz 1000 /home/zz1/qyq/dmelt.xyz')
    mol.lmp.command('dump_modify dmelt element C')
    mol.lmp.command('restart 10000 /home/zz1/qyq/dmelt.xyz')

    mol.lmp.command('run 1400000')


def quench():
    mol = ci.Molecule.read_from('/home/zz1/qyq/dmelt.xyz')
    mol.lmp_setup()
    mol.lmp.commands_string(
        """
        units metal
        dimension 3
        atom_style full
        """
    )
    mol.lmp.read_main_data()
    mol.lmp.command("pair_style tersoff")
    mol.lmp.command("pair_coeff * * /home/zz1/qyq/SiC.tersoff C""")

    mol.lmp.command('thermo_style    custom step temp pe etotal press vol density')
    mol.lmp.command('thermo          10000')
    mol.lmp.command('timestep 0.00007')

    mol.lmp.command('dump xyz all xyz 1000 /home/zz1/qyq/quench.xyz')
    mol.lmp.command('dump_modify xyz element C')

    mol.lmp.command("fix quench all  nvt temp 10000.0 3000.0 7.0")
    mol.lmp.command('run 100000')
    mol.lmp.command('fix quench all  nvt temp 3000.0 3000.0 7.0')
    mol.lmp.command('run 100000')
    mol.lmp.command('fix quench all  nvt temp 3000.0 300.0 7.0')
    mol.lmp.command('run 100000')
    mol.lmp.command('fix quench all  nvt temp 3000.0 300.0 7.0')
    mol.lmp.command('run 100000')
    mol.lmp.command('fix quench all  nvt temp 300.0 300.0 7.0')
    mol.lmp.command('run 100000')


def melt_quench(
        m: ci.Molecule,
        path_dump_to: str,
        origin_temp: float = 298.15,
        relax_temp: float = 3500,
        melt_temp: float = 10000,
        time_step: float = 0.00007
):
    m.lmp_setup()

    # initialization
    m.lmp.commands_string(
        """
        units metal
        dimension 3
        atom_style full
        """
    )

    # Read molecule into LAMMPS
    m.lmp.read_main_data()

    # Configure the force field
    m.lmp.command("pair_style tersoff")
    m.lmp.command("pair_coeff * * /home/zz1/qyq/SiC.tersoff C""")

    # Specify the thermodynamical output to screen
    m.lmp.command('thermo_style    custom step temp pe etotal press vol density')
    m.lmp.command('thermo          1000')

    # the step interval of integral
    m.lmp.command(f'timestep {time_step}')

    # Specify the dump configuration
    m.lmp.command(f'dump xyz all xyz 100 {path_dump_to}')
    m.lmp.command('dump_modify xyz element C')

    # Initialize the temperature for system
    m.lmp.command(f'velocity all create {origin_temp} {random.randint(100000, 999999)}')

    # temperature rise up
    m.lmp.command(f'fix 0 all nvt temp {origin_temp} {melt_temp} 0.7')
    m.lmp.command(f'run 10000')

    m.lmp.command(f'fix 0 all nvt temp {melt_temp} {melt_temp} 1.4')
    m.lmp.command(f'run 20000')

    # Relax
    m.lmp.command('thermo          250')
    m.lmp.command(f'fix 0 all nvt temp {relax_temp} {relax_temp} 1000.0')
    while m.lmp.eval('temp') > relax_temp*1.05:
        m.lmp.command(
            f'velocity all scale {relax_temp}'
        )
        m.lmp.command(f'run 2000')

    m.lmp.command('thermo          1000')
    m.lmp.command(f'run 20000')

    # Quench
    m.lmp.command('thermo          250')
    m.lmp.command(f'fix 0 all nvt temp {origin_temp} {origin_temp} 1000.0')
    while m.lmp.eval('temp') > 400:
        m.lmp.command(
            f'velocity all create {(m.lmp.eval("temp")-origin_temp) / 2 + origin_temp}'
        )
        m.lmp.command(f'run 2000')

    return m


def gcmc():
    """"""
    p = 52.5573
    iodine = ci.Molecule()
    iodine.add_pseudo_atom('iodine', 253.808946, (0., 0., 0.))

    mol = ci.Molecule.read_from('/home/zz1/qyq/mq.cif')
    mol.lmp_setup()

    mol.lmp('units real')
    mol.lmp('dimension 3')
    mol.lmp('atom_style full')

    mol.lmp('read_data /home/zz1/qyq/main.data group frame extra/atom/types 1')
    mol.lmp('mass 2 253.808946')
    # mol.lmp.read_main_data(group='frame')

    # write molecule read file
    # iodine.writefile('lmpmol', '/home/zz1/qyq/lmpmol')
    mol.lmp('molecule iodine /home/zz1/qyq/lmpmol toff 1')
    mol.lmp('group Igas empty')

    mol.lmp('pair_style lj/cut 5.5')
    mol.lmp(f'pair_coeff 1 1 {52.83*0.001987} 3.43')
    mol.lmp(f'pair_coeff 2 2 {550.0*0.001987} 4.982')

    mol.lmp('fix stand frame setforce 0.0 0.0 0.0')

    mol.lmp(
        'fix gcmc Igas gcmc 1 $((20+abs(count(Igas)-20))/2) $((20+abs(count(Igas)-20))/2)'
        ' 0 354568 298.15 100 10 mol iodine pressure 52.5573 tfac_insert 15'
    )

    mol.lmp(
        f'fix out all print 1 "$(step),$(count(Igas)),$(mass(Igas)),$(mass(Igas)/mass(frame))" '
        f'file /home/zz1/qyq/I2.csv '
        'title "step,I2_count,I2_mass,uptake(g/g)" screen no'
    )
    mol.lmp('variable uptake equal mass(Igas)/mass(frame)')
    mol.lmp('fix ave all ave/time 1 50 50 v_uptake file /home/zz1/qyq/ave')

    mol.lmp('thermo_style    custom step temp pe etotal press vol density')
    mol.lmp('thermo          1000')
    mol.lmp('compute_modify thermo_temp dynamic/dof yes')

    mol.lmp(f'dump mq all xyz 100 /home/zz1/qyq/I2.xyz')
    mol.lmp.command(f'dump_modify mq element C, I')

    mol.lmp('timestep 0.0001')
    mol.lmp('run 15000')

    return mol


if __name__ == '__main__':
    from hotpot.tanks.lmp.gcmc import LjGCMC
    import os
    import tqdm
    from time import time as t
    frame = ci.Molecule.read_from('/home/zz1/qyq/mq.cif')
    # frame.build_bonds()
    # # ff = frame + frame
    #
    # # script = frame.dump('lmpdat')
    # #
    # # g1 = ci.Molecule.read_from('O=O', 'smi')
    # # g1.build_conformer()
    # g2 = ci.Molecule.read_from('O=C=O', 'smi')
    # g2.build_conformer()
    #
    # g2 + frame
    # g3 = ci.Molecule.read_from('N#N', 'smi')
    # g3.build_conformer()
    # g4 = ci.Molecule.read_from('O', 'smi')
    # g4.build_conformer()
    #
    # work_dir = f'/home/zz1/qyq/g1g2g3'
    # gc = LjGCMC(frame, 'UFF/LJ.json', g2, g4, work_dir=work_dir, T=298.15, P=0.00001)
    # gc.run()
