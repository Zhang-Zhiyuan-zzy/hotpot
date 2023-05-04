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

import src.cheminfo as ci


def melt():
    mol = ci.Molecule.readfile('examples/struct/aCarbon.xyz')

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


def solve_Peng_Robinson():
    import pandas as pd
    from src.tanks.lmp.eos import PengRobinson, solve_ideal_gas_equation

    critical_params = dict(
        water={'tc': 647.096, 'pc': 22064000.0, 'w': 0.3442920843},
        iodine={'tc': 819, 'pc': 11700000, 'w': 0.123},
        ammonia={'tc': 405.4, 'pc': 11333000, 'w': 0.25601},
        co2={'tc': 304.21, 'pc': 7375000, 'w': 0.225},
    )

    with pd.ExcelWriter("/home/zz1/qyq/pt_curve.xlsx") as writer:
        for gas, params in critical_params.items():
            water_sheet = pd.read_excel('/home/zz1/qyq/saturation_vapor.xlsx', sheet_name=gas).values
            t = water_sheet[:, 0]
            p = water_sheet[:, 1]

            pr = PengRobinson(**params)
            name, vm = pr(t=t, p=p)
            name, f = solve_ideal_gas_equation(v=vm, t=t)

            sheet = np.array([t, p, f]).T

            df = pd.DataFrame(sheet, columns=['Temperature', 'Pressure', 'Fugacity'])
            df.to_excel(writer, sheet_name=gas)


def gcmc():
    """"""


if __name__ == '__main__':

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
    mol.lmp('group Ig empty')

    mol.lmp('pair_style lj/cut 12.5')
    mol.lmp('pair_coeff 1 1 52.83 3.43')
    mol.lmp('pair_coeff 2 2 550.0 4.982')

    mol.lmp('fix stand frame setforce 0.0 0.0 0.0')

    mol.lmp('fix gcmc Ig gcmc 10 100 100 0 354568 298.15 1 10 mol iodine pressure 5800')
    mol.lmp('fix ber all temp/berendsen 298.15 298.15 0.1')
    mol.lmp('fix nvt Ig nvt temp 298.15 298.15 10000')

    mol.lmp('thermo_style    custom step temp pe etotal press vol density')
    mol.lmp('thermo          1000')

    mol.lmp(f'dump mq all xyz 100 /home/zz1/qyq/gcmc.xyz')
    # mol.lmp.command(f'dump_modify mq element C I')

    mol.lmp('timestep 0.0001')
    mol.lmp('run 100000')
