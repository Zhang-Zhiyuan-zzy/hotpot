"""
python v3.7.9
@Project: hotpot
@File   : test_lmp.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/24
@Time   : 21:42
"""
import math
import random
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


if __name__ == '__main__':
    mol = ci.Molecule.read_from('examples/struct/aCarbon.xyz')
    result_mol = melt_quench(mol.copy(), '/home/zz1/qyq/mq.xyz')
