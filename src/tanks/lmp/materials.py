"""
python v3.7.9
@Project: hotpot
@File   : materials.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/26
@Time   : 1:50
Notes:
    This package is to perform some specific tasks base on the LAMMPS
"""
import os
import os.path as osp
import json
from os.path import join as ptj
from typing import *
import math
import random
import numpy as np
import src
import src.cheminfo as ci


dir_force_field = osp.abspath(ptj(src.data_root, 'force_field'))
contents_force_field: dict = json.load(open(ptj(dir_force_field, 'contents.json')))

# Constants
avogadro = 6.02214076e23  # Avogadro numbers
angstrom = 1e-8  # cm


class AmorphousMaker:
    """ To make Amorphous Materials """
    def __init__(
            self,
            element_composition: Dict[str, float],
            force_field: Union[str, os.PathLike],
            density: float = 1.0,  # g/cm^3
            a: float = 25., b: float = 25., c: float = 25.,
            alpha: float = 90., beta: float = 90., gamma: float = 90.,
    ):
        """"""
        # Check the arguments
        # Determine the path of force field.
        if isinstance(force_field, os.PathLike):
            pff = str(force_field)
        elif osp.exists(force_field):
            pff = force_field
        else:
            pff = ptj(dir_force_field, force_field)

        # Make sure the existence of force field file
        if not osp.exists(pff):
            raise FileNotFoundError('the given force field file is not found!')

        pff = osp.abspath(pff)
        path_found = pff.find(dir_force_field)
        if path_found == 0:
            built_in_names = pff[len(dir_force_field):].split(os.sep)

            # Check if a specification for the force field about its applicable elements in the contents
            contents = contents_force_field
            for name in built_in_names:
                contents = contents.get(name)
                # if not have specification
                if not contents:
                    break

                # Get the specification about the applicable elements
                if contents:
                    app_ele = contents.get('elements')
                    # if given elements are not compatible for force field
                    if app_ele and any(e not in app_ele for e in element_composition):
                        raise AttributeError(
                            'the specified force field is inapplicable to all demanded elements'
                            f'the force field {"/".join(built_in_names)} just apply for {",".join(app_ele)}'
                        )

        # assign attrs
        sum_freq = sum(f for f in element_composition.values())
        self.elements = {e: f/sum_freq for e, f in element_composition.items()}  # Normalize the elements' frequency
        self.path_force_field = pff
        self.density = density
        self.cryst_params = (a, b, c, alpha, beta, gamma)

    @staticmethod
    def calc_cryst_density(cryst):
        """ Calculation the density for Crystal object """
        mol: ci.Molecule = cryst.molecule
        sum_mass = sum(ci.periodic_table[a.atom_type]['atomic_mass'] for a in mol.atoms)

        # Density, g/cm^3
        return (sum_mass * avogadro) / (cryst.volume * angstrom ** 3)

    @staticmethod
    def density2atom_numbers(ratio_elements: dict, density: float, cryst):
        """
        Calculate the round atom numbers in a crystal.
        Args:
            ratio_elements: Ratio of elements in the crystal
            density: the demand density in the crystal
            cryst: the crystal

        Returns:
            int, the number of atoms
        """
        # Convert the dict of elements and possibility to numpy array
        elements = np.array(list(ratio_elements.keys()))
        possibility = np.array(list(ratio_elements.values()))
        possibility = possibility / possibility.sum()  # Normalize

        average_mol_mass = sum(ci.periodic_table[e]['atomic_mass'] * p for e, p in zip(elements, possibility))

        # TODO: Check ..., the formula may be wrong
        # Terms:   [Mole in Crystal(Total Mass in Crystal (gram)/Average Mole Mass)]/Avogadro Number
        # Units:      g/cm^3     angstrom^3  angstrom/cm             g/mol             _
        num_atom = ((density * (cryst.volume * angstrom ** 3)) / average_mol_mass) * avogadro

        v1, v2, v3 = cryst.vector  # crystal vectors
        dv1, dv2, dv3 = np.sqrt(sum(v1 ** 2)), np.sqrt(sum(v2 ** 2)), np.sqrt(sum(v3 ** 2))  # length of crystal vectors
        min_dv = min(dv1, dv2, dv3)  # the min length in these vectors
        r_dv1, r_dv2, r_dv3 = dv1 / min_dv, dv2 / min_dv, dv3 / min_dv  # the ratios of length of vector to min vector

        # If all vectors are replaced with the smallest one,
        # how many times is the actual crystal volume after replacement
        fold_min_dv_volume = r_dv1 * r_dv2 * r_dv3

        # number of grid points in the direction of min vector
        min_point = math.pow(num_atom / fold_min_dv_volume, 1 / 3)

        # number of point for in the direction of v1, v2, v3
        num_pv1, num_pv2, num_pv3 = int(r_dv1 * min_point), int(r_dv2 * min_point), int(r_dv3 * min_point)
        # the coordinate of grad point in the bases of v1, v2, v3
        pv1, pv2, pv3 = np.meshgrid(
            np.linspace(0, 1, num_pv1, endpoint=False),
            np.linspace(0, 1, num_pv2, endpoint=False),
            np.linspace(0, 1, num_pv3, endpoint=False)
        )

        fraction_coordinates = np.stack((pv1.flatten(), pv2.flatten(), pv3.flatten()))

        # the actual coordinates in the cartesian coordinates
        cartesian_coordinates = np.matmul(cryst.vector, fraction_coordinates).T

        num_atom = len(cartesian_coordinates)
        atomic_symbols = np.random.choice(elements, num_atom, p=possibility)
        atomic_numbers = np.array([ci.periodic_table[symbol]["number"] for symbol in atomic_symbols])

        return atomic_numbers, cartesian_coordinates

    def load_atoms(self):
        mol = ci.Molecule()
        mol.make_crystal(*self.cryst_params)

        cryst = mol.crystal()
        atomic_number, coordinates = self.density2atom_numbers(
            self.elements, self.density, cryst
        )

        mol.quick_build_atoms(atomic_number)
        mol.set(all_coordinates=coordinates.reshape((-1, len(coordinates), 3)))

        mol.configure_select(0)

        return mol

    def melt_quench(
            self, *ff_args, mol=None, path_writefile: Optional[str] = None,
            origin_temp: float = 298.15, melt_temp: float = 4000., highest_temp: float = 10000,
            time_step: float = 0.0001, path_dump_to: Optional[str] = None, dump_every: int = 100,
    ):
        """
        Perform melt-quench process to manufacture a amorphous materials
        Args:
            *ff_args: the arguments the force file requried, refering the LAMMPS pair_coeff:
             "pair_coeff I J args" url: https://docs.lammps.org/pair_coeff.html
            mol: the molecule to be performed melt-quench. if not given, initialize according to elemental
             compositions
            path_writefile: the path to write the final material (screenshot) to file, if not specify, not save.
            origin_temp: the initial temperature before melt
            melt_temp: the round melting point to the materials
            highest_temp: the highest temperature to liquefy the materials
            time_step: time interval between path integrals when performing melt-quench
            path_dump_to: the path to save the trajectory of the melt-quench process, if not specify not save
            dump_every: the step interval between each dumps
        Returns:
            Molecule obj after melt-quench.
        """
        if not isinstance(mol, ci.Molecule):
            mol = self.load_atoms()

        mol.lmp_setup(units='metal')

        # initialization
        mol.lmp.commands_string(
            """
            units metal
            dimension 3
            atom_style full
            """
        )

        # Read molecule into LAMMPS
        mol.lmp.read_main_data()

        # Configure the force field
        mol.lmp.command("pair_style tersoff")
        mol.lmp.command(f"pair_coeff * * {self.path_force_field} {' '.join(ff_args)}""")

        # Specify the thermodynamical output to screen
        mol.lmp.command('thermo_style    custom step temp pe etotal press vol density')
        mol.lmp.command('thermo          1000')

        # the step interval of integral
        mol.lmp.command(f'timestep {time_step}')

        # Specify the dump configuration
        if path_dump_to:
            dump_fmt = path_dump_to.split('.')[-1]  # the dump fmt is the suffix of file name
            mol.lmp.command(f'dump mq all {dump_fmt} {dump_every} {path_dump_to}')
            mol.lmp.command(f'dump_modify mq element {" ".join(set(mol.atomic_symbols))}')

        # Initialize the temperature for system
        mol.lmp.command(f'velocity all create {origin_temp} {random.randint(100000, 999999)}')

        # Melt
        mol.lmp.command(f'fix 0 all nvt temp {origin_temp} {highest_temp} 0.7')
        mol.lmp.command(f'run 10000')

        mol.lmp.command(f'fix 0 all nvt temp {highest_temp} {highest_temp} 1000')
        while mol.lmp.eval('temp') < highest_temp * 0.95:
            mol.lmp.command(f'run 1000')

        mol.lmp.command(f'run 10000')

        # Relax
        mol.lmp.command('thermo          250')
        mol.lmp.command(f'fix 0 all nvt temp {melt_temp} {melt_temp} 1000.0')
        while mol.lmp.eval('temp') > melt_temp * 1.05:
            mol.lmp.command(
                f'velocity all scale {melt_temp}'
            )
            mol.lmp.command(f'run 2000')

        mol.lmp.command('thermo          1000')
        mol.lmp.command(f'run 20000')

        # Quench
        mol.lmp.command('thermo          250')
        mol.lmp.command(f'fix 0 all nvt temp {origin_temp} {origin_temp} 1000.0')
        while mol.lmp.eval('temp') > origin_temp*1.05:
            mol.lmp.command(
                f'velocity all scale {(mol.lmp.eval("temp") - origin_temp) / 2 + origin_temp}'
            )
            mol.lmp.command(f'run 2000')

        if not path_writefile:
            pwf = ptj(os.getcwd(), 'write_dump.xyz')
            write_fmt = 'xyz'
        else:
            pwf = path_writefile
            write_fmt = path_writefile.split('.')[-1]

        mol.lmp.command(f'write_dump all {write_fmt} {pwf} modify element {" ".join(set(mol.atomic_symbols))}')
        made_mol = ci.Molecule.read_from(pwf)
        if not path_writefile:
            os.remove(pwf)

        made_mol.create_crystal_by_matrix(mol.lmp.cryst_matrix)

        return made_mol

