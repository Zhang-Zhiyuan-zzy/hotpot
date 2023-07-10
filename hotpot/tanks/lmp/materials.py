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
from os.path import join as ptj
from typing import *
import math
import random
import numpy as np
from scipy import spatial
import openbabel.openbabel as ob
import hotpot
import hotpot.cheminfo as ci


dir_force_field = osp.abspath(ptj(hotpot.data_root, 'force_field'))

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

        # assign attrs
        sum_freq = sum(f for f in element_composition.values())
        self.elements = {e: f/sum_freq for e, f in element_composition.items()}  # Normalize the elements' frequency
        self.path_force_field = pff
        self.density = density * 1.123592147466166
        self.cryst_params = (a, b, c, alpha, beta, gamma)

    @staticmethod
    def calc_cryst_density(cryst):
        """ Calculation the density for Crystal object """
        return (cryst.molecule.weight * avogadro) / (cryst.volume * angstrom ** 3)  # Density, g/cm^3

    @staticmethod
    def _density2atom_numbers(ratio_elements: dict, density: float, cryst):
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

        average_mol_mass = sum(ob.GetMass(ob.GetAtomicNum(e)) * p for e, p in zip(elements, possibility))

        # Terms:       [Mole in Crystal(Total Mass in Crystal (gram)/Average Mole Mass)]/Avogadro Number
        # Units:           g/cm^3     angstrom^3  angstrom/cm             g/mol             _
        num_atom = round(((density * (cryst.volume * angstrom ** 3)) / average_mol_mass) * avogadro)

        # calculate the fraction coordinates in the crystal cell
        fraction_coordinates = np.array([np.random.uniform(size=3) for _ in range(num_atom)])

        # the actual coordinates in the cartesian coordinates
        cartesian_coordinates = np.matmul(cryst.vector, fraction_coordinates.T).T

        distance_matrix = spatial.distance_matrix(cartesian_coordinates, cartesian_coordinates)
        distance_matrix = np.tril(distance_matrix, k=-1)
        distance_matrix[distance_matrix > 0.5] = 0

        # loose the closing points
        try_number = 0
        while distance_matrix.any() and try_number < 100:
            points1_idx, points2_idx = rows, cols = distance_matrix.nonzero()

            points1_coords = cartesian_coordinates[points1_idx]
            points2_coords = cartesian_coordinates[points2_idx]

            vector12 = points2_coords - points1_coords

            e12 = vector12/np.tile(np.linalg.norm(vector12, axis=1).reshape(-1, 1), 3)
            scalar_displace = np.tile((0.51-np.linalg.norm(vector12, axis=1)).reshape(-1, 1)/2, 3)

            displacement12 = e12 * scalar_displace
            displacement21 = -displacement12

            cartesian_coordinates[points1_idx] += displacement21
            cartesian_coordinates[points2_idx] += displacement12

            distance_matrix = spatial.distance_matrix(cartesian_coordinates, cartesian_coordinates)
            distance_matrix = np.tril(distance_matrix, k=-1)
            distance_matrix[distance_matrix > 0.5] = 0

            try_number += 1

        atomic_symbols = np.random.choice(elements, num_atom, p=possibility)
        atomic_numbers = np.array([ob.GetAtomicNum(symbol) for symbol in atomic_symbols])

        return atomic_numbers, cartesian_coordinates

    def load_atoms(self):
        mol = ci.Molecule()
        mol.make_crystal(*self.cryst_params)

        cryst = mol.crystal()
        atomic_number, coordinates = self._density2atom_numbers(
            self.elements, self.density, cryst
        )

        mol.quick_build_atoms(atomic_number)
        mol.set(all_coordinates=coordinates.reshape((-1, len(coordinates), 3)))

        mol.conformer_select(0)

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
        mol.lmp("pair_style tersoff")
        mol.lmp(f"pair_coeff * * {self.path_force_field} {' '.join(ff_args)}")

        # Specify the thermodynamical output to screen
        mol.lmp('thermo_style    custom step temp pe etotal press vol density')
        mol.lmp('thermo          1000')

        # the step interval of integral
        mol.lmp(f'timestep {time_step}')

        # Specify the dump configuration
        if path_dump_to:
            dump_fmt = path_dump_to.split('.')[-1]  # the dump fmt is the suffix of file name
            mol.lmp(f'dump mq all {dump_fmt} {dump_every} {path_dump_to}')
            mol.lmp(f'dump_modify mq element {" ".join(set(mol.atomic_symbols))}')

        # Initialize the temperature for system
        mol.lmp(f'velocity all create {origin_temp} {random.randint(100000, 999999)}')

        # Melt
        mol.lmp(f'fix 0 all nvt temp {origin_temp} {highest_temp} 0.7')
        mol.lmp(f'run 10000')

        mol.lmp(f'fix 0 all nvt temp {highest_temp} {highest_temp} 1000')
        while mol.lmp.eval('temp') < highest_temp * 0.95:
            mol.lmp(f'run 1000')

        mol.lmp(f'run 10000')

        # Relax
        mol.lmp('thermo          250')
        mol.lmp(f'fix 0 all nvt temp {melt_temp} {melt_temp} 1000.0')
        while mol.lmp.eval('temp') > melt_temp * 1.05:
            mol.lmp(f'velocity all scale {melt_temp}')
            mol.lmp(f'run 2000')

        mol.lmp('thermo          1000')
        mol.lmp(f'run 20000')

        # Quench
        mol.lmp('thermo          250')
        mol.lmp(f'fix 0 all nvt temp {origin_temp} {origin_temp} 1000.0')
        while mol.lmp.eval('temp') > origin_temp*1.05:
            mol.lmp(f'velocity all scale {(mol.lmp.eval("temp") - origin_temp) / 2 + origin_temp}')
            mol.lmp(f'run 2000')

        if not path_writefile:
            pwf = ptj(os.getcwd(), 'write_dump.xyz')
            write_fmt = 'xyz'
        else:
            pwf = path_writefile
            write_fmt = path_writefile.split('.')[-1]

        mol.lmp(f'write_dump all {write_fmt} {pwf} modify element {" ".join(set(mol.atomic_symbols))}')
        made_mol = ci.Molecule.read_from(pwf)
        if not path_writefile:
            os.remove(pwf)

        made_mol.create_crystal_by_matrix(mol.lmp.cryst_matrix)

        return made_mol

