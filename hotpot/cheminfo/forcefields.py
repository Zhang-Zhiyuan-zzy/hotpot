"""
python v3.9.0
@Project: hotpot
@File   : forcefields
@Auther : Zhiyuan Zhang
@Data   : 2024/12/14
@Time   : 21:26


This module, `forcefields.py`, is a part of the `hotpot` project, designed for advanced molecular structure
simulations and optimizations. It integrates with Open Babel to provide tools for generating, optimizing,
and manipulating molecular geometries using various force fields and algorithms.

Key Features:
1. **Complex Building and Optimization**:
   - Functions like `complexes_build` and `_run_complexes_build` aid in creating 3D molecular complexes
    by iteratively optimizing molecular geometries. These utilize multiprocessing for parallel computations
    and ensure optimized, valid geometries.

2. **Force Field Management**:
   - Classes `OBFF` and `OBFF_` serve as wrappers for Open Babel's force fields (e.g., UFF, MMFF94, GAFF).
    They allow setup and optimization of molecular geometries with fine control over constraints,
    perturbations, equilibrium detection, and other parameters.

3. **Structure Building**:
   - The `OBBuilder` class and `ob_build` function simplify the construction and manipulation of molecular
   structures using Open Babel's OBBuilder tools.

4. **Utilities for Force Field Operations**:
   - The `ob_optimize` function integrates molecular force field optimization using specified force fields,
   providing energy calculations alongside updated coordinates.

5. **Constraint Management**:
   - Support for constraints on atoms, bonds, angles, and torsions during geometry optimizations ensures
    robust modeling capabilities for complex molecular systems.

This module is particularly useful for scientists and researchers in computational chemistry and molecular
modeling domains. It allows for fine-grained customizations and automation of molecular structure optimizations,
leveraging Open Babel's powerful capabilities.
"""
import time
from copy import copy
import logging
from typing import Literal, Optional, Union
import multiprocessing as mp

import numpy as np
from openbabel import openbabel as ob, pybel as pb

from .obconvert import extract_obmol_coordinates, set_obmol_coordinates


def _run_complexes_build(
        mol, queue: mp.Queue,
        build_times=5,
        init_opt_steps=500,
        second_opt_steps=1000,
        min_energy_opt_steps=3000,
):
    """
    Runs the process to build molecular complexes and optimizes their geometries
    using specified force fields (MMFF94s or UFF). The algorithm performs iterative
    geometry optimizations with multiple configurations and selects the one with
    lowest energy. If the geometrical configuration contains issues (e.g., bond
    ring intersection), the component is rebuilt and re-optimized until maximum
    rebuild attempts are reached or a valid geometry is found.

    Parameters:
    mol : object
        The molecular structure object to be processed. Its components are iteratively
        optimized and modified during the function execution.

    queue : mp.Queue
        A multiprocessing queue used to store the final optimized molecular geometry
        and conformers after completing the optimization process.

    build_times : int, default 5
        The number of times a geometry is built and tested for each component to
        identify the lowest energy configuration.

    init_opt_steps : int, default 500
        The number of steps for the initial geometry optimization phase.

    second_opt_steps : int, default 1000
        The number of steps to perform during the secondary optimization phase,
        which occurs after initial optimizations.

    min_energy_opt_steps : int, default 3000
        The number of steps to execute for the final optimization phase, where the
        lowest-energy configuration is refined.

    Raises:
    TimeoutError
        If the maximum number of attempts to rebuild geometrically invalid
        components is exceeded. This indicates an inability to generate valid
        molecular structure within the iteration limits.

    Returns:
    None
        This function does not return any value but places processed coordinates and
        conformers into a multiprocessing queue for further usage.
    """
    clone = copy(mol)
    clone.hide_metal_ligand_bonds()

    max_time = 10
    for component in clone.components:
        if not component.has_metal:
            lst_coords = []
            lst_energy = []
            build_ff = 'MMFF94s'
            rebuild_time = 0
            current_length = 0
            while len(lst_coords) < build_times:
                ob_build(component)

                try:
                    ob_optimize(component, build_ff, init_opt_steps)
                except RuntimeError:
                    ob_optimize(component, 'UFF', init_opt_steps)
                    build_ff = 'UFF'

                if component.has_bond_ring_intersection:
                    rebuild_time += 0
                    # print(len(list(lst_energy)))
                    if len(lst_energy) > current_length:
                        print(min(lst_energy), np.mean(lst_energy), max(lst_energy))
                        current_length = len(lst_energy)
                        rebuild_time = 0

                    if rebuild_time > max_time:
                        raise TimeoutError

                    continue


                energy = ob_optimize(component, build_ff, second_opt_steps)
                lst_energy.append(energy)
                lst_coords.append(component.coordinates)

            component.coordinates = lst_coords[np.argmin(lst_energy)]
            ob_optimize(component, build_ff, min_energy_opt_steps)

            clone.update_atoms_attrs_from_id_dict({a.id: {'coordinates': a.coordinates} for a in component.atoms})

    queue.put((clone.coordinates, clone.conformers))


def complexes_build(
        mol,
        build_times=5,
        init_opt_steps=500,
        second_opt_steps=1000,
        min_energy_opt_steps=3000,
        timeout: int = 1000,
        rm_polar_hs: bool = True,
        **kwargs
):
    """
    Builds 3D complexes of a molecular structure by generating and optimizing
    conformers in multiple steps. The function utilizes multiprocessing to
    perform the task in a separate process and imposes a timeout for the operation.

    Attributes:
        rm_polar_hs (bool): A flag to remove polar hydrogens before starting
        the conformer-building process. Defaults to True if not specified.

    Args:
        mol: The molecular structure object that the function operates on.
        It should support operations like adding hydrogens, refreshing atom
        IDs, and storing calculated coordinates and conformers.
        build_times (int): Number of times to attempt building conformers.
        Defaults to 5 iterations.
        init_opt_steps (int): The number of optimization steps to perform
        during the initial stage of conformer generation. Defaults to 500 steps.
        second_opt_steps (int): The number of optimization steps in the
        second stage. Defaults to 1000 steps.
        min_energy_opt_steps (int): The number of final optimization
        steps to stabilize conformers at minimum energy. Defaults to 3000 steps.
        timeout (int): Maximum time (in seconds) to wait for the conformer
        generation process to complete. Defaults to 1000 seconds.
        kwargs: Additional keyword arguments for customization of the
        complex-building process.

    Raises:
        TimeoutError: If the conformer generation process fails to complete
        within the specified timeout period.

    Returns:
        None. The function modifies the provided molecular structure object
        in place by adding optimized 3D coordinates and conformers.
    """
    mol.add_hydrogens(rm_polar_hs=rm_polar_hs)
    mol.refresh_atom_id()

    queue = mp.Queue()
    process = mp.Process(
        target=_run_complexes_build,
        args=(mol, queue, build_times, init_opt_steps, second_opt_steps, min_energy_opt_steps)
    )

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        raise TimeoutError('Timed out waiting for build complex 3D conformer!')

    mol.coordinates, mol._conformers = queue.get()
    process.terminate()


class OBFF_:
    """ A Wrapper of OpenBabel's ForceField """
    def __init__(
            self,
            ff: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']],
            algorithm: Literal["steepest", "conjugate"] = "conjugate",
            steps: Optional[int] = 100,
            step_size: int = 100,
            equilibrium: bool = False,
            equi_check_steps: int = 5,
            equi_max_displace: float = 1e-4,
            equi_max_energy: float = 1e-4,
            perturb_steps: Optional[int] = None,
            perturb_sigma: float = 0.5,
            save_screenshot: bool = False,
            increasing_Vdw: bool = False,
            Vdw_cutoff_start: float = 0.0,
            Vdw_cutoff_end: float = 12.5,
            print_energy: Optional[int] = None,
            **kwargs
    ):
        self.ff = ob.OBForceField.FindType(ff)
        self.algorithm = algorithm
        self.steps = steps
        self.step_size = step_size
        self.equilibrium = equilibrium
        self.equi_check_steps = equi_check_steps
        self.equi_max_displace = equi_max_displace
        self.equi_max_energy = equi_max_energy
        self.save_screenshot = save_screenshot
        self.perturb_steps = perturb_steps
        self.perturb_sigma = perturb_sigma
        self.increasing_Vdw = increasing_Vdw
        self.Vdw_cutoff_start = Vdw_cutoff_start
        self.Vdw_cutoff_end = Vdw_cutoff_end
        self.print_energy = print_energy

        self.constraints = None

        if increasing_Vdw:
            self.ff.SetVDWCutoff(self.Vdw_cutoff_start)

    def _perturb(self, coords):
        perturb = np.random.normal(0, self.perturb_sigma, coords.shape)
        perturb[perturb > 2*self.perturb_sigma] = self.perturb_sigma
        return perturb

    def _get_optimizer(self, mol):
        obmol = mol.to_obmol()
        if not self.ff.Setup(obmol, self.constraints):
            raise RuntimeError('Fail to initialize the forcefield!!')

        if self.algorithm == "steepest":
            optimizer = self.ff.SteepestDescent
        elif self.algorithm == "conjugate":
            optimizer = self.ff.ConjugateGradients
        else:
            raise NotImplementedError(f"Unknown optimization algorithm {self.algorithm}")

        return obmol, optimizer

    def setup(self, mol):
        obmol = mol.to_obmol()
        if not self.ff.Setup(obmol, self.constraints):
            raise RuntimeError('Fail to initialize the forcefield!!')
        return obmol

    def ob_setup(self, obmol):
        if not self.ff.Setup(obmol, self.constraints):
            raise RuntimeError('Fail to initialize the forcefield!!')
        return obmol

    def optimize(self, mol):
        self._add_constraints(mol)

        obmol, optimizer = self._get_optimizer(mol)

        lst_coords = []
        lst_energy = []
        for s in range(self.steps):
            optimizer(self.step_size)
            energy = self.ff.Energy()
            self.ff.GetCoordinates(obmol)
            coords = extract_obmol_coordinates(obmol)

            lst_coords.append(coords)
            lst_energy.append(energy)

            # Break the optimization, if the system has equilibrium.
            if self.equilibrium and len(lst_energy) > self.equi_check_steps:
                max_displace = max(np.abs(np.array(lst_coords[-self.equi_check_steps-1: -1]) - np.array(lst_coords[-self.equi_check_steps:])))
                max_diff_energy = max(np.abs(np.array(lst_energy[-self.equi_check_steps-1: -1]) - np.array(lst_energy[-self.equi_check_steps:])))

                if max_displace < self.equi_max_displace and max_diff_energy < self.equi_max_energy:
                    break

                if s == self.steps - 1:
                    logging.info(RuntimeWarning("Max iterations reached"))

            # Perturb the system
            if self.perturb_steps and s % self.perturb_steps == 0 and s != 0:
                coords += self._perturb(coords)
                set_obmol_coordinates(obmol, coords)
                self.ob_setup(obmol)

            # Adjust Vdw cutoff
            if self.increasing_Vdw:
                self.ff.SetVDWCutoff(s+1/self.steps*(self.Vdw_cutoff_end-self.Vdw_cutoff_start) + self.Vdw_cutoff_start)

            # Print information
            if self.print_energy and s % self.print_energy == 0:
                logging.debug(f"Energy in step {s}: {lst_energy[-1]}")

        mol.coordinates = lst_coords[-1]
        mol.energy = lst_energy[-1]

        if self.save_screenshot:
            mol.conformer_add(lst_coords, lst_energy)

    def _add_constraints(self, mol):
        """"""
        self.constraints = ob.OBFFConstraints()
        for atom in mol.atoms:
            if atom.constraint:
                self.constraints.AddAtomConstraint(atom.idx)
            else:
                if atom.x_constraint:
                    self.constraints.AddAtomXConstraint(atom.idx)
                if atom.y_constraint:
                    self.constraints.AddAtomYConstraint(atom.idx)
                if atom.z_constraint:
                    self.constraints.AddAtomZConstraint(atom.idx)

        for bond in mol.bonds:
            if bond.constraint:
                self.constraints.AddDistanceConstraint(bond.a1idx, bond.a2idx, bond.length)

        for angle in mol.angles:
            if angle.constraint:
                self.constraints.AddAngleConstraint(angle.a1idx, angle.a2idx, angle.a3idx, angle.degrees)

        for torsion in mol.torsions:
            if torsion.constraint:
                self.constraints.AddTorsionConstraint(
                    torsion.a1idx,
                    torsion.a2idx,
                    torsion.a3idx,
                    torsion.a4idx,
                    torsion.degree
                )


class OBFF:
    """
    A Wrapper of OpenBabel's ForceField
    Class to handle molecular geometry optimization using specific force fields and algorithms.

    This class provides functionalities for setting up and optimizing molecular geometries
    using various classical force fields and optimization algorithms. It allows adjustments
    of molecule constraints during optimization, performs equilibrium search, and can apply
    specific perturbations to atom coordinates. The class supports customizable parameters
    such as steps, equilibrium thresholds, maximum iterations, and van der Waals cutoff scaling.

    Attributes:
        ff (Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']]): The force field type used for optimization.
        algorithm (Literal["steepest", "conjugate"]): The optimization algorithm to be used.
        steps (Optional[int]): Number of steps for the optimization process.
        equilibrium (bool): Indicates whether to find an equilibrium geometry.
        equi_threshold (float): Threshold for determining equilibrium.
        max_iter (int): Maximum number of iterations per optimization cycle.
        save_screenshot (bool): Flag to save intermediate conformers.
        perturb_steps (Optional[int]): Number of perturbation cycles applied during optimization.
        perturb_sigma (float): Gaussian spread intensity for atom perturbation.
        increasing_Vdw (bool): Flag to incrementally scale van der Waals cutoff.
        Vdw_cutoff_start (float): Initial van der Waals cutoff value.
        Vdw_cutoff_end (float): Final van der Waals cutoff value.
    """
    def __init__(
            self,
            ff: Optional[Literal['UFF', 'MMFF94', 'MMFF94s', 'GAFF', 'Ghemical']],
            algorithm: Literal["steepest", "conjugate"] = "conjugate",
            steps: Optional[int] = None,
            equilibrium: bool = False,
            equi_threshold: float = 1e-4,
            max_iter: int = 100,
            save_screenshot: bool = False,
            perturb_steps: Optional[int] = None,
            perturb_sigma: float = 1e-2,
            increasing_Vdw: bool = False,
            Vdw_cutoff_start: float = 0.0,
            Vdw_cutoff_end: float = 12.5,
            **kwargs
    ):
        self.ff = ob.OBForceField.FindType(ff)
        self.constraints = None
        self.algorithm = algorithm
        self.equilibrium = equilibrium
        self.equi_threshold = equi_threshold
        self.max_iter = max_iter if isinstance(max_iter, int) else 1
        self.save_screenshot = save_screenshot
        self.perturb_steps = perturb_steps
        self.perturb_sigma = perturb_sigma
        self.increasing_Vdw = increasing_Vdw
        self.Vdw_cutoff_start = Vdw_cutoff_start
        self.Vdw_cutoff_end = Vdw_cutoff_end

        if steps:
            self.steps = steps
        elif not equilibrium:
            self.steps = 2000
        else:
            self.steps = max((20, 10000 // max_iter))

    def _perturb(self, coords):
        perturb = np.random.normal(0, self.perturb_sigma, coords.shape)
        perturb[perturb > 2*self.perturb_sigma] = self.perturb_sigma
        return perturb

    def _get_optimizer(self, mol):
        obmol = mol.to_obmol()
        if not self.ff.Setup(obmol, self.constraints):
            raise RuntimeError('Fail to initialize the forcefield!!')

        if self.algorithm == "steepest":
            optimizer = self.ff.SteepestDescent
        elif self.algorithm == "conjugate":
            optimizer = self.ff.ConjugateGradients
        else:
            raise NotImplementedError(f"Unknown optimization algorithm {self.algorithm}")

        return obmol, optimizer

    def setup(self, mol):
        obmol = mol.to_obmol()
        if not self.ff.Setup(obmol, self.constraints):
            raise RuntimeError('Fail to initialize the forcefield!!')
        return obmol

    def ob_setup(self, obmol):
        if not self.ff.Setup(obmol, self.constraints):
            raise RuntimeError('Fail to initialize the forcefield!!')
        return obmol

    def optimize(self, mol):
        """
        Optimizes the molecular geometry using specified constraints, forcefield, and optimization settings.

        Performs geometry optimization for a given molecule, either under equilibrium conditions
        or through perturbative steps if specified. The method uses constraint management, molecular
        coordinate manipulation, and an optimization routine based on the chosen forcefield.
        Parameters
        ----------
        mol : Molecule
            A molecular object containing atomic coordinates and chemical information.

        Raises
        ------
        RuntimeWarning
            Issued when the maximum number of iterations is reached without convergence.

        Returns
        -------
        None
        """
        self._add_constraints(mol)

        obmol, optimizer = self._get_optimizer(mol)
        if not self.equilibrium:
            # obmol, optimizer = self._get_optimizer(mol)
            optimizer(self.steps)
            self.ff.GetCoordinates(obmol)
            mol.coordinates = extract_obmol_coordinates(obmol)

        else:
            _opti_times = 0
            coords = mol.coordinates
            while not _opti_times or (isinstance(self.perturb_steps, int) and _opti_times < self.perturb_steps):

                if isinstance(self.perturb_steps, int):
                    # mol.coordinates += self._perturb(mol)
                    coords += self._perturb(coords)
                    set_obmol_coordinates(obmol, coords)
                    self.ob_setup(obmol)

                # obmol, optimizer = self._get_optimizer(mol)
                for i in range(self.max_iter):
                    optimizer(self.steps)
                    self.ff.GetCoordinates(obmol)
                    mol.coordinates = coords_ = extract_obmol_coordinates(obmol)

                    if self.save_screenshot:
                        mol.conformer_add()

                    max_displacement = max(np.linalg.norm(coords_ - coords, axis=1))
                    coords = coords_

                    if max_displacement < self.equi_threshold:
                        break

                if i == self.max_iter - 1:
                    print(RuntimeWarning("Max iterations reached"))

                print(f"Final Energy: {self.ff.Energy()}")
                _opti_times += 1

    def _add_constraints(self, mol):
        """
        Add constraints to a molecular force field based on specified conditions in the input molecule.

        This method initializes the OBFFConstraints object and applies various types of
        constraints such as atom, bond, angle, and torsion constraints. These constraints
        are derived from the properties of the input molecule's atoms, bonds, angles, and
        torsions.

        Parameters:
            mol (Molecule): Input molecule containing atoms, bonds, angles, and torsions
            each potentially having constraint attributes.

        Raises:
            None
        """
        self.constraints = ob.OBFFConstraints()
        for atom in mol.atoms:
            if atom.constraint:
                self.constraints.AddAtomConstraint(atom.idx)
            else:
                if atom.x_constraint:
                    self.constraints.AddAtomXConstraint(atom.idx)
                if atom.y_constraint:
                    self.constraints.AddAtomYConstraint(atom.idx)
                if atom.z_constraint:
                    self.constraints.AddAtomZConstraint(atom.idx)

        for bond in mol.bonds:
            if bond.constraint:
                self.constraints.AddDistanceConstraint(bond.a1idx, bond.a2idx, bond.length)

        for angle in mol.angles:
            if angle.constraint:
                self.constraints.AddAngleConstraint(angle.a1idx, angle.a2idx, angle.a3idx, angle.degrees)

        for torsion in mol.torsions:
            if torsion.constraint:
                self.constraints.AddTorsionConstraint(
                    torsion.a1idx,
                    torsion.a2idx,
                    torsion.a3idx,
                    torsion.a4idx,
                    torsion.degree
                )


def ob_build(mol):
    _builder = ob.OBBuilder()

    obmol = mol.to_obmol()
    _builder.Build(obmol)

    mol.coordinates = extract_obmol_coordinates(obmol)


def ob_optimize(mol, ff='UFF', steps: int = 100) -> float:
    ff = ob.OBForceField.FindType(ff)
    obmol = mol.to_obmol()

    if not ff.Setup(obmol):
        raise RuntimeError('Fail to initialize the forcefield!!')

    ff.SteepestDescent(steps)
    ff.GetCoordinates(obmol)
    mol.coordinates = extract_obmol_coordinates(obmol)

    return ff.Energy()


class OBBuilder:
    """
    Handles the construction and manipulation of molecular structures.

    The OBBuilder class serves as a wrapper around the Open Babel OBBuilder
    to facilitate the building of molecular structures. It initializes
    an OBBuilder instance and provides methods to build structures and
    update molecule coordinates based on the processed OBMol instance.

    Attributes:
        _builder: An instance of Open Babel's OBBuilder used to
            handle molecular structure construction.
    """
    def __init__(self):
        self._builder = ob.OBBuilder()

    def build(self, mol):
        obmol = mol.to_obmol()
        self._builder.Build(obmol)

        mol.coordinates = extract_obmol_coordinates(obmol)


class ForceFields:
    def __init__(self, forcefield: str):
        self.name = forcefield
        self._ff = ob.OBForceField.FindType(forcefield)

