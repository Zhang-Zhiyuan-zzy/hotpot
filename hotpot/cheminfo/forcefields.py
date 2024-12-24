"""
python v3.9.0
@Project: hotpot
@File   : forcefields
@Auther : Zhiyuan Zhang
@Data   : 2024/12/14
@Time   : 21:26
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
        correct_hydrogens: bool = True,
        timeout: int = 1000,
        **kwargs
):
    mol.add_hydrogens(remove_excess=correct_hydrogens)
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
    """ A Wrapper of OpenBabel's ForceField """
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

