from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt


Coordinate = Tuple[float, float, float]


class RuleStage(Enum):
    PRE_BUILD: RuleStage
    PRE_FORCEFIELD_SETUP: RuleStage


class BondKind(Enum):
    SINGLE: BondKind
    DOUBLE: BondKind
    TRIPLE: BondKind
    AROMATIC: BondKind
    ZERO: BondKind
    DATIVE: BondKind
    UNKNOWN: BondKind


class RuleDescriptor:
    rule_id: str
    version: str
    stage: RuleStage
    priority: int


class HybridizationChange:
    atom_index: int
    before: int
    after: int


class CoordinateChange:
    atom_index: int
    before: Coordinate
    after: Coordinate


class RuleApplication:
    rule_id: str
    version: str
    stage: RuleStage
    priority: int
    atom_indices: List[int]
    metric_before: Optional[float]
    hybridization_changes: List[HybridizationChange]
    coordinate_changes: List[CoordinateChange]


class RulePlan:
    stage: RuleStage
    applications: List[RuleApplication]


class ForceFieldSetupError(RuntimeError):
    forcefield: str
    stage: str


class MoleculeData:
    def __init__(
        self,
        schema_version: int,
        atomic_numbers: npt.NDArray[np.int32],
        formal_charges: npt.NDArray[np.int32],
        partial_charges: npt.NDArray[np.float64],
        coordinates: npt.NDArray[np.float64],
        atom_aromatic: npt.NDArray[np.uint8],
        bond_indices: npt.NDArray[np.int32],
        bond_orders: npt.NDArray[np.float64],
        bond_kinds: npt.NDArray[np.uint8],
        bond_aromatic: npt.NDArray[np.uint8],
        unit_cell: Optional[npt.NDArray[np.float64]] = ...,
    ) -> None: ...

    atom_count: int
    bond_count: int


class BuildResult:
    succeeded: bool
    coordinates: npt.NDArray[np.float64]
    rules: RulePlan


class SingleOptimizationResult:
    coordinates: npt.NDArray[np.float64]
    energy_kj_mol: float
    backend_energy_unit: str
    exploded: bool
    rules: RulePlan


class OptimizationFrame:
    coordinates: npt.NDArray[np.float64]
    energy: float
    rms_gradient: float
    max_gradient: float
    exploded: bool
    converged: bool
    epoch_index: int
    segment_epochs_completed: int
    segment_index: int
    energy_change: Optional[float]
    max_displacement: Optional[float]


class OptimizationResult:
    coordinates: npt.NDArray[np.float64]
    terminal_coordinates: npt.NDArray[np.float64]
    frames: List[OptimizationFrame]
    selected_frame_index: int
    best_epoch: int
    final_energy: float
    best_energy: float
    rms_gradient: float
    max_gradient: float
    exploded: bool
    converged: bool
    epochs_completed: int
    steps_submitted: int
    initialization_steps: int
    selected_segment_epochs_completed: int
    backend_energy_unit: str
    termination_reason: str
    terminal_converged: bool
    energy_changes: List[float]
    max_displacements: List[float]
    epoch_energies: List[float]
    rules: RulePlan


class RuntimeInfo:
    compiled_openbabel_version: str
    runtime_openbabel_version: str
    cxx11_abi: int
    openbabel_library_path: str
    babel_libdir: str
    babel_datadir: str


def runtime_info() -> RuntimeInfo: ...

def seed_random(seed: int) -> None: ...


def available_rules(stage: Optional[RuleStage] = ...) -> List[RuleDescriptor]: ...

def build(
    molecule: MoleculeData,
    stereo_warnings: Optional[bool] = ...,
) -> BuildResult: ...

def single_optimize(
    molecule: MoleculeData,
    forcefield: str,
    steps: int,
    singularity_threshold: float = ...,
    repair_angle_radians: float = ...,
) -> SingleOptimizationResult: ...

def optimize(
    molecule: MoleculeData,
    forcefield: str,
    algorithm: str,
    epochs: int,
    steps_per_epoch: int,
    perturb_interval: Optional[int] = ...,
    perturbation_offsets: Optional[npt.NDArray[np.float64]] = ...,
    retain_frames: bool = ...,
    retain_epoch_history: bool = ...,
    increasing_vdw: bool = ...,
    vdw_cutoff_start: float = ...,
    vdw_cutoff_end: float = ...,
    energy_tolerance: float = ...,
    stopping_window: Optional[int] = ...,
    maximum_energy_change_kj_mol: float = ...,
    maximum_atom_displacement_angstrom: float = ...,
    maximum_rms_gradient_kj_mol_angstrom: float = ...,
    maximum_gradient_kj_mol_angstrom: float = ...,
    singularity_threshold: float = ...,
    repair_angle_radians: float = ...,
) -> OptimizationResult: ...
