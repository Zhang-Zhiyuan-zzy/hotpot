from enum import Enum
from typing import List, Optional, Sequence, Tuple


Coordinate = Tuple[float, float, float]


class RuleStage(Enum):
    PRE_BUILD: RuleStage
    PRE_FORCEFIELD_SETUP: RuleStage


class AtomSnapshot:
    def __init__(
        self,
        atomic_number: int,
        formal_charge: int,
        hybridization: int,
        is_metal: bool,
    ) -> None: ...

    @property
    def atomic_number(self) -> int: ...

    @property
    def formal_charge(self) -> int: ...

    @property
    def hybridization(self) -> int: ...

    @property
    def is_metal(self) -> bool: ...


class BondSnapshot:
    def __init__(
        self,
        begin: int,
        end: int,
        order: int,
        aromatic: bool,
    ) -> None: ...

    @property
    def begin(self) -> int: ...

    @property
    def end(self) -> int: ...

    @property
    def order(self) -> int: ...

    @property
    def aromatic(self) -> bool: ...


class RuleDescriptor:
    @property
    def rule_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def stage(self) -> RuleStage: ...

    @property
    def priority(self) -> int: ...


class HybridizationChange:
    @property
    def atom_index(self) -> int: ...

    @property
    def before(self) -> int: ...

    @property
    def after(self) -> int: ...


class CoordinateChange:
    @property
    def atom_index(self) -> int: ...

    @property
    def before(self) -> Coordinate: ...

    @property
    def after(self) -> Coordinate: ...


class RuleApplication:
    @property
    def rule_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def stage(self) -> RuleStage: ...

    @property
    def priority(self) -> int: ...

    @property
    def atom_indices(self) -> List[int]: ...

    @property
    def metric_before(self) -> Optional[float]: ...

    @property
    def hybridization_changes(self) -> List[HybridizationChange]: ...

    @property
    def coordinate_changes(self) -> List[CoordinateChange]: ...


class RulePlan:
    @property
    def stage(self) -> RuleStage: ...

    @property
    def applications(self) -> List[RuleApplication]: ...


class RuleApplicationLimitExceeded(RuntimeError): ...


def available_rules(
    stage: Optional[RuleStage] = ...,
) -> List[RuleDescriptor]: ...


def plan_build(
    atoms: Sequence[AtomSnapshot],
    bonds: Sequence[BondSnapshot],
) -> RulePlan: ...


def plan_optimization(
    atoms: Sequence[AtomSnapshot],
    bonds: Sequence[BondSnapshot],
    coordinates: Sequence[Coordinate],
    singularity_threshold: float,
    repair_angle_radians: float,
) -> RulePlan: ...
