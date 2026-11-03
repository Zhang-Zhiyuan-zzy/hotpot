"""Immutable molecular-topology values shared by force-field consumers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple, Union


__all__ = (
    "AtomIdentity",
    "BondTopologyRevision",
    "ForceFieldFrame",
    "CoordinationEnvironment",
    "AtomTopologySignature",
    "BondTopology",
    "TopologyReference",
    "TrajectoryStart",
    "TrajectoryStage",
    "TrajectoryEvent",
    "RingFrameEvidence",
    "CoordinationFrameEvidence",
    "OptimizationFrameEvidence",
    "FrameEvidence",
)


@dataclass(frozen=True)
class AtomTopologySignature:
    """Stable identity and chemistry for an atom present before optimization."""

    index: int
    atom_id: int
    atomic_number: int
    formal_charge: int


@dataclass(frozen=True, order=True)
class BondTopology:
    """One bond in a topology revision, addressed by atom indices."""

    atom_indices: Tuple[int, int]
    bond_order: float
    bond_kind: str

    def __post_init__(self) -> None:
        atom1_index, atom2_index = sorted(self.atom_indices)
        object.__setattr__(self, "atom_indices", (atom1_index, atom2_index))

    @classmethod
    def from_bond(cls, bond: "Bond") -> "BondTopology":
        """Capture one Hotpot bond without retaining mutable objects."""
        return cls(
            atom_indices=(int(bond.atom1.idx), int(bond.atom2.idx)),
            bond_order=float(bond.bond_order),
            bond_kind=str(bond.bond_kind.value),
        )


@dataclass(frozen=True)
class TopologyReference:
    """Immutable topology snapshot used by force-field transactions."""

    atoms: Tuple[AtomTopologySignature, ...]
    bonds: Tuple[BondTopology, ...]
    allow_added_hydrogens: bool = True


class TrajectoryStart(str, Enum):
    """Earliest force-field stage retained in a trajectory."""

    LIGAND_BUILD = "ligand_build"
    COORDINATION_RESTORATION = "coordination_restoration"
    COMPLEX_UNTANGLING = "complex_untangling"
    FINAL_OPTIMIZATION = "final_optimization"


class TrajectoryStage(str, Enum):
    """Force-field workflow stage that produced a frame."""

    LIGAND_BUILD = "ligand_build"
    COORDINATION_RESTORATION = "coordination_restoration"
    COMPLEX_UNTANGLING = "complex_untangling"
    FINAL_OPTIMIZATION = "final_optimization"


class TrajectoryEvent(str, Enum):
    """Observable event represented by one trajectory frame."""

    INITIAL = "initial"
    BUILD_COMPLETE = "build_complete"
    WARMUP_COMPLETE = "warmup_complete"
    COORDINATION_READY = "coordination_ready"
    BOND_TRIAL = "bond_trial"
    BOND_ACCEPTED = "bond_accepted"
    BOND_REJECTED = "bond_rejected"
    BOND_ROLLBACK = "bond_rollback"
    BOND_FORCED = "bond_forced"
    METAL_RELOCATION_TRIAL = "metal_relocation_trial"
    METAL_RELOCATED = "metal_relocated"
    METAL_RELOCATION_FAILED = "metal_relocation_failed"
    TOPOLOGY_CHECKPOINT = "topology_checkpoint"
    RING_OPENED = "ring_opened"
    PERTURBED = "perturbed"
    OPTIMIZED = "optimized"
    RING_CLOSED = "ring_closed"
    SETTLED = "settled"
    ROLLED_BACK = "rolled_back"
    EPOCH_COMPLETE = "epoch_complete"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class RingFrameEvidence:
    """Ring--bond observations already computed by the workflow."""

    confirmed_piercing_count: int
    uncertain_relation_count: Optional[int] = 0
    ring_scope: Optional[Literal["full_graph", "ligand_skeleton"]] = None
    max_ring_size: Optional[int] = None
    selected_ring_count: Optional[int] = None
    excluded_ring_count: Optional[int] = None
    candidate_pair_count: Optional[int] = None
    aabb_separated_pair_count: Optional[int] = None
    exact_pair_count: Optional[int] = None
    does_not_pierce_pair_count: Optional[int] = None
    scan_complete: Optional[bool] = None


@dataclass(frozen=True)
class CoordinationFrameEvidence:
    """Outcome of one metal--ligand bond restoration observation."""

    bond_atom_indices: Optional[Tuple[int, int]]
    accepted: Optional[bool]
    pending_bond_count: int = 0
    forced: bool = False
    piercing_relation_count: int = 0
    undetermined_relation_count: int = 0
    excluded_ring_count: int = 0
    metal_atom_index: Optional[int] = None
    relocation_status: Optional[Literal["relocated", "infeasible"]] = None
    relocation_candidates_evaluated: int = 0
    safe_donor_atom_indices: Tuple[int, ...] = ()
    minimum_normalized_clearance: Optional[float] = None
    coordination_distance_deviation: Optional[float] = None


@dataclass(frozen=True)
class OptimizationFrameEvidence:
    """Numerical observations associated with an optimization frame."""

    converged: bool
    exploded: bool
    finite_coordinates: bool
    finite_energy: bool
    finite_gradients: bool
    rms_gradient_kj_mol_angstrom: Optional[float] = None
    max_gradient_kj_mol_angstrom: Optional[float] = None
    energy_change_kj_mol: Optional[float] = None
    max_displacement_angstrom: Optional[float] = None


FrameEvidence = Union[
    RingFrameEvidence,
    CoordinationFrameEvidence,
    OptimizationFrameEvidence,
]


@dataclass(frozen=True, order=True)
class AtomIdentity:
    """Atom identity fixed for the lifetime of one trajectory."""

    index: int
    atom_id: int
    atomic_number: int
    formal_charge: int
    symbol: str

    @classmethod
    def from_atom(
        cls,
        atom: "Atom",
        *,
        index: Optional[int] = None,
    ) -> "AtomIdentity":
        """Capture the stable identity fields of a Hotpot atom."""
        return cls(
            index=int(atom.idx if index is None else index),
            atom_id=int(atom.id),
            atomic_number=int(atom.atomic_number),
            formal_charge=int(atom.formal_charge),
            symbol=str(atom.symbol),
        )


@dataclass(frozen=True)
class BondTopologyRevision:
    """Immutable complete bond table for one trajectory topology."""

    index: int
    bonds: Tuple[BondTopology, ...]


@dataclass(frozen=True)
class ForceFieldFrame:
    """Immutable reference to one coordinate/topology state."""

    index: int
    stage: TrajectoryStage
    event: TrajectoryEvent
    coordinate_revision: int
    topology_revision: int
    energy_kj_mol: Optional[float] = None
    component_index: Optional[int] = None
    attempt: Optional[int] = None
    step: Optional[int] = None
    evidence: Optional[FrameEvidence] = None


@dataclass(frozen=True)
class CoordinationEnvironment:
    metal_idx: int
    donor_indices: Tuple[int, ...]
    coordination_number: int
    metal_atomic_number: int
    metal_formal_charge: int
    donor_atomic_numbers: Tuple[int, ...]
    chelate_groups: Tuple[Tuple[int, ...], ...]
