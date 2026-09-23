"""Force-field trajectory records with coordinate and topology revisions.

The trajectory model records facts produced by force-field workflows.  It does
not decide whether a workflow should retry, roll back, or move to another
stage.  In particular, topology-changing coordination and ring-untangling
steps remain distinguishable even when two frames share identical coordinates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np


if TYPE_CHECKING:
    from ..core import Atom, Bond, Molecule


__all__ = (
    "TrajectoryStart",
    "TrajectoryStage",
    "TrajectoryEvent",
    "AtomIdentity",
    "BondTopology",
    "BondTopologyRevision",
    "RingFrameEvidence",
    "CoordinationFrameEvidence",
    "OptimizationFrameEvidence",
    "FrameEvidence",
    "ForceFieldFrame",
    "ForceFieldTrajectory",
    "ForceFieldTrajectoryArchive",
)


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
    RING_OPENED = "ring_opened"
    PERTURBED = "perturbed"
    OPTIMIZED = "optimized"
    RING_CLOSED = "ring_closed"
    SETTLED = "settled"
    ROLLED_BACK = "rolled_back"
    EPOCH_COMPLETE = "epoch_complete"
    TERMINAL = "terminal"


_TRAJECTORY_STAGE_ORDER = {
    TrajectoryStage.LIGAND_BUILD: 0,
    TrajectoryStage.COORDINATION_RESTORATION: 1,
    TrajectoryStage.COMPLEX_UNTANGLING: 2,
    TrajectoryStage.FINAL_OPTIMIZATION: 3,
}


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
class BondTopologyRevision:
    """Immutable complete bond table for one trajectory topology."""

    index: int
    bonds: Tuple[BondTopology, ...]


@dataclass(frozen=True)
class RingFrameEvidence:
    """Ring--bond observations already computed by the workflow."""

    confirmed_piercing_count: int
    uncertain_relation_count: int = 0


@dataclass(frozen=True)
class CoordinationFrameEvidence:
    """Outcome of one metal--ligand bond restoration observation."""

    bond_atom_indices: Optional[Tuple[int, int]]
    accepted: bool
    pending_bond_count: int = 0
    forced: bool = False


@dataclass(frozen=True)
class OptimizationFrameEvidence:
    """Quality observations associated with an optimization frame."""

    accepted: bool
    converged: bool
    rms_gradient_kj_mol_angstrom: Optional[float] = None
    max_gradient_kj_mol_angstrom: Optional[float] = None
    failed_checks: Tuple[str, ...] = ()


FrameEvidence = Union[
    RingFrameEvidence,
    CoordinationFrameEvidence,
    OptimizationFrameEvidence,
]


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


class ForceFieldTrajectory:
    """A topology-aware sequence of force-field workflow frames."""

    _FORMAT_VERSION = 1

    def __init__(
        self,
        atoms: Sequence[AtomIdentity],
        *,
        start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    ) -> None:
        self._atoms = tuple(atoms)
        self._start = start
        self._coordinate_revisions: list[np.ndarray] = []
        self._coordinate_index: dict[bytes, list[int]] = {}
        self._topology_revisions: list[BondTopologyRevision] = []
        self._topology_index: dict[Tuple[BondTopology, ...], int] = {}
        self._frames: list[ForceFieldFrame] = []
        self._selected_index: Optional[int] = None

    @classmethod
    def from_molecule(
        cls,
        mol: "Molecule",
        *,
        start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    ) -> "ForceFieldTrajectory":
        """Create an empty trajectory with the molecule's atom identity."""
        return cls(
            tuple(
                AtomIdentity.from_atom(atom, index=index)
                for index, atom in enumerate(mol.atoms)
            ),
            start=start,
        )

    @property
    def atoms(self) -> Tuple[AtomIdentity, ...]:
        return self._atoms

    @property
    def start(self) -> TrajectoryStart:
        return self._start

    @property
    def frames(self) -> Tuple[ForceFieldFrame, ...]:
        return tuple(self._frames)

    @property
    def topology_revisions(self) -> Tuple[BondTopologyRevision, ...]:
        return tuple(self._topology_revisions)

    @property
    def coordinate_revision_count(self) -> int:
        return len(self._coordinate_revisions)

    @property
    def topology_revision_count(self) -> int:
        return len(self._topology_revisions)

    @property
    def selected_index(self) -> Optional[int]:
        return self._selected_index

    @property
    def selected_frame(self) -> Optional[ForceFieldFrame]:
        if self._selected_index is None:
            return None
        return self._frames[self._selected_index]

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[ForceFieldFrame]:
        return iter(self._frames)

    def __getitem__(self, index: int) -> ForceFieldFrame:
        return self._frames[index]

    def records(self, stage: TrajectoryStage) -> bool:
        """Return whether the configured trajectory includes ``stage``."""
        start_stage = TrajectoryStage(self._start.value)
        return _TRAJECTORY_STAGE_ORDER[stage] >= _TRAJECTORY_STAGE_ORDER[start_stage]

    def coordinates(self, frame_index: int) -> np.ndarray:
        """Return an independent coordinate array for a frame."""
        frame = self._frames[frame_index]
        return self._coordinate_revisions[frame.coordinate_revision].copy()

    def topology(self, frame_index: int) -> BondTopologyRevision:
        """Return the immutable topology revision for a frame."""
        frame = self._frames[frame_index]
        return self._topology_revisions[frame.topology_revision]

    def record(
        self,
        coordinates: np.ndarray,
        bonds: Sequence[BondTopology],
        *,
        stage: TrajectoryStage,
        event: TrajectoryEvent,
        energy_kj_mol: Optional[float] = None,
        component_index: Optional[int] = None,
        attempt: Optional[int] = None,
        step: Optional[int] = None,
        evidence: Optional[FrameEvidence] = None,
    ) -> ForceFieldFrame:
        """Record one factual frame without affecting workflow control."""
        coordinate_revision = self._pool_coordinates(coordinates)
        topology_revision = self._pool_topology(bonds)
        stored_energy = (
            None
            if energy_kj_mol is None or not np.isfinite(energy_kj_mol)
            else float(energy_kj_mol)
        )
        frame = ForceFieldFrame(
            index=len(self._frames),
            stage=stage,
            event=event,
            coordinate_revision=coordinate_revision,
            topology_revision=topology_revision,
            energy_kj_mol=stored_energy,
            component_index=component_index,
            attempt=attempt,
            step=step,
            evidence=evidence,
        )
        self._frames.append(frame)
        return frame

    def record_molecule(
        self,
        mol: "Molecule",
        *,
        stage: TrajectoryStage,
        event: TrajectoryEvent,
        energy_kj_mol: Optional[float] = None,
        component_index: Optional[int] = None,
        attempt: Optional[int] = None,
        step: Optional[int] = None,
        evidence: Optional[FrameEvidence] = None,
    ) -> ForceFieldFrame:
        """Capture coordinates and the currently active Hotpot bond table."""
        self._validate_molecule_atoms(mol)
        return self.record(
            mol.coordinates,
            self._molecule_bonds(mol),
            stage=stage,
            event=event,
            energy_kj_mol=energy_kj_mol,
            component_index=component_index,
            attempt=attempt,
            step=step,
            evidence=evidence,
        )

    def select(self, frame_index: int) -> ForceFieldFrame:
        """Mark a recorded frame as the workflow-selected result."""
        frame = self._frames[frame_index]
        self._selected_index = frame.index
        return frame

    def materialize(self, mol: "Molecule", *, keep_all: bool) -> None:
        """Expose trajectory coordinates through ``Molecule.conformers``.

        ``Molecule.conformers`` cannot represent changing bond topology.  It is
        therefore a coordinate-only compatibility view; this trajectory remains
        the authoritative record for topology-changing frames.
        """
        self._validate_molecule_atoms(mol)
        if not self._frames:
            mol.conformer_clear()
            return

        selected_index = (
            self._selected_index
            if self._selected_index is not None
            else len(self._frames) - 1
        )
        frame_indices = tuple(range(len(self._frames))) if keep_all else (selected_index,)
        coordinates = np.stack(
            tuple(self.coordinates(frame_index) for frame_index in frame_indices)
        )
        energies = np.asarray(
            tuple(
                np.nan
                if self._frames[frame_index].energy_kj_mol is None
                else self._frames[frame_index].energy_kj_mol
                for frame_index in frame_indices
            ),
            dtype=float,
        )
        mol.conformer_clear()
        mol.conformer_add(coordinates, energies)
        materialized_index = frame_indices.index(selected_index)
        mol.conformer_load(materialized_index)

    def write(self, path: Union[str, Path], *, include_sdf: bool = True) -> None:
        """Write a lossless JSON/NPZ record and an optional topology-aware SDF."""
        _TrajectoryWriter.write_trajectory(Path(path), self, include_sdf=include_sdf)

    @classmethod
    def read(cls, path: Union[str, Path]) -> "ForceFieldTrajectory":
        """Read a trajectory written by :meth:`write`."""
        return _TrajectoryWriter.read_trajectory(Path(path))

    def write_sdf(self, path: Union[str, Path]) -> None:
        """Write every frame as an SDF record with its own bond topology."""
        _TrajectoryWriter.write_sdf(Path(path), self)

    def _pool_coordinates(self, coordinates: np.ndarray) -> int:
        array = np.ascontiguousarray(coordinates, dtype=np.float64)
        expected_shape = (len(self._atoms), 3)
        if array.shape != expected_shape:
            raise ValueError(
                f"Coordinates must have shape {expected_shape}, got {array.shape}"
            )
        digest = hashlib.blake2b(memoryview(array), digest_size=16).digest()
        for revision_index in self._coordinate_index.get(digest, ()):
            if np.array_equal(self._coordinate_revisions[revision_index], array):
                return revision_index

        stored = array.copy()
        stored.setflags(write=False)
        revision_index = len(self._coordinate_revisions)
        self._coordinate_revisions.append(stored)
        self._coordinate_index.setdefault(digest, []).append(revision_index)
        return revision_index

    def _pool_topology(self, bonds: Sequence[BondTopology]) -> int:
        normalized = tuple(sorted(bonds))
        if normalized in self._topology_index:
            return self._topology_index[normalized]
        revision_index = len(self._topology_revisions)
        revision = BondTopologyRevision(revision_index, normalized)
        self._topology_revisions.append(revision)
        self._topology_index[normalized] = revision_index
        return revision_index

    def _validate_molecule_atoms(self, mol: "Molecule") -> None:
        observed = tuple(
            AtomIdentity.from_atom(atom, index=index)
            for index, atom in enumerate(mol.atoms)
        )
        if observed != self._atoms:
            raise ValueError("Molecule atom identity differs from this trajectory")

    @staticmethod
    def _molecule_bonds(mol: "Molecule") -> Tuple[BondTopology, ...]:
        atom_indices = {
            id(atom): index
            for index, atom in enumerate(mol.atoms)
        }
        return tuple(
            BondTopology(
                atom_indices=(
                    atom_indices[id(bond.atom1)],
                    atom_indices[id(bond.atom2)],
                ),
                bond_order=float(bond.bond_order),
                bond_kind=str(bond.bond_kind.value),
            )
            for bond in mol.bonds
        )

    def _coordinate_view(self, revision_index: int) -> np.ndarray:
        return self._coordinate_revisions[revision_index]


@dataclass(frozen=True)
class ForceFieldTrajectoryArchive:
    """Main continuous trajectory plus optional ligand-build branches."""

    main: ForceFieldTrajectory
    ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = ()

    def write(self, path: Union[str, Path], *, include_sdf: bool = True) -> None:
        """Write the main trajectory and every independent build branch."""
        _TrajectoryWriter.write_archive(Path(path), self, include_sdf=include_sdf)

    @classmethod
    def read(cls, path: Union[str, Path]) -> "ForceFieldTrajectoryArchive":
        """Read an archive written by :meth:`write`."""
        return _TrajectoryWriter.read_archive(Path(path))


class _TrajectoryWriter:
    """Disk representation for force-field trajectories."""

    @classmethod
    def write_trajectory(
        cls,
        directory: Path,
        trajectory: ForceFieldTrajectory,
        *,
        include_sdf: bool,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = cls._trajectory_manifest(trajectory)
        (directory / "trajectory.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        coordinate_stack = (
            np.stack(tuple(trajectory._coordinate_revisions))
            if trajectory._coordinate_revisions
            else np.empty((0, len(trajectory.atoms), 3), dtype=np.float64)
        )
        np.savez_compressed(directory / "coordinates.npz", coordinates=coordinate_stack)
        if include_sdf:
            cls.write_sdf(directory / "trajectory.sdf", trajectory)
        else:
            (directory / "trajectory.sdf").unlink(missing_ok=True)

    @classmethod
    def read_trajectory(cls, directory: Path) -> ForceFieldTrajectory:
        manifest_object = json.loads(
            (directory / "trajectory.json").read_text(encoding="utf-8")
        )
        manifest = cast(Mapping[str, object], manifest_object)
        format_version = int(cast(int, manifest["format_version"]))
        if format_version != ForceFieldTrajectory._FORMAT_VERSION:
            raise ValueError(f"Unsupported trajectory format version {format_version}")

        atoms = tuple(
            AtomIdentity(**cast(Mapping[str, object], atom_data))
            for atom_data in cast(Sequence[object], manifest["atoms"])
        )
        trajectory = ForceFieldTrajectory(
            atoms,
            start=TrajectoryStart(str(manifest["start"])),
        )
        topologies = tuple(
            tuple(
                BondTopology(
                    atom_indices=cast(Tuple[int, int], tuple(bond_data["atom_indices"])),
                    bond_order=float(cast(float, bond_data["bond_order"])),
                    bond_kind=str(bond_data["bond_kind"]),
                )
                for bond_object in cast(Sequence[object], topology_object)
                for bond_data in (cast(Mapping[str, object], bond_object),)
            )
            for topology_object in cast(Sequence[object], manifest["topologies"])
        )
        with np.load(directory / "coordinates.npz", allow_pickle=False) as coordinate_data:
            coordinate_revisions = np.asarray(coordinate_data["coordinates"])

        for frame_object in cast(Sequence[object], manifest["frames"]):
            frame_data = cast(Mapping[str, object], frame_object)
            coordinate_revision = int(cast(int, frame_data["coordinate_revision"]))
            topology_revision = int(cast(int, frame_data["topology_revision"]))
            frame = trajectory.record(
                coordinate_revisions[coordinate_revision],
                topologies[topology_revision],
                stage=TrajectoryStage(str(frame_data["stage"])),
                event=TrajectoryEvent(str(frame_data["event"])),
                energy_kj_mol=cls._optional_float(frame_data["energy_kj_mol"]),
                component_index=cls._optional_int(frame_data["component_index"]),
                attempt=cls._optional_int(frame_data["attempt"]),
                step=cls._optional_int(frame_data["step"]),
                evidence=cls._evidence_from_data(frame_data["evidence"]),
            )
            if frame.coordinate_revision != coordinate_revision:
                raise ValueError("Coordinate revision order changed while reading trajectory")
            if frame.topology_revision != topology_revision:
                raise ValueError("Topology revision order changed while reading trajectory")

        selected_index = cls._optional_int(manifest["selected_index"])
        if selected_index is not None:
            trajectory.select(selected_index)
        return trajectory

    @classmethod
    def write_archive(
        cls,
        directory: Path,
        archive: ForceFieldTrajectoryArchive,
        *,
        include_sdf: bool,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        cls.write_trajectory(directory / "main", archive.main, include_sdf=include_sdf)
        attempt_paths = []
        for attempt_index, trajectory in enumerate(archive.ligand_build_attempts):
            relative_path = Path("ligand_build_attempts") / f"{attempt_index:04d}"
            cls.write_trajectory(
                directory / relative_path,
                trajectory,
                include_sdf=include_sdf,
            )
            attempt_paths.append(relative_path.as_posix())
        manifest: dict[str, object] = {
            "format_version": ForceFieldTrajectory._FORMAT_VERSION,
            "main": "main",
            "ligand_build_attempts": attempt_paths,
        }
        (directory / "archive.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def read_archive(cls, directory: Path) -> ForceFieldTrajectoryArchive:
        manifest_object = json.loads(
            (directory / "archive.json").read_text(encoding="utf-8")
        )
        manifest = cast(Mapping[str, object], manifest_object)
        main = cls.read_trajectory(directory / str(manifest["main"]))
        attempts = tuple(
            cls.read_trajectory(directory / str(relative_path))
            for relative_path in cast(
                Sequence[object],
                manifest["ligand_build_attempts"],
            )
        )
        return ForceFieldTrajectoryArchive(main, attempts)

    @classmethod
    def write_sdf(cls, path: Path, trajectory: ForceFieldTrajectory) -> None:
        records = tuple(
            cls._sdf_record(trajectory, frame)
            for frame in trajectory.frames
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(records), encoding="utf-8")

    @classmethod
    def _trajectory_manifest(
        cls,
        trajectory: ForceFieldTrajectory,
    ) -> dict[str, object]:
        return {
            "format_version": trajectory._FORMAT_VERSION,
            "start": trajectory.start.value,
            "atoms": [asdict(atom) for atom in trajectory.atoms],
            "topologies": [
                [asdict(bond) for bond in revision.bonds]
                for revision in trajectory.topology_revisions
            ],
            "frames": [cls._frame_data(frame) for frame in trajectory.frames],
            "selected_index": trajectory.selected_index,
        }

    @classmethod
    def _frame_data(cls, frame: ForceFieldFrame) -> dict[str, object]:
        return {
            "index": frame.index,
            "stage": frame.stage.value,
            "event": frame.event.value,
            "coordinate_revision": frame.coordinate_revision,
            "topology_revision": frame.topology_revision,
            "energy_kj_mol": frame.energy_kj_mol,
            "component_index": frame.component_index,
            "attempt": frame.attempt,
            "step": frame.step,
            "evidence": cls._evidence_data(frame.evidence),
        }

    @staticmethod
    def _evidence_data(evidence: Optional[FrameEvidence]) -> Optional[dict[str, object]]:
        if evidence is None:
            return None
        data: dict[str, object] = asdict(evidence)
        if isinstance(evidence, RingFrameEvidence):
            data["type"] = "ring"
        elif isinstance(evidence, CoordinationFrameEvidence):
            data["type"] = "coordination"
        else:
            data["type"] = "optimization"
        return data

    @staticmethod
    def _evidence_from_data(data: object) -> Optional[FrameEvidence]:
        if data is None:
            return None
        evidence_data = dict(cast(Mapping[str, object], data))
        evidence_type = str(evidence_data.pop("type"))
        if evidence_type == "ring":
            return RingFrameEvidence(
                confirmed_piercing_count=int(
                    cast(int, evidence_data["confirmed_piercing_count"])
                ),
                uncertain_relation_count=int(
                    cast(int, evidence_data["uncertain_relation_count"])
                ),
            )
        if evidence_type == "coordination":
            bond_indices = evidence_data["bond_atom_indices"]
            return CoordinationFrameEvidence(
                bond_atom_indices=(
                    None
                    if bond_indices is None
                    else cast(Tuple[int, int], tuple(cast(Sequence[int], bond_indices)))
                ),
                accepted=bool(evidence_data["accepted"]),
                pending_bond_count=int(cast(int, evidence_data["pending_bond_count"])),
                forced=bool(evidence_data["forced"]),
            )
        if evidence_type == "optimization":
            return OptimizationFrameEvidence(
                accepted=bool(evidence_data["accepted"]),
                converged=bool(evidence_data["converged"]),
                rms_gradient_kj_mol_angstrom=_TrajectoryWriter._optional_float(
                    evidence_data["rms_gradient_kj_mol_angstrom"]
                ),
                max_gradient_kj_mol_angstrom=_TrajectoryWriter._optional_float(
                    evidence_data["max_gradient_kj_mol_angstrom"]
                ),
                failed_checks=tuple(
                    str(item)
                    for item in cast(Sequence[object], evidence_data["failed_checks"])
                ),
            )
        raise ValueError(f"Unknown frame evidence type {evidence_type!r}")

    @staticmethod
    def _optional_float(value: object) -> Optional[float]:
        return None if value is None else float(cast(float, value))

    @staticmethod
    def _optional_int(value: object) -> Optional[int]:
        return None if value is None else int(cast(int, value))

    @classmethod
    def _sdf_record(
        cls,
        trajectory: ForceFieldTrajectory,
        frame: ForceFieldFrame,
    ) -> str:
        coordinates = trajectory._coordinate_view(frame.coordinate_revision)
        topology = trajectory.topology_revisions[frame.topology_revision]
        atoms = trajectory.atoms
        if len(atoms) > 999 or len(topology.bonds) > 999:
            raise ValueError("Topology-resolved SDF export currently supports V2000 limits")

        lines = [
            f"Hotpot force-field frame {frame.index}",
            "  Hotpot trajectory",
            "",
            f"{len(atoms):>3}{len(topology.bonds):>3}  0  0  0  0            999 V2000",
        ]
        for atom, coordinate in zip(atoms, coordinates):
            x, y, z = coordinate
            lines.append(
                f"{x:>10.4f}{y:>10.4f}{z:>10.4f} "
                f"{atom.symbol:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
            )
        for bond in topology.bonds:
            atom1_index, atom2_index = bond.atom_indices
            lines.append(
                f"{atom1_index + 1:>3}{atom2_index + 1:>3}"
                f"{cls._sdf_bond_order(bond):>3}  0  0  0  0"
            )

        charged_atoms = tuple(
            (atom.index + 1, atom.formal_charge)
            for atom in atoms
            if atom.formal_charge
        )
        for offset in range(0, len(charged_atoms), 8):
            group = charged_atoms[offset:offset + 8]
            charge_fields = "".join(
                f"{atom_index:>4}{formal_charge:>4}"
                for atom_index, formal_charge in group
            )
            lines.append(f"M  CHG{len(group):>3}{charge_fields}")
        lines.append("M  END")

        cls._append_sdf_property(lines, "HOTpot Frame Index", str(frame.index))
        cls._append_sdf_property(lines, "HOTpot Stage", frame.stage.value)
        cls._append_sdf_property(lines, "HOTpot Event", frame.event.value)
        cls._append_sdf_property(
            lines,
            "HOTpot Topology Revision",
            str(frame.topology_revision),
        )
        cls._append_sdf_property(
            lines,
            "HOTpot Selected",
            str(frame.index == trajectory.selected_index),
        )
        if frame.energy_kj_mol is not None:
            cls._append_sdf_property(
                lines,
                "Energy (kJ/mol)",
                repr(frame.energy_kj_mol),
            )
        cls._append_sdf_property(
            lines,
            "HOTpot Bond Kinds",
            json.dumps(
                [
                    {
                        "atom_indices": bond.atom_indices,
                        "bond_kind": bond.bond_kind,
                    }
                    for bond in topology.bonds
                ],
                separators=(",", ":"),
            ),
        )
        if frame.evidence is not None:
            cls._append_sdf_property(
                lines,
                "HOTpot Evidence",
                json.dumps(cls._evidence_data(frame.evidence), separators=(",", ":")),
            )
        lines.append("$$$$")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _sdf_bond_order(bond: BondTopology) -> int:
        if bond.bond_kind == "aromatic" or bond.bond_order == 1.5:
            return 4
        rounded = int(round(bond.bond_order))
        return rounded if rounded in {1, 2, 3} else 1

    @staticmethod
    def _append_sdf_property(lines: list[str], name: str, value: str) -> None:
        lines.extend((f">  <{name}>", value, ""))
