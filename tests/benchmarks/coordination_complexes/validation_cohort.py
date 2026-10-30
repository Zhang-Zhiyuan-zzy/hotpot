"""Current-code semantic cohort for the Plan-17 force-field validation.

The cohort deliberately contains graph semantics only.  It is reconstructed
once from the tracked 187-extractant source and current CBond inference, then
shared by every validation route without retaining CBond runtime objects or
coordinates.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

from .configuration import STANDARD_SETTINGS, BenchmarkSettings

SCHEMA_VERSION = 2
NATIVE_GRAPH_SCHEMA_VERSION = 2
EXPECTED_SOURCE_COUNT = 187
EXPECTED_COMPLEX_COUNT = 182
EXPECTED_NO_CBOND_INDICES = (139, 182, 185, 186, 187)
CANONICAL_SOURCE_SHA256 = (
    "e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_SOURCE_PATH = REPOSITORY_ROOT / "molecules" / "extractant" / "extractants.smi"

CBondBuilder = Callable[[int, str, str, BenchmarkSettings], object | None]


def canonical_digest(value: object) -> str:
    """Return the SHA-256 of normalized JSON scientific content."""

    content = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _hydrogen_state_document(state: object | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "implicit_hydrogens": state.implicit_hydrogens,
        "bracket_explicit_hydrogens": state.bracket_explicit_hydrogens,
        "no_implicit": state.no_implicit,
    }


def _empty_stereo_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "records": [],
        "atom_transport": [],
        "bond_transport": [],
    }


def _stereo_document(payload: object | None) -> dict[str, object]:
    if payload is None:
        return _empty_stereo_document()

    records = []
    for record in payload.records:
        group = record.group
        records.append(
            {
                "anchor": {
                    "kind": _enum_value(record.anchor.kind),
                    "atom_rows": [int(row) for row in record.anchor.atom_indices],
                },
                "carriers": [
                    {
                        "kind": _enum_value(carrier.kind),
                        "atom_row": (
                            None if carrier.atom is None else int(carrier.atom)
                        ),
                    }
                    for carrier in record.carriers
                ],
                "descriptor": record.descriptor,
                "group": (
                    None
                    if group is None
                    else {
                        "scope": group.scope,
                        "kind": _enum_value(group.kind),
                        "id": int(group.id),
                    }
                ),
                "source_namespace": record.source_namespace,
                "source_code": record.source_code,
            }
        )

    atom_transport = [
        {
            "atom_row": int(fact.atom),
            "isotope": fact.isotope,
            "radical_electrons": fact.radical_electrons,
            "current_hydrogens": _hydrogen_state_document(fact.current_hydrogens),
            "source_hydrogens": _hydrogen_state_document(fact.source_hydrogens),
            "source_namespace": fact.source_namespace,
        }
        for fact in payload.atom_facts
    ]
    bond_transport = [
        {
            "atom_rows": [int(row) for row in fact.atom_indices],
            "display_direction": fact.display_direction,
        }
        for fact in payload.bond_facts
    ]
    return {
        "schema_version": int(payload.schema_version),
        "records": records,
        "atom_transport": atom_transport,
        "bond_transport": bond_transport,
    }


def _cohort_atom_ids(molecule: object) -> tuple[int, ...]:
    atom_ids = tuple(int(atom.id) for atom in molecule.atoms)
    if len(set(atom_ids)) == len(atom_ids) and all(
        atom_id >= 0 for atom_id in atom_ids
    ):
        return atom_ids
    return tuple(range(len(atom_ids)))


def _graph_document(
    molecule: object,
    atom_ids: tuple[int, ...] | None = None,
    *,
    atom_count: int | None = None,
) -> dict[str, object]:
    import numpy as np

    from hotpot import Molecule
    from hotpot.cheminfo.structure._fields import FIELD_INDEX

    if isinstance(molecule, Molecule):
        record, _ = molecule._capture_factual_record(include_frames=False)
        count = record.atom_count if atom_count is None else atom_count
        values = np.frombuffer(record.atom_values, dtype="<f8").reshape(
            record.atom_count, len(FIELD_INDEX)
        )[:count]
        presence = np.frombuffer(record.atom_presence, dtype="<u4")[:count]
        if not np.all(presence & (1 << FIELD_INDEX["implicit_hydrogens"])):
            raise ValueError("graph serialization requires present implicit-H facts")
        atom_ids = (
            tuple(int(value) for value in values[:, FIELD_INDEX["id"]])
            if atom_ids is None else atom_ids
        )
        atom_documents = [
            {
                "row": row,
                "atom_id": atom_ids[row],
                "atomic_number": int(value[FIELD_INDEX["atomic_number"]]),
                "formal_charge": int(value[FIELD_INDEX["formal_charge"]]),
                "is_aromatic": bool(value[FIELD_INDEX["is_aromatic"]]),
                "implicit_hydrogens": int(value[FIELD_INDEX["implicit_hydrogens"]]),
            }
            for row, value in enumerate(values)
        ]
        bond_values = np.frombuffer(record.bond_values, dtype="<f8").reshape(-1, 3)
        bond_documents = [
            {
                "row": row,
                "begin": record.bond_endpoints[index][0],
                "end": record.bond_endpoints[index][1],
                "bond_order": float(bond_values[index, 0]),
                "bond_kind": record.bond_kinds[index],
                "is_aromatic": bool(
                    record.bond_kinds[index] == "aromatic" or record.bond_aromatic[index]
                ),
                "bond_direction": record.bond_directions[index],
            }
            for row, index in enumerate(
                index for index, partition in enumerate(record.bond_partitions)
                if partition == 0 and all(
                    endpoint < count for endpoint in record.bond_endpoints[index]
                )
            )
        ]
        return {
            "atoms": atom_documents,
            "bonds": bond_documents,
            "stereo": _stereo_document(record.stereo_payload),
        }

    atoms = tuple(molecule.atoms)[:atom_count]
    atom_ids = tuple(int(atom.id) for atom in atoms) if atom_ids is None else atom_ids
    atom_rows = {id(atom): row for row, atom in enumerate(atoms)}
    atom_documents = [
        {
            "row": row,
            "atom_id": atom_ids[row],
            "atomic_number": int(atom.atomic_number),
            "formal_charge": int(atom.formal_charge),
            "is_aromatic": bool(atom.is_aromatic),
            "implicit_hydrogens": int(atom.implicit_hydrogens),
        }
        for row, atom in enumerate(atoms)
    ]
    bond_documents = []
    selected_bonds = molecule.bonds
    if atom_count is not None:
        selected_bonds = (
            bond for bond in selected_bonds
            if id(bond.atom1) in atom_rows and id(bond.atom2) in atom_rows
        )
    for row, bond in enumerate(selected_bonds):
        bond_kind = str(_enum_value(bond.bond_kind))
        bond_documents.append(
            {
                "row": row,
                "begin": atom_rows[id(bond.atom1)],
                "end": atom_rows[id(bond.atom2)],
                "bond_order": float(bond.bond_order),
                "bond_kind": bond_kind,
                "is_aromatic": bool(bond_kind == "aromatic" or bond.is_aromatic),
                "bond_direction": bond.bond_direction,
            }
        )
    return {
        "atoms": atom_documents,
        "bonds": bond_documents,
        "stereo": _stereo_document(getattr(molecule, "_stereo_payload", None)),
    }


def _coordination_document(
    molecule: object,
    donor_indices: object,
    metal: str,
    atom_ids: tuple[int, ...] | None = None,
) -> dict[str, object]:
    atoms = tuple(molecule.atoms)
    atom_ids = tuple(int(atom.id) for atom in atoms) if atom_ids is None else atom_ids
    metal_rows = [row for row, atom in enumerate(atoms) if str(atom.symbol) == metal]
    if len(metal_rows) != 1:
        raise ValueError(
            f"CBond result must contain exactly one {metal} atom; "
            f"found {len(metal_rows)}"
        )
    metal_row = metal_rows[0]
    metal_atom = atoms[metal_row]
    donors = sorted(int(row) for row in donor_indices)
    return {
        "metal": {
            "row": metal_row,
            "atom_id": atom_ids[metal_row],
            "atomic_number": int(metal_atom.atomic_number),
            "symbol": metal,
        },
        "donor_rows": donors,
    }


def _source_cases(path: Path) -> tuple[str, tuple[tuple[int, str], ...]]:
    content = path.read_bytes()
    source_sha256 = hashlib.sha256(content).hexdigest()
    if source_sha256 != CANONICAL_SOURCE_SHA256:
        raise ValueError(
            "extractant source SHA-256 differs from the canonical corpus: "
            f"expected {CANONICAL_SOURCE_SHA256}, found {source_sha256}"
        )
    physical_lines = content.decode("utf-8").splitlines()
    if len(physical_lines) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_COUNT} physical source lines, "
            f"found {len(physical_lines)}"
        )
    if any(
        not line.strip() or line.lstrip().startswith("#") for line in physical_lines
    ):
        raise ValueError("canonical extractant source contains a blank/comment line")
    return source_sha256, tuple(
        (index, line.split(maxsplit=1)[0])
        for index, line in enumerate(physical_lines, start=1)
    )


def _current_cbond(
    index: int,
    smiles: str,
    metal: str,
    settings: BenchmarkSettings,
) -> object | None:
    """Run the maintained CBond path once for one source ligand."""

    del index
    from hotpot import read_mol
    from hotpot.cheminfo.AImodels.cbond.apply import (
        auto_build_cbond,
        get_cbond_runtime,
    )

    ligand = read_mol(smiles, fmt="smi")
    try:
        return auto_build_cbond(
            ligand,
            metal,
            threshold=settings.subsequent_cbond_threshold,
            first_threshold=settings.first_cbond_threshold,
            runtime=get_cbond_runtime("cpu"),
            return_details=True,
        )
    except ValueError as error:
        if str(error).startswith("No coordination bond exceeded"):
            return None
        raise


def build_validation_cohort(
    *,
    source_path: Path = CANONICAL_SOURCE_PATH,
    settings: BenchmarkSettings = STANDARD_SETTINGS,
    metal: str = "Am",
    cbond_builder: CBondBuilder | None = None,
) -> dict[str, object]:
    """Build the normalized 187/182 current-code semantic cohort.

    ``cbond_builder`` is an injection seam for focused tests.  The callable is
    invoked exactly once for each physical source case and returns either a
    CBond result (with ``molecule`` and ``donor_indices``) or ``None``.
    """

    from hotpot import Molecule
    from hotpot.cheminfo.operations.chemistry import _require_implicit_hydrogens

    if metal != "Am":
        raise ValueError("the Plan-17 semantic cohort is fixed to Am")
    if settings != STANDARD_SETTINGS:
        raise ValueError("the Plan-17 semantic cohort requires standard settings")
    source_sha256, source_cases = _source_cases(Path(source_path))
    builder = _current_cbond if cbond_builder is None else cbond_builder
    settings_document = dict(settings.to_manifest())
    cases = []
    positive_indices = []
    negative_indices = []

    for index, smiles in source_cases:
        seed = int(settings.seed) + index
        result = builder(index, smiles, metal, settings)
        base_case = {
            "index": index,
            "smiles": smiles,
            "seed": seed,
            "settings": dict(settings_document),
        }
        if result is None:
            negative_indices.append(index)
            cases.append(
                {
                    **base_case,
                    "status": "no_cbond",
                    "reason": "no_coordination_bond_above_threshold",
                }
            )
            continue

        if isinstance(result.molecule, Molecule):
            _require_implicit_hydrogens(result.molecule)
        atom_ids = _cohort_atom_ids(result.molecule)
        graph = _graph_document(result.molecule, atom_ids)
        coordination = _coordination_document(
            result.molecule,
            result.donor_indices,
            metal,
            atom_ids,
        )
        semantic_graph_sha256 = canonical_digest(
            {"graph": graph, "coordination": coordination}
        )
        positive_indices.append(index)
        cases.append(
            {
                **base_case,
                "status": "cbond",
                "graph": graph,
                "coordination": coordination,
                "semantic_graph_sha256": semantic_graph_sha256,
            }
        )

    if tuple(negative_indices) != EXPECTED_NO_CBOND_INDICES:
        raise ValueError(
            "current CBond partition differs from the expected no-CBond cases: "
            f"expected {EXPECTED_NO_CBOND_INDICES}, found {tuple(negative_indices)}"
        )
    if len(positive_indices) != EXPECTED_COMPLEX_COUNT:
        raise ValueError(
            f"expected {EXPECTED_COMPLEX_COUNT} CBond-positive cases, "
            f"found {len(positive_indices)}"
        )

    content: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "sha256": source_sha256,
            "physical_line_count": len(source_cases),
        },
        "protocol": {
            "metal": metal,
            "settings": settings_document,
        },
        "partition": {
            "cbond_count": len(positive_indices),
            "no_cbond_count": len(negative_indices),
            "cbond_indices": positive_indices,
            "no_cbond_indices": negative_indices,
        },
        "cases": cases,
    }
    return {**content, "canonical_sha256": canonical_digest(content)}


def validate_validation_cohort(cohort: Mapping[str, object]) -> None:
    """Validate schema, identity, and the complete fixed partition."""

    if int(cohort["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported validation cohort schema {cohort['schema_version']}"
        )
    source = cohort["source"]
    if source["sha256"] != CANONICAL_SOURCE_SHA256:
        raise ValueError("validation cohort source SHA-256 is not canonical")
    if int(source["physical_line_count"]) != EXPECTED_SOURCE_COUNT:
        raise ValueError("validation cohort source count is not 187")
    protocol = cohort["protocol"]
    if protocol["metal"] != "Am":
        raise ValueError("validation cohort metal is not Am")
    if protocol["settings"] != dict(STANDARD_SETTINGS.to_manifest()):
        raise ValueError("validation cohort settings are not standard")
    partition = cohort["partition"]
    if tuple(partition["no_cbond_indices"]) != EXPECTED_NO_CBOND_INDICES:
        raise ValueError("validation cohort no-CBond partition differs")
    expected_cbond_indices = tuple(
        index
        for index in range(1, EXPECTED_SOURCE_COUNT + 1)
        if index not in EXPECTED_NO_CBOND_INDICES
    )
    if tuple(partition["cbond_indices"]) != expected_cbond_indices:
        raise ValueError("validation cohort CBond-positive partition differs")
    if int(partition["cbond_count"]) != EXPECTED_COMPLEX_COUNT:
        raise ValueError("validation cohort CBond-positive count is not 182")
    if int(partition["no_cbond_count"]) != len(EXPECTED_NO_CBOND_INDICES):
        raise ValueError("validation cohort no-CBond count is not five")
    cases = cohort["cases"]
    if len(cases) != EXPECTED_SOURCE_COUNT:
        raise ValueError("validation cohort does not contain 187 cases")
    if [int(case["index"]) for case in cases] != list(
        range(1, EXPECTED_SOURCE_COUNT + 1)
    ):
        raise ValueError("validation cohort case order differs")

    _, source_cases = _source_cases(CANONICAL_SOURCE_PATH)
    source_smiles = dict(source_cases)

    content = dict(cohort)
    expected_digest = str(content.pop("canonical_sha256"))
    if canonical_digest(content) != expected_digest:
        raise ValueError("validation cohort canonical SHA-256 differs")

    for case in cases:
        index = int(case["index"])
        if case["smiles"] != source_smiles[index]:
            raise ValueError(f"case {index} SMILES differs from the source")
        if int(case["seed"]) != STANDARD_SETTINGS.seed + index:
            raise ValueError(f"case {index} seed differs")
        if case["settings"] != dict(STANDARD_SETTINGS.to_manifest()):
            raise ValueError(f"case {index} settings differ")
        if case["status"] != "cbond":
            if index not in EXPECTED_NO_CBOND_INDICES:
                raise ValueError(f"case {index} is unexpectedly classified as no-CBond")
            continue
        if index in EXPECTED_NO_CBOND_INDICES:
            raise ValueError(f"case {index} is unexpectedly CBond-positive")
        graph = case["graph"]
        atoms = graph["atoms"]
        bonds = graph["bonds"]
        if [int(atom["row"]) for atom in atoms] != list(range(len(atoms))):
            raise ValueError(f"case {index} atom rows are not contiguous")
        atom_ids = [int(atom["atom_id"]) for atom in atoms]
        if len(set(atom_ids)) != len(atom_ids):
            raise ValueError(f"case {index} atom IDs are not unique")
        if any(
            type(atom.get("implicit_hydrogens")) is not int
            or atom["implicit_hydrogens"] < 0
            for atom in atoms
        ):
            raise ValueError(
                f"case {index} atom implicit-hydrogen counts are invalid"
            )
        if [int(bond["row"]) for bond in bonds] != list(range(len(bonds))):
            raise ValueError(f"case {index} bond rows are not contiguous")
        bond_keys = []
        for bond in bonds:
            begin = int(bond["begin"])
            end = int(bond["end"])
            if not (0 <= begin < len(atoms) and 0 <= end < len(atoms)):
                raise ValueError(f"case {index} has an invalid bond endpoint")
            if begin == end:
                raise ValueError(f"case {index} has a self bond")
            bond_keys.append(tuple(sorted((begin, end))))
        if len(set(bond_keys)) != len(bond_keys):
            raise ValueError(f"case {index} has duplicate bond endpoints")
        coordination = case["coordination"]
        metal_row = int(coordination["metal"]["row"])
        donor_rows = tuple(int(row) for row in coordination["donor_rows"])
        if not (0 <= metal_row < len(atoms)):
            raise ValueError(f"case {index} metal row is invalid")
        if int(atoms[metal_row]["atomic_number"]) != 95:
            raise ValueError(f"case {index} metal row is not Am")
        if len(set(donor_rows)) != len(donor_rows) or any(
            row == metal_row or not 0 <= row < len(atoms) for row in donor_rows
        ):
            raise ValueError(f"case {index} donor rows are invalid")
        if any(tuple(sorted((row, metal_row))) not in bond_keys for row in donor_rows):
            raise ValueError(f"case {index} donor bond is absent")
        semantic_digest = canonical_digest(
            {
                "graph": graph,
                "coordination": coordination,
            }
        )
        if semantic_digest != case["semantic_graph_sha256"]:
            raise ValueError(f"case {index} semantic graph SHA-256 differs")


def write_validation_cohort(path: Path, cohort: Mapping[str, object]) -> None:
    """Write one validated, deterministic JSON cohort document."""

    validate_validation_cohort(cohort)
    Path(path).write_text(
        json.dumps(
            cohort,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_validation_cohort(path: Path) -> dict[str, object]:
    """Load and validate one round-tripped cohort document."""

    cohort = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_validation_cohort(cohort)
    return cohort


def derive_c1_c2_native_graph(
    case: Mapping[str, object],
) -> dict[str, object]:
    """Project one positive semantic case into C1/C2 native graph JSON.

    Atom and stereo documents are copied unchanged. Coordination bonds are
    represented as directed donor-to-metal dative edges only in this native
    generation projection because pinned RDKit requires that representation.
    The source bonds and canonical semantic digest remain explicit, so the
    projection cannot redefine the Hotpot business graph.
    """

    if case["status"] != "cbond":
        raise ValueError("C1/C2 native graphs require a CBond-positive case")
    source_bonds = deepcopy(case["graph"]["bonds"])
    metal_row = int(case["coordination"]["metal"]["row"])
    donor_rows = frozenset(int(row) for row in case["coordination"]["donor_rows"])
    generation_bonds = []
    for source_bond in source_bonds:
        bond = deepcopy(source_bond)
        endpoints = {int(bond["begin"]), int(bond["end"])}
        donor_matches = tuple(row for row in donor_rows if row in endpoints)
        if metal_row in endpoints and len(donor_matches) == 1:
            bond["begin"] = donor_matches[0]
            bond["end"] = metal_row
            bond["bond_order"] = 1.0
            bond["bond_kind"] = "dative"
            bond["is_aromatic"] = False
            bond["bond_direction"] = "atom1_to_atom2"
        generation_bonds.append(bond)
    return {
        "schema_version": NATIVE_GRAPH_SCHEMA_VERSION,
        "case_index": int(case["index"]),
        "semantic_graph_sha256": case["semantic_graph_sha256"],
        "atoms": deepcopy(case["graph"]["atoms"]),
        "bonds": generation_bonds,
        "source_bonds": source_bonds,
        "stereo": deepcopy(case["graph"]["stereo"]),
        "coordination": deepcopy(case["coordination"]),
        "generation_transform": "coordination_edges_to_donor_metal_dative",
    }


def _hydrogen_state_from_document(value: Mapping[str, object] | None):
    if value is None:
        return None
    from hotpot.cheminfo.stereo import HydrogenTransportState

    return HydrogenTransportState(
        implicit_hydrogens=value["implicit_hydrogens"],
        bracket_explicit_hydrogens=value["bracket_explicit_hydrogens"],
        no_implicit=value["no_implicit"],
    )


def _stereo_from_document(value: Mapping[str, object]):
    from hotpot.cheminfo.stereo import (
        AtomTransportRecord,
        BondTransportRecord,
        StereoAnchor,
        StereoAnchorKind,
        StereoCarrier,
        StereoCarrierKind,
        StereoGroupKind,
        StereoGroupRef,
        StereoPayload,
        StereoRecord,
    )

    records = []
    for record in value["records"]:
        group = record["group"]
        records.append(
            StereoRecord(
                anchor=StereoAnchor(
                    kind=StereoAnchorKind(record["anchor"]["kind"]),
                    atom_indices=tuple(record["anchor"]["atom_rows"]),
                ),
                carriers=tuple(
                    StereoCarrier(
                        kind=StereoCarrierKind(carrier["kind"]),
                        atom=carrier["atom_row"],
                    )
                    for carrier in record["carriers"]
                ),
                descriptor=record["descriptor"],
                group=(
                    None
                    if group is None
                    else StereoGroupRef(
                        scope=group["scope"],
                        kind=StereoGroupKind(group["kind"]),
                        id=int(group["id"]),
                    )
                ),
                source_namespace=record["source_namespace"],
                source_code=record["source_code"],
            )
        )
    return StereoPayload(
        schema_version=int(value["schema_version"]),
        records=tuple(records),
        atom_facts=tuple(
            AtomTransportRecord(
                atom=int(fact["atom_row"]),
                isotope=fact["isotope"],
                radical_electrons=fact["radical_electrons"],
                current_hydrogens=_hydrogen_state_from_document(
                    fact["current_hydrogens"]
                ),
                source_hydrogens=_hydrogen_state_from_document(
                    fact["source_hydrogens"]
                ),
                source_namespace=fact["source_namespace"],
            )
            for fact in value["atom_transport"]
        ),
        bond_facts=tuple(
            BondTransportRecord(
                atom_indices=tuple(fact["atom_rows"]),
                display_direction=fact["display_direction"],
            )
            for fact in value["bond_transport"]
        ),
    )


def semantic_case_molecule(case: Mapping[str, object]):
    """Reconstruct a Hotpot graph from one case without invoking CBond."""

    if case["status"] != "cbond":
        raise ValueError("a molecule can only be built from a CBond-positive case")
    from dataclasses import replace

    import numpy as np

    from hotpot.cheminfo.core import Molecule
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX

    graph = case["graph"]
    molecule = Molecule()
    for atom in graph["atoms"]:
        if int(atom["row"]) != len(molecule.atoms):
            raise ValueError("semantic atom rows are not contiguous")
        molecule._create_atom(
            id=int(atom["atom_id"]),
            atomic_number=int(atom["atomic_number"]),
            formal_charge=int(atom["formal_charge"]),
            implicit_hydrogens=int(atom["implicit_hydrogens"]),
        )
    for bond in graph["bonds"]:
        molecule._add_bond(
            int(bond["begin"]),
            int(bond["end"]),
            bond_order=float(bond["bond_order"]),
            bond_kind=str(bond["bond_kind"]),
            bond_direction=bond["bond_direction"],
        )
    molecule._update_topology()
    molecule.charge = molecule.sum_atoms_charge
    record, metadata = molecule._capture_factual_record()
    values = np.frombuffer(record.atom_values, dtype="<f8").reshape(record.atom_count, 19).copy()
    presence = np.frombuffer(record.atom_presence, dtype="<u4").copy()
    sources = np.frombuffer(record.atom_sources, dtype="u1").reshape(record.atom_count, 19).copy()
    field = FIELD_INDEX["is_aromatic"]
    values[:, field] = [bool(atom["is_aromatic"]) for atom in graph["atoms"]]
    presence |= 1 << field
    sources[:, field] = ATOM_SOURCE_CODES["explicit"]
    return Molecule._build_molecule(replace(
        record, atom_values=values.tobytes(), atom_presence=presence.tobytes(),
        atom_sources=sources.tobytes(),
        bond_aromatic=tuple(bool(bond["is_aromatic"]) for bond in graph["bonds"]),
        stereo_payload=_stereo_from_document(graph["stereo"]),
    ), bond_metadata=metadata)


__all__ = (
    "CANONICAL_SOURCE_PATH",
    "CANONICAL_SOURCE_SHA256",
    "EXPECTED_COMPLEX_COUNT",
    "EXPECTED_NO_CBOND_INDICES",
    "EXPECTED_SOURCE_COUNT",
    "build_validation_cohort",
    "canonical_digest",
    "derive_c1_c2_native_graph",
    "load_validation_cohort",
    "semantic_case_molecule",
    "validate_validation_cohort",
    "write_validation_cohort",
)
