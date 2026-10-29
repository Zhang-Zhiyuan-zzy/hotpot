"""
python v3.9.0
@Project: hotpot
@File   : rdconvert
@Auther : Zhiyuan Zhang
@Data   : 2024/12/18
@Time   : 8:56
"""
from uuid import uuid4
from dataclasses import replace

import numpy as np

import rdkit
from rdkit import Chem

from .elements import elements
from .structure._builder import _freeze_value
from .structure._commit import _FactTransaction
from .structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS

from .stereo import (
    AtomTransportRecord,
    BondTransportRecord,
    HydrogenTransportState,
    StereoAnchor,
    StereoAnchorKind,
    StereoCarrier,
    StereoCarrierKind,
    StereoGroupKind,
    StereoGroupRef,
    StereoPayload,
    StereoRecord,
    _remap_stereo_payload,
    get_stereo_payload,
    set_stereo_payload,
)

_hphyb2rdkithyb = {
    -1: rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
    0: rdkit.Chem.rdchem.HybridizationType.S,
    1: rdkit.Chem.rdchem.HybridizationType.SP,
    2: rdkit.Chem.rdchem.HybridizationType.SP2,
    3: rdkit.Chem.rdchem.HybridizationType.SP3,
    4: rdkit.Chem.rdchem.HybridizationType.SP2D,
    5: rdkit.Chem.rdchem.HybridizationType.SP3D,
    6: rdkit.Chem.rdchem.HybridizationType.SP3D2,
    7: rdkit.Chem.rdchem.HybridizationType.OTHER
}

_RDKIT_DATIVE_BOND_TYPES = frozenset(
    bond_type for bond_type in (
        getattr(Chem.BondType, "DATIVE", None),
        getattr(Chem.BondType, "DATIVEONE", None),
        getattr(Chem.BondType, "DATIVEL", None),
        getattr(Chem.BondType, "DATIVER", None),
    )
    if bond_type is not None
)

_RDKIT_STEREO_NAMESPACE = "rdkit-2026.09.1"
_RDKIT_GROUP_KINDS = {
    Chem.StereoGroupType.STEREO_ABSOLUTE: StereoGroupKind.ABS,
    Chem.StereoGroupType.STEREO_AND: StereoGroupKind.AND,
    Chem.StereoGroupType.STEREO_OR: StereoGroupKind.OR,
}
_HOTPOT_GROUP_KINDS = {value: key for key, value in _RDKIT_GROUP_KINDS.items()}
_RDKIT_TETRAHEDRAL_CODES = frozenset(
    {"CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"}
)
_IMPLICIT_H_TOKEN = (StereoCarrierKind.IMPLICIT_H, None)


def _new_rdkit_scope():
    return f"rdkit:{uuid4().hex}"


def _enum_name(value):
    return str(value)


def _enum_value(enum_type, name, category):
    value = enum_type.names.get(name)
    if value is None:
        raise ValueError(f"Unsupported RDKit {category} code: {name}")
    return value


def _atom_carriers(atom, implicit_hydrogens, chiral_tag):
    if chiral_tag == "CHI_UNSPECIFIED":
        return ()
    carriers = tuple(
        StereoCarrier(
            StereoCarrierKind.ATOM,
            bond.GetOtherAtomIdx(atom.GetIdx()),
        )
        for bond in atom.GetBonds()
    )
    if chiral_tag in _RDKIT_TETRAHEDRAL_CODES and implicit_hydrogens == 1:
        carriers += (StereoCarrier(StereoCarrierKind.IMPLICIT_H),)
    return carriers


def _add_group_member(mapping, key, group_ref, category):
    if key in mapping:
        raise ValueError(
            f"RDKit source contains conflicting {category} stereo groups at "
            f"index {key}"
        )
    mapping[key] = group_ref


def _extract_stereo_payload(rdmol):
    scope = _new_rdkit_scope()
    hydrogen_view = Chem.Mol(rdmol)
    hydrogen_view.UpdatePropertyCache(strict=False)
    atom_groups = {}
    bond_groups = {}
    for group in rdmol.GetStereoGroups():
        group_kind = _RDKIT_GROUP_KINDS.get(group.GetGroupType())
        if group_kind is None:
            raise ValueError(
                "Unsupported RDKit enhanced stereo group type: "
                f"{group.GetGroupType()}"
            )
        group_ref = StereoGroupRef(
            scope=scope,
            kind=group_kind,
            id=group.GetReadId(),
        )
        for atom in group.GetAtoms():
            _add_group_member(
                atom_groups,
                atom.GetIdx(),
                group_ref,
                "atom",
            )
        for bond in group.GetBonds():
            _add_group_member(
                bond_groups,
                bond.GetIdx(),
                group_ref,
                "bond",
            )

    records = []
    atom_facts = []
    for atom in rdmol.GetAtoms():
        atom_index = atom.GetIdx()
        chiral_tag = _enum_name(atom.GetChiralTag())
        implicit_hydrogens = (
            hydrogen_view.GetAtomWithIdx(atom_index).GetNumImplicitHs()
        )
        records.append(
            StereoRecord(
                anchor=StereoAnchor(StereoAnchorKind.ATOM, (atom_index,)),
                carriers=_atom_carriers(
                    atom,
                    implicit_hydrogens,
                    chiral_tag,
                ),
                descriptor=chiral_tag,
                group=atom_groups.get(atom_index),
                source_namespace=_RDKIT_STEREO_NAMESPACE,
                source_code=chiral_tag,
            )
        )
        hydrogens = HydrogenTransportState(
            implicit_hydrogens=implicit_hydrogens,
            bracket_explicit_hydrogens=atom.GetNumExplicitHs(),
            no_implicit=atom.GetNoImplicit(),
        )
        atom_facts.append(
            AtomTransportRecord(
                atom=atom_index,
                isotope=atom.GetIsotope(),
                radical_electrons=atom.GetNumRadicalElectrons(),
                current_hydrogens=hydrogens,
                source_hydrogens=hydrogens,
                source_namespace=_RDKIT_STEREO_NAMESPACE,
            )
        )

    bond_facts = []
    for bond in rdmol.GetBonds():
        stereo_code = _enum_name(bond.GetStereo())
        records.append(
            StereoRecord(
                anchor=StereoAnchor(
                    StereoAnchorKind.BOND,
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                ),
                carriers=tuple(
                    StereoCarrier(StereoCarrierKind.ATOM, atom_index)
                    for atom_index in bond.GetStereoAtoms()
                ),
                descriptor=stereo_code,
                group=bond_groups.get(bond.GetIdx()),
                source_namespace=_RDKIT_STEREO_NAMESPACE,
                source_code=stereo_code,
            )
        )
        bond_facts.append(
            BondTransportRecord(
                atom_indices=(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                ),
                display_direction=_enum_name(bond.GetBondDir()),
            )
        )

    return StereoPayload(
        schema_version=1,
        records=tuple(records),
        atom_facts=tuple(atom_facts),
        bond_facts=tuple(bond_facts),
    )


def _payload_atom_facts(payload, atom_count=None):
    facts = {}
    for fact in payload.atom_facts:
        if fact.source_namespace != _RDKIT_STEREO_NAMESPACE:
            raise ValueError(
                "RDKit export cannot represent atom transport namespace: "
                f"{fact.source_namespace}"
            )
        if atom_count is not None and not 0 <= fact.atom < atom_count:
            raise ValueError(
                f"RDKit atom transport references missing atom {fact.atom}"
            )
        if fact.atom in facts:
            raise ValueError(
                f"Duplicate RDKit atom transport record for atom {fact.atom}"
            )
        facts[fact.atom] = fact
    return facts


def _validate_payload_references(payload, atom_count, bond_indices):
    bond_keys = {frozenset(indices) for indices in bond_indices}
    stereo_anchors = set()
    for record in payload.records:
        if record.anchor.kind is StereoAnchorKind.ATOM:
            if len(record.anchor.atom_indices) != 1:
                raise ValueError("RDKit atom stereo anchor requires one atom")
            atom_index = record.anchor.atom_indices[0]
            if not 0 <= atom_index < atom_count:
                raise ValueError(
                    f"RDKit atom stereo references missing atom {atom_index}"
                )
            anchor_key = (StereoAnchorKind.ATOM, atom_index)
        elif record.anchor.kind is StereoAnchorKind.BOND:
            if len(record.anchor.atom_indices) != 2:
                raise ValueError("RDKit bond stereo anchor requires two atoms")
            first, second = record.anchor.atom_indices
            if not 0 <= first < atom_count or not 0 <= second < atom_count:
                raise ValueError(
                    "RDKit bond stereo anchor references a missing atom"
                )
            bond_key = frozenset((first, second))
            if bond_key not in bond_keys:
                raise ValueError(
                    "RDKit stereo payload references a missing bond: "
                    f"{record.anchor.atom_indices}"
                )
            anchor_key = (StereoAnchorKind.BOND, bond_key)
        else:
            raise ValueError(
                "RDKit export cannot represent stereo anchor kind: "
                f"{record.anchor.kind.value}"
            )
        if anchor_key in stereo_anchors:
            raise ValueError(
                f"Duplicate RDKit stereo anchor: {record.anchor.atom_indices}"
            )
        stereo_anchors.add(anchor_key)

        for carrier in record.carriers:
            if carrier.kind is StereoCarrierKind.ATOM and (
                carrier.atom is None or not 0 <= carrier.atom < atom_count
            ):
                raise ValueError("RDKit stereo carrier references a missing atom")
        if record.group is not None:
            if record.group.kind not in _HOTPOT_GROUP_KINDS:
                raise ValueError(
                    "RDKit cannot represent enhanced stereo group kind: "
                    f"{record.group.kind}"
                )
            if not 0 <= record.group.id <= 0xFFFFFFFF:
                raise ValueError(
                    "RDKit enhanced stereo group id is out of range: "
                    f"{record.group.id}"
                )

    bond_facts = set()
    for fact in payload.bond_facts:
        if len(fact.atom_indices) != 2:
            raise ValueError("RDKit bond transport requires two atoms")
        first, second = fact.atom_indices
        if not 0 <= first < atom_count or not 0 <= second < atom_count:
            raise ValueError("RDKit bond transport references a missing atom")
        bond_key = frozenset((first, second))
        if bond_key not in bond_keys:
            raise ValueError(
                "RDKit bond transport references a missing bond: "
                f"{fact.atom_indices}"
            )
        if bond_key in bond_facts:
            raise ValueError(
                f"Duplicate RDKit bond transport record: {fact.atom_indices}"
            )
        bond_facts.add(bond_key)


def _apply_atom_fact(rda, fact):
    if fact.isotope is not None:
        rda.SetIsotope(fact.isotope)
    if fact.radical_electrons is not None:
        rda.SetNumRadicalElectrons(fact.radical_electrons)
    hydrogens = fact.current_hydrogens
    if hydrogens is not None:
        if hydrogens.bracket_explicit_hydrogens is not None:
            rda.SetNumExplicitHs(hydrogens.bracket_explicit_hydrogens)
        if hydrogens.no_implicit is not None:
            rda.SetNoImplicit(hydrogens.no_implicit)


def _bond_for_anchor(rdmol, atom_indices):
    first, second = atom_indices
    bond = rdmol.GetBondBetweenAtoms(first, second)
    if bond is None:
        raise ValueError(
            "RDKit stereo payload references a missing bond: "
            f"{atom_indices}"
        )
    return bond


def _transported_atom_carriers(record):
    tokens = []
    for carrier in record.carriers:
        if carrier.kind is StereoCarrierKind.ATOM:
            tokens.append((carrier.kind, carrier.atom))
        elif carrier.kind is StereoCarrierKind.IMPLICIT_H:
            tokens.append(_IMPLICIT_H_TOKEN)
        else:
            raise ValueError(
                "RDKit tetrahedral atom stereo cannot represent lone-pair "
                "or vacancy carriers"
            )
    if _IMPLICIT_H_TOKEN in tokens and tokens[-1] != _IMPLICIT_H_TOKEN:
        raise ValueError(
            "RDKit implicit-H stereo carrier must be the final carrier"
        )
    if len(set(tokens)) != len(tokens):
        raise ValueError("RDKit atom stereo carriers contain duplicates")
    return tuple(tokens)


def _native_atom_carriers(rdmol, atom):
    bracket_explicit_hydrogens = atom.GetNumExplicitHs()
    implicit_hydrogens = atom.GetNumImplicitHs()
    if bracket_explicit_hydrogens > 1 or implicit_hydrogens > 1:
        raise ValueError(
            "RDKit atom stereo cannot represent multiple virtual-H carriers"
        )
    if bracket_explicit_hydrogens and implicit_hydrogens:
        raise ValueError(
            "RDKit atom stereo has conflicting bracket-explicit and implicit "
            "hydrogen references"
        )
    if (
        atom.GetDegree()
        + bracket_explicit_hydrogens
        + implicit_hydrogens
        != 4
    ):
        raise ValueError(
            "RDKit tetrahedral atom stereo requires exactly four graph or "
            "virtual neighbors"
        )
    tokens = tuple(
        (StereoCarrierKind.ATOM, bond.GetOtherAtomIdx(atom.GetIdx()))
        for bond in atom.GetBonds()
    )
    if implicit_hydrogens == 1:
        tokens += (_IMPLICIT_H_TOKEN,)
    return tokens


def _odd_permutation(source, target):
    if len(source) != len(target):
        raise ValueError(
            "RDKit atom stereo carrier count does not match target neighbor "
            "order"
        )
    if set(source) != set(target):
        raise ValueError(
            "RDKit atom stereo carriers are not a permutation of target "
            "neighbors"
        )
    positions = {token: index for index, token in enumerate(target)}
    permutation = tuple(positions[token] for token in source)
    inversions = sum(
        permutation[first] > permutation[second]
        for first in range(len(permutation))
        for second in range(first + 1, len(permutation))
    )
    return inversions % 2 == 1


def _apply_atom_stereo(rdmol, atom, record):
    chiral_tag = _enum_value(
        Chem.ChiralType,
        record.source_code,
        "chiral-tag",
    )
    if record.source_code == "CHI_UNSPECIFIED":
        if record.carriers:
            raise ValueError(
                "CHI_UNSPECIFIED atom stereo cannot carry an ordered sequence"
            )
        atom.SetChiralTag(chiral_tag)
        return
    if record.source_code not in _RDKIT_TETRAHEDRAL_CODES:
        raise ValueError(
            "RDKit atom stereo source_code requires an unfrozen permutation "
            f"property: {record.source_code}"
        )

    transported = _transported_atom_carriers(record)
    native = _native_atom_carriers(rdmol, atom)
    if len(native) not in (3, 4) or len(transported) not in (3, 4):
        raise ValueError(
            "RDKit tetrahedral atom stereo requires three or four carriers"
        )
    if _odd_permutation(transported, native):
        chiral_tag = (
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW
            else Chem.ChiralType.CHI_TETRAHEDRAL_CW
        )
    atom.SetChiralTag(chiral_tag)


def _apply_bond_stereo(rdmol, bond, record):
    carrier_atoms = []
    for carrier in record.carriers:
        if carrier.kind is not StereoCarrierKind.ATOM:
            raise ValueError("RDKit bond stereo requires atom carriers")
        carrier_atoms.append(carrier.atom)
    if carrier_atoms and len(carrier_atoms) != 2:
        raise ValueError(
            "RDKit bond stereo requires exactly two atom carriers"
        )
    if len(set(carrier_atoms)) != len(carrier_atoms):
        raise ValueError("RDKit bond stereo carriers contain duplicates")

    first, second = record.anchor.atom_indices
    anchor_reversed = bond.GetBeginAtomIdx() != first
    if anchor_reversed and record.source_code in (
        "STEREOATROPCW",
        "STEREOATROPCCW",
    ):
        raise ValueError(
            "RDKit atropisomer stereo has no frozen reversed-anchor "
            "transformation"
        )
    if carrier_atoms:
        if (
            carrier_atoms[0] in record.anchor.atom_indices
            or rdmol.GetBondBetweenAtoms(first, carrier_atoms[0]) is None
            or carrier_atoms[1] in record.anchor.atom_indices
            or rdmol.GetBondBetweenAtoms(second, carrier_atoms[1]) is None
        ):
            raise ValueError(
                "RDKit bond stereo carriers do not match ordered anchor sides"
            )
        if anchor_reversed:
            carrier_atoms.reverse()
        try:
            bond.SetStereoAtoms(*carrier_atoms)
        except Exception as error:
            raise ValueError(
                f"RDKit rejected bond stereo carriers: {error}"
            ) from error
    bond.SetStereo(
        _enum_value(Chem.BondStereo, record.source_code, "bond-stereo")
    )


def _display_direction_for_target(bond, fact):
    direction = fact.display_direction
    if direction is None or bond.GetBeginAtomIdx() == fact.atom_indices[0]:
        return direction
    if direction in ("NONE", "EITHERDOUBLE"):
        return direction
    if direction == "ENDDOWNRIGHT":
        return "ENDUPRIGHT"
    if direction == "ENDUPRIGHT":
        return "ENDDOWNRIGHT"
    raise ValueError(
        "RDKit cannot preserve endpoint-anchored bond display direction "
        f"{direction} when bond {fact.atom_indices} is reversed"
    )


def _apply_stereo_payload(rdmol, payload):
    if payload.records:
        rdmol.UpdatePropertyCache(strict=False)
    groups = {}
    for record in payload.records:
        if record.source_namespace != _RDKIT_STEREO_NAMESPACE:
            raise ValueError(
                "RDKit export cannot represent stereo namespace: "
                f"{record.source_namespace}"
            )
        if record.anchor.kind is StereoAnchorKind.ATOM:
            atom = rdmol.GetAtomWithIdx(record.anchor.atom_indices[0])
            _apply_atom_stereo(rdmol, atom, record)
            group_item = ("atom", atom.GetIdx())
        elif record.anchor.kind is StereoAnchorKind.BOND:
            bond = _bond_for_anchor(rdmol, record.anchor.atom_indices)
            _apply_bond_stereo(rdmol, bond, record)
            group_item = ("bond", bond.GetIdx())
        else:
            raise ValueError(
                "RDKit export cannot represent stereo anchor kind: "
                f"{record.anchor.kind.value}"
            )
        if record.group is not None:
            groups.setdefault(record.group, []).append(group_item)

    seen_bond_facts = set()
    for fact in payload.bond_facts:
        bond_key = frozenset(fact.atom_indices)
        if bond_key in seen_bond_facts:
            raise ValueError(
                f"Duplicate RDKit bond transport record: {fact.atom_indices}"
            )
        seen_bond_facts.add(bond_key)
        bond = _bond_for_anchor(rdmol, fact.atom_indices)
        display_direction = _display_direction_for_target(bond, fact)
        if display_direction is None:
            continue
        bond.SetBondDir(
            _enum_value(
                Chem.BondDir,
                display_direction,
                "bond-direction",
            )
        )

    stereo_groups = []
    for group_ref, items in groups.items():
        atom_ids = [index for kind, index in items if kind == "atom"]
        bond_ids = [index for kind, index in items if kind == "bond"]
        stereo_groups.append(
            Chem.CreateStereoGroup(
                _HOTPOT_GROUP_KINDS[group_ref.kind],
                rdmol,
                atom_ids,
                bond_ids,
                group_ref.id,
            )
        )
    rdmol.SetStereoGroups(stereo_groups)


def _verify_current_implicit_hydrogens(rdmol, atom_facts):
    if not atom_facts:
        return
    rdmol.UpdatePropertyCache(strict=False)
    for atom_index, fact in atom_facts.items():
        hydrogens = fact.current_hydrogens
        if (
            hydrogens is not None
            and hydrogens.implicit_hydrogens is not None
            and rdmol.GetAtomWithIdx(atom_index).GetNumImplicitHs()
            != hydrogens.implicit_hydrogens
        ):
            raise ValueError(
                "RDKit public API cannot reproduce current implicit hydrogen "
                f"count for atom {atom_index}: requested "
                f"{hydrogens.implicit_hydrogens}, derived "
                f"{rdmol.GetAtomWithIdx(atom_index).GetNumImplicitHs()}"
            )


def _rdkit_bond_type(bond):
    kind_to_type = {
        "zero": Chem.BondType.ZERO,
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
    }
    if bond.bond_kind.value == "dative":
        raw_type_name = bond.bond_source_metadata.get("raw_bond_type")
        raw_type = (
            getattr(Chem.BondType, raw_type_name, None)
            if raw_type_name is not None
            else None
        )
        if raw_type in _RDKIT_DATIVE_BOND_TYPES:
            return raw_type
        return Chem.BondType.DATIVE
    if bond.bond_kind.value == "aromatic" or bond.is_aromatic:
        return Chem.BondType.AROMATIC
    if bond.bond_kind.value in kind_to_type:
        return kind_to_type[bond.bond_kind.value]
    return Chem.BondType.values[bond.bond_order]


def _rdkit_bond_kind(bond_type, is_aromatic):
    if is_aromatic or bond_type == Chem.BondType.AROMATIC:
        return "aromatic"
    if bond_type == Chem.BondType.ZERO:
        return "zero"
    if bond_type in _RDKIT_DATIVE_BOND_TYPES:
        return "dative"
    return {
        Chem.BondType.SINGLE: "single",
        Chem.BondType.DOUBLE: "double",
        Chem.BondType.TRIPLE: "triple",
    }.get(bond_type, "unknown")


def to_rdmol(mol, kekulize: bool = True, sanitize: bool = False):
    rdmol = Chem.RWMol()
    payload = get_stereo_payload(mol)
    atoms = mol.atoms
    bonds = mol.bonds
    _validate_payload_references(
        payload,
        len(atoms),
        tuple((bond.atom1.idx, bond.atom2.idx) for bond in bonds),
    )
    atom_facts = _payload_atom_facts(payload, len(atoms))

    # Create atoms in the molecule and set their properties
    row_to_idx = {}
    for i, atom in enumerate(atoms):

        rda = Chem.Atom(atom.atomic_number)
        rda.SetFormalCharge(atom.formal_charge)
        atom_fact = atom_facts.get(i)
        if atom_fact is None:
            rda.SetNumExplicitHs(atom.implicit_hydrogens)
        else:
            _apply_atom_fact(rda, atom_fact)
        rda.SetDoubleProp('_GasteigerCharge', atom.partial_charge)
        rda.SetIsAromatic(atom.is_aromatic)
        rda.SetHybridization(_hphyb2rdkithyb[atom.hyb])

        idx = rdmol.AddAtom(rda)
        row_to_idx[i] = idx

    conf = Chem.Conformer(len(atoms))
    for i, (x, y, z) in enumerate(mol.coordinates):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    rdmol.AddConformer(conf)

    for bond in bonds:
        begin_atom_idx, end_atom_idx = bond.atom1.idx, bond.atom2.idx
        if bond.bond_kind.value == "dative":
            if bond.bond_direction is None:
                raise ValueError(
                    "RDKit dative-bond export requires a bond direction"
                )
            if bond.bond_direction == "atom2_to_atom1":
                begin_atom_idx, end_atom_idx = end_atom_idx, begin_atom_idx
            elif bond.bond_direction != "atom1_to_atom2":
                raise ValueError(
                    f"Unsupported dative-bond direction: {bond.bond_direction}"
                )
        rdmol.AddBond(
            row_to_idx[begin_atom_idx],
            row_to_idx[end_atom_idx],
            _rdkit_bond_type(bond),
        )

    rdmol = rdmol.GetMol()
    if kekulize:
        Chem.Kekulize(rdmol)

    if sanitize:
        Chem.SanitizeMol(rdmol)

    rdmol = Chem.RWMol(rdmol)
    export_payload = _remap_stereo_payload(payload, row_to_idx)
    _apply_stereo_payload(rdmol, export_payload)
    _verify_current_implicit_hydrogens(
        rdmol,
        _payload_atom_facts(export_payload, len(atoms)),
    )

    return rdmol.GetMol()


def from_rdmol(rdmol, mol):
    """Populate a Hotpot molecule from an RDKit molecule."""
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["rdkit"]):
        if mol._atoms or mol._bonds or mol._hided_metal_bonds or mol._hided_covalent_bonds:
            raise ValueError("Factual import requires an empty Atom and complete-Bond target")
        initial = mol._capture_factual_record()[0]
        rdmol = Chem.Mol(rdmol)
        stereo_payload = _extract_stereo_payload(rdmol)
        source_bond_metadata = {
            bond.GetIdx(): {
                "raw_bond_type": str(bond.GetBondType()),
                "raw_bond_type_code": int(bond.GetBondType()),
                "is_aromatic": bool(bond.GetIsAromatic()),
                "is_conjugated": bool(bond.GetIsConjugated()),
                "stereo": str(bond.GetStereo()),
                "bond_dir": str(bond.GetBondDir()),
            }
            for bond in rdmol.GetBonds()
        }
        source_bond_types = {
            bond.GetIdx(): bond.GetBondType()
            for bond in rdmol.GetBonds()
        }
        Chem.Kekulize(rdmol)
        conformer = rdmol.GetConformer() if rdmol.GetNumConformers() else None

        atoms = tuple(rdmol.GetAtoms())
        idx_to_row = {atom.GetIdx(): row for row, atom in enumerate(atoms)}
        values = np.zeros((len(atoms), 19), dtype="<f8")
        presence = np.zeros(len(atoms), dtype="<u4")
        sources = np.zeros((len(atoms), 19), dtype="u1")
        for row, atom in enumerate(atoms):
            values[row, 0] = atom.GetAtomicNum()
            values[row, 1:7] = elements.electron_configs[atom.GetAtomicNum()]
            values[row, 7] = atom.GetFormalCharge()
            values[row, 9] = atom.GetIsAromatic()
            values[row, 13] = atom.GetTotalValence()
            values[row, 14] = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
            supplied = [0, 7, 9, 13, 14]
            if atom.HasProp('_GasteigerCharge'):
                values[row, 8] = atom.GetDoubleProp('_GasteigerCharge')
                supplied.append(8)
            if conformer is not None:
                values[row, 10:13] = tuple(conformer.GetAtomPosition(atom.GetIdx()))
                supplied.extend((10, 11, 12))
            presence[row] = sum(FIELD_PRESENCE_BITS[index] for index in (*supplied, 1, 2, 3, 4, 5, 6))
            sources[row, supplied] = ATOM_SOURCE_CODES["rdkit"]
            sources[row, 1:7] = ATOM_SOURCE_CODES["electron_configuration"]
        bonds = tuple(rdmol.GetBonds())
        raw_bonds = np.zeros((len(bonds), 3), dtype="<f8")
        raw_bonds[:, 0] = [bond.GetBondTypeAsDouble() for bond in bonds]
        record = replace(
            initial, atom_count=len(atoms), atom_values=values.tobytes(),
            atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
            bond_endpoints=tuple((idx_to_row[bond.GetBeginAtomIdx()], idx_to_row[bond.GetEndAtomIdx()]) for bond in bonds),
            bond_values=raw_bonds.tobytes(), bond_partitions=(0,) * len(bonds),
            bond_kinds=tuple(_rdkit_bond_kind(source_bond_types[bond.GetIdx()],
                             source_bond_metadata[bond.GetIdx()]["is_aromatic"]) for bond in bonds),
            bond_directions=tuple("atom1_to_atom2" if source_bond_types[bond.GetIdx()] in _RDKIT_DATIVE_BOND_TYPES
                                  else None for bond in bonds),
            bond_sources=("rdkit",) * len(bonds),
            bond_metadata=tuple(_freeze_value(source_bond_metadata[bond.GetIdx()]) for bond in bonds),
            charge=(True, Chem.GetFormalCharge(rdmol), "rdkit"),
            stereo_payload=_remap_stereo_payload(stereo_payload, idx_to_row),
            legacy_frame_values=(("none",),) * 9,
        )
        return mol._install_molecule(record)
