"""RDKit translation of the existing immutable stereo transport records."""

from uuid import uuid4
from ...structure.stereo import (
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
)

_RDKIT_STEREO_NAMESPACE = "rdkit-2026.09.1"
_RDKIT_TETRAHEDRAL_CODES = frozenset({"CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"})
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
    from rdkit import Chem
    _RDKIT_GROUP_KINDS = {
        Chem.StereoGroupType.STEREO_ABSOLUTE: StereoGroupKind.ABS,
        Chem.StereoGroupType.STEREO_AND: StereoGroupKind.AND,
        Chem.StereoGroupType.STEREO_OR: StereoGroupKind.OR,
    }

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
        source_only = (fact.source_namespace in (None, "openbabel")
                       and fact.radical_electrons is None
                       and fact.current_hydrogens is None)
        if fact.source_namespace == "openbabel" and fact.source_hydrogens is not None:
            source = fact.source_hydrogens
            source_only = (source_only and source.bracket_explicit_hydrogens is None
                           and source.no_implicit is None)
        if fact.source_namespace != _RDKIT_STEREO_NAMESPACE and not source_only:
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
    _HOTPOT_GROUP_KINDS = (StereoGroupKind.ABS, StereoGroupKind.AND, StereoGroupKind.OR)

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


def _native_atom_carriers(rdmol, atom, *, lone_pair=False):
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
        and not (lone_pair and atom.GetDegree() == 3
                 and not bracket_explicit_hydrogens and not implicit_hydrogens)
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
    from rdkit import Chem

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
    native = _native_atom_carriers(rdmol, atom, lone_pair=len(transported) == 3)
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
    from rdkit import Chem

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


def _apply_native_ob_stereo(rdmol, record):
    from rdkit import Chem
    from ..openbabel.stereo_transport import _native_code

    family, value, specified = _native_code(record)
    if family == 'SP':
        raise ValueError('RDKit square-planar SP projection has no registered native permutation translation')
    if record.group is not None:
        raise ValueError('Native Open Babel stereo cannot carry enhanced groups')
    if family == 'TH':
        if record.anchor.kind is not StereoAnchorKind.ATOM or len(record.anchor.atom_indices) != 1:
            raise ValueError('Native Open Babel TH requires an atom anchor')
        atom = rdmol.GetAtomWithIdx(record.anchor.atom_indices[0])
        native = _native_atom_carriers(rdmol, atom, lone_pair=any(
            carrier.kind is StereoCarrierKind.LONE_PAIR for carrier in record.carriers))
        if len(native) == 3:
            kind = (StereoCarrierKind.IMPLICIT_H if atom.GetNumExplicitHs() or atom.GetNumImplicitHs()
                    else StereoCarrierKind.LONE_PAIR)
            native += ((kind, None),)
        transported = tuple((carrier.kind, carrier.atom) for carrier in record.carriers)
        if len(transported) != 4:
            raise ValueError('Native Open Babel TH requires four ordered carriers')
        odd = _odd_permutation(transported, native)
        if not specified or value == 'UNKNOWN':
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        else:
            clockwise = (value == 'CW') != odd
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW if clockwise
                             else Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        return False

    if record.anchor.kind is not StereoAnchorKind.BOND or len(record.carriers) != 4:
        raise ValueError('Native Open Babel CT requires a bond and four ordered carriers')
    begin, end = record.anchor.atom_indices
    bond = _bond_for_anchor(rdmol, (begin, end))
    if bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError('Native Open Babel CT anchor requires a double bond')
    selected = []
    positions = []
    for side, (center, other) in enumerate(((begin, end), (end, begin))):
        atom = rdmol.GetAtomWithIdx(center)
        expected = tuple((StereoCarrierKind.ATOM, neighbor.GetIdx())
                         for neighbor in atom.GetNeighbors() if neighbor.GetIdx() != other)
        hydrogens = atom.GetNumExplicitHs() + atom.GetNumImplicitHs()
        if hydrogens > 1 or len(expected) + hydrogens > 2:
            raise ValueError('Native Open Babel CT has unsupported graph or virtual neighbor count')
        if len(expected) == 1:
            kind = (StereoCarrierKind.IMPLICIT_H if hydrogens
                    else StereoCarrierKind.LONE_PAIR)
            expected += ((kind, None),)
        pair = record.carriers[2 * side:2 * side + 2]
        transported = tuple((carrier.kind, carrier.atom) for carrier in pair)
        if len(expected) != 2 or set(transported) != set(expected) or len(set(transported)) != 2:
            raise ValueError('Native Open Babel CT carriers do not match ordered anchor sides')
        position = next(index for index, carrier in enumerate(pair) if carrier.kind is StereoCarrierKind.ATOM)
        selected.append(pair[position].atom)
        positions.append(2 * side + position)
    if bond.GetBeginAtomIdx() != begin:
        selected.reverse()
    bond.SetStereoAtoms(*selected)
    bond.SetStereo(Chem.BondStereo.STEREOANY if not specified else
                   Chem.BondStereo.STEREOTRANS if sum(positions) % 2 == 0 else Chem.BondStereo.STEREOCIS)
    return True


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
    from rdkit import Chem
    _HOTPOT_GROUP_KINDS = {
        StereoGroupKind.ABS: Chem.StereoGroupType.STEREO_ABSOLUTE,
        StereoGroupKind.AND: Chem.StereoGroupType.STEREO_AND,
        StereoGroupKind.OR: Chem.StereoGroupType.STEREO_OR,
    }

    if payload.records:
        rdmol.UpdatePropertyCache(strict=False)
    groups = {}
    native_alkene = False
    for record in payload.records:
        if record.source_namespace == 'openbabel-stereo/1':
            native_alkene = _apply_native_ob_stereo(rdmol, record) or native_alkene
            continue
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

    if native_alkene:
        Chem.SetDoubleBondNeighborDirections(rdmol)
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


def _verify_current_implicit_hydrogens(rdmol, atom_facts, prepared_hydrogens, row_to_idx):
    """Verify current I and the complete prepared R after the real cache update."""
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

    for row, expected in enumerate(prepared_hydrogens):
        atom = rdmol.GetAtomWithIdx(row_to_idx[row])
        actual = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
        if actual != expected:
            raise ValueError(
                "RDKIT.HYDROGEN_INVENTORY: non-graph hydrogen count for "
                f"atom row {row}: requested {expected}, reproduced {actual} "
                f"(implicit={atom.GetNumImplicitHs()}, "
                f"bracket={atom.GetNumExplicitHs()})"
            )
