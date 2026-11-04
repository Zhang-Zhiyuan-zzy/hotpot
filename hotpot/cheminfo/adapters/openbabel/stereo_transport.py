"""Native Open Babel stereo records; toolkit imports occur only at OB calls."""
from typing import NamedTuple

from ...structure.stereo import (
    StereoAnchor, StereoAnchorKind, StereoCarrier, StereoCarrierKind, StereoRecord,
)


_OPENBABEL_STEREO_NAMESPACE = 'openbabel-stereo/1'
_RDKIT_STEREO_NAMESPACE = 'rdkit-2026.09.1'
_NATIVE_CODES = {
    f'{family}/{value}/{specified}': (family, value, bool(specified))
    for family, values in (('TH', ('CW', 'CCW', 'UNKNOWN')), ('CT', ('U',)), ('SP', ('U',)))
    for value in values for specified in (0, 1)
}


def _native_code(record):
    try:
        return _NATIVE_CODES[record.source_code]
    except KeyError:
        raise ValueError(f'Unsupported Open Babel stereo code: {record.source_code}') from None


def _atom_carrier(row):
    return StereoCarrier(StereoCarrierKind.ATOM, row)


def _virtual_carrier(hydrogens):
    return StereoCarrier(StereoCarrierKind.IMPLICIT_H if hydrogens else StereoCarrierKind.LONE_PAIR)


def _extract_stereo_records(obmol):
    from openbabel import openbabel as ob

    atoms = tuple(ob.OBMolAtomIter(obmol))
    id_to_row = {atom.GetId(): row for row, atom in enumerate(atoms)}
    implicit = int(ob.OBStereo.ImplicitRef) & 0xffffffff
    def row(reference):
        try:
            return id_to_row[int(reference)]
        except KeyError:
            raise ValueError(f'Unsupported Open Babel stereo atom reference: {reference}') from None
    def carrier(reference, center, *, square_planar=False):
        if int(reference) & 0xffffffff == implicit:
            if square_planar:
                raise ValueError('Open Babel SP implicit references are unsupported')
            return _virtual_carrier(atoms[row(center)].GetImplicitHCount())
        return _atom_carrier(row(reference))

    records = []
    winding_codes = {ob.OBStereo.Clockwise: 'CW', ob.OBStereo.AntiClockwise: 'CCW',
                     ob.OBStereo.UnknownWinding: 'UNKNOWN'}
    for data in obmol.GetAllData(ob.StereoData):
        family = ob.toStereoBase(data).GetType()
        if family == ob.OBStereo.Tetrahedral:
            config = ob.toTetrahedralStereo(data).GetConfig()
            if config.view != ob.OBStereo.ViewFrom or config.winding not in winding_codes:
                raise ValueError('Unsupported Open Babel tetrahedral view/winding')
            anchor = StereoAnchor(StereoAnchorKind.ATOM, (row(config.center),))
            references = (config.from_or_towards, *config.refs)
            carriers = tuple(carrier(reference, config.center) for reference in references)
            code = f'TH/{winding_codes[config.winding]}/{int(config.specified)}'
        elif family == ob.OBStereo.CisTrans:
            config = ob.toCisTransStereo(data).GetConfig()
            if config.shape != ob.OBStereo.ShapeU:
                raise ValueError('Unsupported Open Babel cis/trans shape')
            anchor = StereoAnchor(StereoAnchorKind.BOND, (row(config.begin), row(config.end)))
            carriers = tuple(carrier(reference, config.begin if index < 2 else config.end)
                             for index, reference in enumerate(config.refs))
            code = f'CT/U/{int(config.specified)}'
        elif family == ob.OBStereo.SquarePlanar:
            config = ob.toSquarePlanarStereo(data).GetConfig()
            if config.shape != ob.OBStereo.ShapeU:
                raise ValueError('Unsupported Open Babel square-planar shape')
            anchor = StereoAnchor(StereoAnchorKind.ATOM, (row(config.center),))
            carriers = tuple(carrier(reference, config.center, square_planar=True)
                             for reference in config.refs)
            code = f'SP/U/{int(config.specified)}'
        else:
            raise ValueError(f'Unsupported Open Babel stereo family/type: {family}')
        if len(carriers) != 4:
            raise ValueError('Open Babel stereo requires four ordered references')
        records.append(StereoRecord(anchor, carriers, code, None, _OPENBABEL_STEREO_NAMESPACE, code))
    return tuple(records)


def _anchor(record, kind, count):
    if record.anchor.kind is not kind or len(record.anchor.atom_indices) != count:
        raise ValueError('Unsupported Open Babel stereo anchor')
    return record.anchor.atom_indices


def _carrier_tokens(carriers):
    return tuple((carrier.kind, carrier.atom) for carrier in carriers)


def _check_carriers(carriers, expected):
    tokens = _carrier_tokens(carriers)
    if len(tokens) != len(expected) or len(set(tokens)) != len(tokens) or set(tokens) != set(_carrier_tokens(expected)):
        raise ValueError('Open Babel stereo carriers do not match target graph or virtual references')


class _StereoConfig(NamedTuple):
    """Transient toolkit-independent row references for one OB configuration."""

    family: str
    value: str
    specified: bool
    anchors: tuple[int, ...]
    references: tuple[int | None, ...]
    hydrogens: tuple[int, ...]


def _project_stereo_payload(payload, atomic_numbers, bonds, hydrogens):
    """Translate source declarations once, against the actual target graph."""
    adjacency = {row: [] for row in atomic_numbers}
    bond_orders = {}
    for begin, end, order in bonds:
        adjacency[begin].append(end)
        adjacency[end].append(begin)
        bond_orders[frozenset((begin, end))] = order

    def atom_row(row):
        if row not in adjacency:
            raise ValueError(f'Open Babel stereo references missing atom row: {row}')
        return row

    def reference(carrier):
        if carrier.kind is StereoCarrierKind.ATOM:
            return atom_row(carrier.atom)
        if carrier.kind in (StereoCarrierKind.IMPLICIT_H, StereoCarrierKind.LONE_PAIR) and carrier.atom is None:
            return None
        raise ValueError('Unsupported Open Babel stereo carrier')

    def neighbors(center, other=None):
        return tuple(_atom_carrier(row) for row in adjacency[atom_row(center)] if row != other)

    def inventory(center, other=None):
        actual = neighbors(center, other)
        count = 4 if other is None else 2
        if hydrogens[center] > 1:
            raise ValueError('Open Babel stereo cannot represent multiple virtual-H carriers')
        if len(actual) + hydrogens[center] > count:
            raise ValueError('Open Babel stereo has unsupported graph or virtual neighbor count')
        if len(actual) == count - 1:
            actual += (_virtual_carrier(hydrogens[center]),)
        if len(actual) != count:
            raise ValueError('Open Babel stereo has unsupported target neighbor count')
        return actual

    configs = []
    for record in payload.records:
        if record.group is not None:
            raise ValueError('Open Babel cannot preserve enhanced stereo groups')
        if record.source_namespace == _OPENBABEL_STEREO_NAMESPACE:
            family, value, specified = _native_code(record)
            carriers = record.carriers
        elif record.source_namespace == _RDKIT_STEREO_NAMESPACE:
            code = record.source_code
            if code in ('CHI_UNSPECIFIED', 'STEREONONE'):
                if record.carriers:
                    raise ValueError('Unspecified RDKit stereo code has ordered carriers')
                continue
            if code in ('CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'):
                family, value, specified = 'TH', code.removeprefix('CHI_TETRAHEDRAL_'), True
                center, = _anchor(record, StereoAnchorKind.ATOM, 1)
                carriers = record.carriers
                if len(carriers) == 3 and all(carrier.kind is StereoCarrierKind.ATOM for carrier in carriers):
                    remaining = tuple(carrier for carrier in neighbors(center) if carrier not in carriers)
                    if len(remaining) == 1 and atomic_numbers[remaining[0].atom] == 1:
                        carriers += remaining
                    else:
                        carriers += (_virtual_carrier(hydrogens[center]),)
            elif code in ('STEREOE', 'STEREOZ', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'):
                family, value, specified = 'CT', 'U', code != 'STEREOANY'
                begin, end = _anchor(record, StereoAnchorKind.BOND, 2)
                if len(record.carriers) not in ((0, 2) if not specified else (2,)):
                    raise ValueError('RDKit cis/trans stereo requires two selected atom carriers')
                pairs = []
                for side, (center, other) in enumerate(((begin, end), (end, begin))):
                    actual = inventory(center, other)
                    selected = record.carriers[side] if record.carriers else actual[0]
                    if selected.kind is not StereoCarrierKind.ATOM or selected not in actual:
                        raise ValueError('RDKit cis/trans stereo carriers do not match ordered anchor sides')
                    pairs.append((selected, *(carrier for carrier in actual if carrier != selected)))
                if code in ('STEREOZ', 'STEREOCIS'):
                    pairs[1] = pairs[1][::-1]
                carriers = (*pairs[0], *pairs[1])
            else:
                raise ValueError(f'Unsupported RDKit stereo source code for Open Babel: {code}')
        else:
            raise ValueError(f'Open Babel cannot represent stereo namespace: {record.source_namespace}')

        if family == 'TH':
            center, = _anchor(record, StereoAnchorKind.ATOM, 1)
            _check_carriers(carriers, inventory(center))
            anchors = (center,)
        elif family == 'CT':
            begin, end = _anchor(record, StereoAnchorKind.BOND, 2)
            if bond_orders.get(frozenset((begin, end))) != 2:
                raise ValueError('Open Babel cis/trans anchor requires a double bond')
            _check_carriers(carriers[:2], inventory(begin, end))
            _check_carriers(carriers[2:], inventory(end, begin))
            anchors = (begin, end)
        else:
            center, = _anchor(record, StereoAnchorKind.ATOM, 1)
            if any(carrier.kind is not StereoCarrierKind.ATOM for carrier in carriers):
                raise ValueError('Open Babel SP virtual references are unsupported')
            _check_carriers(carriers, neighbors(center))
            if len(carriers) != 4:
                raise ValueError('Open Babel square-planar stereo requires four atom carriers')
            anchors = (center,)
        configs.append(_StereoConfig(family, value, specified, anchors,
                                     tuple(reference(carrier) for carrier in carriers),
                                     tuple(hydrogens[row] for row in anchors)))
    return tuple(configs)


def _apply_stereo_payload(obmol, payload, row_to_idx, hydrogens):
    from openbabel import openbabel as ob

    idx_to_row = {index: row for row, index in row_to_idx.items()}
    bonds = tuple((idx_to_row[bond.GetBeginAtomIdx()], idx_to_row[bond.GetEndAtomIdx()],
                   bond.GetBondOrder()) for bond in ob.OBMolBondIter(obmol))
    atomic_numbers = {row: obmol.GetAtom(index).GetAtomicNum()
                      for row, index in row_to_idx.items()}
    configs = _project_stereo_payload(payload, atomic_numbers, bonds, hydrogens)
    row_to_id = {row: obmol.GetAtom(index).GetId() for row, index in row_to_idx.items()}
    implicit = int(ob.OBStereo.ImplicitRef) & 0xffffffff
    for projected in configs:
        references = tuple(implicit if row is None else row_to_id[row] for row in projected.references)
        if projected.family == 'TH':
            config = ob.OBTetrahedralConfig()
            config.center = row_to_id[projected.anchors[0]]
            config.from_or_towards = references[0]
            config.refs = ob.OBStereo.MakeRefs(*references[1:])
            config.view = ob.OBStereo.ViewFrom
            config.winding = {'CW': ob.OBStereo.Clockwise, 'CCW': ob.OBStereo.AntiClockwise,
                              'UNKNOWN': ob.OBStereo.UnknownWinding}[projected.value]
            data = ob.OBTetrahedralStereo(obmol)
        elif projected.family == 'CT':
            config = ob.OBCisTransConfig()
            config.begin, config.end = (row_to_id[row] for row in projected.anchors)
            config.refs = ob.OBStereo.MakeRefs(*references)
            config.shape = ob.OBStereo.ShapeU
            data = ob.OBCisTransStereo(obmol)
        else:
            config = ob.OBSquarePlanarConfig()
            config.center = row_to_id[projected.anchors[0]]
            config.refs = ob.OBStereo.MakeRefs(*references)
            config.shape = ob.OBStereo.ShapeU
            data = ob.OBSquarePlanarStereo(obmol)
        config.specified = projected.specified
        data.SetConfig(config)
        obmol.CloneData(data)
    obmol.SetChiralityPerceived(True)
