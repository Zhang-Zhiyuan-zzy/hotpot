"""Native Open Babel stereo records; toolkit imports occur only at OB calls."""
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


def _apply_stereo_payload(obmol, payload, row_to_idx, hydrogens):
    from openbabel import openbabel as ob

    row_to_id = {row: obmol.GetAtom(index).GetId() for row, index in row_to_idx.items()}
    idx_to_row = {index: row for row, index in row_to_idx.items()}
    implicit = int(ob.OBStereo.ImplicitRef) & 0xffffffff
    def atom_id(row):
        try:
            return row_to_id[row]
        except KeyError:
            raise ValueError(f'Open Babel stereo references missing atom row: {row}') from None
    def reference(carrier):
        if carrier.kind is StereoCarrierKind.ATOM:
            return atom_id(carrier.atom)
        if carrier.kind in (StereoCarrierKind.IMPLICIT_H, StereoCarrierKind.LONE_PAIR) and carrier.atom is None:
            return implicit
        raise ValueError('Unsupported Open Babel stereo carrier')
    def neighbors(center, other=None):
        atom_id(center)
        return tuple(_atom_carrier(idx_to_row[atom.GetIdx()])
                     for atom in ob.OBAtomAtomIter(obmol.GetAtom(row_to_idx[center]))
                     if idx_to_row[atom.GetIdx()] != other)
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
            config = ob.OBTetrahedralConfig()
            config.center = atom_id(center)
            config.from_or_towards = reference(carriers[0])
            config.refs = ob.OBStereo.MakeRefs(*(reference(carrier) for carrier in carriers[1:]))
            config.view = ob.OBStereo.ViewFrom
            config.winding = {'CW':ob.OBStereo.Clockwise, 'CCW':ob.OBStereo.AntiClockwise,
                              'UNKNOWN':ob.OBStereo.UnknownWinding}[value]
            data = ob.OBTetrahedralStereo(obmol)
        elif family == 'CT':
            begin, end = _anchor(record, StereoAnchorKind.BOND, 2)
            bond = obmol.GetBond(row_to_idx[begin], row_to_idx[end])
            if bond is None or bond.GetBondOrder() != 2:
                raise ValueError('Open Babel cis/trans anchor requires a double bond')
            _check_carriers(carriers[:2], inventory(begin, end))
            _check_carriers(carriers[2:], inventory(end, begin))
            config = ob.OBCisTransConfig()
            config.begin, config.end = atom_id(begin), atom_id(end)
            config.refs = ob.OBStereo.MakeRefs(*(reference(carrier) for carrier in carriers))
            config.shape = ob.OBStereo.ShapeU
            data = ob.OBCisTransStereo(obmol)
        else:
            center, = _anchor(record, StereoAnchorKind.ATOM, 1)
            if any(carrier.kind is not StereoCarrierKind.ATOM for carrier in carriers):
                raise ValueError('Open Babel SP virtual references are unsupported')
            _check_carriers(carriers, neighbors(center))
            if len(carriers) != 4:
                raise ValueError('Open Babel square-planar stereo requires four atom carriers')
            config = ob.OBSquarePlanarConfig()
            config.center = atom_id(center)
            config.refs = ob.OBStereo.MakeRefs(*(reference(carrier) for carrier in carriers))
            config.shape = ob.OBStereo.ShapeU
            data = ob.OBSquarePlanarStereo(obmol)
        config.specified = specified
        data.SetConfig(config)
        obmol.CloneData(data)
    obmol.SetChiralityPerceived(True)
