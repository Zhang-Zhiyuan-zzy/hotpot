"""Named bonding preparations and atomic representation assignments."""

from dataclasses import replace

import numpy as np

from ..structure._commit import (
    _FactTransaction, _commit_aromatic_assignment, _commit_atom_facts,
    _set_group_status,
)
from ..structure._dependencies import (
    GROUP_INDEX, _FactStatus, _ensure_group, _expand_dirty, _component_rows, _neighbour_rows,
)
from ..structure._fields import (
    ATOM_FIELDS, ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS,
)
from ..structure._identity import (
    _capture_source, _check_source, _hydrogen_inputs, _hydrogen_transport_inputs,
)
from ..structure._mutation import _normalize_atom_fields
from ..structure.snapshots import _BondingSnapshot, _AromaticAssignment
from ..rules.chemistry.elements import _electron_configuration, _is_metal
from ..rules.chemistry.bonding.valence import _calculate_valence, _hybridization
from ..rules.chemistry.bonding.formal_charge import (
    _calculate_formal_charge, _sum_formal_charges, _default_molecular_charge,
)
from ..rules.chemistry.bonding.hydrogen import _calculate_implicit_hydrogens
from ..rules.chemistry.bonding.bond_order import _sum_orders
from ..rules.chemistry.bonding.kekule import (
    _calculate_aromatic_assignment, _check_joint_ring_kekulization, _aromatic_flags_assignment,
)


__all__ = ()

_EXPLICIT = frozenset(ATOM_SOURCE_CODES[name] for name in (
    'explicit', 'legacy_array', 'rdkit', 'openbabel', 'ccdc', 'mca_dto', 'cclib', 'tmqm',
))
_AROMATIC_PRODUCERS = frozenset(ATOM_SOURCE_CODES[name] for name in ('aromaticity', 'kekule', 'bond_order'))


def _local_inputs(atom):
    mol = atom.mol
    entry = mol._topology_index.get(id(atom))
    if entry is None:
        return atom.atomic_number, (), (), bool(atom._detached_table.read(0, 'is_aromatic'))
    table = mol._atom_table
    neighbours, orders = [], []
    for bond in entry[1]:
        other = bond.atom2 if bond.atom1 is atom else bond.atom1
        neighbours.append(int(table._columns[0][mol._topology_index[id(other)][0]]))
        orders.append(float(bond._attrs[0]))
    return (int(table._columns[0][entry[0]]), tuple(neighbours), tuple(orders),
            bool(table._columns[FIELD_INDEX['is_aromatic']][entry[0]]))


def _atom_valence(atom):
    with atom.mol._mutation_lock:
        number, neighbours, orders, _ = _local_inputs(atom)
        covalent = _sum_orders(number, neighbours, orders, 'covalent')
        return _calculate_valence(number, neighbours, covalent)


def _atom_formal_charge(atom):
    with atom.mol._mutation_lock:
        number, neighbours, orders, _ = _local_inputs(atom)
        return _calculate_formal_charge(number, neighbours,
                                       _sum_orders(number, neighbours, orders, 'covalent'), atom.symbol)


def _atom_order_sum(atom, scope):
    with atom.mol._mutation_lock:
        number, neighbours, orders, _ = _local_inputs(atom)
        return _sum_orders(number, neighbours, orders, scope)


def _atom_hybridization(atom):
    with atom.mol._mutation_lock:
        number, neighbours, orders, _ = _local_inputs(atom)
        return _hybridization(number, orders, atom.is_aromatic,
                              _sum_orders(number, neighbours, orders, 'heavy'))


def _atom_implicit_proposal(atom):
    with atom.mol._mutation_lock:
        transport = {entry[0]: entry[1:] for entry in _hydrogen_transport_inputs(
            tuple(member._uid for member in atom.mol._atoms), atom.mol._stereo_payload)}
        return _calculate_implicit_hydrogens(*_hydrogen_inputs(atom, transport))


def _hydrogen_display_value(table, row, *, pending=False):
    field, group = FIELD_INDEX['implicit_hydrogens'], GROUP_INDEX['IMPLICIT_H']
    status = table._group_status[row, group]
    if status == _FactStatus.CONFLICT:
        raise RuntimeError('Implicit-H representation conflicts with current facts; call calc_implicit_hydrogens() explicitly')
    if (status == _FactStatus.CURRENT and int(table._presence[row]) & FIELD_PRESENCE_BITS[field]
            and (not pending or int(table._sources[row, field]) in _EXPLICIT)):
        return table._columns[field][row]
    return 0


def _implicit_hydrogen_read(atom):
    """Integer display; unresolved input never becomes a zero H fact."""
    with atom.mol._mutation_lock:
        entry = atom.mol._topology_index.get(id(atom))
        if entry is None:
            table = atom._detached_table
            pending = any(atom._uid in seeds for seeds in table._dirty_seeds[:2])
            return int(_hydrogen_display_value(table, 0, pending=pending))
        _prepare_hydrogens(atom.mol, rows=(entry[0],))
        return int(_hydrogen_display_value(atom.mol._atom_table, entry[0]))


def _require_implicit_hydrogens(mol, *, rows=None):
    """Prepare one requested H group for a consumer requiring finite counts."""
    with mol._mutation_lock:
        selected = tuple(range(len(mol._atoms))) if rows is None else tuple(rows)
        if not selected:
            return ()
        _prepare_hydrogens(mol, rows=selected)
        table = mol._atom_table
        statuses = table._group_status[list(selected), GROUP_INDEX['IMPLICIT_H']]
        if any(status == _FactStatus.CONFLICT for status in statuses):
            raise RuntimeError('Implicit-H representation conflicts with current facts; call calc_implicit_hydrogens() explicitly')
        if any(status != _FactStatus.CURRENT for status in statuses):
            raise ValueError('This operation requires resolved implicit-H input; supply the missing elemental or neighbour facts')
        field = FIELD_INDEX['implicit_hydrogens']
        return tuple(int(table._columns[field][row]) for row in selected)


def _require_aromatic_representation(mol, *, rows=None):
    with mol._mutation_lock:
        selected = tuple(range(len(mol._atoms))) if rows is None else tuple(rows)
        if not selected:
            return
        _prepare_aromatic_representation(mol, rows=selected)
        statuses = mol._atom_table._group_status[list(selected), GROUP_INDEX['AROMATIC_REPRESENTATION']]
        if any(status != _FactStatus.CURRENT for status in statuses):
            raise RuntimeError('Aromatic representation is unresolved or conflicting; call determine_rings_aromatic() explicitly')


def _atom_projection(mol, *, fields=ATOM_FIELDS, rows=None, strict_hydrogens=False):
    """One independent numeric projection, selecting only the requested fields."""
    with mol._mutation_lock:
        fields = tuple(fields)
        selected = tuple(range(len(mol._atoms))) if rows is None else tuple(rows)
        if 'implicit_hydrogens' in fields:
            if strict_hydrogens:
                _require_implicit_hydrogens(mol, rows=selected)
            else:
                _prepare_hydrogens(mol, rows=selected)
        if 'is_aromatic' in fields:
            _require_aromatic_representation(mol, rows=selected)
        values = mol._atom_table.select(fields, rows=selected)
        electron_fields = tuple((column, FIELD_INDEX[field] - 1) for column, field in enumerate(fields)
                                if field in ('n', 's', 'p', 'd', 'f', 'g'))
        if electron_fields:
            configurations = tuple(_electron_configuration(mol._atoms[row].atomic_number) for row in selected)
            for column, index in electron_fields:
                values[:, column] = tuple(configuration[index] for configuration in configurations)
        for column, field in enumerate(fields):
            if field == 'valence':
                values[:, column] = tuple(_atom_valence(mol._atoms[row]) for row in selected)
            elif field == 'implicit_hydrogens':
                # Preserve current float64 bytes; absent/stale slots display
                # the legacy placeholder without creating a current H fact.
                for position, row in enumerate(selected):
                    table = mol._atom_table
                    if (table._group_status[row, GROUP_INDEX['IMPLICIT_H']] != _FactStatus.CURRENT
                            or not int(table._presence[row]) & FIELD_PRESENCE_BITS[FIELD_INDEX[field]]):
                        values[position, column] = _hydrogen_display_value(table, row)
        return values


def _prepare_hydrogens(mol, *, rows=None, force=False):
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['implicit_hydrogens']):
        selected = tuple(range(len(mol._atoms))) if rows is None else tuple(rows)
        table = mol._atom_table
        field, group = FIELD_INDEX['implicit_hydrogens'], GROUP_INDEX['IMPLICIT_H']
        if not force and not any(table._dirty_seeds[:3]) and all(
                table._group_status[row, group] in (_FactStatus.CURRENT, _FactStatus.UNRESOLVED, _FactStatus.CONFLICT)
                and table._group_status[row, GROUP_INDEX['AROMATIC_REPRESENTATION']] == _FactStatus.CURRENT
                for row in selected):
            return
        dirty = set().union(*mol._atom_table._dirty_seeds[:2])
        _prepare_aromatic_representation(mol, rows=selected)
        dirty.update(set().union(*mol._atom_table._dirty_seeds[:2]))
        _expand_dirty(mol, 'IMPLICIT_H')
        dirty.update(mol._atoms[row]._uid for row in _neighbour_rows(mol, dirty))
        table = mol._atom_table
        if len(selected) < len(mol._atoms):
            requested = set(selected)
            selected += tuple(row for row, atom in enumerate(mol._atoms) if row not in requested
                and (atom._uid in dirty or table._group_status[row, group]
                     in (_FactStatus.NEEDS_RECALC, _FactStatus.ABSENT)))
        proposals, current, unresolved, conflicts = [], [], [], []
        binding = None
        transport = None
        for row in selected:
            atom = mol._atoms[row]
            status = table._group_status[row, group]
            present = bool(int(table._presence[row]) & FIELD_PRESENCE_BITS[field])
            explicit = present and int(table._sources[row, field]) in _EXPLICIT
            if explicit:
                current.append(row)
                continue
            aromatic_status = table._group_status[row, GROUP_INDEX['AROMATIC_REPRESENTATION']]
            if aromatic_status == _FactStatus.CONFLICT:
                conflicts.append(row)
                continue
            if aromatic_status == _FactStatus.UNRESOLVED:
                unresolved.append(row)
                continue
            if not force and atom._uid not in dirty and status in (
                    _FactStatus.CURRENT, _FactStatus.UNRESOLVED, _FactStatus.CONFLICT):
                continue
            if binding is None:
                binding = _capture_source(mol, ('bonding_inputs',))
                transport = {entry[0]: entry[1:] for entry in _hydrogen_transport_inputs(
                    tuple(member._uid for member in mol._atoms), mol._stereo_payload)}
            inputs = _hydrogen_inputs(atom, transport)
            value = _calculate_implicit_hydrogens(*inputs)
            if inputs != _hydrogen_inputs(atom, transport) or mol._topology_index.get(id(atom), (None,))[0] != row:
                raise RuntimeError('Hydrogen proposal source changed; recalculate before committing')
            if value is None:
                unresolved.append(row)
            else:
                proposals.append((row, value))
        if proposals:
            update_rows = tuple(row for row, _ in proposals)
            update = _normalize_atom_fields(binding, update_rows,
                tuple(mol._atoms[row]._uid for row in update_rows),
                {'implicit_hydrogens': tuple(value for _, value in proposals)},
                producer=ATOM_SOURCE_CODES['implicit_hydrogens'])
            _commit_atom_facts(mol, update)
        elif binding is not None:
            _check_source(mol, binding)
        absent = tuple(row for row in unresolved if int(table._presence[row]) & FIELD_PRESENCE_BITS[field]
                       or int(table._sources[row, field]) != ATOM_SOURCE_CODES['default'])
        if absent:
            _commit_atom_facts(mol, _normalize_atom_fields(
                _capture_source(mol), absent, tuple(mol._atoms[row]._uid for row in absent),
                {'implicit_hydrogens': tuple(table._columns[field][row] for row in absent)},
                producer=ATOM_SOURCE_CODES['default'], present=False))
        _set_group_status(mol, 'IMPLICIT_H', current, _FactStatus.CURRENT)
        _set_group_status(mol, 'IMPLICIT_H', unresolved, _FactStatus.UNRESOLVED)
        _set_group_status(mol, 'IMPLICIT_H', conflicts, _FactStatus.CONFLICT)


def _prepare_formal_charge_coverage(mol):
    table = mol._atom_table
    if not len(table):
        return
    group = GROUP_INDEX['FORMAL_CHARGE_COVERAGE']
    if not table._dirty_seeds[group] and table._group_status[0, group] in (
            _FactStatus.CURRENT, _FactStatus.UNRESOLVED, _FactStatus.CONFLICT):
        return
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['explicit']):
        _expand_dirty(mol, 'FORMAL_CHARGE_COVERAGE')
        field = FIELD_INDEX['formal_charge']
        covered = (table._presence[:len(table)] & FIELD_PRESENCE_BITS[field]) != 0
        if not covered.all():
            status = _FactStatus.UNRESOLVED
        elif mol._charge_present and _sum_formal_charges(table.column('formal_charge')) != mol._charge_value:
            status = _FactStatus.CONFLICT
        else:
            status = _FactStatus.CURRENT
        _set_group_status(mol, 'FORMAL_CHARGE_COVERAGE', range(len(table)), status)


def _atom_charge_sum(mol):
    with mol._mutation_lock:
        _prepare_formal_charge_coverage(mol)
        return _sum_formal_charges(mol._atom_table.column('formal_charge'))


def _calculate_default_molecular_charge(mol):
    from copy import copy
    _prepare_valence_and_hydrogens(mol, None)
    _require_implicit_hydrogens(mol)
    clone = copy(mol)
    if not clone.has_hydrogens:
        clone.add_hydrogens()
    if clone.is_organic:
        components = (clone,)
    else:
        clone.hide_metal_ligand_bonds()
        components = clone.components
    return _default_molecular_charge(tuple(
        (component.is_organic, component.sum_explicit_hydrogens, component.sum_implicit_hydrogens,
         component.atoms[0].get_formal_charge() if len(component.atoms) == 1 and not component.is_organic else 0)
        for component in components))


def _prepare_valences(mol):
    with mol._mutation_lock:
        for atom in mol._atoms:
            _atom_valence(atom)


def _materialize_current_hydrogens(atom):
    """Transfer current virtual/bracket H to graph H, keeping source history."""
    from ..stereo import get_stereo_payload, set_stereo_payload
    payload = get_stereo_payload(atom.mol)
    row = atom.idx
    facts = tuple(replace(fact, current_hydrogens=replace(fact.current_hydrogens,
                      implicit_hydrogens=0, bracket_explicit_hydrogens=0))
                  if fact.atom == row and fact.current_hydrogens is not None else fact
                  for fact in payload.atom_facts)
    if facts != payload.atom_facts:
        set_stereo_payload(atom.mol, replace(payload, atom_facts=facts))


def _prepare_hydrogen_expansions(mol, atoms):
    """Capture all automatic additions before the first graph/payload write."""
    counts = _require_implicit_hydrogens(mol, rows=tuple(atom.idx for atom in atoms))
    source = _capture_source(mol, ('bonding_inputs',))
    transport = {entry[0]: entry[1:] for entry in _hydrogen_transport_inputs(
        tuple(atom._uid for atom in mol._atoms), mol._stereo_payload)}
    table = mol._atom_table
    prepared = []
    for atom, count in zip(atoms, counts):
        row = atom.idx
        producer = int(table._sources[row, FIELD_INDEX['implicit_hydrogens']])
        inputs = _hydrogen_inputs(atom, transport)
        inferred = (table._group_status[row, GROUP_INDEX['IMPLICIT_H']] == _FactStatus.CURRENT
                    and bool(int(table._presence[row]) & FIELD_PRESENCE_BITS[FIELD_INDEX['implicit_hydrogens']])
                    and producer == ATOM_SOURCE_CODES['implicit_hydrogens']
                    and inputs[0] > 1 and not _is_metal(inputs[0]))
        graph_hydrogens = sum(other.atomic_number == 1 for other in atom.neighbours)
        prepared.append((count, graph_hydrogens, transport[atom._uid][1], producer,
                         inferred, inputs))
    _check_source(mol, source)
    return tuple(prepared)


def _add_atom_hydrogens(atom, num, rm_polar_hs, rng, *, prepared=None):
    with _FactTransaction(atom.mol, producer=ATOM_SOURCE_CODES['implicit_hydrogens']):
        neighbours = atom.neighbours
        hydrogens = [other for other in neighbours if other.atomic_number == 1]
        inferred = False
        if num is None:
            if prepared is None:
                prepared, = _prepare_hydrogen_expansions(atom.mol, (atom,))
            count, graph_count, bracket, producer, inferred, inputs = prepared
            num = max(count - graph_count, 0) if inferred else count - graph_count
            if not inferred and atom.polar_hydrogen_site:
                num -= sum(other.is_metal for other in neighbours)
        added = []
        if num > 0:
            added = [atom._add_atom(atom_attrs={'coordinates': np.asarray(atom.coordinates)
                     + atom.random_point_on_sphere(1.05, rng=rng)}) for _ in range(num)]
        if inferred:
            _materialize_current_hydrogens(atom)
        if added:
            return 1, added
        if num < 0 and atom.polar_hydrogen_site and hydrogens and rm_polar_hs:
            atom.mol._remove_atoms_from_topology(hydrogens[:abs(num)])
            return -1, hydrogens[:abs(num)]
        return 0, []


def _add_hydrogens(mol, rm_polar_hs, rng):
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['implicit_hydrogens']):
        atoms = tuple(atom for atom in mol.atoms if not (atom.is_hydrogen or atom.is_metal))
        prepared = _prepare_hydrogen_expansions(mol, atoms)
        modified = False
        for atom, inputs in zip(atoms, prepared):
            change, _ = _add_atom_hydrogens(atom, None, rm_polar_hs, rng, prepared=inputs)
            modified |= bool(change)
        if modified:
            mol._update_topology()


def _bonding_snapshot(mol, *, ligand_skeleton=True):
    from .topology_queries import _ring_cycles
    table = mol._atom_table
    numbers = tuple(int(value) for value in table.column('atomic_number'))
    bonds = tuple(bond for bond in mol._bonds if not ligand_skeleton or not bond.is_metal_ligand_bond)
    edges = tuple((bond._uid, mol._topology_index[id(bond.atom1)][0],
                   mol._topology_index[id(bond.atom2)][0], float(bond._attrs[0]),
                   bond._bond_kind.value, bond._is_aromatic) for bond in bonds)
    rings = tuple(_ring_cycles(mol, ligand_skeleton=ligand_skeleton))
    hfield = FIELD_INDEX['implicit_hydrogens']
    hydrogens = tuple(int(table._columns[hfield][row]) if
        int(table._presence[row]) & FIELD_PRESENCE_BITS[hfield] else None for row in range(len(table)))
    return _BondingSnapshot(_capture_source(mol, ('bonding_inputs',)),
        tuple(atom._uid for atom in mol._atoms), numbers, hydrogens,
        tuple(bool(value) for value in table.column('is_aromatic')), edges,
        tuple(tuple(row) for row in table.coordinates()), rings,
        # Preserve the characterized Ring.has_3d predicate. Its legacy tuple
        # versus Atom comparison is always true, even for coincident rows.
        (True,) * len(rings))


def _assign_aromaticity(mol, *, kekule, force=None):
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['kekule' if kekule else 'aromaticity']):
        snapshot = _bonding_snapshot(mol)
        if force is False:
            return
        if force is None and not kekule and (not snapshot.rings or any(
                all(snapshot.aromatic[row] for row in ring) for ring in snapshot.rings)):
            return
        proposal = _calculate_aromatic_assignment(snapshot, kekule=kekule)
        if proposal is None:
            _set_group_status(mol, 'AROMATIC_REPRESENTATION', range(len(mol._atoms)), _FactStatus.UNRESOLVED)
            return
        _commit_aromatic_assignment(mol, proposal, snapshot=snapshot)


def _prepare_valence_and_hydrogens(mol, assign_aromatic):
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['implicit_hydrogens']):
        _assign_aromaticity(mol, kekule=False, force=assign_aromatic)
        _prepare_hydrogens(mol, force=True)


def _prepare_aromatic_representation(mol, *, rows=None):
    """Revalidate supplied flags; only inferred representations are recalculated."""
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['aromaticity']):
        selected = tuple(range(len(mol._atoms))) if rows is None else tuple(rows)
        table = mol._atom_table
        group = GROUP_INDEX['AROMATIC_REPRESENTATION']
        if not any(table._dirty_seeds[:3]) and all(table._group_status[row, group]
                in (_FactStatus.CURRENT, _FactStatus.UNRESOLVED, _FactStatus.CONFLICT) for row in selected):
            return
        dirty = set().union(*mol._atom_table._dirty_seeds[:3])
        affected = set(_component_rows(mol, dirty)) if dirty else set()
        statuses = _ensure_group(mol, 'AROMATIC_REPRESENTATION')
        changed = tuple(row for row, status in enumerate(statuses)
                        if row in affected or status in (_FactStatus.NEEDS_RECALC, _FactStatus.ABSENT))
        rows = tuple(_component_rows(mol, tuple(mol._atoms[row]._uid for row in changed))) if changed else ()
        if not rows:
            return
        field = FIELD_INDEX['is_aromatic']
        # No selected aromatic representation is the existing false default;
        # querying a topology does not opt into aromaticity perception.
        opted = tuple(mol._atoms[row]._uid for row in rows if bool(table._columns[field][row])
                      or int(table._sources[row, field]) in _AROMATIC_PRODUCERS)
        if not opted:
            _set_group_status(mol, 'AROMATIC_REPRESENTATION', rows, _FactStatus.CURRENT)
            return
        perceived = set(_component_rows(mol, opted))
        _set_group_status(mol, 'AROMATIC_REPRESENTATION',
                          tuple(row for row in rows if row not in perceived), _FactStatus.CURRENT)
        rows = tuple(row for row in rows if row in perceived)
        snapshot = _bonding_snapshot(mol)
        cyclic = set(row for ring in snapshot.rings for row in ring)
        explicit = {row for row in rows if int(table._sources[row, field]) in _EXPLICIT}
        conflicts = tuple(row for row in explicit if snapshot.aromatic[row] and row not in cyclic)
        if conflicts:
            affected = _component_rows(mol, tuple(mol._atoms[row]._uid for row in conflicts))
            _set_group_status(mol, 'AROMATIC_REPRESENTATION', affected, _FactStatus.CONFLICT)
            affected = set(affected)
            rows = tuple(row for row in rows if row not in affected)
            if not rows:
                return
            explicit.intersection_update(rows)
        inferred = tuple(row for row in rows if row not in explicit)
        inferred_rows = set(inferred)
        flags = tuple(False if row in inferred_rows else value for row, value in enumerate(snapshot.aromatic))
        proposal = _calculate_aromatic_assignment(replace(snapshot, aromatic=flags), kekule=False)
        if proposal is None:
            _set_group_status(mol, 'AROMATIC_REPRESENTATION', inferred, _FactStatus.UNRESOLVED)
            return
        proposed_flags = tuple(proposal.atom_aromatic[row] if row in inferred_rows else snapshot.aromatic[row]
                               for row in range(len(snapshot.atom_uids)))
        proposal = _aromatic_flags_assignment(snapshot, proposed_flags)
        covered = set(rows)
        selected = tuple(index for index, bond in enumerate(snapshot.bonds)
                         if bond[1] in covered and bond[2] in covered)
        _commit_aromatic_assignment(mol, replace(proposal, atom_aromatic=proposed_flags,
            bond_uids=tuple(proposal.bond_uids[index] for index in selected),
            bond_aromatic=tuple(proposal.bond_aromatic[index] for index in selected),
            bond_orders=tuple(proposal.bond_orders[index] for index in selected),
            bond_kinds=tuple(proposal.bond_kinds[index] for index in selected),
            covered_rows=rows), snapshot=snapshot, preserve_sources=True)


def _atom_aromatic_read(atom):
    with atom.mol._mutation_lock:
        entry = atom.mol._topology_index.get(id(atom))
        if entry is None:
            return bool(atom._detached_table.read(0, 'is_aromatic'))
        table = atom.mol._atom_table
        group = GROUP_INDEX['AROMATIC_REPRESENTATION']
        if any(table._dirty_seeds[index] for index in (0, 1, 2)) or table._group_status[entry[0], group] in (_FactStatus.NEEDS_RECALC, _FactStatus.ABSENT):
            _prepare_aromatic_representation(atom.mol, rows=(entry[0],))
        status = table._group_status[entry[0], group]
        if status != _FactStatus.CURRENT:
            raise RuntimeError('Aromatic representation is unresolved or conflicting; call determine_rings_aromatic() explicitly')
        return bool(atom.mol._atom_table._columns[FIELD_INDEX['is_aromatic']][entry[0]])


def _bond_aromatic_read(bond):
    with bond.mol._mutation_lock:
        table = bond.mol._atom_table
        group = GROUP_INDEX['AROMATIC_REPRESENTATION']
        rows = tuple(atom.idx for atom in bond.atoms)
        if any(table._dirty_seeds[index] for index in (0, 1, 2)) or any(
                table._group_status[row, group] in (_FactStatus.NEEDS_RECALC, _FactStatus.ABSENT) for row in rows):
            _prepare_aromatic_representation(bond.mol, rows=rows)
        for atom in bond.atoms:
            status = bond.mol._atom_table._group_status[atom.idx, GROUP_INDEX['AROMATIC_REPRESENTATION']]
            if status != _FactStatus.CURRENT:
                raise RuntimeError('Aromatic representation conflicts with current topology; call determine_rings_aromatic() explicitly')
        return bool(bond._is_aromatic)


def _ring_aromatic_read(ring):
    from ..views._source import _check_ring_source
    with ring.mol._mutation_lock:
        _check_ring_source(ring, prepare=True)
        table = ring.mol._atom_table
        rows = tuple(ring.mol._topology_index[id(atom)][0] for atom in ring.atoms)
        if any(table._group_status[row, GROUP_INDEX['AROMATIC_REPRESENTATION']] != _FactStatus.CURRENT for row in rows):
            raise RuntimeError('Ring aromatic representation conflicts with current facts; call determine_rings_aromatic() explicitly')
        return all(bool(table._columns[FIELD_INDEX['is_aromatic']][row]) for row in rows)


def _assign_bond_orders(mol):
    from ..obconvert import _perceive_bond_orders
    from ..structure._builder import _thaw_canonical
    from ..structure._commit import _commit_bond_facts
    with mol._mutation_lock:
        source = _capture_source(mol, ('bonding_inputs', 'cell'))
        snapshot = replace(_bonding_snapshot(mol), source=source)
        atoms = tuple(mol._atoms)
        bonds = tuple(bond for bond in mol._bonds if not bond.is_metal_ligand_bond)
        by_pair = {frozenset((bond.atom1.idx, bond.atom2.idx)): bond for bond in bonds}
        atom_uids = tuple(atom._uid for atom in atoms)
    aromatic, perceived = _perceive_bond_orders(mol, bonds)
    if len(aromatic) != len(atoms) or {frozenset(entry[0]) for entry in perceived} != by_pair.keys():
        raise ValueError('Bond-order perception changed the selected member mapping')
    ordered = tuple((by_pair[frozenset(entry[0])], entry) for entry in perceived)
    proposal = _AromaticAssignment(source, atom_uids, tuple(bond._uid for bond, _ in ordered),
        aromatic, tuple(entry[6] for _, entry in ordered), tuple(float(entry[1]) for _, entry in ordered),
        tuple(entry[2] for _, entry in ordered), tuple(range(len(atoms))), 'bond_order')
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['bond_order']):
        _commit_aromatic_assignment(mol, proposal, snapshot=snapshot)
        for bond, entry in ordered:
            _commit_bond_facts(mol, bond, _capture_source(mol), raw_values=bond._attrs,
                kind=bond._bond_kind, direction=entry[3], source_namespace=entry[4],
                metadata=_thaw_canonical(entry[5]), aromatic=entry[6])


def _check_joint_ring(joint):
    with joint.mol._mutation_lock:
        joint._check_source()
        hydrogens = _require_implicit_hydrogens(joint.mol, rows=tuple(atom.idx for atom in joint.atoms))
        joint._check_source()
        return _check_joint_ring_kekulization(tuple(atom.atomic_number for atom in joint.atoms),
            tuple(atom.sum_heavy_cov_orders for atom in joint.atoms), hydrogens)


def _restore_assembly_aromaticity(mol, changes):
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['aromaticity']):
        snapshot = _bonding_snapshot(mol)
        flags = list(snapshot.aromatic)
        for atom, value in changes:
            if atom.mol is not mol or id(atom) not in mol._topology_index:
                raise ValueError('Assembly aromatic assignment requires current owned members')
            flags[mol._topology_index[id(atom)][0]] = bool(value)
        _commit_aromatic_assignment(mol, _aromatic_flags_assignment(snapshot, flags), snapshot=snapshot, preserve_sources=True)
