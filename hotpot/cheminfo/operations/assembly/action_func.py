"""Atomic assembly transformations with action-local hydrogen inventories."""

from dataclasses import replace
from math import isfinite
from typing import Callable, Literal, Union

from hotpot.cheminfo.structure.molecule import Molecule
from ...structure.bond_kind import BondKind
from ...rules.chemistry.bonding.hydrogen import _calculate_implicit_hydrogens
from ...structure._commit import _FactTransaction, _commit_atom_facts, _commit_bond_facts
from ...structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from ...structure._identity import _capture_source, _hydrogen_inputs, _hydrogen_transport_inputs
from ...structure._mutation import _normalize_atom_fields
from ...structure._stereo import _get_stereo_payload
from ...structure.stereo import StereoCarrierKind
from ..chemistry import _require_implicit_hydrogens, _restore_assembly_aromaticity, _set_stereo_payload


DefaultAction = Literal['atom_link', 'bond_shoulder']
ActionFuncTemplate = Union[Callable[[Molecule, list[int], Molecule, list[int]], Molecule], DefaultAction]
_default_actions = {}
_EXPLICIT = ATOM_SOURCE_CODES['explicit']
_TOTAL_SOURCES = {_EXPLICIT, ATOM_SOURCE_CODES['implicit_hydrogens']}
_IMPORTED_SOURCES = {ATOM_SOURCE_CODES['rdkit'], ATOM_SOURCE_CODES['openbabel']}
_UNSPECIFIED_RDKIT = {'CHI_UNSPECIFIED', 'STEREONONE', 'STEREOANY'}


def actions_register(register_key: str):
    def register(action):
        _default_actions[register_key] = action
        return action
    return register


def _problem(code, action, atom, reason):
    raise ValueError(f'ASSEMBLY.{code} action={action} atom_index={atom.idx}: {reason}')


def _count(value, action, atom):
    if not isfinite(value) or value < 0 or value != int(value):
        _problem('HYDROGEN_INVENTORY', action, atom, f'invalid current H count {value!r}')
    return int(value)


def _inputs(atom):
    mol = atom.mol
    transport = {entry[0]: entry[1:] for entry in _hydrogen_transport_inputs(
        tuple(member._uid for member in mol._atoms), _get_stereo_payload(mol))}
    return _hydrogen_inputs(atom, transport)


def _heavy_inputs(atom):
    inputs = _inputs(atom)
    indices = tuple(i for i, number in enumerate(inputs[1]) if number != 1)
    return (inputs[0], *(tuple(inputs[field][i] for i in indices) for field in (1, 2)),
            inputs[3], inputs[4],
            *(tuple(inputs[field][i] for i in indices) for field in (5, 6, 7)),
            *inputs[8:])


def _capacity(atom, action):
    inputs = _heavy_inputs(atom)
    capacity = _calculate_implicit_hydrogens(*inputs[:9], 0, False)
    # Neutral aromatic N does not acquire unspecified H under Stage5. Its
    # declared graph/bracket caps still occupy the ordinary three sigma slots.
    if inputs[0] == 7 and inputs[3] and inputs[4] == 0:
        orders = tuple(1. if flag or kind == 'aromatic' else order
                       for order, kind, flag in zip(inputs[2], inputs[5], inputs[6]))
        kinds = tuple('single' if flag or kind == 'aromatic' else kind
                      for kind, flag in zip(inputs[5], inputs[6]))
        capacity = _calculate_implicit_hydrogens(inputs[0], inputs[1], orders, False,
            inputs[4], kinds, (False,) * len(kinds), inputs[7], inputs[8], 0, False)
    if capacity is None:
        _problem('HYDROGEN_UNAVAILABLE', action, atom, 'final heavy connectivity has unresolved H inputs')
    return capacity


def _inventory(atom, action):
    mol, field = atom.mol, FIELD_INDEX['implicit_hydrogens']
    table, row = mol._atom_table, atom.idx
    if (int(table._presence[row]) & FIELD_PRESENCE_BITS[field]
            and int(table._sources[row, field]) != ATOM_SOURCE_CODES['implicit_hydrogens']):
        _count(float(table._columns[field][row]), action, atom)
    try:
        _require_implicit_hydrogens(mol, rows=(row,))
    except (ValueError, RuntimeError) as error:
        raise ValueError(f'ASSEMBLY.HYDROGEN_UNAVAILABLE action={action} atom_index={row}: {error}') from error
    table = mol._atom_table
    count = _count(float(table._columns[field][row]), action, atom)
    source = int(table._sources[row, field])
    graph = tuple(atom.hydrogens)
    for hydrogen in graph:
        bond = mol.bond(atom.idx, hydrogen.idx)
        if (len(hydrogen.neighbours) != 1 or bond.bond_kind is not BondKind.SINGLE
                or bond._is_aromatic or hydrogen.formal_charge != 0):
            _problem('HYDROGEN_INVENTORY', action, atom, 'graph H is not a neutral single-bond cap')
    if source in _TOTAL_SOURCES:
        total, residual = count, count - len(graph)
    elif source in _IMPORTED_SOURCES:
        total, residual = count + len(graph), count
    else:
        _problem('HYDROGEN_INVENTORY', action, atom, f'unregistered H producer {source}')
    if residual < 0:
        _problem('HYDROGEN_INVENTORY', action, atom, 'site-total H is smaller than graph H')
    payload = _get_stereo_payload(mol)
    facts = () if payload is None else tuple(f for f in payload.atom_facts if f.atom == row)
    current = tuple(f.current_hydrogens for f in facts if f.current_hydrogens is not None)
    brackets = []
    for state in current:
        implicit = _count(0 if state.implicit_hydrogens is None else state.implicit_hydrogens, action, atom)
        bracket = _count(0 if state.bracket_explicit_hydrogens is None else state.bracket_explicit_hydrogens, action, atom)
        if implicit + bracket != residual or (state.no_implicit and implicit):
            _problem('HYDROGEN_INVENTORY', action, atom, 'current H decomposition disagrees with represented inventory')
        brackets.append(bracket)
    inputs = _inputs(atom)
    bracket = inputs[9]
    if any(value != bracket for value in brackets):
        _problem('HYDROGEN_INVENTORY', action, atom, 'current bracket-H occurrences disagree')
    if total > _capacity(atom, action):
        _problem('HYDROGEN_INVENTORY', action, atom, 'represented H overfills the capped site')
    return {'total': total, 'residual': residual, 'bracket': bracket,
            'graph': graph, 'no_implicit': inputs[10]}


def _assembly_hydrogen_total(atom):
    """Physical H at a coherent assembly site; no global field reinterpretation."""
    with _FactTransaction(atom.mol, producer=_EXPLICIT):
        return _inventory(atom, 'site-selection')['total']


def _check_stereo(mol, atoms, action):
    payload = _get_stereo_payload(mol)
    if payload is None:
        return
    rows = {atom.idx for atom in atoms}
    for record in payload.records:
        touched = rows.intersection(record.anchor.atom_indices) or any(
            carrier.kind is StereoCarrierKind.ATOM and carrier.atom in rows
            for carrier in record.carriers)
        unspecified = (record.source_namespace.startswith('rdkit')
                       and record.descriptor in _UNSPECIFIED_RDKIT)
        if touched and not unspecified:
            atom = next(atom for atom in atoms if atom.idx in rows)
            _problem('STEREO_TRANSFORMATION_UNDEFINED', action, atom,
                     f'affected stereo descriptor {record.descriptor!r}')
    for fact in payload.bond_facts:
        if rows.intersection(fact.atom_indices) and fact.display_direction not in (None, 'NONE'):
            _problem('STEREO_TRANSFORMATION_UNDEFINED', action, atoms[0],
                     f'affected bond display {fact.display_direction!r}')


def _remove_caps(atom, graph, number, action):
    if not number:
        return ()
    payload = _get_stereo_payload(atom.mol)
    facts = {} if payload is None else {
        row: tuple(f.isotope for f in payload.atom_facts if f.atom == row)
        for row in (h.idx for h in graph)}
    ordinary, labelled = [], []
    for hydrogen in graph:
        isotopes = tuple(0 if value in (None, 0, 1) else value for value in (facts.get(hydrogen.idx) or (0,)))
        if len(set(isotopes)) != 1:
            _problem('ISOTOPE_AMBIGUITY', action, atom, 'conflicting graph-H isotope occurrences')
        (ordinary if isotopes[0] == 0 else labelled).append((hydrogen, isotopes[0]))
    selected = ordinary[:number]
    remaining = number - len(selected)
    if remaining:
        if remaining < len(labelled) and len({isotope for _, isotope in labelled}) > 1:
            _problem('ISOTOPE_AMBIGUITY', action, atom, 'cap removal yields distinct isotope products')
        selected += labelled[:remaining]
    return tuple(hydrogen for hydrogen, _ in selected)


def _consumption(atom, inventory, action):
    if inventory['total'] < 1:
        _problem('HYDROGEN_UNAVAILABLE', action, atom, 'one capped H is required')
    residual = inventory['residual']
    removed = _remove_caps(atom, inventory['graph'], int(residual == 0), action)
    bracket = min(inventory['bracket'], max(residual - 1, 0))
    return inventory['total'] - 1, bracket, removed


def _write_hydrogens(atom, total, bracket, action):
    mol, row = atom.mol, atom.idx
    residual = total - len(atom.hydrogens)
    if residual < 0:
        _problem('HYDROGEN_INVENTORY', action, atom, 'product has more graph H than its site-total')
    payload = _get_stereo_payload(mol)
    if payload is not None:
        facts = []
        for fact in payload.atom_facts:
            current = fact.current_hydrogens
            if fact.atom == row and current is not None:
                current_bracket = residual if current.no_implicit else min(bracket, residual)
                fact = replace(fact, current_hydrogens=replace(current,
                    implicit_hydrogens=residual - current_bracket,
                    bracket_explicit_hydrogens=current_bracket))
            facts.append(fact)
        _set_stereo_payload(mol, replace(payload, atom_facts=tuple(facts)))
    _commit_atom_facts(mol, _normalize_atom_fields(_capture_source(mol), (row,), (atom._uid,),
        {'implicit_hydrogens': (total,)}, producer=_EXPLICIT))
    _inventory(atom, action)


def _complete(atom, inventory, action):
    capacity = _capacity(atom, action)
    graph = tuple(atom.hydrogens)
    removed = _remove_caps(atom, graph, max(len(graph) - capacity, 0), action)
    if removed:
        _check_stereo(atom.mol, (atom, *removed), action)
        atom.mol.remove_atoms(removed)
    remaining_graph = len(atom.hydrogens)
    bracket = min(inventory['bracket'], max(capacity - remaining_graph, 0))
    inputs = _heavy_inputs(atom)
    inferred = _calculate_implicit_hydrogens(*inputs[:9], bracket, inventory['no_implicit'])
    if inferred is None:
        _problem('HYDROGEN_UNAVAILABLE', action, atom, 'product H completion is unresolved')
    total = max(remaining_graph + bracket, inferred)
    _write_hydrogens(atom, total, bracket, action)


def _transfer_bond(mol, bond, old, new):
    first = new if bond.atom1 is old else bond.atom1
    second = new if bond.atom2 is old else bond.atom2
    copied = mol.add_bond(first, second, bond.bond_order, bond_kind=bond.bond_kind,
        bond_direction=bond.bond_direction, bond_source=bond.bond_source,
        bond_source_metadata=bond.bond_source_metadata)
    _commit_bond_facts(mol, copied, _capture_source(mol),
        raw_values=bond.attrs,
        kind=bond.bond_kind, direction=bond.bond_direction,
        source_namespace=bond.bond_source, metadata=bond.bond_source_metadata,
        aromatic=bond._is_aromatic)


@actions_register('AtomLink')
def atom_link_atom_action(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int]):
    assert len(hit) == len(action_points) == 1
    working = frag.copy()
    with _FactTransaction(mol, producer=_EXPLICIT):
        parent, incoming = mol.atoms[hit[0]], working.atoms[action_points[0]]
        parent_h, incoming_h = _inventory(parent, 'AtomLink'), _inventory(incoming, 'AtomLink')
        parent_plan = _consumption(parent, parent_h, 'AtomLink')
        incoming_plan = _consumption(incoming, incoming_h, 'AtomLink')
        _check_stereo(mol, (parent, *parent_plan[2]), 'AtomLink')
        _check_stereo(working, (incoming, *incoming_plan[2]), 'AtomLink')
        offset = len(mol.atoms)
        mol.add_component(working)
        added = tuple(mol.atoms[offset:])
        attachment = added[incoming.idx]
        removed = (*parent_plan[2], *(added[h.idx] for h in incoming_plan[2]))
        if removed:
            mol.remove_atoms(removed)
        mol.add_bond(parent, attachment, 1., bond_kind=BondKind.SINGLE)
        _write_hydrogens(parent, *parent_plan[:2], 'AtomLink')
        _write_hydrogens(attachment, *incoming_plan[:2], 'AtomLink')
        return mol


@actions_register('BondAdd')
def bond_order_add(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int]):
    assert len(hit) == 2
    with _FactTransaction(mol, producer=_EXPLICIT):
        bond = mol.bond(*hit)
        if bond._is_aromatic or bond.bond_kind not in (BondKind.SINGLE, BondKind.DOUBLE):
            _problem('HYDROGEN_INVENTORY', 'BondAdd', bond.atom1,
                     'only nonaromatic single/double bond increments are defined')
        atoms = bond.atoms
        inventories = tuple(_inventory(atom, 'BondAdd') for atom in atoms)
        plans = tuple(_consumption(atom, inventory, 'BondAdd') for atom, inventory in zip(atoms, inventories))
        removed = tuple(h for plan in plans for h in plan[2])
        _check_stereo(mol, (*atoms, *removed), 'BondAdd')
        if removed:
            mol.remove_atoms(removed)
        bond.bond_order += 1
        for atom, plan in zip(atoms, plans):
            _write_hydrogens(atom, *plan[:2], 'BondAdd')
        return mol


@actions_register('AtomReplace')
def atom_replace(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int]):
    assert len(hit) == 1 and len(frag.atoms) == 1
    working = frag.copy()
    with _FactTransaction(mol, producer=_EXPLICIT):
        atom = mol.atoms[hit[0]]
        inventory = _inventory(atom, 'AtomReplace')
        _check_stereo(mol, (atom, *inventory['graph']), 'AtomReplace')
        atom.atomic_number = working.atoms[0].atomic_number
        _complete(atom, inventory, 'AtomReplace')
        return mol


@actions_register('EdgeShoulder')
def shoulder_bond_action(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int]):
    assert len(hit) == len(action_points) == 2
    working = frag.copy()
    with _FactTransaction(mol, producer=_EXPLICIT):
        old = tuple(mol.atoms[i] for i in hit)
        anchors = tuple(working.atoms[i] for i in action_points)
        assert mol.bond(old[0].idx, old[1].idx) is not None
        old_h = tuple(_inventory(atom, 'EdgeShoulder') for atom in old)
        inventories = tuple(_inventory(atom, 'EdgeShoulder') for atom in anchors)
        deleted = (*old, *(h for inventory in old_h for h in inventory['graph']))
        _check_stereo(mol, deleted, 'EdgeShoulder')
        _check_stereo(working, (*anchors, *(h for inventory in inventories for h in inventory['graph'])), 'EdgeShoulder')
        incoming_aromatic = tuple(atom.is_aromatic for atom in anchors)
        aromatic = tuple(atom.is_aromatic or incoming
            for atom, incoming in zip(old, incoming_aromatic))
        transfers = tuple(tuple(bond for bond in atom.bonds
            if all(other not in old and other.atomic_number != 1 for other in bond.atoms if other is not atom))
            for atom in old)
        offset = len(mol.atoms)
        mol.add_component(working)
        replacements = tuple(mol.atoms[offset + atom.idx] for atom in anchors)
        for original, replacement, bonds in zip(old, replacements, transfers):
            for bond in bonds:
                _transfer_bond(mol, bond, original, replacement)
        mol.remove_atoms(deleted)
        _restore_assembly_aromaticity(mol, tuple(zip(replacements, aromatic)))
        adopted = tuple((atom, flag) for atom, flag, incoming
            in zip(replacements, aromatic, incoming_aromatic) if flag != incoming)
        if adopted:
            # Changed anchor flags are explicit product facts before either H write.
            _commit_atom_facts(mol, _normalize_atom_fields(_capture_source(mol),
                tuple(atom.idx for atom, _ in adopted),
                tuple(atom._uid for atom, _ in adopted),
                {'is_aromatic': tuple(flag for _, flag in adopted)}, producer=_EXPLICIT))
        for atom, inventory in zip(replacements, inventories):
            _complete(atom, inventory, 'EdgeShoulder')
        return mol


@actions_register('RingWedge')
def ring_wedge(mol: Molecule, hit: list[int], frag: Molecule, action_points: list[int]):
    assert len(hit) == len(action_points) == 1
    working = frag.copy()
    with _FactTransaction(mol, producer=_EXPLICIT):
        atom, anchor = mol.atoms[hit[0]], working.atoms[action_points[0]]
        assert len(atom.heavy_neighbours) == 1
        inventory, anchor_h = _inventory(atom, 'RingWedge'), _inventory(anchor, 'RingWedge')
        if inventory['total'] < 2:
            _problem('HYDROGEN_UNAVAILABLE', 'RingWedge', atom, 'at least two capped H are required')
        _check_stereo(mol, (atom, *inventory['graph']), 'RingWedge')
        _check_stereo(working, (anchor, *anchor_h['graph']), 'RingWedge')
        aromatic = anchor.is_aromatic
        offset = len(mol.atoms)
        mol.add_component(working)
        added = tuple(mol.atoms[offset:])
        displaced = added[anchor.idx]
        deleted = (displaced, *(added[h.idx] for h in anchor_h['graph']))
        for bond in tuple(displaced.bonds):
            if all(other.atomic_number != 1 for other in bond.atoms if other is not displaced):
                _transfer_bond(mol, bond, displaced, atom)
        mol.remove_atoms(deleted)
        _restore_assembly_aromaticity(mol, ((atom, aromatic),))
        # The retained site's adopted scaffold flag is an explicit product fact,
        # so H invalidation cannot re-infer it from the previous terminal site.
        _commit_atom_facts(mol, _normalize_atom_fields(_capture_source(mol),
            (atom.idx,), (atom._uid,), {'is_aromatic': (aromatic,)}, producer=_EXPLICIT))
        _complete(atom, inventory, 'RingWedge')
        return mol
