"""Lossless assembly of stereo transport from its canonical fact owners."""

from dataclasses import replace

from .stereo import (
    AtomTransportRecord, BondTransportRecord, HydrogenTransportState,
    StereoPayload,
)

__all__ = ()


def _decompose_stereo(mol, sealed):
    """Split transport occurrences without interpreting their values or order."""

    atom_rows = [[] for _ in mol._atoms]
    bond_rows = [[] for _ in mol._bonds]
    if sealed is None:
        return None, (), ((), ()), tuple(map(tuple, atom_rows)), tuple(map(tuple, bond_rows))

    atom_layout, bond_layout = [], []
    for fact in sealed.atom_facts:
        if not 0 <= fact.atom < len(atom_rows):
            atom_layout.append(fact)
            continue
        row = atom_rows[fact.atom]
        atom_layout.append((mol._atoms[fact.atom]._uid, len(row), fact.source_hydrogens))
        current = fact.current_hydrogens
        row.append((fact.isotope, fact.radical_electrons, current is not None,
                    None if current is None else current.implicit_hydrogens,
                    None if current is None else current.bracket_explicit_hydrogens,
                    None if current is None else current.no_implicit,
                    fact.source_namespace))

    bonds_by_edge = {frozenset((bond.atom1._uid, bond.atom2._uid)): (index, bond)
                     for index, bond in enumerate(mol._bonds)}
    for fact in sealed.bond_facts:
        first, second = fact.atom_indices
        if not (0 <= first < len(atom_rows) and 0 <= second < len(atom_rows)):
            bond_layout.append(fact)
            continue
        first_uid, second_uid = mol._atoms[first]._uid, mol._atoms[second]._uid
        target = bonds_by_edge.get(frozenset((first_uid, second_uid)))
        if target is None:
            bond_layout.append(fact)
            continue
        index, bond = target
        row = bond_rows[index]
        bond_layout.append((bond._uid, len(row)))
        row.append((first_uid, second_uid, fact.display_direction))

    return (sealed.schema_version, sealed.records, (tuple(atom_layout), tuple(bond_layout)),
            tuple(map(tuple, atom_rows)), tuple(map(tuple, bond_rows)))


def _install_stereo_storage(mol, sealed):
    """Install the split on a freshly built table under its caller's owner scope."""

    schema, records, source, atom_rows, bond_rows = _decompose_stereo(mol, sealed)
    mol._atom_table._stereo_facts[:len(atom_rows)] = atom_rows
    mol._stereo_schema, mol._stereo_records, mol._stereo_source = schema, records, source
    for bond, row in zip(mol._bonds, bond_rows):
        bond._display_facts = row


def _get_stereo_payload(mol):
    """Assemble raw transport under the owner lock without preparing chemistry."""

    with mol._mutation_lock:
        if mol._stereo_schema is None:
            return None
        atom_indices = {atom._uid: row for row, atom in enumerate(mol._atoms)}
        bonds = {bond._uid: bond for bond in mol._bonds}
        atom_layout, bond_layout = mol._stereo_source
        atom_facts = []
        for entry in atom_layout:
            if type(entry) is AtomTransportRecord:
                atom_facts.append(entry)
                continue
            uid, occurrence, source_hydrogens = entry
            row = atom_indices[uid]
            isotope, radical, present, implicit, bracket, no_implicit, namespace = (
                mol._atom_table._stereo_facts[row][occurrence])
            current = HydrogenTransportState(implicit, bracket, no_implicit) if present else None
            atom_facts.append(AtomTransportRecord(
                row, isotope, radical, current, source_hydrogens, namespace))
        bond_facts = []
        for entry in bond_layout:
            if type(entry) is BondTransportRecord:
                bond_facts.append(entry)
                continue
            uid, occurrence = entry
            first, second, direction = bonds[uid]._display_facts[occurrence]
            bond_facts.append(BondTransportRecord(
                (atom_indices[first], atom_indices[second]), direction))
        return StereoPayload(mol._stereo_schema, mol._stereo_records,
                             tuple(atom_facts), tuple(bond_facts))


def _stereo_storage_state(mol):
    """Retain exact source and display owners for the existing transaction."""

    return (mol._stereo_schema, mol._stereo_records, mol._stereo_source,
            tuple((bond, bond._display_facts) for bond in mol._bonds))


def _restore_stereo_storage(mol, state):
    """Restore retained owners; native Undo separately restores Atom rows."""

    mol._stereo_schema, mol._stereo_records, mol._stereo_source, displays = state
    for bond, row in displays:
        bond._display_facts = row


def _atom_isotope(atom):
    """Project equal isotope occurrences, reporting ambiguity at scalar use."""

    mol = atom._mol
    with mol._mutation_lock:
        entry = mol._topology_index.get(id(atom))
        table, row = (atom._detached_table, 0) if entry is None else (mol._atom_table, entry[0])
        facts = table._stereo_facts[row]
        if not facts:
            raise AttributeError("isotope")
        isotopes = tuple(fact[0] for fact in facts)
        if any(value != isotopes[0] for value in isotopes[1:]):
            raise ValueError(f"Conflicting isotope declarations for atom row {row}: {isotopes!r}")
        return isotopes[0]


def _replace_isotope(payload, row, value):
    """Replace all row occurrences with an already normalized isotope value."""

    if payload is None:
        payload = StereoPayload(1, (), (), ())
    facts = tuple(replace(fact, isotope=value) if fact.atom == row else fact
                  for fact in payload.atom_facts)
    if not any(fact.atom == row for fact in facts):
        facts += (AtomTransportRecord(row, value, None, None, None, None),)
    return replace(payload, atom_facts=facts)
