"""Current factual topology membership and index construction."""

from .errors import NotInSameMolecule, ObjNotInMolecule


def _selected_atom_rows(mol, atoms, *, public=False):
    rows = []
    seen = set()
    for atom in atoms:
        if atom.mol is not mol:
            if public:
                raise NotInSameMolecule(mol, atom)
            raise ValueError('Selected Atom belongs to another Molecule')
        record = mol._topology_index.get(id(atom))
        if record is None:
            if public:
                raise ObjNotInMolecule(atom, mol)
            raise KeyError(id(atom))
        row = record[0]
        if (atom._uid_owner is not mol._owner_token or mol._atoms[row] is not atom
                or id(atom) in seen):
            raise ValueError('Atom selection requires unique current members')
        seen.add(id(atom))
        rows.append(row)
    return tuple(rows)


def _build_topology_index(atoms, bonds):
    rows = {}
    incident = {}
    for row, atom in enumerate(atoms):
        if id(atom) in rows:
            raise ValueError('a Molecule cannot contain the same Atom twice')
        if atoms and atom.mol is not atoms[0].mol:
            raise ValueError('every Atom member must belong to its Molecule')
        rows[id(atom)] = row
        incident[id(atom)] = []
    for bond in bonds:
        if id(bond.atom1) not in rows or id(bond.atom2) not in rows:
            raise ValueError('every Bond endpoint must belong to its Molecule')
    pairs = set()
    for bond in bonds:
        first, second = rows[id(bond.atom1)], rows[id(bond.atom2)]
        if first == second:
            raise ValueError('a Molecule Bond cannot join an Atom to itself')
        pair = tuple(sorted((first, second)))
        if pair in pairs:
            raise ValueError('a Molecule cannot contain parallel Bonds')
        pairs.add(pair)
        incident[id(bond.atom1)].append(bond)
        incident[id(bond.atom2)].append(bond)
    return {id(atom): (rows[id(atom)], tuple(sorted(incident[id(atom)], key=lambda bond:
                rows[id(bond.atom2 if bond.atom1 is atom else bond.atom1)]))) for atom in atoms}
