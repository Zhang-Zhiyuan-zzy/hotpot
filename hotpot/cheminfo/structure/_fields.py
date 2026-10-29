"""Private Atom projections shared by storage and the existing facade.

Codecs describe scalar reads, not storage casts. The old numeric projection
keeps all 19 float64 values until each owning stage migrates its facts/results.
"""

from types import MappingProxyType


__all__ = ()


ATOM_SCHEMA = (
    ("atomic_number", int),
    ("n", int),
    ("s", int),
    ("p", int),
    ("d", int),
    ("f", int),
    ("g", int),
    ("formal_charge", int),
    ("partial_charge", float),
    ("is_aromatic", bool),
    ("x", float),
    ("y", float),
    ("z", float),
    ("valence", int),
    ("implicit_hydrogens", int),
    ("id", int),
    ("x_constraint", bool),
    ("y_constraint", bool),
    ("z_constraint", bool),
)
ATOM_FIELDS = tuple(name for name, _ in ATOM_SCHEMA)
FIELD_INDEX = MappingProxyType({name: index for index, name in enumerate(ATOM_FIELDS)})
FIELD_CODECS = MappingProxyType(dict(ATOM_SCHEMA))
FIELD_PRESENCE_BITS = tuple(1 << index for index in range(len(ATOM_FIELDS)))
ATOM_PRESENCE_ALL = (1 << len(ATOM_FIELDS)) - 1
ATOM_SOURCE_CODES = MappingProxyType({
    "default": 0,
    "explicit": 1,
    "legacy_array": 2,
    "rdkit": 3,
    "openbabel": 4,
    "ccdc": 5,
    "mca_dto": 6,
    "electron_configuration": 7,
    "valence": 8,
    "implicit_hydrogens": 9,
})
COORDINATE_FIELDS = ("x", "y", "z")
SCALAR_FIELDS = tuple(name for name in ATOM_FIELDS if name not in COORDINATE_FIELDS)
FIELD_ADDRESSES = tuple(
    (1, COORDINATE_FIELDS.index(name)) if name in COORDINATE_FIELDS
    else (0, SCALAR_FIELDS.index(name))
    for name in ATOM_FIELDS
)
ALIAS_TARGETS = MappingProxyType({
    "symbol": ("atomic_number",),
    "coordinates": COORDINATE_FIELDS,
    "constraint": ("x_constraint", "y_constraint", "z_constraint"),
})
# This is the reviewed batch API, not permission to remove legacy setters
# before their owning-stage migration.
ATOM_BATCH_WRITABLE = frozenset({
    "atomic_number", "symbol", "formal_charge", "implicit_hydrogens", "id",
    "coordinates", *COORDINATE_FIELDS,
})
ATOM_BATCH_READONLY = frozenset({
    *ATOM_FIELDS, *ALIAS_TARGETS,
    "attrs", "attrs_enumerator", "electron_configuration", "idx", "mol", "mca",
    "isotope",
}) - ATOM_BATCH_WRITABLE
