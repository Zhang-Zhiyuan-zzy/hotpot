"""Private Atom projections shared by storage and the existing facade.

Codecs describe scalar reads, not storage casts. The historical 19-column
transport/projection order remains; eight derived slots have no value storage.
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
TABLE_FIELDS = ATOM_FIELDS[:16]
CONSTRAINT_FIELDS = ATOM_FIELDS[16:]
TABLE_PRESENCE_ALL = (1 << len(TABLE_FIELDS)) - 1
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
    "cclib": 10,
    "conformer_load": 11,
    "geometry": 12,
    "forcefield": 13,
    "xtb": 14,
    "aromaticity": 15,
    "kekule": 16,
    "partial_charge": 17,
    "tmqm": 18,
    "bond_order": 19,
})
COORDINATE_FIELDS = ("x", "y", "z")
DERIVED_FIELDS = frozenset({"n", "s", "p", "d", "f", "g", "valence", "partial_charge"})
FACT_FIELDS = tuple(name for name in ATOM_FIELDS if name not in DERIVED_FIELDS)
ATOM_SCALAR_READONLY = frozenset({*DERIVED_FIELDS, "is_aromatic", "electron_configuration"})
SCALAR_FIELDS = tuple(name for name in TABLE_FIELDS
                      if name not in DERIVED_FIELDS and name not in COORDINATE_FIELDS)
FIELD_ADDRESSES = tuple(
    None if name in DERIVED_FIELDS else
    (1, COORDINATE_FIELDS.index(name)) if name in COORDINATE_FIELDS
    else (0, SCALAR_FIELDS.index(name))
    for name in TABLE_FIELDS
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


class MolBlock:
    """
    Represents an abstraction of a molecular block.

    The MolBlock class provides an abstraction to represent and manage molecular
    blocks, typically used in chemical or molecular modeling. It provides methods
    and properties to interact with molecular attributes, rings, and other
    chemical properties. The class supports controlled attribute setting,
    retrieval, and enumeration while limiting certain operations like copying.

    Attributes:
        _attrs_dict (dict): Internal mapping of molecular attributes to their
            respective processing functions.
        _attrs_setter (dict): Internal mapping of molecular attributes to their
            respective setter functions.
        _default_attrs (dict): Default molecular attribute values.
        _attrs_enumerator (tuple): A tuple enumerating all attribute names in
            _attrs_dict.

    Raises:
        PermissionError: When attempting to copy an instance of MolBlock.

    Methods:
        __repr__: Returns a string representation of the MolBlock instance.
        __copy__: Restricts copying of the MolBlock instance; raises PermissionError.
        __dir__: Modifies the directory of attributes to include the molecular
            attributes in _attrs_enumerator.
        __getattr__: Retrieves a molecular attribute's value based on its processor
            function or delegates to the parent class for non-existent attributes.
        __setattr__: Sets an attribute using a custom setter function; defaults to
            a custom handler for key-value assignment.
        _default_attr_setter: Default attribute setter function for numerical values.
        attrs_enumerator: Property to get the tuple of molecular attribute names.
        attr_dict: Property to get a dictionary mapping attribute names to values.
        in_ring: Property to check if the molecular block is part of any ring.
        in_organic: Property to check if the molecular block belongs to an
            organic molecule.
        rings: Property to get a list of rings involving the molecular block.
    """
    _attrs_dict = {}
    _attrs_setter = {}
    _default_attrs = {}
    _attrs_enumerator = tuple(_attrs_dict.keys())

    @property
    def _uid(self):
        return self._member_uid

    def __repr__(self):
        return f"{self.__class__.__name__}({self.label})"

    def __copy__(self):
        raise PermissionError(f"The {self.__class__.__name__} not allow to copy")

    def __dir__(self):
        return list(self._attrs_enumerator) + list(super(MolBlock, self).__dir__())

    def __getattr__(self, item):
        from .atom import Atom, _atom_numeric_value
        from .constraints import _bond_constraint_value
        if item in self._attrs_dict:
            if isinstance(self, Atom):
                return _atom_numeric_value(self, item)
            with self.mol._mutation_lock:
                value = (_bond_constraint_value(self._constraint_values[self._uid]) if item == 'constraint'
                         else self._attrs[0 if item == 'bond_order' else 1])
                return self._attrs_dict[item](value)
        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        from .atom import Atom, _atom_fact_engine
        from .bond import Bond, _bond_order_setter, _store_bond_field
        if isinstance(self, Atom):
            _atom_fact_engine()
            Atom.__setattr__(self, key, value)
        elif isinstance(self, Bond) and key in self._attrs_dict:
            if key == 'bond_order':
                _bond_order_setter(self, key, value)
            else:
                _store_bond_field(self, key, value)
        else:
            object.__setattr__(self, key, value)

    @staticmethod
    def _default_attr_setter(self, key, value):
        from .atom import Atom, _store_atom_fields
        from .bond import _store_bond_field
        if isinstance(self, Atom):
            _store_atom_fields(self, {key: value})
        else:
            _store_bond_field(self, key, value)

    @property
    def attrs_enumerator(self) -> tuple:
        return self._attrs_enumerator

    @property
    def attr_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.attrs_enumerator}

    @property
    def in_ring(self):
        """
        Determine if the atom or group is part of one or more rings in the molecule.

        Check each ring in the molecule associated with the object and confirm
        if the object exists within any of those rings.

        @return: Boolean indicating whether the object is part of any ring

        """
        return any(self in r for r in self.mol.rings)

    @property
    def in_organic(self) -> bool:
        return self.mol.is_organic

    @property
    def rings(self):
        """
        Represents a property that computes and returns a list of rings containing the instance.

        This property iterates over the rings of a molecule and checks if the object
        invoking the property is a member of each ring. If the object is part of any
        rings, they are included in the returned list.

        @return: A list of rings containing the invoking instance
        """
        return [r for r in self.mol.rings if self in r]
