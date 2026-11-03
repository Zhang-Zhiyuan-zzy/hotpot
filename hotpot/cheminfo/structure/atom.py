"""The factual Atom carrier and its scalar native dispatch boundary."""

import operator
from typing import Optional, Union

import cython
import numpy as np

from hotpot.utils import types
from .molecule import Molecule
from ._fields import (
    MolBlock, ATOM_FIELDS as _ATOM_FIELDS,
    ATOM_SOURCE_CODES as _ATOM_SOURCE_CODES, FIELD_INDEX as _ATOM_FIELD_INDEX,
    FIELD_CODECS as _ATOM_FIELD_CODECS, DERIVED_FIELDS as _ATOM_DERIVED_FIELDS,
    ATOM_SCALAR_READONLY as _ATOM_SCALAR_READONLY,
    CONSTRAINT_FIELDS as _ATOM_CONSTRAINT_FIELDS,
)
from ._atom_table import _AtomFactTable
from .constraints import _EMPTY_ATOM_CELL, _atom_axis_value, _split_atom_rows
from .errors import NotInSameMolecule, ObjNotInMolecule
from ._identity import _capture_source
from ._results import (
    _current_mca_result, _invalidate_results, _atom_partial_charge,
    _partial_charge_record, _install_partial_charge_record,
)
from ._mutation import _normalize_atom_facts
from ._commit import _FactTransaction, _fact_engine, _commit_atom_sparse
from ._dependencies import _mark_topology_dirty, _seed, GROUP_INDEX
from ._stereo import _atom_isotope
from ._state import _restore_atom_member, _restore_detached_atom


def _atomic_number_setter(self: 'Atom', key, atomic_number):
    _store_atom_fields(self, {'atomic_number': atomic_number})


class Atom(MolBlock):
    """
    Represents a single atom with detailed attributes and properties.

    The class provides atomic data and models an atomic entity, including
    attributes such as position coordinates (x, y, z), atomic number,
    valence, charge details, and its chemical symbol. It also encapsulates
    essential chemical information (e.g., atomic orbitals, periodic table
    data, and default valencies) useful for computational chemistry and
    molecular modeling tasks.

    Attributes
    ----------
    atomic_number : int
        Atomic number of the element.
    formal_charge : int
        Formal charge associated with the atom.
    partial_charge : float
        Partial charge of the atom derived from calculations or estimations.
    x : float
        X-coordinate of the atom in 3D Cartesian space.
    y : float
        Y-coordinate of the atom in 3D Cartesian space.
    z : float
        Z-coordinate of the atom in 3D Cartesian space.
    valence : int
        Valence electrons or bonding capacity of the atom.
    id : int
        Unique identifier for the atom instance.
    symbol : str
        Elemental symbol representing the atom (e.g., "H" for Hydrogen).
    idx : int
        Index of the atom in molecular structures or calculations.

    """
    # Cython define
    atomic_number: int
    formal_charge: int
    partial_charge: Optional[float]
    x: float
    y: float
    z: float
    valence: int
    id: int
    symbol: str
    idx: int
    is_aromatic: bool
    implicit_hydrogens: int

    _coord_getter = operator.attrgetter('x', 'y', 'z')

    _attrs_dict = dict(_ATOM_FIELD_CODECS)

    _attrs_setter = {
        'atomic_number': _atomic_number_setter
    }

    _default_attrs = {
        "atomic_number": 0,
        "is_aromatic": False,
        "formal_charge": 0,
        "coordinates": (0., 0., 0.),
        "x": 0.,
        "y": 0.,
        "z": 0.,
        "valence": 0,
        "implicit_hydrogens": 0,
    }
    _attrs_enumerator = tuple(_attrs_dict.keys())

    # Index Check
    _ELECTRON_N_CONFIG = _attrs_enumerator.index('n')
    _ELECTRON_G_CONFIG = _attrs_enumerator.index('g')

    from ..elements import elements

    def __init__(self, mol: Molecule = None, **kwargs):
        owner = mol if mol is not None else Molecule()
        uid = owner._allocate_atom_uid()
        allowed = {'atomic_number', 'symbol', 'formal_charge', 'implicit_hydrogens',
                   'coordinates', 'x', 'y', 'z', 'id', 'constraint',
                   'x_constraint', 'y_constraint', 'z_constraint'}
        unknown = kwargs.keys() - allowed
        if unknown:
            raise TypeError(f"Unsupported Atom constructor fields: {', '.join(sorted(unknown))}")
        constraints = {name: kwargs.pop(name) for name in tuple(kwargs)
                       if name in ('constraint', 'x_constraint', 'y_constraint', 'z_constraint')}
        with _FactTransaction(owner, producer=_ATOM_SOURCE_CODES['explicit']) as transaction:
            row = len(owner._atom_table)
            update = _normalize_atom_facts(_capture_source(owner), (row,), (uid,), kwargs)
            values = np.zeros(19, dtype=np.float64)
            presence = 0
            sources = np.zeros((1, 19), dtype=np.uint8)
            for field, column in zip(update.fields, update.values):
                index = _ATOM_FIELD_INDEX[field]
                values[index] = column[0]
                presence |= 1 << index
                sources[0, index] = _ATOM_SOURCE_CODES['explicit']
            for axis in ('x_constraint', 'y_constraint', 'z_constraint'):
                if axis in constraints or 'constraint' in constraints:
                    value = constraints.get(axis, constraints.get('constraint'))
                    if 'constraint' in constraints and axis in constraints and bool(value) != bool(constraints['constraint']):
                        raise ValueError(f'Conflicting Atom constructor aliases for {axis}')
                    index = _ATOM_FIELD_INDEX[axis]
                    values[index] = bool(value)
                    presence |= 1 << index
                    sources[0, index] = _ATOM_SOURCE_CODES['explicit']
            table_values, masks, codes, cells = _split_atom_rows(
                values[None, :], presence=np.asarray([presence], dtype=np.uint32), sources=sources)
            _initialize_atom_handle(self, owner, uid, cell=cells[0])
            transaction.touch_registration(self)
            owner._atom_table._append_prepared(table_values, masks, codes)
            owner._atoms.append(self)
            owner._topology_index[id(self)] = (row, ())
            transaction.touch_constraint_value(owner._constraints.atom_axes, uid)
            owner._constraints.atom_axes[uid] = cells[0]
            object.__setattr__(self, '_constraint_values', owner._constraints.atom_axes)
            _invalidate_results(owner, ('representation', 'geometry'), transaction)
            _seed(owner, GROUP_INDEX['FORMAL_CHARGE_COVERAGE'], (uid,))

    @property
    def isotope(self):
        return _atom_isotope(self)

    @property
    def attrs(self):
        from ..operations.chemistry import _atom_projection, _detached_atom_projection
        with self.mol._mutation_lock:
            record = self.mol._topology_index.get(id(self))
            if record is None:
                return _detached_atom_projection(self)
            return _atom_projection(self.mol, rows=(record[0],))[0]

    def __reduce_ex__(self, protocol):
        with self.mol._mutation_lock:
            record = self.mol._topology_index.get(id(self))
            if record is not None:
                return _restore_atom_member, (self.mol, record[0], self._uid)
            state = self._detached_table.__getstate__()
            extra = {key: value for key, value in self.__dict__.items()
                     if key not in ('_mol', '_member_uid', '_uid_owner', '_detached_table', '_constraint_values')}
            return _restore_detached_atom, (self.mol, self._uid, state, self._constraint_values[self._uid]), extra

    def __setstate__(self, state):
        self.__dict__.update(state)


    def _add_atom(
            self,
            atom: Union[str, int, "Atom"] = 1,
            bond_order: int = 1,
            atom_attrs: dict = None,
            bond_attrs: dict = None
    ):
        if isinstance(atom, str):
            from ..reference.properties import _atomic_number_for_symbol
            atom = _atomic_number_for_symbol(atom)
        if isinstance(atom, int):
            atom = self.mol._create_atom(atomic_number=atom, **(atom_attrs or {}))
        else:
            atom = self.mol._add_atom(atom)
            if atom_attrs:
                self.mol.updata_atoms(atoms=(atom,), **atom_attrs)
        self.mol._add_bond(self, atom, bond_order, **(bond_attrs or {}))
        return atom

    def add_atom(
            self,
            atom: Union[str, int, "Atom"] = 1,
            bond_order: int = 1,
            atom_attrs: dict = None,
            bond_attrs: dict = None
    ):
        """
        Add an atom to a molecular graph with optional attributes and bond details.

        This function facilitates the addition of a single atom to the molecular
        graph. The user can specify the atom type, the bond order to the existing
        structure, and additional optional attributes for both the atom and the
        bond. The resulting atom is internally processed and added to the graph,
        and the molecular graph representation is updated afterward.

        Parameters:
            atom: Union[str, int, "Atom"]
                The atom to add. Can be a string (symbol), an integer (atomic number),
                or an Atom object. Defaults to integer 1.
            bond_order: int
                The bond order for connecting the new atom to the existing structure.
                Defaults to 1.
            atom_attrs: dict
                Additional attributes to assign to the newly added atom. Defaults
                to None.
            bond_attrs: dict
                Additional attributes to assign to the bond connecting the new atom.
                Defaults to None.

        Returns:
            atom
                The newly created and added atom object linked to the molecular graph.
        """
        from ..operations.topology import _add_atom
        return _add_atom(self, atom, bond_order, atom_attrs, bond_attrs)

    @staticmethod
    def random_point_on_sphere(
            radius: float = 1.,
            rng: Optional[np.random.Generator] = None,
    ):
        """
        Generates a random point on the surface of a sphere with a given radius.

        The function uses uniformly distributed random numbers to compute a point
        on the sphere by generating the polar angle (theta) and azimuthal angle
        (phi) in spherical coordinates, and converting them into Cartesian
        coordinates (x, y, z). The generated point lies on the sphere's surface.

        Args:
            radius (float): The radius of the sphere. Default is 1.0.
            rng (numpy.random.Generator, optional): Generator used for sampling.
                When omitted, NumPy's process-global random generator is used.

        Returns:
            tuple[float, float, float]: Cartesian coordinates (x, y, z) of the
            point on the sphere's surface.
        """
        # 随机生成极角 theta 和方位角 phi
        from ..operations.geometry import _random_point_on_sphere
        return _random_point_on_sphere(radius, rng)

    @property
    def period(self) -> int:
        return self.n

    @property
    def group(self):
        from ..operations.chemistry import _element_group
        return _element_group(self.atomic_number)

    @property
    def polar_hydrogen_site(self) -> bool:
        """
        Checks if the atomic site is a polar hydrogen site.

        A polar hydrogen site is identified by its atomic number and is determined to
        be polar if the atomic number equals 8 (oxygen) or, in the case of nitrogen
        (atomic number 7), an additional check ensures it is aromatic.

        Returns:
            bool: True if the atomic site is a polar hydrogen site, False otherwise.
        """
        from ..operations.chemistry import _atom_polar_hydrogen_site
        return _atom_polar_hydrogen_site(self)

    def _add_hydrogens(
            self,
            num: int = None,
            rm_polar_hs: bool = True,
            rng: Optional[np.random.Generator] = None,
    ) -> (int, list["Atom"]):
        from ..operations.chemistry import _add_atom_hydrogens
        return _add_atom_hydrogens(self, num, rm_polar_hs, rng)

    def add_hydrogen(self, num: int = None) -> list["Atom"]:
        """
        Adds hydrogen atoms to the molecule.

        This method adds hydrogen atoms to a molecule. The number of hydrogens to add
        can be specified, or the method may add them based on default behavior. After
        adding the hydrogens, the molecule graph is updated to reflect the changes.

        Args:
            num (int, optional): The number of hydrogen atoms to add. If None, the
            method will determine the number of hydrogens automatically.

        Returns:
            list["Atom"]: A list of the hydrogen atoms added to the molecule.
        """
        from ..operations.chemistry import _complete_atom_hydrogens
        return _complete_atom_hydrogens(self, num)

    @property
    def bonds(self) -> list['Bond']:
        with self.mol._mutation_lock:
            record = self.mol._topology_index.get(id(self))
            return [] if record is None else list(record[1])

    def _calc_implicit_hydrogens(self) -> int:
        # TODO: Implement by C++
        from ..operations.chemistry import _atom_implicit_proposal
        return _atom_implicit_proposal(self)


    def calc_implicit_hydrogens(self):
        from ..operations.chemistry import _calculate_hydrogens
        _calculate_hydrogens(self.mol, rows=(self.idx,))

    @property
    def constraint(self) -> bool:
        """
        Returns a boolean indicating whether all constraints (x_constraint,
        y_constraint, z_constraint) are satisfied.

        The method evaluates multiple constraints and determines if they are
        all True. This is commonly used to validate that all conditions
        required by the system are met.

        @return: Boolean value indicating whether all constraints are satisfied
        """
        return all([self.x_constraint, self.y_constraint, self.z_constraint])

    @constraint.setter
    def constraint(self, value: bool):
        _store_atom_fields(self, dict.fromkeys(('x_constraint', 'y_constraint', 'z_constraint'), value))

    @property
    def coordinates(self):
        with self.mol._mutation_lock:
            return Atom._coord_getter(self)

    @coordinates.setter
    def coordinates(self, value: types.ArrayLike):
        x, y, z = value
        _store_atom_fields(self, {'x': x, 'y': y, 'z': z})

    @property
    def exact_mass(self):
        from ..operations.chemistry import _element_exact_mass
        return _element_exact_mass(self.atomic_number)

    @property
    def electronegativity(self) -> Optional[float]:
        from ..operations.chemistry import _element_electronegativity
        return _element_electronegativity(self.atomic_number)

    @property
    def explicit_hydrogens(self) -> int:
        """
        Calculate the number of explicit hydrogen atoms attached.

        Returns the count of hydrogen atoms explicitly bonded as neighbors
        to the current atom in the structure. Only neighbors with an atomic
        number of 1 are considered hydrogen atoms.

        Returns:
            int: The count of explicit hydrogen atoms.
        """
        return len([a for a in self.neighbours if a.atomic_number == 1])

    @property
    def covalent_radius(self) -> float:
        from ..operations.chemistry import _element_covalent_radius
        return _element_covalent_radius(self.atomic_number)

    @property
    def density(self) -> float:
        from ..operations.chemistry import _element_density
        return _element_density(self.atomic_number)

    @property
    def hyb(self) -> int:
        from ..operations.chemistry import _atom_hybridization
        return _atom_hybridization(self)

    @property
    def idx(self) -> int:
        with self.mol._mutation_lock:
            return self.mol._topology_index[id(self)][0]

    @property
    def mca(self) -> float:
        """Predicted methyl cation affinity of this atom in kJ/mol."""
        mol = self.mol
        with mol._mutation_lock:
            entry = mol._topology_index.get(id(self))
            if self._uid_owner is not mol._owner_token or entry is None:
                raise RuntimeError(
                    "MCA Atom membership changed; call hotpot.cheminfo.operations.inference.mca.mca(mol, backend=...) again"
                )
            return _current_mca_result(mol).atom_predictions[entry[0]].mca_kj_mol

    @property
    def is_error_electron_configure(self) -> bool:
        """
        Checks if the electron configuration of an element has errors, considering
        its metallicity and the number of missing electrons.

        Returns:
            bool: Indicates if the electron configuration is erroneous.

        """
        from ..operations.chemistry import _atom_is_error_electron_configure
        return _atom_is_error_electron_configure(self)

    @property
    def is_hydrogen(self) -> bool:
        """
        Determines if the element is hydrogen based on its atomic number.

        Checks whether the atomic number of the element is equal to 1,
        which is the atomic number of hydrogen.

        Returns:
            bool: True if the atomic number is 1; otherwise, False.
        """
        return self.atomic_number == 1

    @property
    def is_polar_hydrogen(self) -> bool:
        """
        Returns a boolean value indicating whether the atom is a polar
        hydrogen. A polar hydrogen is defined as a hydrogen atom that is
        bonded to a polar hydrogen site.

        Raises:
            ImportError: If there are issues accessing the attributes.

        Returns:
            bool: True if the atom is a polar hydrogen, False otherwise.
        """
        from ..operations.chemistry import _atom_is_polar_hydrogen
        return _atom_is_polar_hydrogen(self)

    @property
    def is_lanthanide(self):
        from ..operations.chemistry import _element_is_lanthanide
        return _element_is_lanthanide(self.atomic_number)

    @property
    def is_actinide(self):
        from ..operations.chemistry import _element_is_actinide
        return _element_is_actinide(self.atomic_number)

    @property
    def is_noble_gases(self):
        from ..operations.chemistry import _element_is_noble_gas
        return _element_is_noble_gas(self.atomic_number)

    @property
    def is_halogens(self):
        from ..operations.chemistry import _element_is_halogen
        return _element_is_halogen(self.atomic_number)

    @property
    def is_metal(self):
        from ..operations.chemistry import _element_is_metal
        return _element_is_metal(self.atomic_number)

    @property
    def label(self) -> str:
        """
        Constructs and returns a label combining the symbol and the index.

        The label is generated by concatenating the symbol and string
        representation of the index. It is intended to uniquely identify an
        entity based on these two components.

        Returns:
             str: The concatenated label consisting of symbol followed by
                  the string representation of idx.
        """
        return self.symbol + str(self.idx)

    def link_with(self, other: "Atom"):
        """
        Links the current atom with another atom from a different molecule.

        This method assumes that the two atoms belong to different molecules and
        connects them by creating a bond between the atom in the current molecule
        and a cloned copy of the other atom. The molecule of the other atom is
        integrated into the molecule of the current atom as a separate component.

        Args:
            other ("Atom"): The atom from a different molecule to link with.

        Raises:
            AssertionError: If the two atoms belong to the same molecule.
        """
        from ..operations.topology import _link_atom
        return _link_atom(self, other)

    @property
    def mass(self):
        from ..operations.chemistry import _element_mass
        return _element_mass(self.atomic_number)

    @property
    def mol(self) -> Molecule:
        """Return the immutable molecule owner, including after detachment."""
        return self._mol

    @property
    def molecule(self):
        """
        Property to access the molecule object.

        This property allows retrieving the molecule object associated with the
        current instance. The molecule object represents the data structure
        used to store and manipulate molecular information.

        Returns
        -------
        Any
            The molecule object associated with the instance.
        """
        return self.mol

    @property
    def neigh_idx(self) -> np.ndarray:
        """Return adjacent Atom indices in deterministic order."""
        return np.array([
            self.mol._topology_index[id(
                bond.atom2 if bond.atom1 is self else bond.atom1
            )][0]
            for bond in self.bonds
        ])

    @property
    def hydrogens(self):
        """
        Property to retrieve the list of hydrogen atom neighbors.

        The hydrogens property filters and returns neighbors of the current atom that
        are hydrogen atoms.

        Returns:
            list: A list containing atom objects of all neighbors that are identified
            as hydrogen atoms.
        """
        return [a for a in self.neighbours if a.is_hydrogen]

    @property
    def h_count(self):
        return len(self.hydrogens)

    @property
    def connectivity(self):
        return len(self.bonds)

    @property
    def neighbours(self) -> list['Atom']:
        """Return adjacent Atoms in deterministic index order."""
        return [
            bond.atom2 if bond.atom1 is self else bond.atom1
            for bond in self.bonds
        ]

    @property
    def heavy_neighbours(self) -> list['Atom']:
        """
        Determines and returns a list of all heavy atom neighbors of the
        current atom. Heavy atoms are defined as atoms with an atomic
        number not equal to 1.

        Returns:
            list[Atom]: A list of heavy atom neighbors for the current atom.
        """
        return [a for a in self.neighbours if a.atomic_number != 1]

    @property
    def missing_electrons_element(self):
        """
        Calculates and returns the number of missing electrons for the element
        based on its electron configuration. The method uses the atomic orbital
        data and calculates the difference between the maximum possible number
        of electrons for the orbital and the current configuration.

        Returns:
            int: The number of missing electrons for the element.
        """
        from ..operations.chemistry import _element_missing_electrons
        return _element_missing_electrons(self.atomic_number)

    @property
    def missing_electrons(self):
        """ missing electrons in open shell for this atom. """
        from ..operations.chemistry import _atom_missing_electrons
        return _atom_missing_electrons(self)

    @property
    def open_shell_electrons(self) -> int:
        """
        Calculate and return the number of open-shell electrons in the system.

        The method computes the electron configuration using the
        `calc_electron_config` method and calculates the total number
        of electrons in the open shells from the configuration.

        Returns:
            int: The number of open-shell electrons.
        """
        from ..operations.chemistry import _atom_open_shell_electrons
        return _atom_open_shell_electrons(self)

    @property
    def electron_configuration(self) -> np.ndarray[int]:
        from ..operations.chemistry import _element_electron_configuration
        return np.asarray(_element_electron_configuration(self.atomic_number), dtype=int)


    @property
    def l(self) -> int:
        """
        Gets the angular momentum quantum number (l) from the calculated
        electron configuration.

        The property calculates the electron configuration using the
        `calc_electron_config` method and retrieves only the `l` value
        from the returned configuration data.

        Returns:
            int: The angular momentum quantum number (l) derived from
            the electron configuration calculation.
        """
        n, l, conf = self.calc_electron_config()
        return l

    @property
    def n(self) -> int:
        """
        Returns the principal quantum number (n) which determines the energy
        level of the electron in an atom. This is calculated based on the
        electron configuration.

        Returns:
            int: The principal quantum number of the electron.
        """
        n, l, conf = self.calc_electron_config()
        return n

    def calc_electron_config(self) -> (int, int, list[int]):
        from ..operations.chemistry import _atom_calc_electron_config
        return _atom_calc_electron_config(self)

    @property
    def oxidation_state(self) -> int:
        """
        Determines the oxidation state of an atom based on its properties and bonded environment.

        The oxidation state is calculated using the formal charge, atomic number, bonds,
        electronegativity, and implicit hydrogens of the atom. Specific rules are implemented
        for certain atomic numbers and for atoms that are metals, ensuring a consistent
        computation for a wide range of chemical environments.

        Returns:
            int: The calculated oxidation state of the atom.

        Raises:
            AttributeError: If required properties are missing in the atom or its bonds.
        """
        from ..operations.chemistry import _atom_oxidation_state
        return _atom_oxidation_state(self)

    def get_formal_charge(self) -> cython.int:
        from ..operations.chemistry import _atom_formal_charge
        return _atom_formal_charge(self)

    def get_valence(self) -> cython.int:
        from ..operations.chemistry import _atom_valence
        return _atom_valence(self)

    def set_valence_to_default(self):
        from ..operations.chemistry import _atom_valence
        _atom_valence(self)


    @property
    def sum_bond_orders(self) -> int:
        from ..operations.chemistry import _atom_order_sum
        return _atom_order_sum(self, 'all')

    @property
    def sum_heavy_cov_orders(self) -> int:
        from ..operations.chemistry import _atom_order_sum
        return _atom_order_sum(self, 'heavy')

    @property
    def sum_covalent_orders(self) -> int:
        from ..operations.chemistry import _atom_order_sum
        return _atom_order_sum(self, 'covalent')

    @property
    def symbol(self) -> str:
        """
        The `symbol` property retrieves the chemical symbol of an element based on its atomic number
        from the `_symbols` attribute. This property is used to access the chemical symbol as a string
        representation.

        @property
            Gets the chemical symbol corresponding to the atomic number.

        Returns:
            str: The chemical symbol associated with the element's atomic number.
        """
        from ..reference.properties import _ELEMENT_SYMBOLS
        return _ELEMENT_SYMBOLS[self.atomic_number]

    @symbol.setter
    def symbol(self, value: str):
        """
            Updates the element's atomic number using its chemical symbol.

            The `symbol` property allows setting the chemical symbol of the element,
            and automatically updates the corresponding atomic number.

            Raises:
                ValueError: If the provided symbol is not in the predefined list
                of symbols.

            Args:
                value: A string representing the chemical symbol of the element.
        """
        from ..reference.properties import _ELEMENT_SYMBOLS
        self.atomic_number = _ELEMENT_SYMBOLS.index(value)

    @property
    def vector(self) -> np.ndarray:
        """
        Returns the vector representation of the object's coordinates as a numpy array.

        The property retrieves the coordinates of the object and converts them into a
        one-dimensional numpy array, which can be used for mathematical operations or
        further processing.

        Returns:
            np.ndarray: A numpy array representation of the coordinates.
        """
        return np.array([self.coordinates])



_ATOM_FACT_ENGINE = None


def _atom_numeric_value(atom, field):
    with atom.mol._mutation_lock:
        if field in _ATOM_CONSTRAINT_FIELDS:
            return _ATOM_FIELD_CODECS[field](_atom_axis_value(atom._constraint_values[atom._uid],
                                                            _ATOM_FIELD_INDEX[field] - 16))
        record = atom.mol._topology_index.get(id(atom))
        table, row = (atom._detached_table, 0) if record is None else (atom.mol._atom_table, record[0])
        return _ATOM_FIELD_CODECS[field](table.raw_index(row, _ATOM_FIELD_INDEX[field]))


def _atom_fact_engine():
    global _ATOM_FACT_ENGINE
    if _ATOM_FACT_ENGINE is None:
        engine = _fact_engine()
        engine._install_atom_dispatch(Atom, _atomic_number_setter, _commit_atom_sparse,
                                      NotInSameMolecule, ObjNotInMolecule)
        _ATOM_FACT_ENGINE = engine
    return _ATOM_FACT_ENGINE


def _store_atom_fields(atom, columns, *, producer=None):
    if columns.keys() & _ATOM_SCALAR_READONLY:
        raise AttributeError('Calculated Atom projections are read-only; use the complete Molecule assignment')
    engine = _atom_fact_engine()
    producer = _ATOM_SOURCE_CODES['explicit'] if producer is None else producer
    if len(columns) == 1:
        field, value = next(iter(columns.items()))
        engine._commit_atom_scalar(atom, _ATOM_FIELD_INDEX[field], value, producer=producer)
        return
    engine._commit_atom_fields(atom, columns, producer=producer)


def _initialize_atom_handle(atom, mol, uid, *, cell=_EMPTY_ATOM_CELL):
    object.__setattr__(atom, '_mol', mol)
    object.__setattr__(atom, '_member_uid', uid)
    object.__setattr__(atom, '_uid_owner', mol._owner_token)
    object.__setattr__(atom, '_constraint_values', {uid: cell})


def _new_atom_handle(mol, uid, *, cell=_EMPTY_ATOM_CELL):
    atom = object.__new__(Atom)
    _initialize_atom_handle(atom, mol, uid, cell=cell)
    return atom


def _import_atom_row(mol, values, *, presence=None, sources=None, partial_charge_result=None):
    """Install one supplied factual row through the private import boundary."""
    values = np.asarray(values, dtype=np.float64)
    table_values, masks, codes, cells = _split_atom_rows(values[None, :], presence=presence, sources=sources)
    field = _ATOM_FIELD_INDEX['partial_charge']
    supplied_charge = bool(int(masks[0]) & (1 << field))
    if supplied_charge:
        if partial_charge_result is not None:
            raise ValueError('An explicit partial-charge result excludes supplied Atom charge bits')
        from ._results import PartialChargeSet
        partial_charge_result = (PartialChargeSet(
            (0,), (float(values[field]),), 'supplied', None, (), 'e', (int(codes[0, field]),)), ())
    prepared = _AtomFactTable(table_values, presence=masks, sources=codes)
    with _FactTransaction(mol, producer=_ATOM_SOURCE_CODES['legacy_array']) as transaction:
        empty = not mol._atoms
        if supplied_charge and mol._partial_charge_result is not None:
            previous = mol._partial_charge_result.payload
            if previous.method != 'supplied' or previous.method_version is not None or previous.parameters:
                raise ValueError('A supplied Atom charge row cannot mix methods; import a complete partial-charge set')
        atom = _new_atom_handle(mol, mol._allocate_atom_uid(), cell=cells[0])
        transaction.touch_registration(atom)
        row = mol._atom_table._append_prepared(prepared.matrix(), prepared.presence(), prepared.sources())
        mol._atoms.append(atom)
        mol._topology_index[id(atom)] = (row, ())
        transaction.touch_constraint_value(mol._constraints.atom_axes, atom._uid)
        mol._constraints.atom_axes[atom._uid] = cells[0]
        object.__setattr__(atom, '_constraint_values', mol._constraints.atom_axes)
        _mark_topology_dirty(mol, (atom._uid,))
        if empty and partial_charge_result is not None:
            _install_partial_charge_record(mol, partial_charge_result)
        elif supplied_charge:
            from dataclasses import replace
            historical = _partial_charge_record(mol, tuple(range(row)), whole_context=False)
            supplied, _ = partial_charge_result
            if historical is None:
                payload = replace(supplied, atom_uids=tuple(range(row + 1)),
                    values=(None,) * row + supplied.values,
                    provenance=(_ATOM_SOURCE_CODES['default'],) * row + supplied.provenance)
                marker = ('charge_representation',)
            else:
                previous, marker = historical
                payload = replace(previous, atom_uids=tuple(range(row + 1)),
                    values=previous.values + supplied.values,
                    provenance=previous.provenance + supplied.provenance)
            _install_partial_charge_record(mol, (payload, marker))
        return atom


def _seal_detached_table(table):
    table._blocks = tuple(np.frombuffer(block.tobytes(), dtype=block.dtype).reshape(block.shape)
                          for block in table._blocks)
    from ._atom_table import _column_views
    table._columns = _column_views(table._blocks)
    table._presence = np.frombuffer(table._presence.tobytes(), dtype=table._presence.dtype)
    table._sources = np.frombuffer(table._sources.tobytes(), dtype=table._sources.dtype).reshape(table._sources.shape)
    table._group_status = np.frombuffer(table._group_status.tobytes(), dtype=np.uint8).reshape(table._group_status.shape)
    table._dirty_seeds = tuple(frozenset(seeds) for seeds in table._dirty_seeds)
    return table


def _detach_atom(atom):
    from ._dependencies import _FactStatus
    row, bonds = atom.mol._topology_index[id(atom)]
    table = atom.mol._atom_table.take((row,))
    local_uids = {atom._uid, *(bond.atom2._uid if bond.atom1 is atom else bond.atom1._uid for bond in bonds)}
    if (int(table._sources[0, _ATOM_FIELD_INDEX['implicit_hydrogens']]) == _ATOM_SOURCE_CODES['implicit_hydrogens']
            and table._group_status[0, GROUP_INDEX['IMPLICIT_H']] == _FactStatus.CURRENT
            and any(local_uids.intersection(seeds) for seeds in table._dirty_seeds[:2])):
        table._group_status[0, GROUP_INDEX['IMPLICIT_H']] = _FactStatus.NEEDS_RECALC
    table._dirty_seeds = tuple({atom._uid}.intersection(seeds) for seeds in table._dirty_seeds)
    object.__setattr__(atom, '_detached_table', _seal_detached_table(table))
    object.__setattr__(atom, '_constraint_values', {atom._uid: atom._constraint_values[atom._uid]})


def _numeric_property(field):
    index, codec = _ATOM_FIELD_INDEX[field], _ATOM_FIELD_CODECS[field]

    def read(atom):
        if field == 'partial_charge':
            return _atom_partial_charge(atom)
        if field in _ATOM_DERIVED_FIELDS:
            if field == 'valence':
                return atom.get_valence()
            return int(atom.electron_configuration[index - 1])
        if field == 'is_aromatic':
            from ..operations.chemistry import _atom_aromatic_read
            return _atom_aromatic_read(atom)
        if field == 'implicit_hydrogens':
            from ..operations.chemistry import _implicit_hydrogen_read
            return _implicit_hydrogen_read(atom)
        mol = atom._mol
        with mol._mutation_lock:
            if field in _ATOM_CONSTRAINT_FIELDS:
                return codec(_atom_axis_value(atom._constraint_values[atom._uid], index - 16))
            record = mol._topology_index.get(id(atom))
            if record is None:
                return codec(atom._detached_table._columns[index][0])
            return codec(mol._atom_table._columns[index][record[0]])

    return property(read)



for _field in _ATOM_FIELDS:
    if _field not in Atom.__dict__:
        setattr(Atom, _field, _numeric_property(_field))
del _field
