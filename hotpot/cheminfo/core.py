"""
python v3.9.0
@Project: hotpot0.5.0
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2024/6/1
@Time   : 17:23

Note: for Chemical Informatics
"""
import re
from pathlib import Path
from copy import copy

import cclib
from openbabel import openbabel as ob, pybel as pb
from rdkit import Chem
from rdkit.Chem import AllChem

from scipy.spatial.distance import pdist, squareform
import networkx as nx
import periodictable

from . import ob2chem
from .rdkit2chem import RDKit
from .graph import *


class Conformers:
    """ A class to handle Molecule conformers (coordinates) """
    def __init__(self, mol: "Molecule"):
        self.molecule = mol
        self.mol_conformer_idx = None

        self.coords = None
        self.charges = None
        self.energies = None

        self.idx = None

    def __repr__(self):
        if self.coords is None:
            return f"{self.__class__.__class__}(count=0, atoms=0)"
        else:
            return f"{self.__class__.__class__}(count={len(self.coords)}, atoms={self.coords.shape[-2]})"

    def __len__(self):
        return 0 if not isinstance(self.coords, np.ndarray) else len(self.coords)

    def add_conformer(
            self, coords: [np.ndarray, Sequence],
            charges: [np.ndarray, Sequence] = None,
            energies: Union[float, Sequence, np.ndarray] = None
    ):
        """ Add conformers to the molecule """
        coords = np.array(coords)
        if coords.ndim != 2 and coords.ndim != 3:
            raise ValueError(f"The dimensions of given conformers must be 2 or 3, instead of {coords.ndim}")

        if coords.ndim == 2:
            rows, cols = coords.shape
            c_num = 1
            coords = coords.reshape(c_num, rows, cols)
        else:
            c_num, rows, cols = coords.shape

        if cols != 3:
            raise ValueError(f"the length of last dimension(x, y, z) of given conformers must be 3, instead of {cols})")

        if rows != self.molecule.atom_counts:
            raise ValueError(f"the length of -2 dimensions of given conformers must be same with molecule.atom_counts")

        if self.coords is not None:
            coords = np.vstack((self.coords, coords))

        if charges is None:
            charges = np.zeros((c_num, rows))
        else:
            charges = np.array(charges)

        if charges.shape != (c_num, rows):
            raise ValueError(f"the given charges for conformers must be ({c_num}, {rows}), instead of {charges.shape}")

        if self.charges is not None:
            charges = np.vstack((self.charges, charges))

        if energies is None:
            energies = np.zeros(c_num)
        else:
            energies = np.array(energies).flatten()

        if energies.shape != (c_num,):
            raise ValueError(f"the given energies for conformers must be ({c_num}), instead of {energies.shape}")

        if self.energies is not None:
            energies = np.vstack((self.energies, energies))

        # Setting
        self.coords = coords
        self.charges = charges
        self.energies = energies

    def update_for_mol(self, idx: int):
        if self:
            mol = self.molecule
            mol.energy, mol.atoms_partial_charges, mol.coordinates = \
                (self.energies[idx], self.charges[idx], self.coords[idx])
            self.idx = idx
        else:
            raise AttributeError(f"There are none of conformers stored in the molecule {self.molecule}")

    def store_current_conformer(self, force_current: bool = False) -> int:
        """
         Store current conformer on the molecule to Conformer repository.
         Args:
             force_current: If true, force the current conformer to Conformer repository no matter if it's existent.

         Return:
               The conformer index of stored current conformer
        """
        if force_current or self.idx is None:
            mol = self.molecule
            self.add_conformer(mol.coordinates, mol.atoms_partial_charges.reshape((1, -1)), mol.energy)

            self.idx = len(self) - 1
        else:
            print(RuntimeWarning("The current conformer has already been existent in Conformer repository."))

        return self.idx


class Molecule:
    atom_attrs_enumerator = (
        'atomic_number',
        'formal_charge',
        'partial_charge',
        'is_aromatic',
        'coord_x',
        'coord_y',
        'coord_z',
        'valence',
        'implicit_hydrogens',
        # 'explicit_hydrogens'
    )

    bond_attrs_enumerator = (
        'begin_atom_index',
        'end_atom_index',
        'bond_order',
        'is_aromatic'
    )

    for i, attr_name in enumerate(atom_attrs_enumerator):
        if attr_name == 'is_aromatic':
            enum_atom_is_aromatic = i
        locals()[f'enum_{attr_name}'] = i

    for i, attr_name in enumerate(bond_attrs_enumerator):
        if attr_name == 'is_aromatic':
            enum_bond_is_aromatic = i
        locals()[f'enum_{attr_name}'] = i

    del i, attr_name

    # # determine atom_indices
    # enum_atomic_number = atom_attrs_enumerator.index('atomic_number')
    # enum_formal_charge = atom_attrs_enumerator.index('formal_charge')
    # enum_partial_charge = atom_attrs_enumerator.index('partial_charge')
    # enum_atom_is_aromatic = atom_attrs_enumerator.index('is_aromatic')
    enum_coords = (
        atom_attrs_enumerator.index('coord_x'),
        atom_attrs_enumerator.index('coord_y'),
        atom_attrs_enumerator.index('coord_z')
    )
    # enum_label = atom_attrs_enumerator.index('label')
    # enum_valence = atom_attrs_enumerator.index('valence')
    # enum_implicit_hydrogens = atom_attrs_enumerator.index('implicit_hydrogens')
    # enum_explicit_hydrogens = atom_attrs_enumerator.index('explicit_hydrogens')
    #
    # # determine bond_indices
    # enum_begin_atom_index = bond_attrs_enumerator.index('begin_atom_index')
    # enum_end_atom_index = bond_attrs_enumerator.index('end_atom_index')
    # enum_bond_order = bond_attrs_enumerator.index('bond_order')
    # enum_bond_is_aromatic = bond_attrs_enumerator.index('is_aromatic')

    def __init__(self):
        self._atoms_data = None
        self._bonds_data = None
        self.atom_labels: Union[list, None] = None
        self.charges = 0

        self._atoms = []
        self._bonds = []
        self._conformers = Conformers(self)
        self.energy = 0.
        self.charge = 0

        self.zero_point = None
        self.free_energy = None  # - mol.energy - mol.zero_point  # Hartree to eV
        self.entropy = None
        self.enthalpy = None  # - mol.energy - mol.zero_point
        self.temperature = None
        self.pressure = None
        self.thermal_energy = None  # kcal to ev
        self.capacity = None  # cal to ev

    def __repr__(self):
        try:
            return f'Mol({self.format})'
        except TypeError:
            return f"Mol()"

    def __contains__(self, item: Union['Atom', 'Bond']):
        if isinstance(item, Atom):
            return item.molecule is self and item in self._atoms
        if isinstance(item, Bond):
            return item.molecule is self and item in self._bonds

    def __copy__(self):
        return self.copy()

    def __radd__(self, other: int):
        if isinstance(other, int) and other == 0:
            return self
        else:
            raise TypeError(f'the {self.__class__} can add with {type(other)}')

    def __add__(self, other: Union["Molecule", int]):
        mol = self.__class__()
        mol._atoms_data = np.vstack((self._atoms_data, other._atoms_data))
        mol._bonds_data = np.vstack((self._bonds_data, other._bonds_data))
        mol._bonds_data[len(self._bonds_data):, :2] += self.atom_counts

        mol._update_atom_bond_array()

        return mol

    @property
    def conformer_index(self) -> int:
        return self._conformers.idx

    def store_current_conformer(self):
        self._conformers.store_current_conformer()

    def add_conformers(
            self,
            coords: Union[np.ndarray, Sequence],
            charges: Union[np.ndarray, Sequence],
            energies: Union[np.ndarray, float]
    ):
        self._conformers.add_conformer(coords, charges, energies)

    @property
    def has_conformers(self) -> bool:
        return bool(self._conformers)

    @property
    def conformers_count(self) -> int:
        return 0 if not self.has_conformers else len(self._conformers)

    @property
    def conformers_index(self) -> int:
        return self._conformers.idx

    def _update_atom_bond_array(self):
        self._atoms = [Atom(self) for _ in range(self.atom_counts)]

        if self.link_matrix.size > 0:
            self._bonds = [Bond(self, i, j) for i, j in self.link_matrix]
        else:
            self._bonds = []

    def assign_atom_total_valence(self):
        """"""
        for a in self._atoms:
            if a.is_metal:
                a.total_valence = 0
            elif a.symbol == 'S':
                a.total_valence = a.bond_orders + a.bond_orders % 2
            elif a.symbol == 'P':
                if a.bond_orders <= 3:
                    a.total_valence = 3
                elif a.bond_orders <= 5:
                    a.total_valence = 5
            else:
                a.total_valence = Atom._default_valence[a.atomic_number]

    @staticmethod
    def calc_pair_covalent_dist(element1, element2):
        return getattr(periodictable, element1).covalent_radius + getattr(periodictable, element2).covalent_radius

    @property
    def covalent_dist_matrix(self):
        cov_radius1 = Atom.covalent_radii[np.int_(self.atoms_data[:, 0])]
        cov_radius2 = Atom.covalent_radii[np.int_(self.atoms_data[:, 0])].reshape(-1, 1)

        return cov_radius1 + cov_radius2

    def has_3d(self):
        return np.any(np.bool_(self.coordinates))

    def is_disorder(self):
        """ Checks if any atoms pairs are too closed to each other """
        check_list = np.nonzero(self.dist_matrix < 0.5 * self.covalent_dist_matrix)
        return np.any(check_list[0] != check_list[1])

    def link_atom_cloud(self) -> np.ndarray:
        """ Returns a link matrix according to the position matrix and the covalent distance """
        if not self.has_3d():
            raise AttributeError('The atoms in the molecule has no 3D coordinates')
        elif self.is_disorder():
            raise AttributeError('the position of atoms in the molecule is disordered, some of them are too close')

        adj = np.array(list(
            np.nonzero(
                np.logical_and(
                    0. < self.dist_matrix,
                    self.dist_matrix < 1.0 * self.covalent_dist_matrix
                )
            )
        ))

        return adj.T

    @classmethod
    def read(cls, src, fmt=None, error_raise: bool = True):
        """ Reads a molecule from a file or a string of text"""
        by_cclib = ['g16']
        if fmt in by_cclib:
            return cls.read_by_cclib(src, fmt)
        else:
            return cls.read_by_openbabel(src, fmt, error_raise)

    @classmethod
    def read_by_openbabel(cls, src, fmt=None, error_raise: bool = True) -> Generator['Molecule', None, None]:
        """ Read Molecule information by an OpenBabel """
        obmol_reader = ob2chem.read(src, fmt)

        def reader():
            for obmol in obmol_reader:
                try:
                    mol = cls()
                    mol._atoms_data, mol._bonds_data, idx_to_row = ob2chem.to_arrays(obmol)
                    mol._update_atom_bond_array()
                    mol.assign_atom_total_valence()

                    yield mol

                except Exception as e:
                    if error_raise:
                        raise e
                    else:
                        print(RuntimeWarning('Meeting an error and continue the reader'))
                        continue
        return reader()

    @classmethod
    def read_by_cclib(cls, src, fmt='g16'):
        """ Read Molecule information by cclib package """

        def reader() -> Generator['Molecule', None, None]:
            """ A generator wrapper """
            data = cclib.io.ccopen(src).parse()

            mol = next(cls.read_by_openbabel(src, fmt))
            mol.charge = data.charge
            mol.spin_multiplicity = data.mult

            coords = data.atomcoords
            mulliken_charges = data.atomcharges['mulliken']
            energies = data.scfenergies

            if mulliken_charges.ndim == 1:
                mulliken_charges = np.tile(mulliken_charges, (len(coords), 1))

            mol.add_conformers(coords, mulliken_charges, energies)
            mol.assign_conformers(mol.conformers_count - 1, store_current=False)

            try:
                mol.zero_point = data.zpve * 27.211386245988
                mol.free_energy = data.freeenergy * 27.211386245988  # - mol.energy - mol.zero_point  # Hartree to eV
                mol.entropy = data.entropy * 27.211386245988
                mol.enthalpy = data.enthalpy * 27.211386245988  # - mol.energy - mol.zero_point
                mol.temperature = data.temperature
                mol.pressure = data.pressure
            except AttributeError:
                pass

            # Grab thermal energy, delta capacity at volume, delta entropy
            with open(src) as file:
                lines = file.readlines()

            anchor_line = 0
            title_pattern = re.compile(r'\s+E \(Thermal\)\s+CV\s+S')
            for i, line in enumerate(lines):
                if title_pattern.match(line):
                    anchor_line = i
                    break

            if anchor_line != 0:
                thermal_energy, capacity, _ = map(float, re.split(r'\s+', lines[anchor_line + 2].strip())[1:])
                mol.thermal_energy = 0.043361254529175 * thermal_energy  # kcal to ev
                mol.capacity = 0.043361254529175 * 1e-3 * capacity  # cal to ev

            yield mol

        return reader()

    def dump_by_openbabel(self, fmt, **kwargs) -> Union['str', 'bytes']:
        """ Dump Molecule information by an OpenBabel """
        obmol, _ = self.to_obmol()
        return ob2chem.ob_dump(obmol, fmt, **kwargs)

    def write_by_openbabel(self, fpath: Union[str, Path], fmt, **kwargs):
        content = self.dump_by_openbabel(fmt, **kwargs)
        mode = 'w' if isinstance(content, str) else 'wb'
        with open(fpath, mode) as file:
            file.write(content)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.int_(self._atoms_data[:, 0])

    @property
    def is_organic(self) -> bool:
        """ Whether the molecule is organic compound """
        return bool({'C', 'H'} & set(self.element_counts) and
                    np.all(np.isin(self.atomic_numbers, list(Atom.metal_), invert=True)))

    def to_obmol(self) -> (ob.OBMol, dict[int, int]):
        """ Convert the molecule to an OpenBabel OBMol Object """
        return ob2chem.to_obmol(self)

    def to_rdmol(self) -> (Chem.RWMol, dict[int, int]):
        """ Convert the molecule to a RDKit RWMol Object """
        return RDKit.arrays_to_rdkitmol(self._atoms_data, self._bonds_data)

    @property
    def atoms_partial_charges(self) -> np.ndarray:
        """ The partial charges of the atoms in the molecule. """
        return self._atoms_data[:, self.enum_partial_charge]

    @atoms_partial_charges.setter
    def atoms_partial_charges(self, charges: Union[np.ndarray, Sequence]) -> None:
        """ Set the partial charges of the atoms in the molecule. """
        charges = np.array(charges)
        if charges.shape != (self.atom_counts,):
            raise ValueError(f"the partial charges must have the same number of atoms! "
                             f"{charges.shape} != {(self.atom_counts,)}")

        self._atoms_data[:, self.enum_partial_charge] = np.array(charges)

    @property
    def atoms_data(self):
        return self._atoms_data

    @property
    def bond_data(self):
        return self._bonds_data

    @property
    def link_matrix(self):
        """ the adjacency matrix with the shape of (BN, 2), where BN is the number of bonds """
        # return (self._bonds_data.tolist() and
        #         np.int_(self._bonds_data[:, (self.enum_begin_atom_index, self.enum_end_atom_index)]))
        if self._bonds_data.size > 0:
            return np.int_(self._bonds_data[:, (self.enum_begin_atom_index, self.enum_end_atom_index)])
        else:
            return np.array([[]])
        # return (self._bonds_data and
        #         np.int_(self._bonds_data[:, (self.enum_begin_atom_index, self.enum_end_atom_index)]))

    @property
    def adjacency(self) -> np.ndarray:
        """ the adjacency square matrix """
        return linkmat2adj(self.atom_counts, self.link_matrix)

    def laplacian(self, norm: bool = False) -> np.ndarray:
        """ the laplacian matrix of the molecular graph """
        return adj2laplacian(self.adjacency, norm)

    @atoms_data.setter
    def atoms_data(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError('the "atoms_data" must be a numpy array"')
        if value.shape[1] < len(self.atom_attrs_enumerator):
            raise ValueError(f'the "atoms_data" is a 2D array with "atom count" row '
                             f'and more than {len(self.atom_attrs_enumerator)} columns')

        self._atoms_data = value

    @property
    def atom_counts(self) -> int:
        return len(self._atoms_data)

    def add_atom(
            self, atom: Union['Atom', int, str], *,
            formal_charge: Optional[int] = 0,
            partial_charge: Optional[float] = 0.,
            is_aromatic: Optional[bool] = False,
            coordinates: Optional[Union[Sequence, np.ndarray]] = (0., 0., 0.),
            valence: Optional[int] = None
    ):
        """"""
        if isinstance(atom, Atom):
            atom_attrs = atom.attrs
        elif isinstance(atom, (int, str)):
            atom_attrs = np.zeros(len(self.atom_attrs_enumerator))
            atomic_number = atom_attrs[0] = atom if isinstance(atom, int) else ob.GetAtomicNum(atom)
            atom_attrs[1] = formal_charge
            atom_attrs[2] = partial_charge
            atom_attrs[3] = int(is_aromatic)
            atom_attrs[4:7] = coordinates
            atom_attrs[8] = atom_attrs[7] = valence if valence else Atom._default_valence[atomic_number]

        else:
            raise TypeError('the atom must be a Atom, int, or str!')

        # Update atom data sheet in the Molecule
        self._atoms_data = np.vstack((self._atoms_data, atom_attrs))
        self._atoms.append(Atom(self))

        # return the Atom just add into
        return self._atoms[-1]

    def remove_atom(self, atom: Union['Atom', int, str]):
        """"""
        if isinstance(atom, Atom) and atom in self:
            atom_idx = atom.idx
            bond_idx = atom.bonds_idx

        elif isinstance(atom, int):
            atom_idx = atom
            atom = self.atoms[atom_idx]
            bond_idx = self.atoms[atom_idx].bonds_idx

        elif isinstance(atom, str):
            if not self.have_normalized_labels:
                raise AttributeError('atoms in the Molecule are not normalized before indexing by labels')
            atom_idx = self.atom_labels.index(atom)
            atom = self.atoms[atom_idx]
            bond_idx = self.atoms[atom_idx].bonds_idx

        else:
            raise TypeError('the `atom` must be a Atom, int, or str!')

        # Retrieve atom attribute
        attrs = atom.attrs

        self._atoms.remove(atom)  # Remove atoms list
        self._bonds = np.delete(self._bonds, bond_idx, axis=0).tolist()  # Remove bonds list on the atom

        # Remove atom label
        if self.have_normalized_labels:
            self.atom_labels.pop(atom_idx)

        # Remove related data sheet
        self._atoms_data = np.delete(self._atoms_data, atom_idx, axis=0)
        self._bonds_data = np.delete(self._bonds_data, bond_idx, axis=0)

        # update the indices in the adjacency matrix
        self._bonds_data[:, :2][self.link_matrix > atom_idx] -= 1

        # transfer the removed atom to a new isolate molecule
        atom.molecule = Molecule()
        atom.molecule.atoms_data = attrs.reshape(1, -1)
        atom.molecule._atoms = [atom]

    def bond(self, atom1: int, atom2: int) -> "Bond":
        """ Get the bond between atom1 and atom2 """
        return self._bonds[np.all(np.isin(self.link_matrix, [atom1, atom2]), axis=1)]

    def is_bond(self, atom1: int, atom2: int) -> bool:
        """ Check if bond exists between atom1 and atom2 """
        return bool(np.any(np.all(np.isin(self.link_matrix, [atom1, atom2]), axis=1)))

    def add_bond(
            self, atom1: int, atom2: int, bond_order: int = 1,
            is_aromatic: bool = False, raise_error: bool = True
    ) -> bool:
        if atom1 > self.atom_counts or atom2 > self.atom_counts:
            raise ValueError('the atom indices must be less than the number of total atoms in the molecule')

        if self.is_bond(atom1, atom2):
            if raise_error:
                raise ValueError('bond between atom1 and atom2 are already bonded!')
            else:
                return False

        bond_attrs = np.array([atom1, atom2, bond_order, int(is_aromatic)])
        self._bonds_data = np.vstack((self._bonds_data, bond_attrs))
        self._bonds = self._bonds + [Bond(self, atom1, atom2)]

        return True

    def remove_bond(self, atom1: int, atom2: int) -> bool:
        if not self.is_bond(atom1, atom2):
            return False

        bond_idx = self.bond(atom1, atom2).idx
        self._bonds_data = np.delete(self._bonds_data, bond_idx, axis=0)
        self._bonds.pop(bond_idx)

    def add_hydrogens(self):
        """"""
        implicit_valence = np.array([0 if a.is_metal else a.total_valence - a.valence for a in self.atoms])
        implicit_valence[implicit_valence < 0] = 0

        hydro_attrs = np.array([1, 0, 0., 0, 0., 0., 0., 1, 0])
        hydro_data = np.tile(hydro_attrs, (sum(implicit_valence), 1))
        bonds_data = np.zeros((sum(implicit_valence), len(self.bond_attrs_enumerator)), dtype=int)

        bonds_data[:, self.enum_bond_order] = 1
        bonds_data[:, self.enum_bond_is_aromatic] = 0

        start_index = self.atom_counts
        hydro_count = 0
        for i, v in enumerate(implicit_valence):
            for _ in range(v):
                bonds_data[hydro_count, 0] = i
                bonds_data[hydro_count, 1] = start_index + hydro_count
                hydro_count += 1

        self._atoms_data = np.vstack((self._atoms_data, hydro_data))

        if self._bonds_data.size > 0:
            self._bonds_data = np.vstack((self._bonds_data, bonds_data))
        else:
            self._bonds_data = np.array(bonds_data)

        # Print this code when debug could report an error, but don't worry
        self._atoms = self._atoms + [Atom(self) for _ in range(len(hydro_data))]
        self._bonds = self._bonds + [Bond(self, i, j) for i, j in self.link_matrix[len(self._bonds):]]

    def assign_conformers(self, idx: int, store_current: bool = True) -> None:
        """Assign conformers to the molecule """
        if store_current:
            current_index = self._conformers.store_current_conformer()

        self._conformers.idx = idx
        self._conformers.update_for_mol(idx)

    @property
    def hydrogens(self):
        return [a for a in self._atoms if a.atomic_number == 1]

    def remove_hydrogens(self):
        for h in self.hydrogens:
            self.remove_atom(h)

    @property
    def dist_matrix(self) -> np.ndarray:
        """ the distance matrix for point cloud of atoms """
        return squareform(pdist(self.coordinates))

    def set_aromatic_by_huckle(self):
        """ set aromatic for rings by Huckle's rules """
        for ring in self.simple_rings:
            if ring.is_aromatic:
                ring.set_aromatic()

    @property
    def element_counts(self) -> dict[str, int]:
        atom_num_arr = self._atoms_data[:, self.enum_atomic_number]

        # Get unique elements and their counts
        unique_elements, counts = np.unique(atom_num_arr, return_counts=True)

        # Create a dictionary of element counts
        element_counts = dict(zip(map(lambda e: ob.GetSymbol(int(e)), unique_elements), counts))

        return {e: element_counts[e] for e in sorted(element_counts)}

    @property
    def format(self) -> str:
        return ''.join(f'{e}{c}' for e, c in self.element_counts.items())

    @property
    def atoms(self) -> list['Atom']:
        return copy(self._atoms)

    @property
    def metals(self) -> list['Atom']:
        return [a for a in self.atoms if a.is_metal]

    @property
    def bonds(self) -> list['Bond']:
        return copy(self._bonds)

    @staticmethod
    def node_matcher(n1_attr: dict, n2_attr: dict) -> bool:
        """ Check if two atomic nodes are identical """
        return int(n1_attr['attrs'][0]) == int(n2_attr['attrs'][0])

    @staticmethod
    def edge_matcher(e1_attr: dict, e2_attr: dict) -> bool:
        """ Check if two bond edges are identical """
        return int(e1_attr['attrs'][2]) == int(e2_attr['attrs'][2])

    @property
    def graph(self) -> nx.Graph:
        """ Return a networkx graph """
        graph = nx.Graph()
        graph.add_edges_from(self.link_matrix)

        nx.set_node_attributes(graph, {i: attr for i, attr in enumerate(self._atoms_data)}, 'attrs')
        nx.set_edge_attributes(graph, {(attrs[0], attrs[1]): attrs for attrs in self._bonds_data}, 'attrs')

        return graph

    def dfs(
            self,
            graph: nx.Graph,
            node: int,
            visited: set[int],
            path: list[int],
            paths: list[list[int]],
    ):
        """ Recursively traverse the nodes in the graph by depth first search """
        visited.add(node)
        path.append(node)

        paths.append(path)

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                self.dfs(graph, neighbor, set(visited), list(path), paths)

    def branch_search(self, graph: nx.Graph, main: list[int], exclude_nodes: set[int]) -> list[list[int]]:
        """"""
        visited = set(main) | set(exclude_nodes)
        paths = []
        for node in main:
            path = []
            self.dfs(graph, node, set(visited), list(path), paths)

        return paths

    @staticmethod
    def _branch_form(paths: list[list[int]]):
        """"""
        def _branch_path(nest: dict, p: tuple, start_idx: int):
            end_idx = next((i for i in range(start_idx, len(p)) if nest.get(p[start_idx: i]) is not None), len(p))
            if end_idx == len(p):
                return nest.setdefault(p[start_idx: end_idx], {})

            _branch_path(nest.setdefault(p[start_idx: end_idx]), p, start_idx=end_idx)

        paths = [tuple(p) for p in sorted(paths, key=len)]
        nested_dict = {}
        for path in paths:
            _branch_path(nested_dict, path, 0)

        return nested_dict

    @staticmethod
    def _branch_reduce(nested_dict: dict):

        def _reduce(root_key: tuple, root_dict: dict, anchor_dict: dict):
            if len(root_dict) == 1:
                leaf_key, leaf_dict = list(root_dict.items())[0]
                _reduce(root_key + leaf_key, leaf_dict, anchor_dict)

            else:
                anchor_dict = anchor_dict.setdefault(root_key, {})
                for leaf_key, leaf_dict in root_dict.items():
                    _reduce(leaf_key, leaf_dict, anchor_dict)

        reduced_dict = {}
        for key, val in nested_dict.items():
            _reduce(key, val, reduced_dict)

        return reduced_dict

    def all_simple_paths(self, exclude_hydrogens: bool = True):
        """"""
        graph = self.graph
        paths = []

        excludes_nodes = set((self.atomic_numbers == 1).nonzero()[0]) if exclude_hydrogens else set()
        for node in (graph.nodes() - excludes_nodes):
            visited = set() | excludes_nodes
            path = []
            self.dfs(graph, node, visited, path, paths)

        return paths

    @staticmethod
    def _longest_path(paths: list):
        return max(paths, key=len)

    def longest_simple_path(self, exclude_hydrogens: bool = True):
        return max(self.all_simple_paths(exclude_hydrogens), key=len)

    def _dfs_longest_path(self, graph, node, visited, path, ignore_hydrogens, non_hydrogens):
        visited.add(node)
        path.append(node)

        longest_path = list(path)

        for neighbor in graph.neighbors(node):
            if neighbor not in visited and (not ignore_hydrogens or non_hydrogens[neighbor]):
                current_path = self._dfs_longest_path(graph, neighbor, visited, path, ignore_hydrogens, non_hydrogens)
                if len(current_path) > len(longest_path):
                    longest_path = current_path

        path.pop()
        visited.remove(node)

        return longest_path

    def _dfs_branch_paths(self, graph, start_node, lsp_nodes, visited, path, ignore_hydrogens, non_hydrogens):
        """DFS to find all simple branch paths from start_node, avoiding other LSP nodes."""
        visited.add(start_node)
        path.append(start_node)

        # Yield the path if it's a branch path of non-zero length
        if len(path) > 1:
            yield list(path)

        for neighbor in graph.neighbors(start_node):
            if ((neighbor not in visited and neighbor not in lsp_nodes) and
                    (not ignore_hydrogens or non_hydrogens[neighbor])):

                yield from self._dfs_branch_paths(
                    graph, neighbor, lsp_nodes, visited, path, ignore_hydrogens, non_hydrogens)

        path.pop()
        visited.remove(start_node)

    def _main_branch(self, paths_from_same_anchor, branched_path: dict):
        """"""
        main_path = max(paths_from_same_anchor, key=len)

        branches = branched_path.setdefault(main_path, {})
        for path in paths_from_same_anchor:

            if path == main_path:
                continue

            i = 0
            while path[:i] == main_path[:i]:
                i += 1
                if i > len(path):
                    break

            if path[i:]:
                list_branch = branches.setdefault(main_path[i-2], [])
                list_branch.append(path[i:])

        for anchor, paths_from_same_anchor in branches.items():
            if len(paths_from_same_anchor) > 1:
                branches[anchor] = {}
                self._main_branch(paths_from_same_anchor, branches[anchor])

            else:
                branches[anchor] = paths_from_same_anchor[0]

    def find_branch_paths(self, exclude_hydrogens=True):
        """
        Recursively find all simple branch paths based on the given longest simple paths (LSP).

        In a group of branches that originate from the same node anchored on the LSP, the longest branch is designated
        as the main branch. The other branches are recursively re-anchored to identify their own main branches and
        sub-branches, and this process continues recursively.
        """
        graph = self.graph
        lsp = self.longest_simple_path(exclude_hydrogens)
        exclude_nodes = set((self.atomic_numbers == 1).nonzero()[0]) if exclude_hydrogens else set()
        branch_paths = []

        for node in lsp:
            visited = set(lsp) | exclude_nodes
            path = []

            self.dfs(graph, node, visited, path, branch_paths)

        reduce_path = self._path_reduce(branch_paths)

        branches = {}
        for branch_path in reduce_path:
            list_path = branches.setdefault(branch_path[0], [])
            list_path.append(branch_path)

        for anchor, paths in branches.items():
            if len(paths) == 1:
                branches[anchor] = paths[0]
            else:
                sub_branch = branches[anchor] = {}
                self._main_branch(paths, sub_branch)

        return branches

    @staticmethod
    def _path_reduce(paths: list[list[int]]):
        """ Reduce redundant paths """
        paths = sorted(map(tuple, paths), key=len)
        reduced_paths = []

        while paths:
            longer_path = paths.pop()
            to_remove = set()
            for shorter_path in paths:
                if shorter_path == longer_path[:len(shorter_path)]:
                    to_remove.add(shorter_path)

            paths = sorted(set(paths) - to_remove, key=len)

            reduced_paths.append(longer_path)

        return reduced_paths

    def sequence_representation(self):
        """ Generate a sequence representation for the given molecular graph """
        def _branch(pre: tuple, suffix: dict):
            if not suffix:
                return pre
            else:
                suffix = {k: suffix[k] for k in sorted(suffix, key=len)}


        paths = self.find_branch_paths()

        seq = []
        for key, val in paths.items():
            main = key[0]
            branch_pre = key[1:]
            branch_suffix = val

            seq.append(main)
            branch = _branch(branch_pre, branch_suffix)
            if branch:
                seq.append(branch)

    @property
    def paths(self) -> dict[int, dict[int, list]]:
        return {start_idx: end_and_path for start_idx, end_and_path in nx.all_pairs_shortest_path(self.graph)}

    @property
    def simple_rings(self) -> list["SimpleRing"]:
        return [SimpleRing(self, np.array(r, dtype=int)) for r in nx.cycle_basis(self.graph)]

    @property
    def components(self) -> list['Molecule']:
        """"""
        def update_adj_index(c_old_adj, old_node_idx):
            """ Update the old index of adj matrix in parent molecule to the new index in child component """
            sort_indices = np.argsort(old_node_idx)
            sort_node_idx = old_node_idx[sort_indices]

            indices = np.searchsorted(sort_node_idx, c_old_adj)
            c_new_adj = sort_indices[indices]

            return c_new_adj

        graph = nx.Graph()
        graph.add_edges_from(self.link_matrix)

        components = []
        for c_node_idx in nx.connected_components(graph):
            c_node_idx = np.int_(np.array(list(c_node_idx)))

            c_atoms_data = self._atoms_data[c_node_idx]

            # Transfer atom labels
            if not self.atom_labels:
                c_atom_labels = None
            else:
                c_atom_labels = np.array(self.atom_labels)[c_node_idx].tolist()

            c_bonds_data = self._bonds_data[np.any(np.isin(self.link_matrix, c_node_idx), axis=1)]

            # Update node index in the adj matrix for the new component
            c_bonds_data[:, :2] = update_adj_index(c_bonds_data[:, :2], c_node_idx)

            # Assemble components
            component = Molecule()
            component._atoms_data = c_atoms_data
            component._bonds_data = c_bonds_data
            component.atom_labels = c_atom_labels

            component._update_atom_bond_array()

            components.append(component)

        return components

    @property
    def coordinates(self):
        if isinstance(self._atoms_data, np.ndarray):
            return self._atoms_data[:, self.enum_coords]

    @coordinates.setter
    def coordinates(self, value: Union[np.ndarray, Sequence]):
        value = np.array(value)

        if value.ndim != 2:
            raise ValueError('The dimension of coordinates must be 2.')

        rows, cols = value.shape
        if cols != 3:
            raise ValueError('The columns of coordinates must be 3.')
        if rows != self.atom_counts:
            raise ValueError('the rows of coordinates must be the same with the number of atoms.')

        self._atoms_data[:, self.enum_coords] = value

    @property
    def implicit_hydrogens(self) -> int:
        implicit_valence = np.array([0 if a.is_metal else a.total_valence - a.valence for a in self.atoms])
        implicit_valence[implicit_valence < 0] = 0
        return sum(implicit_valence)

    @property
    def is_hydrogen_complete(self) -> bool:
        return self.implicit_hydrogens == 0

    def copy(self):
        """ Returns a copy of the current Molecule """
        clone = self.__class__()

        clone._atoms_data = np.copy(self._atoms_data)
        clone._bonds_data = np.copy(self._bonds_data)
        clone.atom_labels = copy(self.atom_labels)

        clone._update_atom_bond_array()

        return clone

    def remove_metals(self):
        for a in self.atoms:
            if a.is_metal:
                self.remove_atom(a)

    @property
    def organic_components(self) -> ('Molecule', dict):
        """Returns a copy of the current Molecule after remove
        all metals and get a mapping dict to rebuild the molecule """
        clone = self.copy()
        old_atoms_list = clone.atoms
        clone.remove_metals()
        new_atoms_list = clone.atoms

        atom_new2old_mapping = {}
        for new_index, atom in enumerate(new_atoms_list):
            try:
                old_index = old_atoms_list.index(atom)
            except IndexError:
                continue

            atom_new2old_mapping[new_index] = old_index

        return clone, atom_new2old_mapping

    def build_3d(
            self,
            engine: Literal['openbabel', 'rdkit'] = 'openbabel',
            forcefield: str = 'UFF',
            steps: int = 500,
            embedding: bool = True,
            **kwargs
    ):
        def _rdkit_optimizer():
            rdmol, row_to_idx = self.to_rdmol()

            # Perform the calculation of implicit valence before adding hydrogens
            rdmol.UpdatePropertyCache(strict=False)

            # Add Hydrogens
            # rdmol = Chem.RemoveHs(rdmol)
            rdmol = Chem.AddHs(rdmol)

            AllChem.EmbedMolecule(rdmol)
            AllChem.UFFOptimizeMolecule(rdmol, maxIters=500, vdwThresh=12.5)

            atoms_array, _, idx_to_row = RDKit.rdkitmol_to_arrays(rdmol)

            row_old_to_new = {r: idx_to_row[i] for r, i in row_to_idx.items()}
            new_rows = np.array(list(row_old_to_new.values()))

            coords = atoms_array[new_rows][:, self.enum_coords]
            self._atoms_data[:, self.enum_coords] = coords

        def _transfer_coords(r2i: dict, i2r: dict, new_atoms_array: np.ndarray):
            row_old_to_new = {r: i2r[i] for r, i in r2i.items()}
            new_rows = np.array(list(row_old_to_new.values()))

            coords = new_atoms_array[new_rows][:, self.enum_coords]
            self._atoms_data[:, self.enum_coords] = coords

        if not self.is_hydrogen_complete:
            self.add_hydrogens()

        if not self.have_normalized_labels:
            self.normalize_labels()

        if engine == 'openbabel':
            if forcefield == 'auto':
                if self.is_organic:
                    obmol, row_to_idx = self.to_obmol()
                    pbmol = pb.Molecule(obmol)

                    pbmol.make3D(forcefield='UFF', steps=50)
                    pbmol.localopt(steps=steps)

                    atoms_array, _, idx_to_row = ob2chem.to_arrays(obmol)
                    _transfer_coords(row_to_idx, idx_to_row, atoms_array)

                else:
                    obmol, row_to_idx = self.to_obmol()
                    pbmol = pb.Molecule(obmol)

                    pbmol.make3D(forcefield='UFF', steps=steps)
                    pbmol.localopt(steps=steps, forcefield='UFF')

                    atoms_array, _, idx_to_row = ob2chem.to_arrays(obmol)
                    _transfer_coords(row_to_idx, idx_to_row, atoms_array)

            else:
                obmol, row_to_idx = self.to_obmol()
                pbmol = pb.Molecule(obmol)

                pbmol.make3D(forcefield=forcefield, steps=steps)
                pbmol.localopt(steps=steps)

                atoms_array, _, idx_to_row = ob2chem.to_arrays(obmol)
                _transfer_coords(row_to_idx, idx_to_row, atoms_array)

        elif engine == 'rdkit':
            _rdkit_optimizer()

        else:
            raise ValueError(f'the engine {engine} is not supported, just "auto" or "openbabel" or "rdkit" allowed')

    def normalize_labels(self):
        """ give each atom a unique label """
        element_count = {}
        for atom in self.atoms:
            count = element_count[atom.symbol] = element_count.get(atom.symbol, 0) + 1
            atom.label = atom.symbol + str(count)

    @property
    def have_normalized_labels(self) -> bool:
        return isinstance(self.atom_labels, list) and self.atom_counts == len(set(self.atom_labels))

    @property
    def smiles(self) -> str:
        """ Return an anti-canonical (i.e., randomized) SMILES string """
        return self.dump_by_openbabel('smi', opt={"C": None}).strip()

    @property
    def canonical_smiles(self) -> str:
        """ Return the canonical SMILES """
        return self.dump_by_openbabel('smi', opt={'c': None, 'i': None}).strip()

    def graph_spectrum(self, length=4):
        clone = self.copy()
        clone.add_hydrogens()
        return GraphSpectrum.from_adj_atoms(clone.adjacency, clone.atomic_numbers, length=length)

    def spectral_similarity(self, other, length: int = 4):
        spec1 = self.graph_spectrum(length)
        spec2 = other.graph_spectrum(length)

        return spec1.similarity(spec2)


class Atom:
    attrs_enumerator = Molecule.atom_attrs_enumerator
    enum_atomic_number = Molecule.enum_atomic_number
    enum_formal_charge = Molecule.enum_formal_charge
    enum_partial_charge = Molecule.enum_partial_charge
    enum_is_aromatic = Molecule.enum_atom_is_aromatic
    enum_coords = Molecule.enum_coords
    enum_total_valence = Molecule.enum_valence


    # Element categorize in periodic tabel
    _alkali_metals = [3, 11, 19, 37, 55, 87]  # Group 1
    _alkaline_earth_metals = [4, 12, 20, 38, 56, 88]  # Group 2
    _transition_metals = list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)) + list(range(104, 113))
    _post_transition_metals = [13, 31, 49, 50, 81, 82, 83, 113, 114, 115, 116]
    _lanthanides = list(range(57, 72))
    _actinides = list(range(89, 104))
    metal_ = set(_alkali_metals + _alkaline_earth_metals + _transition_metals
                 + _post_transition_metals + _lanthanides + _actinides)

    _nonmetals = [1, 6, 7, 8, 15, 16, 34]
    _metalloids = [5, 14, 32, 33, 51, 52, 84]
    _noble_gases = [2, 10, 18, 36, 54, 86, 118]
    _halogens = [9, 17, 35, 53, 85, 117]

    covalent_radii = np.array([0.] + [getattr(periodictable, ob.GetSymbol(i)).covalent_radius or 0. for i in range(1, 119)])

    _default_valence = {
        1: 1,    # Hydrogen (H)
        2: 0,    # Helium (He) - inert
        3: 1,    # Lithium (Li)
        4: 2,    # Beryllium (Be)
        5: 3,    # Boron (B)
        6: 4,    # Carbon (C)
        7: 3,    # Nitrogen (N)
        8: 2,    # Oxygen (O)
        9: 1,    # Fluorine (F)
        10: 0,   # Neon (Ne) - inert
        11: 1,   # Sodium (Na)
        12: 2,   # Magnesium (Mg)
        13: 3,   # Aluminium (Al)
        14: 4,   # Silicon (Si)
        15: 3,   # Phosphorus (P)
        16: 2,   # Sulfur (S)
        17: 1,   # Chlorine (Cl)
        18: 0,   # Argon (Ar) - inert
        19: 1,   # Potassium (K)
        20: 2,   # Calcium (Ca)
        21: 3,   # Scandium (Sc)
        22: 4,   # Titanium (Ti)
        23: 5,   # Vanadium (V)
        24: 3,   # Chromium (Cr)
        25: 2,   # Manganese (Mn)
        26: 2,   # Iron (Fe)
        27: 3,   # Cobalt (Co)
        28: 2,   # Nickel (Ni)
        29: 2,   # Copper (Cu)
        30: 2,   # Zinc (Zn)
        31: 3,   # Gallium (Ga)
        32: 4,   # Germanium (Ge)
        33: 3,   # Arsenic (As)
        34: 2,   # Selenium (Se)
        35: 1,   # Bromine (Br)
        36: 0,   # Krypton (Kr) - inert
        37: 1,   # Rubidium (Rb)
        38: 2,   # Strontium (Sr)
        39: 3,   # Yttrium (Y)
        40: 4,   # Zirconium (Zr)
        41: 5,   # Niobium (Nb)
        42: 6,   # Molybdenum (Mo)
        43: 7,   # Technetium (Tc)
        44: 4,   # Ruthenium (Ru)
        45: 3,   # Rhodium (Rh)
        46: 2,   # Palladium (Pd)
        47: 1,   # Silver (Ag)
        48: 2,   # Cadmium (Cd)
        49: 3,   # Indium (In)
        50: 4,   # Tin (Sn)
        51: 3,   # Antimony (Sb)
        52: 2,   # Tellurium (Te)
        53: 1,   # Iodine (I)
        54: 0,   # Xenon (Xe) - inert
        55: 1,   # Cesium (Cs)
        56: 2,   # Barium (Ba)
        57: 3,   # Lanthanum (La)
        58: 3,   # Cerium (Ce)
        59: 3,   # Praseodymium (Pr)
        60: 3,   # Neodymium (Nd)
        61: 3,   # Promethium (Pm)
        62: 3,   # Samarium (Sm)
        63: 3,   # Europium (Eu)
        64: 3,   # Gadolinium (Gd)
        65: 3,   # Terbium (Tb)
        66: 3,   # Dysprosium (Dy)
        67: 3,   # Holmium (Ho)
        68: 3,   # Erbium (Er)
        69: 3,   # Thulium (Tm)
        70: 3,   # Ytterbium (Yb)
        71: 3,   # Lutetium (Lu)
        72: 4,   # Hafnium (Hf)
        73: 5,   # Tantalum (Ta)
        74: 6,   # Tungsten (W)
        75: 5,   # Rhenium (Re)
        76: 4,   # Osmium (Os)
        77: 3,   # Iridium (Ir)
        78: 2,   # Platinum (Pt)
        79: 1,   # Gold (Au)
        80: 2,   # Mercury (Hg)
        81: 3,   # Thallium (Tl)
        82: 4,   # Lead (Pb)
        83: 3,   # Bismuth (Bi)
        84: 2,   # Polonium (Po)
        85: 1,   # Astatine (At)
        86: 0,   # Radon (Rn) - inert
        87: 1,   # Francium (Fr)
        88: 2,   # Radium (Ra)
        89: 3,   # Actinium (Ac)
        90: 4,   # Thorium (Th)
        91: 5,   # Protactinium (Pa)
        92: 6,   # Uranium (U)
        93: 5,   # Neptunium (Np)
        94: 6,   # Plutonium (Pu)
        95: 3,   # Americium (Am)
        96: 3,   # Curium (Cm)
        97: 3,   # Berkelium (Bk)
        98: 3,   # Californium (Cf)
        99: 3,   # Einsteinium (Es)
        100: 3,  # Fermium (Fm)
        101: 3,  # Mendelevium (Md)
        102: 3,  # Nobelium (No)
        103: 3,  # Lawrencium (Lr)
        104: 4,  # Rutherfordium (Rf)
        105: 5,  # Dubnium (Db)
        106: 6,  # Seaborgium (Sg)
        107: 7,  # Bohrium (Bh)
        108: 4,  # Hassium (Hs)
        109: 3,  # Meitnerium (Mt)
        110: 4,  # Darmstadtium (Ds)
        111: 1,  # Roentgenium (Rg)
        112: 2,  # Copernicium (Cn)
        113: 3,  # Nihonium (Nh)
        114: 4,  # Flerovium (Fl)
        115: 3,  # Moscovium (Mc)
        116: 2,  # Livermorium (Lv)
        117: 1,  # Tennessine (Ts)
        118: 0,  # Oganesson (Og) - inert
    }

    def __init__(
            self, mol: Molecule = None,
            # idx: int = None,
            *,
            atomic_number: int = None,
            is_aromatic: int = 0,
            formal_charge: int = 0,
            partial_charge: int = 0.,
            coordinates: tuple[float, float, float] = (0., 0., 0.)
    ):
        # Create a new molecule for isolate atom
        if not isinstance(mol, Molecule):
            mol = Molecule()
            # idx = 0
        else:  # define atom by given molecule
            atomic_number = None

        # if not isinstance(mol, Molecule) or not isinstance(int(idx), int):
        #     raise ValueError("The mol and atomic_number must be given at least one")

        self.molecule = mol
        # self.idx = int(idx)

        # Init the new molecule created by the isolate atom
        if isinstance(atomic_number, int):
            self.molecule.atoms_data = np.zeros((1, len(self.attrs_enumerator)))
            self.atomic_number = atomic_number
            self.is_aromatic = is_aromatic
            self.formal_charge = formal_charge
            self.partial_charge = partial_charge
            self.coordinates = coordinates

    def __repr__(self):
        return f'Atom({self.label})'

    def __copy__(self):
        raise AttributeError("This object cannot be copied.")

    def __deepcopy__(self, memo):
        raise AttributeError("This object cannot be deep-copied.")

    @property
    def idx(self):
        return self.molecule.atoms.index(self)

    @property
    def attrs(self) -> np.ndarray:
        return self.molecule.atoms_data[self.idx]

    @property
    def atomic_number(self) -> int:
        return int(self.attrs[self.enum_atomic_number])

    @atomic_number.setter
    def atomic_number(self, value: int) -> None:
        if isinstance(value, int) and 0 <= value <= 120:
            self.attrs[self.enum_atomic_number] = value
        else:
            raise ValueError('the atomic_number should be an integer between 0 and 120')

    @property
    def symbol(self) -> str:
        return ob.GetSymbol(self.atomic_number)

    @symbol.setter
    def symbol(self, value: str) -> None:
        self.atomic_number = ob.GetAtomicNum(value)

    @property
    def is_in_ring(self):
        return bool(self.simple_rings)

    @property
    def simple_rings(self) -> list["SimpleRing"]:
        return [r for r in self.molecule.simple_rings if self in r]

    @property
    def label(self) -> str:
        if self.molecule.atom_labels and len(self.molecule.atom_labels) == self.molecule.atom_counts:
            return str(self.molecule.atom_labels[self.idx])
        else:
            return self.symbol

    @label.setter
    def label(self, value: str) -> None:
        if self.molecule.atom_labels and len(self.molecule.atom_labels) == self.molecule.atom_counts:
            self.molecule.atom_labels[self.idx] = value
        else:
            self.molecule.atom_labels = [a.symbol for a in self.molecule.atoms]
            self.molecule.atom_labels[self.idx] = value

    @property
    def covalent_radius(self) -> float:
        return float(self.covalent_radii[self.atomic_number])

    @property
    def coordinates(self) -> np.ndarray:
        return self.attrs[self.enum_coords]

    @coordinates.setter
    def coordinates(self, value: Union[Sequence, np.ndarray]) -> None:
        if isinstance(value, tuple):
            value = np.array(value)

        if value.shape != (3,):
            raise ValueError('the coordinates should be a 3-dimensional array or 3-dimensional Sequence')

        self.attrs[self.enum_coords] = value

    @property
    def bond_orders(self) -> int:
        return int(sum(b.bond_order for b in self.bonds))

    @property
    def is_aromatic(self) -> bool:
        return bool(self.attrs[self.enum_is_aromatic])

    @is_aromatic.setter
    def is_aromatic(self, value: bool) -> None:
        self.attrs[self.enum_is_aromatic] = bool(value)

    @property
    def formal_charge(self) -> int:
        return int(self.attrs[self.enum_formal_charge])

    @formal_charge.setter
    def formal_charge(self, value: int) -> None:
        self.attrs[self.enum_formal_charge] = int(value)

    @property
    def is_metal(self) -> bool:
        return self.atomic_number in self.metal_

    @property
    def bonds(self) -> list['Bond']:
        bonds = self.molecule.bonds

        # TODO: Test code
        if self.bonds_idx is None:
            print(self.molecule.canonical_smiles)

        return [bonds[i] for i in self.bonds_idx]

    @property
    def bonds_idx(self) -> np.ndarray:
        if self.molecule.link_matrix.size > 0:
            return np.where(np.any(self.molecule.link_matrix == self.idx, axis=1))[0]
        else:
            return np.array([])

    @property
    def neigh_idx(self) -> np.ndarray:
        mol_adj = self.molecule.link_matrix
        mask_mat = mol_adj == self.idx
        bond_idx = np.where(np.any(mask_mat, axis=1))[0]
        return np.where(mask_mat[bond_idx][:, 0], mol_adj[bond_idx][:, 1], mol_adj[bond_idx][:, 0])

    @property
    def neighbours(self) -> list['Atom']:
        atoms = self.molecule.atoms
        return [atoms[i] for i in self.neigh_idx]

    @property
    def partial_charge(self) -> float:
        return float(self.attrs[self.enum_partial_charge])

    @partial_charge.setter
    def partial_charge(self, value: float) -> None:
        self.attrs[self.enum_partial_charge] = float(value)

    @property
    def total_valence(self) -> int:
        return int(self.attrs[7])

    @total_valence.setter
    def total_valence(self, value: int) -> None:
        self.attrs[7] = int(value)

    @property
    def valence(self) -> int:
        return sum(b.bond_order for b in self.bonds)

    def electron_configuration(self, length = 4):
        n, conf = calc_electron_config(self.atomic_number, length)
        return n, conf


class Bond:
    """"""
    attrs_enumerator = Molecule.bond_attrs_enumerator
    enum_begin_atom_index = Molecule.enum_begin_atom_index
    enum_end_atom_index = Molecule.enum_end_atom_index
    enum_bond_order = Molecule.enum_bond_order
    enum_is_aromatic = Molecule.enum_bond_is_aromatic

    order_to_name = {
        1: "Single",
        2: "Double",
        3: "Triple"
    }

    def __init__(
            self, mol: Molecule,
            begin_atom_index,
            end_atom_index,
    ) -> None:
        self.molecule = mol
        self.begin_atom_index = begin_atom_index
        self.end_atom_index = end_atom_index

    def __repr__(self):
        return f"Bond({self.atom1.label}, {self.atom2.label}, {self.order_to_name[self.bond_order]})"

    @property
    def attrs(self) -> np.ndarray:
        return self.molecule.bond_data[self.idx]

    @property
    def atoms_index(self) -> (int, int):
        return self.begin_atom_index, self.end_atom_index

    @property
    def idx(self) -> int:
        # mask_array = np.isin(self.molecule.adj, self.atoms_index)
        # judge_array = np.nonzero(np.all(mask_array, axis=1))[0]
        #
        # assert judge_array.size == 1
        # return int(judge_array)
        return self.molecule.bonds.index(self)

    @property
    def is_in_ring(self) -> bool:
        return any(np.all(np.isin(self.atoms_index, r)) for r in self.molecule.simple_rings)

    def rings(self) -> list[np.ndarray]:
        return [r for r in self.molecule.simple_rings if np.all(np.isin(self.atoms_index, r))]

    @property
    def atom1(self) -> Atom:
        return self.molecule.atoms[self.begin_atom_index]

    @property
    def atom2(self) -> Atom:
        return self.molecule.atoms[self.end_atom_index]

    @property
    def bond_order(self) -> int:
        return int(self.attrs[self.enum_bond_order])

    @property
    def is_aromatic(self) -> bool:
        return bool(self.attrs[self.enum_is_aromatic])

    @is_aromatic.setter
    def is_aromatic(self, value: bool):
        self.attrs[self.enum_is_aromatic] = int(value)


class SimpleRing:
    def __init__(self, mol: Molecule, rings: Union[list[int], np.ndarray]):
        self.mol = mol
        self.rings = np.array(rings)

    def __contains__(self, item: Union[Atom, Bond]) -> bool:
        return item in self.atoms or item in self.bonds

    @property
    def atoms(self) -> list[Atom]:
        atoms = self.mol.atoms
        return [atoms[i] for i in self.rings]

    @property
    def bonds(self) -> list[Bond]:
        bonds = self.mol.bonds
        mol_bond_idx = np.where(np.all(np.isin(self.mol.link_matrix, self.rings), axis=1))[0]
        return [bonds[i] for i in mol_bond_idx]

    def members(self) -> int:
        return len(self.rings)

    @property
    def is_aromatic(self):
        return all(a.is_aromatic for a in self.atoms) or self.is_huckle

    @property
    def is_huckle(self) -> bool:
        pi_electron = 0
        for a in self.atoms:
            double_bond_num = len([b for b in a.bonds if b.bond_order == 2])
            if double_bond_num == 1:
                pi_electron += 1
            elif double_bond_num > 1:
                return False
            else:
                if a.symbol == 'C':
                    if a.formal_charge == -1:
                        pi_electron += 2

                if a.symbol in {'N', 'O', 'S'}:
                    pi_electron += 2

        return not (pi_electron - 2) % 4

    def set_aromatic(self, check_huckle: bool = True):
        """ Set all atoms and bonds to aromatic in the ring """
        if not check_huckle or self.is_huckle:
            for a in self.atoms:
                a.is_aromatic = True
            for b in self.bonds:
                b.is_aromatic = True

        elif check_huckle:
            print(UserWarning('Set aromatic fail, due to not meet huckle rule'))

    def unset_aromatic(self):
        for a in self.atoms:
            a.is_aromatic = False
        for b in self.bonds:
            b.is_aromatic = False


class Crystal:
    """ a space symmetry operator for Molecule """
