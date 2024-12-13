"""
python v3.9.0
@Project: hotpot0.5.0
@File   : rdkit2chem
@Auther : Zhiyuan Zhang
@Data   : 2024/6/2
@Time   : 9:31
"""
import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

bond_map = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE
}


class RDKit:
    @staticmethod
    def rdkitmol_to_arrays(rdmol):
        # Initialize lists to store atom and bond information
        atoms = []
        bonds = []

        # Get atom indices in a consistent order
        idx_to_row = {atom.GetIdx(): i for i, atom in enumerate(rdmol.GetAtoms())}

        # Iterate over atoms in the molecule
        for idx, atom in enumerate(rdmol.GetAtoms()):
            atomic_number = atom.GetAtomicNum()
            formal_charge = atom.GetFormalCharge()
            partial_charge = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
            is_aromatic = int(atom.GetIsAromatic())
            pos = rdmol.GetConformer().GetAtomPosition(atom.GetIdx())
            x, y, z = pos.x, pos.y, pos.z
            valence = atom.GetTotalValence()
            implicit_hydrogens = atom.GetNumImplicitHs()
            # explicit_hydrogens = atom.GetNumExplicitHs()

            # Append the atom information to the atoms list
            atoms.append([
                atomic_number, formal_charge, partial_charge,
                is_aromatic, x, y, z, valence, implicit_hydrogens,
                # explicit_hydrogens
            ])

        # Iterate over bonds in the molecule
        for bond in rdmol.GetBonds():
            begin_atom_idx = idx_to_row[bond.GetBeginAtomIdx()]  # Map to reordered index
            end_atom_idx = idx_to_row[bond.GetEndAtomIdx()]  # Map to reordered index
            bond_order = bond.GetBondTypeAsDouble()
            is_aromatic = bond.GetIsAromatic()

            # Append the bond information to the bonds list
            bonds.append([begin_atom_idx, end_atom_idx, bond_order, is_aromatic])

        # Convert lists to numpy arrays
        atoms_array = np.array(atoms)
        bonds_array = np.array(bonds)

        return atoms_array, bonds_array, idx_to_row

    @staticmethod
    def arrays_to_rdkitmol(atoms_array, bonds_array):
        mol = Chem.RWMol()

        # Create atoms in the molecule and set their properties
        row_to_idx = {}
        for i, atom_data in enumerate(atoms_array):

            atomic_number, formal_charge, partial_charge, is_aromatic, x, y, z, valence, implicit_hydrogen = atom_data

            atom = Chem.Atom(int(atomic_number))
            atom.SetFormalCharge(int(formal_charge))
            atom.SetDoubleProp('_GasteigerCharge', partial_charge)
            atom.SetIsAromatic(bool(is_aromatic))  # Convert to bool

            idx = mol.AddAtom(atom)
            row_to_idx[i] = idx

        # Create bonds in the molecule
        for bond_data in bonds_array:
            begin_atom_idx, end_atom_idx, bond_order, is_aromatic = bond_data
            mol.AddBond(row_to_idx[begin_atom_idx], row_to_idx[end_atom_idx],
                        Chem.BondType.values[int(bond_order)])
            bond = mol.GetBondBetweenAtoms(row_to_idx[begin_atom_idx], row_to_idx[end_atom_idx])
            bond.SetIsAromatic(bool(is_aromatic))  # Convert to bool

        # Set 3D coordinates
        conf = Chem.Conformer(len(atoms_array))
        for idx, atom_data in enumerate(atoms_array):
            x, y, z = atom_data[[4, 5, 6]]  # (4, 5, 6) is the enumerator of coordinates
            conf.SetAtomPosition(idx, Chem.rdGeometry.Point3D(x, y, z))
        mol.AddConformer(conf)

        return mol, row_to_idx

    @staticmethod
    def nx_to_rdkit_molecule(graph):
        """Convert a NetworkX graph to an RDKit molecule."""
        rdmol = Chem.RWMol()
        atom_idx_map = {}

        # Add atoms
        coordinates = []
        for node, data in graph.nodes(data=True):
            atomic_number = data.get('atomic_number', 'C')  # Default to carbon if no element is specified
            atom = Chem.Atom(atomic_number)
            rd_idx = rdmol.AddAtom(atom)
            atom_idx_map[node] = rd_idx

            atom.SetIsAromatic(data.get('is_aromatic', False))
            coordinates.append(data['coordinates'])

        coordinates = np.array(coordinates)
        RDKit.graph_coordinates_to_rdkit_conformer(rdmol, coordinates)

        # Add bonds
        for u, v, data in graph.edges(data=True):
            bond_type = data.get('bond_type')  # Default to single bond
            rdmol.AddBond(atom_idx_map[u], atom_idx_map[v], bond_map[bond_type])

            bond = rdmol.GetBondBetweenAtoms(atom_idx_map[u], atom_idx_map[v])
            bond.SetIsAromatic(data.get('is_aromatic', False))

        return rdmol, atom_idx_map

    @staticmethod
    def graph_coordinates_to_rdkit_conformer(rdmol, coordinates: np.ndarray):
        if coordinates.shape != (rdmol.GetNumAtoms(), 3):
            raise ValueError("The shape of coordinates array must match the number of atoms in the molecule")

        # Check if the molecule already has a conformer, if not add one
        if rdmol.GetNumConformers() == 0:
            conformer = Chem.Conformer(rdmol.GetNumAtoms())
            rdmol.AddConformer(conformer)
        else:
            conformer = rdmol.GetConformer()

        # Set the coordinates for each atom
        for atom_idx in range(rdmol.GetNumAtoms()):
            x, y, z = coordinates[atom_idx]
            conformer.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))

    @staticmethod
    def rdkit_to_nx(mol):
        """Convert an RDKit molecule to a NetworkX graph, including 3D coordinates if available."""
        graph = nx.Graph()

        # Add nodes with atom properties
        for atom in mol.GetAtoms():
            node_idx = atom.GetIdx()
            graph.add_node(node_idx,
                           atomic_number=atom.GetAtomicNum(),
                           formal_charge=atom.GetFormalCharge(),
                           is_aromatic=atom.GetIsAromatic())

        # Check and include 3D coordinates if available
        if mol.GetNumConformers() > 0:
            conformer = mol.GetConformer()
            for atom in mol.GetAtoms():
                node_idx = atom.GetIdx()
                pos = conformer.GetAtomPosition(node_idx)
                graph.nodes[node_idx]['coords'] = np.array([pos.x, pos.y, pos.z])

        # Add edges with bond properties
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            graph.add_edge(begin_idx, end_idx,
                           bond_type=str(bond_type),
                           is_aromatic=bond.GetIsAromatic())

        return graph

    @staticmethod
    def mol_to_smiles(mol):
        """Convert an RDKit molecule to a SMILES string."""
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)



