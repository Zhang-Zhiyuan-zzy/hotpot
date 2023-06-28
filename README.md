# Hotpot
## Introduction
This Python package has been specifically designed to streamline communication between
commonly used computational tools in chemistry and materials research. The package is
aptly named Hotpot, after the popular dish from Sichuan, China. The defining feature of
Hotpot is its ease of preparation and deliciousness, regardless of the ingredients used.
Similarly, this Hotpot package brings together a variety of computational tools 
(i.e. ingredients) to simplify research related to chemical materials. This allows chemists
and materials scientists to create delectable scientific cuisine with ease.

The following jobs are supported by Hotpot:

    - Molecular Simulation, link to LAMMPS and RASPA
    - Quantum or Ab-initio Calculation, link to Gaussian and ABACUS
    - Feature Extraction and Machine learnig, link to openbabel, Zeo++, RdKit et al.

## Installation

### Requirements

````
python >= 3.9
openbabel >= 3.1.1
cclib
lammps
````

### Install requirement
Before installing the `Hotpot`, you should install the requirements at the first. It is
recommended to create a new conda environment to run the package.
> conda create -n hp python==3.9 openbabel cclib lammps rdkit -c conda-forge

### Install
After the requirements are installed, now the ''Hotpot'' could be installed by pip
> conda activate hp

> pip install hotpot-zzy


## Usage
The Hotpot is very easy to use, the core class of Hotpot is the `Molecule`, which is designed
as the general interface for all functions across the entire the package. In the following
example, we first load a Molecule object by `SMILES` string, and then the build their 3D conformer:

```
import hotpot as hp
mol = hp.Molecule('c1c(O)ccc(C(=O)O)c1', 'smi')  # Load a 4-hydroxybenzoic acid molecule
print(mol.has_3d)  # the molcule is a 2D molcule now, whose all coordinates are (0, 0, 0)

mol.build_conformer(force_field='UFF')  # build the molecule to 3D, by univeral force field
print(m.has_3d)  # Now, the molecule is a 3D molecule, all of atoms have their coordinate

# check the atoms coordinates:
mol.normalize_labels()  # reorder the atom's labels
for atom in mol.atoms:
    print(atom.label, atom.symbol, atom.coordinates)  # get the label, symbol, coordinates of the atom
```

In general, a `Molecule` is consist of many `Atom` and `Bond` objects. One can get the attributes from
the `Molecule`, `Atoms` or `Bonds`.
```
print(mol.atoms)  # get all atoms in the molecule
print(mol.bonds)  # get all bonds in the molecule

atom = mol.atoms[0]
bond = mol.bonds[0]

print(atom.neighbours)  # get all neigh atoms of this atoms
print(bond.atom1, b.atom2)  # get the begin and end atom of this bond
print(bond.type)  # get the bond type
```

### Molecule Read and Write
The `Hotpot` read and write the molecule from string or files by calling the [openbabel](https://github.com/openbabel) 
and [cclib](https://github.com/cclib/cclib) packages, most of formats supported by the two packages are support 
by `Hotpot` too. the Main method to read and parse to `Molecule` object is `read_from()`:

> mol = hp.Molecule.read_from('/path/to/file', fmt='cif')  # read a cif file from disk 

Or, read a `SMILES`, `inchikey` or other string like the example above. 

The arg `fmt` is optional when to read `Molecule` from file, if the suffix of the file are correct:
> mol = hp.Molecule.read_from('/path/to/file.cif')

One also could write the molecule object to formatted file by the `writefile()` method, where the `fmt` is
the first arg and required. the actual format of the output is specified by the `fmt` arg:

> mol.writefile('cif', 'path/to/cif/file')

One could retrieve the formatted string by `dump()` method, where only the `fmt` pass into:
> cif_script = mol.dump('cif')

### Cheminformatics
It is easy to get the `SMILES` or `Inchi` key of the `Molecule` object
> print(mol.smiles)

> print(mol.inchi)

The `Molecule` object could convert to certain fingerprint object, like `FP2`, `FP3`, `FP4` or `MACCS`
> fp = mol.fingerprint(fptype='FP2')

The `Molecule` objects could calculate the similarity between each other based on specified fingerprint
> mol.similarity(other_mol, fptype='FP3')  # calculate the similarity by 'FP3' fingerprint

The 'Molecule' object could retrieve its link_matrix as the input of graph learning
> print(mol.link_matrix)  # get a [2, Nb] matrix, where `Nb` is the number of bonds

### Submit the Molecule to Gaussian16 software
One can directly submit the `Molecule` object to Gaussian16 software. Assuming you want to optimize the
configure of the molecule by Gaussian16
```angular2html
mol.gaussian(
    g16root='path/to/g16root',
    link0='the link0 string',
    route='opt B3LYP/6-311++G**',
    path_log_file='path/to/save/the/log',
    path_err_file='path/to/record/error',
    inplace_attrs=True  # whether to inplace the attribute of the molecule according to the last status of the molecule in the log file
)
print(mol.energy)  # get the SCF energy in the last optimized status
print(mol.coordinates)  # get the coordinates matrix after optimizing by gaussian 16
```

### Submit the Molecule(Framework) to LAMMPS to perform grand canonical Monte-Carlo simulation
Suppose that you want to determine the Uptake of carbon dioxide in a metal-organic framework at 298.15 K and 0.5 bar
```angular2html
work_dir = 'work/dir'  # specify a dir to save the results and log for the GCMC simulation

co2 = hp.Molecule.read_from('O=C=O', 'smi')  # load a carbon dioxide by SMILES
frame = hp.Molecule.read_from('path/to/mof/file.cif')  # load a mof file as the framework

# Run GCMC simulation
frame.gcmc(
    co2, 
    force_field='path/to/force/field',  # by default, the force field is the LJ potential from UFF 
    work_dir=work_dir, 
    T=298.15, P=0.5  # specify the external environment
)
```
When perform the GCMC, the chemical potential `mu` or fugacity coefficient `phi` should be given. Fortunately, in
the `mu` or `phi` could be estimated by equation of equation. For some common substance `gcmc()` method can calculate 
the `mu` and `phi` automatically, by `Peng-Robinson` equation by default.

### Access the property of substance for common substance
For certain common substance, we can access its thermodynamical property, like critical temperature `Tc` and
saturation vapor pressure `Psat` by [thermo](https://pypi.org/project/thermo/) package:
```angular2html
mol = hp.Molecule.read_from('c1ccc(O)cc1', 'smi')  # read a phenol by SMILES
mol.thermo_init()  # some kwargs could pass into, see documentation
print(mol.thermo.Tc)  # the critical temperature
print(mol.thermo.Psat)  # the saturation vapor pressure
```