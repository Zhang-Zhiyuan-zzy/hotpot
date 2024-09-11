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
> conda create -n hp python==3.9 openbabel cclib lammps -c conda-forge

### Install
After the requirements are installed, now the ''Hotpot'' could be installed by pip
> conda activate hp

> pip install hotpot-zzy

or you can install from this github repository:
```angular2html
git clone https://github.com/Zhang-Zhiyuan-zzy/hotpot.git
pip install build  # install `build` package
python -m build
pip install dist/hotpot_zzy-`VERSION`-py3-none-any.whl
```


## Usage
The Hotpot is very easy to use, the core class of Hotpot is the `Molecule`, which is designed
as the general interface for all functions across the entire the package. In the following
example, we first load a Molecule object by `SMILES` string, and then the build their 3D conformer:

```pycon
import hotpot as hp
mol = hp.Molecule.read_from('c1c(O)ccc(C(=O)O)c1', 'smi')  # Load a 4-hydroxybenzoic acid molecule
print(mol.has_3d)  # the molcule is a 2D molcule now, whose all coordinates are (0, 0, 0)

mol.build_3d(force_field='UFF')  # build the molecule to 3D, by univeral force field
print(m.has_3d)  # Now, the molecule is a 3D molecule, all of atoms have their coordinate

# check the atoms coordinates:
mol.normalize_labels()  # reorder the atom's labels
for atom in mol.atoms:
    print(atom.label, atom.symbol, atom.coordinate)  # get the label, symbol, coordinates of the atom
```

In general, a `Molecule` is consist of many `Atom` and `Bond` objects. One can get the attributes from
the `Molecule`, `Atoms` or `Bonds`.
```pycon
print(mol.atoms)  # get all atoms in the molecule
print(mol.bonds)  # get all bonds in the molecule

atom = mol.atoms[0]
bond = mol.bonds[0]

print(atom.neighbours)  # get all neigh atoms of this atoms
print(bond.atom1, bond.atom2)  # get the begin and end atom of this bond
print(bond.type)  # get the bond type
```

### Molecule Read and Write
The `Hotpot` read and write the molecule from string or files by calling the [openbabel](https://github.com/openbabel) 
and [cclib](https://github.com/cclib/cclib) packages, most formats supported by the two packages are support 
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
conformer of the molecule by Gaussian16

```pycon
mol.gaussian(
    g16root='path/to/g16root',
    link0='the link0 string',
    route='opt B3LYP/6-311++G**',
    path_log_file='path/to/save/the/log',
    path_err_file='path/to/record/error',
    inplace_attrs=True  # whether to inplace the attribute of the molecule according to the last status of the molecule in the log file
    debugger='auto'  # Handle the Gaussian Error by the default method
)
print(mol.energy)  # get the SCF energy in the last optimized status
print(mol.coordinate)  # get the coordinates matrix after optimizing by gaussian 16
```
The Gaussian program will run and handle some common error report automatically. To handle errors with more elaborate
methods, user can custom a new debugger by inherit from the hotpot.tanks.quantum.GaussErrorHandle, seeing 
documentation for more details.

### Submit the Molecule(Framework) to LAMMPS to perform grand canonical Monte-Carlo simulation
Suppose that you want to determine the Uptake of carbon dioxide in a metal-organic framework at 298.15 K and 0.5 bar
```pycon
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
the `mu` or `phi` could be estimated by state of equation. For some common substance `gcmc()` method can calculate 
the `mu` and `phi` automatically, by `Peng-Robinson` equation by default.

### Access the property of substance for common substance
For certain common substance, we can access its thermodynamical property, like critical temperature `Tc` and
saturation vapor pressure `Psat` by [thermo](https://pypi.org/project/thermo/) package:

```pycon
mol = hp.Molecule.read_from('c1ccc(O)cc1', 'smi')  # read a phenol by SMILES
mol.get_thermo()  # some kwargs could pass into, see documentation
print(mol.thermo.Tc)  # the critical temperature
print(mol.thermo.Psat)  # the saturation vapor pressure
```

### Handle molecules in large scale
In the era of artificial intelligence, chemical information needs to be processed and utilized on a large scale. 
`Hotpot` provides an interface called `MolBundle` for processing data on a large scale. For instance, if there 
is a large number of single-point energy results computed using `Gaussian` stored somewhere on a disk, and we 
want to create a dataset to train a [deep potential](https://tutorials.deepmodeling.com/en/latest/Tutorials/DeePMD-kit/learnDoc/Introduction.html)
model using this data, we can utilize "MolBundle" to efficiently read all the `Gaussian` computation data on a large
scale and convert it into the required dataset [System](https://docs.deepmodeling.com/projects/deepmd/en/master/data/system.html) 
format for training the model:

```pycon
import hotpot as hp
from hotpot.bundle import DeepModelBundle

path_raw_data = 'path/to/gaussian/log'
path_system = 'path/to/system'

bundle = hp.MolBundle.read_from(
    'g16log', path_raw_data, '*/*.log', nproc=32
)

# Convert to DeepModelBundle object with method to organize the molecular structures to System dataset
bundle: DeepModelBundle = bundle.to('DeepModelBundle')
bundle.to_dpmd_sys(path_system, validate_ratio=0.1)

# Or, the user could get the System object export from the Molecule directly
```

`hotpot` is currently making every effort to support the use of various computational tools from the Deep Modeling
community. In addition to organize the quantum calculation data and save them to disk directly, the `hotpot`
now allowed build `Molecule` object from dpdata [System] and [LabeledSystem] object.

```python
from pathlib import Path

import hotpot as hp
from hotpot.plugins.deepmd import read_system

data_root_dir = "path/to/data"

# Read MultiSystem object
ms = read_system(data_root_dir, file_pattern='**/*.log', fmt="gaussian/md")

mols = []
for ls in ms:
    mol = hp.Molecule.build_from_dpdata_system(ls)
    mols.append(mol)

# Supposed that I want to know the process of breaking and generating of bonds of the first Molecule
struct_dir = Path('path/to/struct/save')
img_dir = Path('path/to/img/save')
mol = mols[0]
# Iterating each conformer in the quantum chemistry calculation
for i in range(mol.conformer_counts):
    mol.conformer_select(i)
    mol.remove_bonds(*mol.bonds)  # Clear all pre-build bonds
    mol.build_bonds()  # rebuild bonds according to the point cloud of atoms
    mol.assign_bond_types()

    mol.writefile(struct_dir.joinpath(f"{i}.mol2"))  # Save the 3D mol structure with built bonds to mol2 file
    mol.save_2d_img(img_dir.joinpath(f'{i}.png'))  # Save the 2d img structure to png file
```

## TroubleShooting
### 1) Missing dependent dynamic libs
When installing the package, you might meet some errors from missing dependent libs, like the message:
*ImportError: libXrender.so.1: cannot open shared object file: No such file or directory*. 
This trouble is caused by the lacking of the `libxrender1` lib and could be solved by run the following command
(supposing an Ubuntu system):
> sudo apt-get install libxrender1

The similar trouble should be solved like the above.