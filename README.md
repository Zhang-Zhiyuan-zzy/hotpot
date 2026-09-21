![banner](https://raw.githubusercontent.com/Zhang-Zhiyuan-zzy/hotpot/main/doc/picture/banner.png)
# 🥘Hotpot(火锅): AI-Driven Infrastructure for Chemistry

> **Bridging the gap between Chemical Intuition and Artificial Intelligence.** *From Empirical Rules to Data-Driven Foundation Models.*  
> **In Hotpot, every ingredient is cookable.** *什么都能涮*  
> **In Data-Driven, every problem is computable.** *什么都能算*  
> *(The Chinese phrases are just the Chinese versions of the English lines. “涮 / shuàn” (“to dip in hotpot”) and “算 / suàn” (“to compute”) form a wordplay because of their similar sound.)*


## Contents
- [Introduction](#-introduction)
- [Key Features](#-key-features--architecture)
- [Installation](#-installation)
- [Usage Examples](#usage-examples)

## 📖 Introduction

Hotpot is not just a chemical informatics toolkit; it is a **research-grade infrastructure** designed to 
digitize, model, and analyze chemistry environments.

Unlike traditional tools (e.g., RDKit, OpenBabel) that rely heavily on explicit valence rules—which 
often fail in metal-ligand scenarios—Hotpot adopts a **Data-Driven Philosophy**. It seamlessly 
integrates a robust chemical kernel with modern deep learning pipelines, enabling "Fuzzy Modeling" 
for complex chemical intuition that cannot be captured by simple mathematical formulas.

**Crucially, Hotpot abstracts the complexity of Artificial Intelligence into a silent, high-performance backend.**

To the user, Hotpot feels like the familiar tools you already use. It simulates manipulating actual 
chemical entities -- whether a single `Molecule` or a periodic `Crystal` Lattice. You interact solely with 
intuitive `Molecule` and `Crystal` objects—the standard vernacular of chemistry. The massive AI training 
frameworks and complex inference engines run entirely behind the scenes, invisible and automated.

+ **Minimal-AI Code**: Users typically do not need to touch tensors, write training loops, or configure neural networks.
+ **Seamless Adaptation**: For standard tasks, the default models work out of the box. For specific domains, 
you simply organize your private data into Molecule objects; Hotpot ingests the data and refines the 
engine automatically.


## 🏗️ Key Features & Architecture

Hotpot is built on a modular architecture designed to hide complexity. It consists of a robust 
**Chemical Kernel** for data handling and a silent **AI Engine** for intelligence.

### 1. The Chemical Kernel (`hotpot.core`)
*The robust foundation that digitizes chemistry.*

+ **Chemist-Centric Interface**:
  - **Intuitive Operations**: Operates in the natural vernacular of chemistry. You interact with `Molecule`,
    `Atom`, and `Bond` objects directly—manipulating structures in code feels exactly like building models in a lab.
  - Plays nicely with existing cheminformatics tools and workflows, preserving the interfaces users are already used to.
+ **Multi-Scale Property Integration**:
  - **Micro to Macro**: A unified interface for managing diverse physical properties. Effortlessly manage microscopic descriptors
    (`Atom.elements`, `Molecule.descriptors`) alongside macroscopic observables (`Molecule.get_thermo()`).
+ **Universal I/O Bridge**: 
  - **Read/Write Common Formats**: Seamlessly handles standard chemistry formats such as `.mol2`, `.cif`, `.xyz`, and Gaussian `.gjf`.  
  - **AI-Ready Graphs**: Transparently converts structures into graph representations suitable for modern deep learning
    models, without exposing low-level details to the user. 

### 2. Data-Driven Analysis (Pre-trained & Ready)
*Intelligence baked into the `Molecule` object (especially, for Coordination Chemistry).*

+ **Coordination pattern determination**:
  - **`AIModel.cbond`**: Surpasses traditional valence rules by using deep learning to predict 
  coordinate bonds in complex transition metal environments.
+ **Site-resolved methyl cation affinity**:
  - **`MCAPredictor`**: Predicts MCA values in kJ/mol for every heavy atom and
    separately identifies important nucleophilic sites from a SMILES string, an RDKit molecule, a hotpot
    `Molecule`, or a lightweight molecular graph. CPU inference is always
    available and CUDA is selected automatically when a compatible ONNX
    Runtime provider is installed.
  - Predictions can be attached directly to atoms through the calculator API:
    ```python
    from hotpot import read_mol
    from hotpot.calculator import mca

    mol = read_mol("c1ccccc1CN")
    mca(mol)
    for atom in mol.atoms:
        print(atom.mca)  # all-atom MCA prediction in kJ/mol
    for atom, value in mol.mca_sites.items():
        print(atom, value)  # important, reliably classified sites
    ```
  - The same predictor is available from the command line. Output is a plain
    atom table suitable for terminal display or redirection:
    ```bash
    hotpot mca 'c1ccccc1CN'
    hotpot mca molecules.sdf -o mca.txt
    hotpot mca 'c1ccccc1CN' --plot mca.png
    hotpot mca 'c1ccccc1CN' --plot mca-all.png --all-site
    ```
+ **Coordination-bond CLI**:
  - Build a metal-ligand coordination graph from a ligand SMILES or molecule
    file and print its canonical SMILES:
    ```bash
    $ hotpot cbond Eu 'O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C(C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3'
    $ hotpot cbond Eu ligand.mol2 -o europium-complex.smi
    $ hotpot cbond Eu ligand.mol2 --all-structures --bond-detail
    $ hotpot cbond --doc
    ```
  - The default raw-logit threshold is `-0.125`. Ranked probabilities from
    `--all-structures` are normalized path weights, not calibrated physical
    probabilities.
+ **3D Structure Initialization** (`Molecule.build3d`):
  - **Complex-aware 3D build**: A specialized force-field pipeline for generating metal complexes
  - **Topology-aware optimization**: Adds continuous topological inspection during geometry optimization 
  and applies tailored breaking / reconstruction strategies, preventing common failures in metal complex 
  3D generation, such as tangled chain, interlocked rings, and other non-optimizable artifacts.
  - Missing hydrogens are added on a transactional working copy. Energies are
    reported in kJ/mol, and `quality_level="standard"` is used by default.
  - Complex force-field requests currently resolve to UFF. This workflow does
    not infer oxidation states or guarantee a ligand-field geometry.
+ **Connecting microscopic models with macroscopic observables**:
  - Macroscopic properties (`logβ`, `logD`, ...) are typically statistical constructs emerging from ensembles of microscopic 
  states, rather than from any single configuration. Relying on a small number of static microscopic 
  models to infer macroscopic behavior can therefore introduce substantial bias and be misleading.
  - Hotpot combines approximate microscopic models with rich molecular representations and environmental
  variables to make this micro–macro connection more reliable. Embracing the idea that “all models are wrong, 
  but some are useful”, Hotpot uses AI-based fuzzy modeling to improve the robustness and accuracy of inferring
  macroscopic observables from microscopic model.
+ **Oxidation state identification** - *coming soon ...*
+ **Other important chemical problems**  
  - If there is a core chemistry task you think should be “built-in” to the `Molecule` object, feel free to open an issue and describe your use case.

### 3. Assembly & Generation of Virtual Molecules
*From fragment-based enumeration to AI-driven molecular design.*

+ **High-throughput fragment-based assembly**  
  - Assemble virtual molecules from scaffolds and fragments at scale, enabling grid-like exploration of targeted
  chemical spaces (e.g. focused libraries around a given scaffold or motif).

+ **AI-based molecular generation**  
  - **Molecular generation**: Generate new candidate molecules by learning from a small set of example structures, 
  proposing novel analogues in the same “chemical family” or design space.
  - **Conditional molecular generation**: Generate molecules under explicit goals or constraints — e.g. guided by
  target properties, property predictors, or user-defined objective functions — to search for structures that optimize
  (maximize / minimize) desired performance while respecting structural patterns of the examples.

### 4. Optimization of Wet Experiments
*Close the loop between computation and lab experiments.*

+ **Multiple optimization strategies**  
  - Supports a range of optimization backends, including Bayesian optimization (BO) and evolutionary algorithms (EA), 
  for efficient exploration of experimental parameter spaces.

+ **Structure-aware experimental optimization**  
  - Combines experimental parameters with optional structural / molecular representations, enabling joint optimization
  over both reaction conditions and molecular features.

+ **Mixed-type design spaces**  
  - Handles continuous and discrete variables in a unified framework, suitable for real experimental design problems
  (temperatures, pH, solvents, ligands, catalysts, etc.).

+ **Manifold / parameter-space visualization**  
  - Provides visualization of the explored parameter manifold and optimization trajectory to help chemists understand
  where the optimizer is searching and why.

+ **CLI integration**
  - Exposed via a simple command-line interface, e.g. `hotpot optimize ...`, so optimization workflows can be scripted
  and automated without additional boilerplate.
---

## 📥 Installation

Python 3.9–3.14 is supported by the chemical kernel, search layer and ONNX
inference path. Python 3.10–3.14 use Open Babel 3.2.x; Python 3.9 remains a
compatibility target and uses `openbabel-wheel` 3.1.1.23 because Open Babel
3.2.x does not publish a Python 3.9 package.

### PyPI installation

```bash
conda create -n hp python=3.11 pip -y
conda activate hp
python -m pip install --upgrade pip
python -m pip install hotpot-zzy
```

The PyPI installation includes the chemical kernel, MCA/CBond ONNX inference
and molecular plotting. On Python 3.10 or newer it installs the official
`openbabel` 3.2.x package; Python 3.9 installs `openbabel-wheel` 3.1.1.23.
Do not mix PyPI `openbabel`, `openbabel-wheel`, and Conda `openbabel` in one
environment because they provide the same Python modules and native libraries.

### Editable source installation

```bash
git clone https://github.com/Zhang-Zhiyuan-zzy/hotpot.git
cd hotpot
conda create -n hp python=3.11 pip -y
conda activate hp
python -m pip install --upgrade pip
python -m pip install -e .
```

For the fixed Python 3.11/Open Babel 3.2 development environment described by
`environment.yml`, run the following from the repository root:

```bash
conda env create -f environment.yml
conda activate hp
```

Available optional dependency groups are:

| Extra | Installation | Scope |
|---|---|---|
| `optimize` | `pip install 'hotpot-zzy[optimize]'` | optimization and classical ML workflows |
| `datasets` | `pip install 'hotpot-zzy[datasets]'` | downloads, HDF5 and PyG datasets |
| `complexformer` | `pip install 'hotpot-zzy[complexformer]'` | ComplexFormer training and LoRA support |
| `onnx-export` | `pip install 'hotpot-zzy[onnx-export]'` | ONNX export and inspection tools |
| `legacy-search` | `pip install 'hotpot-zzy[legacy-search]'` | archived SymPy-based search modules |
| `dev` | `pip install -e '.[dev]'` | tests, linting and package builds |
| `all` | `pip install -e '.[all,dev]'` | all pip-installable optional components |

`torch-cluster` and `torch-scatter` in the `complexformer` extra may require a
PyTorch/CUDA-specific wheel index. CCDC is proprietary, while Gaussian, xTB,
Zeo++ and LAMMPS are external programs; none can be installed reliably as a
portable PyPI dependency.

For GPU ONNX inference, replace the CPU runtime after installation:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```
---
## 📌 Usage examples
### 1.Building a metal-ligand pair
```python
import hotpot as hp
smi = 'O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C(C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3'  # (CyMe4)Pyz-PrMe-DIPhen extractant
ligand = hp.read_mol(smi)

pair = ligand.auto_pair_metal('Eu')
print(pair.smiles)
```

Generate and optimize 3D coordinates through the canonical `build3d` method.
Complex building uses a spawned worker, so executable scripts should use the
standard Python main guard. By default, the first chemically acceptable ligand
geometry is refined. Set `candidate_count` to a positive integer only when an
explicit multi-conformer search is wanted; if the search budget cannot supply
the requested count, Hotpot warns and refines the acceptable candidates that
were found:

```python
import hotpot as hp

SMILES = (
    "O=C(N(C)CCC)C(C=C1)=NC2=C1C=CC3=C2N=C("
    "C4=NC(C(C)(C)CCC5(C)C)=C5N=N4)C=C3"
)


def main():
    pair = hp.read_mol(SMILES).auto_pair_metal("Eu")
    report = pair.build3d(
        seed=20260916,
        max_attempts=20,
        epochs=20,
        steps_per_epoch=500,
    )
    print(report.optimization.best_energy, report.optimization.energy_unit)
    pair.write("./Eu-pair.mol2")


if __name__ == "__main__":
    main()
```
The [mol2 file](https://raw.githubusercontent.com/Zhang-Zhiyuan-zzy/hotpot/main/doc/mol_file/Eu-pair.mol2) 
and [movie](https://raw.githubusercontent.com/Zhang-Zhiyuan-zzy/hotpot/main/doc/picture/Eu-pair.gif) after coordination generation.

Coordination-bond candidates are proposed by the AI model. The resulting 3D
complex is built and screened by the topology-aware force-field workflow.

### 2.Cheminformatics support
The `Molecule` object is designed to be a familiar, standard cheminformatics tool for chemists.
You can access the `Atom`, `Bond`, `Rings`, and fragment `Molecule` objects directly through
the *properties* of `Molecule`.

##### Properties
Continuing with the *Eu-ligand pair* example:
```pycon
print(pair.atoms)
print(pair.bonds)
print(pair.rings)                       # full-graph Relevant Cycles
print(pair.ligand_rings)                # ligand-skeleton Relevant Cycles
print(pair.cycle_basis_rings)           # explicit legacy cycle basis

assert len(pair.components) == 1
pair.hide_metal_ligand_bonds()          # Hide the coordination bonds temporarily
assert len(pair.components) == 2        # Now appears as two fragments: [ligand, metal]
pair.recover_hided_metal_ligand_bonds()
assert len(pair.components) == 1        # Restored to a whole pair

eu_metal = pair.metals[0]
print(eu_metal.neighbours)              # [Atom(N), Atom(N), Atom(N), Atom(O)]

print(pair.link_matrix)                 # Connectivity graph table
```

Relevant-Cycle access returns the complete requested family or raises
`RelevantCycleLimitExceeded` at the default 10,000-cycle safety limit; it never
returns a silently truncated family. Use
`rings_for_scope(..., max_cycles=None)` only when unbounded enumeration is
intentional. Existing trained ring-feature models continue to use the explicit
legacy cycle-basis APIs for their ring tensors; aromaticity perception itself
uses Relevant Cycles.

##### SMARTS Support & Extensions

Searching for coordination centers using SMARTS patterns:
```pycon
hits = pair.search_substructure('[Ln](n)(n)(n)O')  # [Ln] --> lanthanide
print(len(hits))  # == 1
print(hits[0].atoms)  # [Atom(N32), Atom(O0), Atom(Eu67), Atom(N10), Atom(N17)]

hits = pair.search_substructure('[Ln](n)(n)O')
print(len(hits))  # == 3

hits = pair.search_substructure('[An](n)(n)(n)O')  # [An] --> actinide
print(len(hits))  # == 0
```
Hotpot features a built-in SMARTS parser ([API](https://raw.githubusercontent.com/Zhang-Zhiyuan-zzy/hotpot/main/hotpot/cheminfo/smarts.md))
that compiles atom, bond, logical and anchored recursive expressions into the
NetworkX-backed `Searcher` objects. Unsupported stereochemical, directional-bond
and isotope constraints fail explicitly instead of being treated as unconstrained.

Topology-sensitive SMARTS can select one of two named semantics profiles. The
default preserves the complete molecular graph; the ligand profile removes
metal--nonmetal edges only when calculating ligand-local descriptors and never
mutates the molecule or replaces the NetworkX search backend:

```pycon
from hotpot import SmartsSemantics

full_hits = pair.search_substructure(
    "[N;D4;X4]", semantics=SmartsSemantics.FULL_GRAPH
)
ligand_hits = pair.search_substructure(
    "[N;D3;X3]", semantics=SmartsSemantics.LIGAND_SKELETON
)
```

| Profile | `D` / `X` | `v` | `R` / `r` |
|:--------|:----------|:----|:----------|
| `FULL_GRAPH` (default) | All graph neighbours; `X` also includes implicit H | Sum of numeric bond orders plus implicit H | Relevant Cycles from `Molecule.rings` |
| `LIGAND_SKELETON` | Non-metal atoms exclude metal--ligand edges; metal centres retain their full coordination number | Uses the same ligand view and counts only `SINGLE`, `DOUBLE`, `TRIPLE`, and `AROMATIC` bond kinds | Relevant Cycles from `Molecule.ligand_rings` |

Both profiles use the `Atom.implicit_hydrogens` produced by the input reader;
switching profiles does not reperceive or recalculate hydrogens. For example,
Open Babel 3.1 and 3.2 assign the same coordinated amine donor zero implicit H
from the supplied MOL2 fixture but one implicit H from its SDF counterpart, so
their `X` and `v` values remain format-dependent even in `LIGAND_SKELETON`.

Bond matching uses semantic `BondKind` metadata. `-` and an implicit aliphatic
single bond match `SINGLE`, not `DATIVE`, `UNKNOWN`, or `ZERO`; `~` matches any
edge. Open Babel 3.1 and 3.2 collapse MOL2 `du`, `un`, and `nc` bond tokens to
order zero and do not retain which token was present, so Hotpot conservatively
records those imported edges as `UNKNOWN`.

The MCA calculator uses `LIGAND_SKELETON` to classify organic motifs, but its
reported reliable sites deliberately exclude metal atoms and atoms directly
bound to a metal. Per-atom model output and reliable-site selection are
separate concepts; the latter is an applicability-domain decision rather than
a general SMARTS rule.

To specifically address the demand in **Coordination Chemistry**, the syntax has been extended with custom 
wildcards for metals and periodic table properties:

| Symbol      | Definition | Description                                      | Example            |
|:------------|:-----------|:-------------------------------------------------|:-------------------|
| **`M`**     | Metal      | Matches any metal atom                           | `[M]~[O]`          |
| **`!M`**    | Non-Metal  | Matches any non-metal atom                       | `[!M]`             |
| **`Ln`**    | Lanthanide | Matches Lanthanide series (La-Lu)                | `[Ln](n)(n)(n)`    |
| **`An`**    | Actinide   | Matches Actinide series (Ac-Lr)                  | `[An]~[O]`         |
| **`NP<n>`** | Period     | Matches elements in Period *n* (supports ranges) | `[NP4]`, `[NP3-5]` |
| **`NG<n>`** | Group      | Matches elements in Group *n* (supports ranges)  | `[NG1]`, `[NG1-2]` |

##### Conversion with `RDKit` and `OpenBabel`

Interfacing with other cheminformatics tools:
```pycon
obMol = pair.to_obmol()
rdMol = pair.to_rdmol()

# The shared input converter also accepts SMILES, paths, RDKit Mol, OBMol,
# Pybel Molecule, and objects exposing to_rdmol().
assert hp.to_hotpot_mol(pair) is pair
from_rdkit = hp.to_hotpot_mol(rdMol)
from_openbabel = hp.to_hotpot_mol(obMol)
from_smiles = hp.to_hotpot_mol("CCN")
```
External molecules are copied while preserving source atom order. Convert the
whole molecule first, then retrieve a corresponding Hotpot atom by source index;
isolated external atoms are deliberately not converted without their graph.
Converting to [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) Data:
```pycon
data = pair.to_pyg_data()
print(data.x)                       # Tensor of atom attributes
print(data.x_names)                 # atom attribute name
print(data.edge_index)
print(data.edge_attr)
print(data.edge_attr_names)
print(data.pair_index)              # atom pairs indices
print(data.pair_attr)               # pair attrs
print(data.pair_attr_names)
print(data.rings_node_index)
print(data.rings_attr)              # Tensor with shape [rings_num, 2]
print(data.rings_attr_names)        # ['is_aromatic', 'has_metal']
print(data.rings_node_nums)         # How many atoms in a ring
print(data.mol_rings_node_nums)     # How many rings in the molecule
print(data.coordinates)
```

See the [cheminfo.core API Documentation](https://raw.githubusercontent.com/Zhang-Zhiyuan-zzy/hotpot/main/doc/cheminfo.md) for more details.

### 3.Molecular properties, descriptors, and representation
Extracting thermodynamic properties using [`thermo`](https://thermo.readthedocs.io/) library:
```pycon
import hotpot as hp
mol = hp.read_mol('c1ccc(O)cc1', 'smi')  # read a phenol by SMILES
thermo = mol.get_thermo(temp=298.15, pressure=101325)
print(thermo.Tc)  # the critical temperature (K)
print(thermo.Psat)  # the saturation vapor pressure 
print(...)
```
Extracting the Graph-Spectral representation:
```python
import hotpot as hp
mol1 = hp.read_mol('c1ccc(O)cc1', 'smi')
mol2 = hp.read_mol('c1ccccc1C(=O)O', 'smi')
mol1_ = hp.read_mol('c1ccccc1O', 'smi')     # Same molecule, different atom ordering

spectral1 = mol1.graph_spectral()
spectral2 = mol2.graph_spectral()
spectral1_ = mol1_.graph_spectral()

similarity_diff = spectral1 | spectral2
print(similarity_diff)                      # Similarity in graph spectrum: 0.907590226292854
similarity_same = spectral1 | spectral1_
print(similarity_same)                      # Similarity in graph spectrum: 1.0

print(spectral1.vectors.shape)              # numpy array: shape=[6, 13]
print(spectral2.vectors.shape)              # numpy array: shape=[6, 15]
```

### 4.Molecular assembly
The molecular assembly is handled by the standalone module 
[`hotpot.MolAssembly`](https://raw.githubusercontent.com/Zhang-Zhiyuan-zzy/hotpot/main/hotpot/cheminfo/mol_assemble/README.md) temporarily.
### Generic description
The molecular assembly (`hotpot.cheminfo.mol_assemble`) module iteratively generates virtual 
molecular structures based on the user-specified molecular Framework (`hotpot.Molecule`) and
assembly fragments `hotpot.cheminfo.mol_assemble.Fragment`. The Framework is a standard 
`Molecule` object, while the assembly operation is specifically implemented using the `Fragment`. 

An instantiated `Fragment` must specify the following four factors:
1) The 2D molecular structure of the fragment (a `Molecule` object)
2) The atom(s) (specified by index) on the fragment used for connection with the Framework
3) The searcher for locating connection sites on the Framework (a `hotpot.cheminfo.search.Searcher` object)
4) The specific connection operation (specified in an `action_func` function) between the `Fragment` and 
the Framework at the connection sites.

The `Fragment` provides users the flexibility to customize their own assembly strategies. 
Of course, `Hotpot` has predefined some common molecular assembly `Fragment` (named `Assembler`).
When handling the `Assembler`, users only need to specify its fragment structure and indicate the
(optional) `action_points` indices (i.e., specify which Fragmental atoms as the `"reaction site"` to
react with the frame `Molecule`).

So far, the predefined `Assembler` include (see the following `Scheme 1` for details):
1) EdgeShoulder (required two `action_points`)
2) AtomLink (required one `action_points`)
3) BondAdding (No `action_points` required)
4) AtomReplace (No `action_points` required)
5) AlkylGraft (No `action_points` required, just a specific `AtomLink`)
6) RingWedge (required one `action_points`)

![Scheme of Assemblers](https://raw.githubusercontent.com/Zhang-Zhiyuan-zzy/hotpot/main/hotpot/cheminfo/mol_assemble/Assemblers.svg)

***Scheme 1** Illustration of Assembly of Molecule by different Assemblers*


### 5.Wet-lab experimental optimization

**Hotpot** also integrates a module for optimizing the *wet-lab experiments* using an active learning scheme.
For pure parameter optimization, you can use the CLI interface:
```bash
hotpot optimize [input_excel] [output_dir] --flags args ...
hotpot optimize --help  # for help
```
Simply follow the instructions in the command‑line interface to obtain the optimized recommended parameters.
The results and the manifold visualization of the explored parameter space are saved in `output_dir`.
The `input_excel` file should be organized as follows:

| feature1 | feature2 | ... | featureN | target |
|----------|----------|-----|----------|--------|
| 0.64654  | 148.792  | ... | -30.897  | 0.3433 |
| ...      | ...      | ... | ...      | ...    |
---------------------------------------------------

For optimization involving molecule structures:
```python
import numpy as np
import hotpot as hp

list_smi = [
    'c1cccc1',
    'c1cccc1C(=O)O',
    # ...
]

mol_space = [hp.read_mol(smi) for smi in list_smi]
samples = [
    hp.read_mol(list_smi[i]) for i in np.random.randint(2, size=100).tolist()
]
for mol, params in zip(samples, np.random.randn(100, 3)):
    mol.add_envs(params, name=['T', 'P', 'Conc.'])

bundle = hp.MolBundle(samples)

result = bundle.optimize(
    mol_space=mol_space,  # Optional
    env_space=...,        # Optional
    maximize=True,        # Default
    n_trails=20,
    batch_size=5,
    mol_repr='ComplexFormer_nano',  # Optional[rdkit, fp, spectrum], The optimize method automatically selects a suitable representation.
    visualize=True
)

print(result.mol.smiles)
print(result.env)
result.fig.show()  # Displays the manifold visualization
```


## 🛤️ Roadmap & Project Evolution

Hotpot initially started as a more Pythonic wrapper around OpenBabel and RDKit, aiming to:

- provide a cleaner, chemist-friendly interface on the Python side, and  
- avoid low-level C++ issues (e.g., segmentation faults / exit code 139) ..., and the unnatural modeling of metal complexes.

During development, it became clear that heuristic, rule-based logic is not sufficient for many real chemical problems,
especially in coordination chemistry. Many chemical and biological insights are empirical and resist explicit coding.

Hotpot is therefore evolving from a **rule-based wrapper** into a **data-driven infrastructure** that tries to capture
such *tacit knowledge* through large-scale pre-training on coordination chemistry and related databases.

**Current Status**

- The current `main` branch focuses on a stable, chemist-centric core (`hotpot.cheminfo.core`) and classical utilities.
- Several advanced AI-backed components described in this README currently live in  research branches and
  internal prototypes, and will be merged step by step.
- Public APIs in `hotpot.cheminfo.core` will be kept as stable as possible to ensure backward compatibility as new 
  models and pipelines are integrated.

**Planned Timeline**

A large part of the AI backend is closely tied to ongoing Ph.D. research work.  
Major model components and pipelines are planned to be merged into the public repository progressively as the
research is completed and stabilized (target: around late 2026).
