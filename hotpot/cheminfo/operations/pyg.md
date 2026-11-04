# General chemical information and PyG Data

`Molecule.to_pyg()` and `Crystal.to_pyg()` return ordinary editable PyG `Data`.
Their `from_pyg(data, field_map=None)` classmethods reconstruct the supported
chemical information for querying and review. General conversion performs no
chemistry perception, calculation or deployed-model inference. CBond and MCA
retain their own customized model input interfaces.

## An ordinary mixed training batch

```python
from hotpot import Crystal, Molecule, SpaceGroup
from torch_geometric.loader import DataLoader

mol = Molecule()
mol.create_atom(atomic_number=6, formal_charge=0,
                implicit_hydrogens=4, coordinates=(1., 2., 3.))
crystal = Crystal(5, 6, 7, 90, 90, 90, molecule=mol,
                  space_group=SpaceGroup(1))
plain = Molecule()
plain.create_atom(atomic_number=8, formal_charge=0,
                  implicit_hydrogens=2, coordinates=(0., 0., 0.))
batch = next(iter(DataLoader([plain.to_pyg(), crystal.to_pyg()], batch_size=2)))
graphs = batch.to('cpu').to_data_list()
restored = Crystal.from_pyg(graphs[1])
assert restored.molecule is not mol
assert restored.space_group == SpaceGroup(1)
```

A tensor may be moved to CUDA with ordinary `Data`/`Batch.to(device)`. Split a
multi-graph Batch with `to_data_list()` before reconstruction. `torch.save` and
trusted-local `torch.load(..., weights_only=False)` preserve these ordinary
primitive/tensor fields. A Data object contains no retained source Molecule,
locks or hidden pristine restoration payload.

## Current fields and edits

The single current schema is `hotpot.molecule/2`. Node fields carry atomic
number, formal charge, total-H target, raw aromatic representation, source id,
Cartesian angstrom `pos`, presence/source, representation availability and
constraints. Edge columns carry one complete chemical Bond each, with order,
kind, direction, aromatic representation, id, constraints and mapping key.
Wrapped structured fields retain exact stereo/H/isotope occurrences, source
namespaces, portable annotations and scientific observations. Discrete facts
stay integer/bool; missing numeric storage never becomes a chemical fact.

Crystal fields add an oriented angstrom row basis `cell_matrix` (1x3x3), actual
cell presence, directed integer `bond_image` (Mx3) and its presence, occupancy
and its presence, and useful numeric group R/t views. Wrapped `hotpot_crystal`
holds the one exact group/setting, site relations, disorder and historical
source binding/maps. Numeric group tensors must agree with the exact group.
Unknown group has zero operations; it is never fabricated as P1. Standalone
cell-only Crystal and an empty contained Molecule remain distinct.

Imported tensor edits are authoritative. Reorder all node fields/atom_key
and remap edge/constraint/pair row indices together. Reorder edge fields and
bond_key together; reverse a directed image when reversing its edge. Wrapped
current references join by atom_key/bond_key, while historical source indices
stay historical. Deleted/new members need complete updated context mappings;
stale or contradictory maps reject locally instead of restoring old facts.
No rule runs to fill absent H, aromaticity, charges, Bonds or symmetry.

Observations become attributable historical annotations on generic import,
including partial charges, MCA and selected/generated frame observations.
They are never silently installed as current calculated attachments. Main
coordinates follow edited pos rather than a historical frame. Source frames
retain their separate existing semantics. Historical annotations describe
their original subject namespace and initial import order; they do not promise
a live join after later edits, components or merges.

## Explicit foreign mappings

Foreign Data requires a field map; columns are never guessed from embeddings.
For example, select atomic numbers and positions directly:

```python
import torch
from torch_geometric.data import Data

foreign = Data(num_nodes=1, element=torch.tensor([8]),
               xyz=torch.tensor([[0., 0., 0.]]))
water_site = Molecule.from_pyg(foreign, field_map={
    'fields': {'z': 'element', 'pos': 'xyz'},
    'edge_mode': 'single', 'coordinate_unit': 'angstrom'})
assert len(water_site.atoms) == 1
assert len(water_site.bonds) == 0
```

A selector is an attribute name or `(attribute, column)`. Length units are
angstrom, nanometer or bohr; explicit cell_matrix uses that same selection.
Select actual periodic images/occupancies and their presence fields as needed.
An optional `fields.crystal_context` selects the shared full portable Crystal
record; selected numeric projections must agree. Missing image is unavailable,
not explicit zero. `edge_mode='paired'` requires exactly one opposite arc per
chemical edge, matching nondirectional facts and opposite direction/images.
The current Molecule graph cannot hold self/parallel quotient edges; those
full projections reject without collapsing them. Unmapped foreign material
is reported by exact attribute path. Unsupported actual metadata rejects
locally rather than using an arbitrary-object pickle fallback.
