# Plan 19 baseline dependency inventory

Revision: `2704f3c9e0ca449f0d9d1445d6320714d5dbbc0c`

This inventory is the Stage 0 deletion gate.  Historical files under `plan/`
and NetworkX imports used solely by independent tests are evidence, not
production dependencies.

## Production NetworkX imports

| File | Baseline role | Disposition |
| --- | --- | --- |
| `hotpot/cheminfo/core.py` | persistent graph, components, paths, DFS/Z-matrix, Paton basis | migrate to topology index and graph algorithms; remove import |
| `hotpot/cheminfo/search/search.py` | query graph cache and public constructor | migrate tests to query atoms/bonds; delete surface and import |
| `hotpot/cheminfo/AImodels/mca/site_detection.py` | 1-WL adjacency and rooted exact isomorphism | native exact anchored matching; remove import |
| `hotpot/cheminfo/forcefields/coordination.py` | ligand components | direct component labels; remove import |
| `hotpot/cheminfo/forcefields/working_copy.py` | graph-typed rollback snapshot | snapshot authority only and rebuild derived topology |
| `hotpot/cheminfo/graph/traversal.py` | NetworkX-typed DFS helpers | explicit node/edge traversal or deletion |
| `hotpot/cheminfo/ob2chem.py` | unused import | delete import |

Runtime declarations to delete after production imports reach zero:

- `pyproject.toml` project dependency;
- `requirements.txt` base dependency; and
- `hotpot/ds/public/_implementation.py` runtime-distribution identity.

`tests/requirements-inference.txt` retains NetworkX as a test dependency.

## Duplicated topology and updater consumers

The Atom-specific `_neighbours`/`_bonds`, `Molecule._graph`,
`_update_graph()`, and `_update_atom_neighbours_bonds()` are consumed by:

- Core construction, mutation, copy, component extraction, paths, rings, and
  internal-coordinate traversal;
- RDKit, Open Babel, CCDC, and MCA DTO converters;
- force-field working-copy snapshot, rollback, and commit;
- hidden-bond, complex-hydrogen, topology-determinism, geometry, ring,
  SMARTS, search-compiler, and kekulize tests.

Every mutation path must converge on `_update_topology()` before those fields
and methods are deleted.  `AtomSeq._bonds` is a different constituent-domain
field and must remain.

## Legacy cycle-basis consumers

Production/model/example consumers requiring migration:

- `hotpot/cheminfo/AImodels/data_extract.py`;
- `hotpot/plugins/PyG/data/utils.py`;
- `hotpot/cheminfo/forcefields/working_copy.py`;
- `examples/BayesianDesign/PhenMol/pair_link.py`; and
- Core Molecule/Atom properties and caches.

Active tests requiring semantic rewrites rather than aliases:

- `tests/test_cheminfo/test_complex_hydrogens.py`;
- `tests/test_cheminfo/test_geometry.py`;
- `tests/test_cheminfo/test_geometry_quality.py`;
- `tests/test_cheminfo/test_ligand_rings.py`; and
- `tests/test_cheminfo/test_relevant_ring_integration.py`.

Active documentation/example references requiring migration:

- `README.2026.md`;
- `doc/cheminfo.md`;
- `skills/usage.claude.md`;
- `hotpot/cheminfo/kekulize/Kekulize.md`;
- `hotpot/cheminfo/graph/README.md`; and
- `examples/BayesianDesign/PhenMol/pair_link.py`.

Final deletion set, with no aliases:

- `Molecule.cycle_basis_rings`;
- `Molecule.ligand_cycle_basis_rings`;
- `Atom.cycle_basis_rings`;
- `_cycle_basis_rings`;
- `_ligand_cycle_basis_rings`; and
- `_uncached_cycle_basis_rings`.

## Public Graph-like surfaces

| Surface | Active baseline users | Disposition |
| --- | --- | --- |
| `Molecule.graph`, `simple_graph`, `atom_bond_graph` | Core algorithms, force-field coordination, SMARTS adapters, geometry/ring/kekulize tests | migrate scientific behavior, then delete |
| `Molecule.adjacency_matrix` | none found | delete; preserve `graph_spectral()` numerical path |
| `Molecule.super_acyclic_graph()` | none; always raises | delete |
| `Substructure.construct_graph()` and `query_graph` | SMARTS adapters/tests | build independent test graph from query atoms/bonds; delete production cache/API |
| MCA `MoleculeGraph` | MCA public/import/serialization tests | migrate repository callers to supported molecule inputs; delete export/wrapper |
| NetworkX-typed traversal helpers | Core Z-matrix and legacy graph tests | explicit node/edge input or private implementation |

## Model parity gate

Both ring feature extractors currently remove metal-ligand Bonds and feed the
legacy NetworkX Paton basis, including its cycle and outer-basis order, into
trained models.  They must call the same new deterministic native algorithm
on coordination-filtered edges.  Stage 0 freezes:

- basis cycle sets and sequences;
- `mol_rings_nums`, `rings_node_index`, `rings_node_nums`, and
  `mol_rings_node_nums`;
- ring attributes; and
- accepted prediction digests.

Cubane's baseline model representation is five cycles with lengths
`[4, 4, 6, 4, 4]`; the six four-member Relevant Cycles are scientifically and
operationally distinct.
