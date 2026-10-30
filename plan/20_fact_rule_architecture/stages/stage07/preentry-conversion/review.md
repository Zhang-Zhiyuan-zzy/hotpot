ANALYSIS_RESULT
task_id=p20-s7a-preentry-consumer-inventory-704283-35
revision=1
verification_code=704283
task_hash=3ee1cec59629068e8857ea2f16bc529864d07bd917135eb35068d27be38d109e
source_commit=25735ab07ca9a2b30f5a395abe4fd10133852913
diff_or_tree_hash=a6891525f4f834287666f8bb031a265dca6dd5728ecf4cd1581bd8778d19e75f
capture_identity_kind=SHA256 canonical capture-manifest.json raw bytes; NOT Git tree
status=PREENTRY_INVENTORY_ONLY

## `_obmol` / `_row2idx` ownership and lifecycle

| Role | Current factual behavior |
|---|---|
| Owner/storage | `Molecule.__init__` creates persistent carrier fields `_obmol = None` and `_row2idx = None` ([core.py:145-173], excerpt SHA `17f0c5db…`, source SHA `bc44ba23…`). |
| Materialization | `Molecule.to_obmol()` calls `obconvert.mol2obmol(self)` only when `_obmol` is falsey, stores both returned values, and returns the stored mutable `OBMol` ([core.py:2066-2079], SHA `c6a730bd…`). For a normal nonempty molecule, repeated calls expose the same external mutable object. |
| Mapping use | `_row2idx` is populated beside `_obmol`, journaled, invalidated and excluded from pickle state. The exact current search found no production read after storage. Direct callers needing a mapping instead consume the mapping returned by `obconvert.mol2obmol()`. |
| Topology invalidation | `_update_topology()` sets both fields to `None` after rebuilding or reusing `_topology_index` ([core.py:474-504], SHA `17d148b9…`). Factual installation calls `_update_topology(..., _prepared_index=...)`, so complete imports also clear them. |
| Coordinate invalidation | `commit_xtb_coordinates()` explicitly clears both after committing coordinates ([plugins/xtb/adapter.py:396-413], SHA `10e11ad…`). The supplied exact search exposes no other explicit non-topology invalidator. General coordinate/fact commit coverage therefore remains unproven from this packet. |
| Partial-charge synchronization | `assign_partial_charge()` does not invalidate the cache; when it exists, it mutates its Open Babel atoms to mirror newly assigned partial charges ([core.py:1415-1440], SHA `af1c6bc…`). |
| Transaction rollback | `_CACHE_FIELDS` is exactly `("_obmol", "_row2idx")`. `touch_caches()` retains original objects and shallow contents only for list/dict values, then touches frames and constraints ([structure/_commit.py:227-237], SHA `690d0b98…`). Rollback restores the original references and `_row2idx` dictionary contents ([286-353], SHA `7998b226…`). It does not clone or restore the interior of a mutable `OBMol` changed in place. |
| Topology journal | `_TOPOLOGY_FIELDS` also includes `_obmol`, `_row2idx` and `_crystal`; full topology rollback restores those carrier fields. `create_atom()` explicitly journals caches before `_update_topology()` invalidates them. |
| Pickle/copy | `_capture_molecule_pickle()` lists `_obmol` and `_row2idx` among owned fields excluded from arbitrary state ([core.py:4764-4788], SHA `fa427f25…`). Factual capture serializes canonical molecule facts, not these external values. Reconstructed carriers therefore rematerialize them rather than pickle the cache. |

Current readers and mutation exposure:

- `get_partial_charge()` first copies `ob.OBMol(self.to_obmol())`, so charge calculation mutates a copy.
- `InChi`, `smiles`, and `kekulize_smiles` read through the cached value ([core.py:1899-1946]).
- `_io.MolWriter._ob_write()` gets `mol.to_obmol()` once, then calls `set_obmol_coordinates()` on it for every conformer. Under current ownership this mutates the carrier-held cache and leaves it at the final written frame ([`_io/_io.py:392-404`], source SHA `eeccb15c…`).
- `Molecule.to_pybel_mol()` bypasses the cache and wraps a fresh `obconvert.mol2obmol(self)[0]`.
- `obconvert.write_by_pybel()`, `_perceive_bond_orders()` and direct `mol2obmol()` users operate on freshly created values.
- Python benchmark/readme consumers call `to_obmol()` and then run `OBBuilder` or force fields on that object. Although held in a local variable, it is currently the carrier-owned cached object.
- `test_ob2chem_compat.py:37-40` proves the method delegates to `obconvert.mol2obmol`, but only checks one call; it does not establish fresh identity.
- xTB tests assert explicit invalidation. The working-copy rollback test asserts both fields are `None`, but does not test restoration of a previously populated cache.

The native `OBMol` occurrences are separate ownership domains:

- `obWrappers/_native/native_engine.cpp` creates operation-local `OBMol` values for build/check/optimization calls.
- Force-field `StructureSession` owns an `OBMol` for the lifetime of an opaque native session, with its own coordinate/topology revisions and runtime mutex. It is not stored on a Python `Molecule`.
- `test_session_factories_keep_one_opaque_native_structure` verifies no public `.obmol` attribute and rejects copying the session.
- Source Open Babel objects in conversion tests are test-local inputs rather than carrier cache state.

## Crystal ownership and consumers

Current carrier shape:

- `Crystal` stores six public parameters, a `mol` back-reference, and `_obcell = None` ([crystal.py:8-16], source SHA `b212225e…`).
- `obcell` constructs a fresh `OBUnitCell` on every access. `_obcell` is not read by this property and the exact current search found no production reader of `_obcell`.
- `Molecule.create_crystal()` constructs `Crystal(..., mol=self)`; factual installation does the same from a six-value `record.cell`.
- `_capture_factual_record()` records only `(a,b,c,alpha,beta,gamma)`.
- `_identity`’s `cell` domain hashes only those six values. The source-identity test confirms assigning an arbitrary `_obcell` does not affect cell identity, while changing `a` does.
- `obconvert._factual_obmol()` is the sole supplied Python consumer of `mol.crystal.obcell`, cloning the resulting toolkit cell into a fresh `OBMol`.
- Native Open Babel packing independently consumes a six-value unit-cell buffer and constructs an operation-local `OBUnitCell`.
- Pickle capture additionally saves `dict(mol._crystal.__dict__)`; restore updates the newly installed Crystal with that dictionary. Thus current pickle state can carry `mol`, `_obcell`, and other instance entries in addition to the canonical six-value record. No current runtime pickle behavior was executed here.

Current versus historical labels:

- Current `crystal.py` has no `LatticeGroup`, `operators`, `space_group`, or `lattice_group`. Those entries remain only in the accepted historical Stage1 inventory and are stale as current-source labels.
- `_obcell`, `mol`, and the module-level Open Babel import remain current.
- The characterized matrix conversion is already extracted to `rules/mathematics/cell.py`; `Crystal.from_matrix()` delegates through `_matrix_to_params`.
- `tests/test_cheminfo/structure/test_crystal_convention.py` currently exists at SHA `46ed382d…`, although the maintained Stage7A map still labels it `planned_not_created`. That status is stale.
- The current test fixes alpha as the angle between the first two rows, verifies orthogonal lengths/angles and subclass dispatch, and asserts no `matrix` property. This agrees with the accepted Stage1 decision and does not reopen it.
- Supplied current tests do not establish a complete Crystal Open Babel import/export round trip, pickle cleanup, absence of back-references, or installed-wheel adapter behavior.

## Conversion, I/O and topology utility boundaries

`convert.py` currently:

- imports Open Babel, Pybel, RDKit, `Molecule`, `read_mol`, and both toolkit converters at module load;
- returns a Hotpot `Molecule` unchanged;
- converts RDKit, `OBMol`, and Pybel values into new carriers;
- routes strings/paths through `read_mol`;
- accepts an object with callable `to_rdmol()`, verifies its result is `Chem.Mol`, then imports it;
- exports only `is_molecule_input` and `to_hotpot_mol`.

Direct production consumers supplied are CBond CLI/stage and MCA pipeline/serialization. `tests/test_molecule_conversion.py` covers all dispatch categories, unsupported objects, order, charge, aromaticity, hydrogens, conformers and root-export identity.

`core_utils.py` currently contains only:

- `read_mol()`, a one-result wrapper around `MolReader`;
- `atom_idx_pair_to_bond_idx()`, which constructs unordered endpoint pairs from `mol.bonds` and uses `list.index()` for each requested pair.

The exact boundary search found one production consumer of the topology helper: `Substructure.from_mol()` in `search/search.py:293-317`. `read_mol` has a much wider surface: the supplied exact index records 543 references across production, examples and tests. That grouped index is consumer evidence, not a reviewed set of 543 implementation bodies.

I/O coupling:

- `cheminfo.__init__` exports `MolReader`, `MolWriter`, core types, `read_mol`, and conversion dispatch.
- `_io/_io.py` eagerly imports Open Babel, `obconvert`, PubChem and factual-builder dependencies. It conditionally imports `core` through the package when the core module is absent.
- Open Babel and RDKit factual imports already build immutable records and install them through the Stage3 builder path; the factual-import tests verify one topology build, retained owner token/lock and selected preconfigured facts.
- `obconvert.mol2obmol()` creates a fresh projection and returns its row mapping. `_perceive_bond_orders()` creates a local projection, invokes Open Babel perception and returns factual proposals.
- `rdconvert.to_rdmol()` creates a fresh RDKit value per call. `from_rdmol()` and `obmol2mol()` require an empty complete target and install factual records.
- Existing RD stereo tests are connected round-trip coverage, but their scientific expectations were not re-reviewed in this inventory.

## Thermo and visualization boundaries

Thermo:

- `core.py` imports `Thermo` at module scope and `Molecule.get_thermo()` constructs it directly ([core.py:823-824], SHA `c870e9e…`).
- `call_thermo.py` eagerly imports pandas, `thermo.chemical.Chemical`, and RDKit.
- `Thermo` is caller-owned and stores the source molecule, temperature, pressure, and an external `_chem`; it is not retained on `Molecule`.
- It retains the accepted declared 181-property surface and unrestricted `__getattr__` forwarding to `_chem`.
- `get_chem()` creates one `Chemical` per supplied identifier, catches per-identifier/property failures and returns a pandas DataFrame.
- `mol_to_thermo()` tries, in order, kekulized RDKit SMILES, InChI, `mol.cas`, and PubChem CID. Each RDKit route calls the fresh `to_rdmol()` adapter.
- No Thermo-specific technical test appears in the maintained Stage7A selection or supplied connected test bodies. Preservation of all declared properties, dynamic forwarding, caller ownership, fallback order and optional-dependency behavior therefore lacks current source/wheel evidence.

Visualization:

- `draw.py` eagerly imports RDKit, CairoSVG, Matplotlib and the legacy core class; it also reads packaged fonts.
- `draw_single_mol()` and `draw_grid()` accept RDKit values directly and otherwise convert objects exposing `.smiles`; for a Hotpot molecule this invokes its current Open Babel-backed `smiles` path rather than `to_rdmol()`.
- Production use includes MCA CLI’s function-local import. Supplied direct consumers also include several examples and ChineseMedical/BayesianDesign scripts.
- The Stage7A maintained test list contains no focused visualization contract, packaged-font, output-format, or installed-wheel test.

## Import and cold-source constraints

The current import chain remains eager:

1. `import hotpot` explicitly preloads RDKit.
2. `hotpot` wildcard-imports `cheminfo`.
3. `cheminfo` imports `_io`, legacy core and conversion dispatch.
4. Those paths import Open Babel, RDKit, Thermo/pandas and other adapter dependencies at module load.

The two supplied fresh-process tests prove selected coexistence/import orders succeed on their recorded source, but do not prove lazy loading or operation when optional dependencies are absent:

- `test_import_safety.py` imports Hotpot before both toolkits.
- `rdWrappers/test_import_order.py` tests Hotpot↔RDKit order, not a cold minimal `structure` import or absent Open Babel/Thermo/Cairo/Matplotlib environment.

No installed-wheel cold import, module-origin, optional-dependency, font/resource, or external-object retention measurement was executed or supplied as completed evidence.

## Connected extraction checklist for the later owner stage

Derived from the existing Section 9 and Stage7A mapping only:

- Remove all carrier/journal/pickle/xTB ownership of `_obmol` and `_row2idx`; migrate callers that currently rely on repeated or mutable `to_obmol()` results.
- Preserve explicit local reuse where one coarse operation needs it, notably multi-conformer writing and benchmark/force-field flows, without confusing native session-owned `OBMol` with Python carrier storage.
- Rebind `convert.py`, `_io`, `obconvert`, `rdconvert`, CBond, MCA, search, benchmark and public facade imports to their actual post-Stage6 owners.
- Move `read_mol` and the sole identified topology-helper consumer according to the existing adapter/topology disposition; the broad `read_mol` path index must be rechecked after imports move.
- Preserve the accepted six-value Crystal and matrix convention while removing the remaining back-reference, `_obcell` field and toolkit import/property boundary; cover factual import/copy/pickle/source identity and toolkit/native round trips.
- Preserve the accepted caller-owned Thermo behavior and complete declared/dynamic surface while separating its external imports from carriers.
- Preserve drawing entry points and current direct consumers while validating optional imports, packaged resources and source/wheel output behavior.
- Update the maintained map: `test_crystal_convention.py` exists; `test_conversion_freshness.py` and `test_adapter_boundaries.py` remain uncreated in the supplied capture.
- Add the already existing but currently unmapped connected tests where appropriate: xTB invalidation, working-copy rollback, factual imports, source identity, bond metadata and `test_ob2chem_compat.py`.
- Freeze source and installed-wheel selections, cold-import module origins, conversion identity/mutation isolation, repeated-workflow memory/work evidence, and exact skip/optional-dependency accounting before any Stage7A gate claim.

## Evidence limits and mandatory rebind

This is static PREENTRY inventory only. No imports, tests, collection, builds, wheels, memory checks, runtime probes, benchmarks, Git inspection, chemistry execution or performance measurements were performed.

Primary exact-search bindings:

- Python identifiers: SHA `9f9fbc6b…`
- Non-Python identifiers: SHA `936869c2…`
- Direct boundary consumers: SHA `57328339…`
- Generic `read_mol` index: SHA `69f64fbe…`
- Crystal consumers: SHA `5435855a…`

The supplied source commit is root-provided; actual byte hashes govern this bounded capture.

Stage6 production and transitive imports may change these paths. This capture and every affected source/installed-wheel selection must be rebound after Stage6 changes and before Stage7A entry or implementation ownership release. This requirement is not a dependency for Stage6 acceptance and does not authorize Stage7A entry, design, implementation, scientific change, or gate closure.