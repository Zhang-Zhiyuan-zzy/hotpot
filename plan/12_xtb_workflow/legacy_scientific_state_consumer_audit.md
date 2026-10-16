# Legacy scientific-state consumer audit

Date: 2026-10-17

Scope: read-only inventory required by
[`scientific_state_implementation.md`](scientific_state_implementation.md)
before claiming project-wide scientific-state migration.

## 1. Conclusion

The xTB CLI, pipeline and benchmark now share the canonical resolver and
backend projector. Gaussian input generation does not. It remains a documented
second state path and prevents a project-wide migration claim.

No Gaussian behavior is changed by this audit. The legacy path must first gain
a named Gaussian backend context and an explicit adapter from
`ScientificStateBundle`/`BackendStateProjection`; deleting the old methods
before that migration would break an existing writer.

## 2. Active and inactive legacy surfaces

| Surface | Consumer | Current behavior | Assessment |
|---|---|---|---|
| `Molecule.calc_mol_default_charge()` | `cheminfo/_io/g16.py::to_gjf_preprocess()` | Recalculates atom valence, clones the molecule, adds H when absent, hides metal-ligand bonds, sums organic H differences and single-atom formal charges, and silently contributes zero for other inorganic components | Active hidden charge estimator; not attributable and not equivalent to the canonical resolver |
| `Molecule.default_spin_mult` | `cheminfo/_io/g16.py::to_gjf()` | Uses electron parity only, so every even-electron system becomes a singlet and every odd-electron system a doublet | Active execution fallback; physically legal in simple cases but not a scientific state decision |
| `Molecule.default_unpaired_electrons` | no repository consumer | Returns the same parity as a raw UHF count | Exposed but currently unused; do not route a new backend through it |
| `Molecule.charge` / conformer `spin_mult` | Gaussian writer and readers | Mutable scalar state without a facts identity or decision provenance | Transport storage, not a resolver contract |
| `plugins/qm/gaussian/_works.py::extract_log_info()` | Gaussian work helper | Reads `mol.spin_multiplicity`, while the Molecule/conformer surface is named `spin_mult` | Apparent stale attribute name; requires a focused Gaussian test before repair |

## 3. Current call flow

```text
Molecule.write(..., format="gjf")
  -> MolWriter GJF pre-plugin
     -> unless both skip flags are true
        -> Molecule.calc_mol_default_charge()
        -> mutate Molecule.charge
  -> Open Babel text generation
  -> MolWriter GJF post-plugin
     -> charge = Molecule.charge
     -> multiplicity = Molecule.spin_mult or Molecule.default_spin_mult
     -> emit Gaussian charge/multiplicity line
```

The pre-plugin condition uses `not call_flag OR not writer_flag`; therefore one
true skip flag does not suppress charge estimation. Both flags must be true.
This is existing behavior, not a compatibility requirement established by a
test.

Gaussian output flows in the opposite direction:

```text
cclib / G16 log parser
  -> charge and multiplicity scalars
  -> Molecule.charge and conformer spin_mult
```

Parsed execution facts may later become attributable backend history, but they
must not be silently promoted to a scientific ground-state claim.

## 4. Separation from the canonical path

| Concern | Canonical calculator/xTB path | Legacy Gaussian path |
|---|---|---|
| Chemical boundary | immutable `MolecularFactsIdentity` | mutable `Molecule` |
| Charge | attributable candidate and hard validation | H/valence heuristic or mutable scalar |
| Scientific spin | molecular $2S$ and multiplicity evidence | absent |
| Backend state | named projection convention and effective electrons | parity-derived multiplicity |
| Conflict/abstention | typed result | scalar fallback |
| Provenance | scientific and backend identities | none |

The legacy path must not be described as an implementation of the new default
rules. It predates them and can disagree on coordination compounds, radicals,
transition metals and f-block systems.

## 5. Required migration boundary

A later Gaussian-specific change should:

1. define a Gaussian `BackendContext` with all-electron/ECP effective-electron
   semantics and a named multiplicity projection convention;
2. accept one validated `BackendStateProjection` at the GJF writer boundary;
3. keep explicit user charge/multiplicity authoritative through the canonical
   request and conflict path;
4. remove writer-side mutation and parity fallback only after focused GJF
   tests lock the new contract;
5. repair or retire the unconsumed `spin_multiplicity` helper reference; and
6. then remove `calc_mol_default_charge`, `default_spin_mult` and
   `default_unpaired_electrons` only if repository-wide consumers are zero.

Until that work is approved and tested, project documentation must state
“canonical scientific state for xTB consumers,” not “all computational
backends.”
