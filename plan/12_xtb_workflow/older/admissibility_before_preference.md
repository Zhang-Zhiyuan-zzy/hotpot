# Admissibility before preference: charge, spin, and oxidation-state policy

> **OLDER / NON-NORMATIVE.** This external-review-derived policy is retained as
> an audit source. The active owner-approved contract is
> [`scientific_state_implementation.md`](../scientific_state_implementation.md)
> and the normative specifications indexed there.

Status: implementation guidance for the next Stage 12 development cycle. This
document records policy and acceptance boundaries; it does not claim that the
described recovery work is implemented.

## 1. Decision

Hotpot must resolve charge, spin, and oxidation state in two ordered steps:

1. establish the states that are admissible under the available identity,
   conservation, electron-count, backend, and explicit-source constraints;
2. select a practical default only from that admissible set.

This is the **admissibility-before-preference** rule. A preferred state may be
uncertain or not globally stable, but it must never violate a hard constraint.
Convergence is additional execution evidence; it neither legalizes an
inadmissible state nor proves that a legal state is the physical ground state.

The rule supports practical automation. Hotpot may choose one default run when
several states remain legal, provided that it records the candidate set,
selection policy, backend context, assumptions, and source evidence. It does
not have to perform a state scan for every ordinary calculation.

## 2. Why the three inference problems share one control flow

Net charge, spin, and oxidation state are scientifically independent results,
but their orchestration has the same shape:

```text
bind exact chemical and backend identity
  -> collect attributable source facts
  -> apply hard admissibility constraints
  -> retain every surviving candidate
  -> apply an explicit selection policy
  -> validate the selected backend representation
  -> execute and preserve the outcome
```

They must not be collapsed into one inference algorithm:

- a molecular net charge does not uniquely assign atom formal charges;
- a net charge does not uniquely assign metal oxidation states;
- an oxidation state does not uniquely determine molecular spin;
- a successful SCF or xTB calculation does not prove a ground state; and
- one backend's multiplicity representation is not automatically another
  backend's `UHF` representation.

The shared framework belongs in orchestration and result contracts. Charge,
oxidation-state, and spin modules remain independently testable and must not
call sibling inference implementations.

## 3. Result axes that must remain separate

Every inference result must expose at least these independent axes:

| Axis | Meaning |
|---|---|
| identity | Exact graph, protonation, atom map, component, and backend context to which the result applies |
| candidates | All states surviving the current evidence and hard constraints |
| admissibility | `valid`, `invalid`, or `indeterminate`, with reason codes |
| selection | Selected candidate, or `None` when selection is not justified |
| selection source | Explicit user/source fact, system policy, lowest-legal fallback, scan result, or unresolved |
| assumptions | Every convention or scientific prior used |
| execution | Not run, rejected before launch, failed, converged, or nonconverged |
| verification | Whether execution and external evidence confirm the requested state |

The implementation may use domain-specific classes, but it must preserve
these semantics. Recommended domain results are `NetChargeResult`,
`OxidationStateInferenceResult`, and `SpinStateInferenceResult`, composed by a
state resolver. Operative results should use a closed status vocabulary such
as `PRESERVED`, `VALIDATED`, `INFERRED`, `AMBIGUOUS`, `CONFLICT`, and
`UNSUPPORTED` rather than overloading `None` or zero.

## 4. Authority and selection order

The common authority order is:

```text
verified explicit user setting
  > verified executed/source setting
  > dataset- or system-specific default
  > general legal fallback
  > unresolved
```

An earlier source is not used blindly. It must remain admissible for the exact
chemical identity and target backend. When an authoritative source conflicts
with a hard target-backend constraint, Hotpot reports `CONFLICT`; it does not
silently rewrite the state.

The general fallback differs by scientific quantity:

| Quantity | Permitted practical fallback |
|---|---|
| net charge | No numeric "lowest charge" rule. Select only when graph/source constraints make one charge unique or a declared system profile supplies it; otherwise remain unresolved. |
| spin | Choose the lowest legal `UHF` only after target-backend electron parity is known, unless a valid system-specific reference/default has higher authority. |
| oxidation state | No numeric "lowest oxidation state" rule. Select only a unique constrained assignment or a declared, attributable system policy; otherwise return ranked candidates or unresolved. |

## 5. Hard admissibility by domain

### 5.1 Net charge

The total molecular charge `Q` must be an integer bound to an exact molecular
identity. At minimum:

- authoritative imported or user-supplied `Q` is preserved and validated;
- atom and fragment formal-charge sums, when claimed, must equal `Q`;
- component charges must sum to `Q`;
- the target backend must have a nonnegative effective electron count;
- protonation, added/removed atoms, and redox changes invalidate a state packet
  until it is rebound; and
- a graph that does not identify `Q` yields candidates or `UNRESOLVED`, not an
  implicit neutral state.

Lewis/formal-charge allocation and net-charge inference remain separate
results. Coordination cleavage, CBC bookkeeping, and oxidation-state
assignment must never silently change the intact molecule's `Q`.

### 5.2 Spin

For a backend electron count `N_e` and an xTB-style unpaired-electron value
`u`:

```text
N_alpha = (N_e + u) / 2
N_beta  = (N_e - u) / 2
u >= 0
u <= N_e
u mod 2 == N_e mod 2
multiplicity = u + 1
```

Violating parity is not a poor scientific choice; it is an illegal electronic
state. For example, an odd-electron doublet has `u=1`. Replacing it with
`u=0` is invalid even before relative stability is considered.

After the legal set is established:

1. preserve a verified source multiplicity or `UHF` when it can be translated
   to the target backend;
2. otherwise apply a declared system-specific default, such as a
   metal/oxidation-state execution prior;
3. otherwise choose the lowest legal `u` for one automatic run; and
4. retain optional state scanning as an explicit higher-cost policy.

The selected default is an execution state, not a universal ground-state
claim. Eu(III) `UHF=6`, for example, may be a useful xTB recovery prior, but
the retained recovery experiments do not justify making it an unconditional
rule for every Eu(III) calculation.

### 5.3 Oxidation state

Oxidation-state candidates must obey the selected counting convention and all
available conservation constraints:

- the sum of assigned fragment/metal ionic charges must be compatible with
  molecular `Q` under that convention;
- element candidate domains must be explicit and versioned;
- ligand protonation, radical, non-innocent, and redox alternatives remain
  visible;
- multimetal assignments are solved globally rather than independently;
- an explicit ion/source assignment is preserved when it is chemically and
  identity compatible; and
- ambiguity is returned when different legal assignments survive.

Geometry, bond valence, population analysis, or a converged electronic
calculation may rank candidates only within its declared domain. None is a
universal oxidation-state oracle.

## 6. Backend-specific electronic-state representations

The chemical system state and the backend execution representation must be
stored separately. A state packet must include or bind:

```text
chemical_identity_digest
atom_map_digest
net_charge
candidate/selected scientific state
backend
method
basis_or_parameter_set
ECP and removed-core-electron definition, when applicable
effective_electron_count
backend multiplicity or UHF
source and assumptions
```

The current xTB spin implementation computes parity from the full explicit
nuclear-charge sum and `Q`. That is a valid lowest-legal fallback for its
declared xTB execution context. It is not a generic Gaussian/ECP rule.

Gaussian calculations in the incoming archive use ECPs such as `MWB49` and
`MWB53`. Removing an odd number of core electrons changes effective-electron
parity. A preliminary all-nuclear-charge comparison flags 60 GJF settings that
would appear parity-incompatible without their ECP context. These are not
automatically erroneous; they demonstrate that ECP identity is a mandatory
part of state translation.

At a pipeline boundary:

- GFN-FF consumes total charge and topology evidence but does not apply spin;
- GFN1/2-xTB consumes total charge and target-backend `UHF`;
- Gaussian consumes total charge, multiplicity, method, basis, and ECP; and
- no node may reuse a predecessor's integer spin field without translating and
  revalidating it for the target backend.

## 7. GJF/LOG reconciliation policy

Raw GJF and LOG files are immutable evidence. A normalized calculation record
must distinguish the requested input state from the state actually executed.

When GJF charge or multiplicity conflicts with LOG:

1. verify file digests and source aliases;
2. compare atom count and element multiset;
3. establish an atom mapping and compare ordered elements where available;
4. compare an explicit molecular graph when either format supplies one;
5. otherwise compare a clearly labelled inferred graph and geometry/distance
   evidence without pretending that coordinate inference is an explicit bond
   record; and
6. decide `same`, `different`, or `indeterminate` chemical structure without
   using charge or multiplicity as an identity shortcut.

If structure identity is `same`:

- LOG is authoritative for the state actually executed;
- extract LOG charge, multiplicity, `N_alpha`, `N_beta`, wavefunction type,
  `N_alpha-N_beta`, and available `<S^2>` diagnostics;
- retain the original GJF requested values as provenance;
- create a corrected derivative input or normalized state projection from the
  LOG values rather than editing the raw source bytes; and
- record an audited `input_state_overridden_by_executed_log` relationship.

If structure identity is `different` or `indeterminate`:

- detach the GJF and LOG;
- preserve them as separate source artifacts;
- mark GJF charge and multiplicity `at_risk` because the claimed successful
  calculation no longer verifies them;
- do not transfer LOG energy, geometry, convergence, or spin diagnostics to
  the GJF Case; and
- require rematching or owner review before benchmark admission.

This rule treats LOG as execution truth only after structure identity has been
established. A filename or shared numeric prefix is insufficient.

## 8. Audit of the incoming Gaussian archive

The archive contains 483 nominal GJF/LOG pairs. The previous archive audit
checked path matching, termination, optimization, frequency completion,
significant imaginary frequencies, and file hashes, but did not compare GJF
and LOG charge/multiplicity.

The current audit found:

- 472 pairs have matching requested and executed charge/multiplicity;
- 11 pairs conflict;
- every LOG uses one internally stable charge/multiplicity across its recorded
  optimization/frequency jobs;
- all 11 conflicts have the same atom count, element multiset, ordered element
  sequence, and distance-inferred graph between GJF and the first LOG
  orientation;
- 10 have effectively identical starting distance matrices; and
- one Eu/SCRF pair has the same inferred graph but a different conformer
  (`0.625 A` distance-matrix RMS difference, approximately), so it remains a
  same-graph relationship with lower geometry-identity confidence.

The state transitions are:

| GJF `(Q, M)` | LOG `(Q, M)` | Count |
|---|---|---:|
| `(3, 2)` | `(3, 1)` | 4 |
| `(1, 2)` | `(3, 1)` | 2 |
| `(2, 1)` | `(2, 2)` | 2 |
| `(0, 2)` | `(0, 1)` | 1 |
| `(3, 1)` | `(0, 1)` | 1 |
| `(4, 1)` | `(3, 2)` | 1 |

The affected archive records are listed explicitly so that intake tests can
bind the finding to source content rather than only reproducing aggregate
counts:

| Archive record | GJF `(Q, M)` | LOG `(Q, M)` | Structural decision |
|---|---:|---:|---|
| `61_60_C22H26EuN4O2__3a0ba549f1__a262ca5846` | `(1, 2)` | `(3, 1)` | same graph and starting geometry |
| `25_17_PDALC_Dy__ea6a34b71a__f0beb0749d` | `(3, 2)` | `(3, 1)` | same graph and starting geometry |
| `27_17_PDALC_Er__e4620942bf__e3028739ab` | `(3, 2)` | `(3, 1)` | same graph and starting geometry |
| `47_C4-PTA__41f45f5b19__0bcaa7cf5a` | `(0, 2)` | `(0, 1)` | same graph and starting geometry |
| `Co__e25e811909__38ecddf217` | `(2, 1)` | `(2, 2)` | same graph and starting geometry |
| `61_60_C22H26EuN4O2__c20e95cc32__df76a98604` | `(1, 2)` | `(3, 1)` | same graph, different conformer |
| `48_C6-PTA__34cfb75f9f__101f8a379c` | `(3, 1)` | `(0, 1)` | same graph and starting geometry |
| `21_17_PDALC_Sm__5bb7d5117e__866ff96ec0` | `(3, 2)` | `(3, 1)` | same graph and starting geometry |
| `31_17_PDALC_Cu__0b8295f229__78b3e2ef16` | `(2, 1)` | `(2, 2)` | same graph and starting geometry |
| `54_53_C32H30N4O2Ce+4__b8ae97adb6__e9b746499b` | `(4, 1)` | `(3, 2)` | same graph and starting geometry |
| `Er__6d5ea0df5a__7494773a3a` | `(3, 2)` | `(3, 1)` | same graph and starting geometry |

Under the owner-approved rule above, all 11 remain structurally associated and
their normalized execution state is corrected from LOG. The raw GJF remains
unchanged. The archive therefore contains 41 GJF files requesting a doublet,
but 37 LOG files actually executing a doublet: 34 agree with a doublet GJF and
three correct a singlet GJF. The executed doublets use unrestricted PBE0 with
`N_alpha-N_beta=1`; their available spin diagnostics provide stronger evidence
than the GJF multiplicity line alone.

This audit must be reproduced by the intake implementation rather than copied
as an unverified constant.

## 9. How the new reference data may be used

The Gaussian archive adds valuable source facts:

- verified net-charge configurations and backend-specific initial spin
  references;
- actual executed alpha/beta electron counts and spin diagnostics from LOG;
- successful optimized structures, forces, frequencies, and energies;
- manually checked DOI-linked experimental log-beta values; and
- direct overlap with the Stage 12 extractant corpus.

Current overlap with the 187-ligand corpus is:

| Reference relation | Count |
|---|---:|
| identical canonical ligand SMILES | 94 |
| archived free-ligand Gaussian calculation | 57 |
| archived Eu-complex Gaussian calculation | 43 |
| both free ligand and Eu complex | 42 |
| ligand with a Eu log-beta observation | 57 |

These data can validate state transport, charge/multiplicity legality,
atom/graph conservation, geometry gates, and FF/xTB-to-DFT structural
consistency. They cannot establish equality between GFN and PBE0 energies,
turn a converged state into a ground-state proof, or make log-beta values
comparable across different solvent/media/concentration conditions.

## 10. Current-code boundary

At the time of this document:

- `infer_lowest_spin()` implements the lowest legal parity fallback;
- `resolve_electronic_state()` accepts explicit charge and unpaired-electron
  overrides but still exposes the earlier combined state contract;
- `prepare_xtb_input()` writes XYZ for every xTB method;
- the proposed independent `NetChargeResult`,
  `OxidationStateInferenceResult`, and
  `CoordinationElectronCountingResult` do not exist in production code;
- oxidation-state-aware spin selection is not implemented; and
- the recovery phases in
  `xtb_failure_root_cause_and_recovery_plan.md` remain proposals.

The original Stage 12 execution-wrapper acceptance remains valid within its
declared scope. It must not be reported as completion of this broader inference
and cross-backend policy.

## 11. Required implementation boundaries

The next implementation must:

1. add independent immutable result contracts before changing inference;
2. bind every result to exact molecular and backend identity;
3. preserve source candidates even when a default is selected;
4. reject illegal charge/spin combinations before backend launch;
5. make the automatic default policy explicit and attributable;
6. revalidate state after every atom, protonation, graph, component, or backend
   transition;
7. treat GJF requested state and LOG executed state as separate facts;
8. prohibit filename-only GJF/LOG association;
9. retain failures and unresolved states without fabricating a fallback;
10. keep execution success, geometry acceptance, and physical-state confidence
    independent; and
11. rerun the complete Stage 12 corpus after each contract or state-policy
    change.

It must not:

- use charge zero as evidence of an explicitly neutral user choice;
- select the numerically smallest charge or oxidation state as a fallback;
- use ordinary GFN2 energy alone to declare a spin ground state;
- copy Gaussian multiplicity directly into xTB `UHF` without target-backend
  validation;
- use convergence to overwrite contradictory source evidence;
- mutate raw GJF/LOG evidence to hide reconciliation; or
- merge coordinate-identical assets into one scientific Case without an
  identity decision.

## 12. Minimum acceptance tests

### Contracts and orchestration

- each inference module runs using constructed inputs without calling sibling
  estimators;
- candidate, admissibility, selection, source, assumptions, and backend
  identity survive serialization;
- explicit source settings outrank defaults only when target-admissible;
- an automatic selection never removes unselected legal candidates; and
- unresolved states stop only the backend operation that requires a unique
  state.

### Charge

- authoritative positive, negative, and neutral total charges;
- fragment and atom-charge sum invariants;
- protonation/redox identity invalidation;
- graph-indeterminate charge returns candidates or unresolved; and
- representation-equivalent SMILES/SDF/MOL/MOL2 inputs preserve the net-charge
  decision or report explicit information loss.

### Spin

- even-electron `u=1` and odd-electron `u=0` are rejected;
- multiplicity and `UHF` explicit inputs are exactly equivalent after backend
  translation;
- lowest-legal fallback produces `u=0` for even and `u=1` for odd target
  electron counts;
- ECP effective-electron parity is tested independently from xTB parity;
- system-specific defaults override the general fallback without becoming
  ground-state claims; and
- state scans retain one independently attributable result per candidate.

### Oxidation state

- assignment sums obey molecular charge under the declared convention;
- Eu(II)/Eu(III), Fe(II)/Fe(III), non-innocent, and multimetal cases retain
  ambiguity where evidence does not decide;
- source ion labels remain attributable evidence; and
- oxidation candidates never silently change net charge or spin.

### Gaussian intake

- all 483 GJF/LOG pairs are independently reparsed;
- the 472 matching pairs remain matching;
- the 11 known state conflicts are detected from content;
- same-structure conflicts project LOG execution state while retaining raw GJF
  provenance;
- deliberately mismatched structures are detached and mark the GJF state at
  risk; and
- `N_alpha`, `N_beta`, wavefunction type, `<S^2>`, method, basis, and ECP are
  stored when present.

## 13. References

Project specifications and evidence:

1. `charge_inference.md`, especially the preserve/validate/infer authority
   order and representation-invariance requirements.
2. `oxidation_state_inference.md`, especially independent contracts, global
   constraint solving, and known limits.
3. `electronic_state_inference.md`, especially electron-parity constraints,
   candidate states, and the Eu diagnostic boundary.
4. `xtb_failure_root_cause_and_recovery_plan.md`, especially the controlled
   Eu `UHF=6`, GFN-FF SDF, and MOL2 charge-drift diagnostics.
5. The incoming GJF/LOG files and their SHA-256 inventory, which are primary
   computational evidence for requested and executed Gaussian states.

External definitions and method boundaries:

1. xTB command-line documentation, definitions of `--uhf` and `$spin`:
   <https://xtb-docs.readthedocs.io/en/latest/commandline.html>.
2. C. Bannwarth et al., GFN2-xTB, *J. Chem. Theory Comput.* 2019,
   DOI `10.1021/acs.jctc.8b01176`.
3. S. Grimme et al., GFN-xTB, *J. Chem. Theory Comput.* 2017,
   DOI `10.1021/acs.jctc.7b00118`.
4. P. Karen et al., IUPAC oxidation-state recommendations,
   DOI `10.1515/pac-2015-1204`.
5. NIST Atomic Spectra Database and Handbook of Basic Atomic Spectroscopic
   Data for attributable atomic configuration and ground-term priors.
6. Gaussian 16 output records: charge/multiplicity echo, alpha/beta electron
   counts, restricted/unrestricted SCF identity, and `<S^2>` diagnostics are
   the execution evidence used by the reconciliation policy.
