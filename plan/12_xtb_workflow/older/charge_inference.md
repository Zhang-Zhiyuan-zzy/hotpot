# Charge inference: principles, contracts and reliability boundary

> **OLDER / NON-NORMATIVE.** This design record is retained for theory,
> limitations and historical alternatives. The active implementation contract
> is [`scientific_state_implementation.md`](../scientific_state_implementation.md)
> and the normative specifications indexed there.

Status: design specification only; the target implementation described here is
not yet production code.

Related specifications:

- [oxidation-state inference](oxidation_state_inference.md);
- [electronic-state inference](electronic_state_inference.md); and
- [xTB failure root-cause and recovery plan](xtb_failure_root_cause_and_recovery_plan.md).

## 1. Scope and decisive conclusion

This document defines two independent charge products for a fixed computational
system:

1. `NetChargeResult`: evidence and candidates for the net integer charge $Q$ of
   the declared computational boundary; and
2. `LewisFormalChargeResult`: full-graph Lewis electron allocations, atom formal
   charges and resonance-equivalence classes consistent with a selected or
   candidate $Q$.

The general problem is not:

```text
remove coordination bonds
-> infer one charge independently for each fragment
-> sum the selected fragment charges
```

That procedure is valid only when the cleavage boundary and ownership of every
boundary electron pair are already defined. The general procedure is:

```text
declare the computational boundary and charge evidence
-> freeze atoms, H inventory, protonation and semantic topology
-> generate joint Lewis electron-allocation candidates
-> apply local and global electron/charge constraints
-> quotient by resonance and symmetry equivalence
-> report a typed unique status, AMBIGUOUS or UNRESOLVED
```

Ionic cleavage, donor-return cleavage and Green CBC/L--X--Z ligand electron
counting are not ordinary charge inference. They belong to the
oxidation-state/coordination-electron-counting layer and must carry an explicit
convention.

Partial charges are transported through the pipeline but are calculated by
separate population-analysis backends.

## 2. Quantities and terminology

| Quantity | Definition | Ownership and invariant |
|---|---|---|
| Computational boundary, $\Omega$ | Exact atoms included in one calculation, including the declared treatment of counterions and solvent | Caller/pipeline fact; changing $\Omega$ changes the meaning of $Q$ |
| Net system charge, $Q$ | Missing electron count relative to the nuclei in $\Omega$ | `NetChargeResult`; integer required by xTB |
| Atom formal charge, $q_i$ | Lewis bookkeeping value $q_i=V_i-N_i-\tfrac12B_i$ for one complete electron allocation | `LewisFormalChargeResult`; representation plus provenance required |
| Connected-component charge, $Q_c$ | $\sum_{i\in c}q_i$ for a connected component $c$ of the original semantic graph | Charge module; meaningful only for the specified full-graph Lewis allocation |
| Lewis solving-block charge | Internal subtotal used to factor a constraint problem | Solver implementation detail, not an independently observed chemical fragment |
| Coordination electron-counting assignment, $E_L^{(\kappa)}$ | Convention-specific ligand/core bookkeeping under $\kappa$ | Oxidation-state/ligand-electron-counting layer, not either charge result; ionic counting may expose $Q_L^{(\kappa)}$, whereas CBC exposes L/X/Z class and donated-electron counts rather than a universal ligand charge |
| Oxidation state, $x_i$ | IUPAC ionic-approximation electron assignment | Oxidation-state module |
| Partial charge, $q_i^{\mathrm{partial}}$ | Method- and geometry-dependent density partition such as Mulliken, Hirshfeld or CM5 | Separate backend; real-valued and not interchangeable with formal charge |

For a complete Lewis allocation:

$$
Q=\sum_i q_i=\sum_cQ_c.
$$

Oxidation states also sum to $Q$, but that does not imply $q_i=x_i$.

### 2.1 Why a coordination-cleavage charge is different

For the adduct $\mathrm{NH_3\rightarrow BF_3}$:

- donor-return cleavage gives neutral $\mathrm{NH_3}$ and neutral
  $\mathrm{BF_3}$; while
- the localized full-adduct single-bond Lewis representation assigns
  $\mathrm{N}^{+}$ and $\mathrm{B}^{-}$.

Both descriptions conserve total charge. They answer different bookkeeping
questions. Therefore deleting a coordination edge and recomputing isolated
fragments cannot be presented as the atom formal charges of the intact graph.

## 3. Identifiability and normalized input contract

### 3.1 Net charge is not generally identifiable from a graph

Fixed atoms, hydrogens and connectivity do not uniquely determine $Q$:

| Shared structural facts | Possible states |
|---|---|
| $\mathrm{CH_3}$ atom inventory and C--H connectivity | $\mathrm{CH_3^+}$, $\mathrm{CH_3^\bullet}$ or $\mathrm{CH_3^-}$ |
| $\mathrm{[Fe(CN)_6]}$ composition and connectivity | $3-$, $4-$ and other electronically distinct states |
| One metal/dioxygen connectivity | neutral dioxygen, superoxo or peroxo assignments |
| One metal/non-innocent-ligand graph | different metal/ligand redox distributions |

Consequently, net charge is primarily an input fact or a candidate set. A
graph-only rule must not silently assume neutrality or a common metal oxidation
state.

### 3.2 Required normalized facts

| Input fact | Requirement | Missing or conflicting input |
|---|---|---|
| Computational boundary | Explicit membership of salt, counterions, solvent and adducts | `INFORMATION_LOSS` |
| Atom identity | Stable atom IDs, atomic numbers and isotope facts when relevant | Reject the request |
| Atom/H inventory | Explicit H or authoritative implicit H, without double counting | `UNRESOLVED` |
| Protonation and tautomer | Fixed before charge inference; the module never moves H | Solve each supplied state separately |
| Connectivity | Explicit adjacency; no silent distance-based bonding | `UNRESOLVED` |
| Bond semantics | Localized covalent, delocalized/aromatic, dative with direction, unknown, multicentre or haptic where applicable | Preserve alternatives or return out-of-domain |
| Net charge or electron count | Strongly recommended and bound to $\Omega$ | Only charge candidates; no default selection for coordination/open-shell systems |
| Local constraints | Optional explicit atom/component charge, bond order or radical facts with provenance | Use only as identified hard evidence |
| Delocalization hypotheses | One or more auditable aromatic/bond-order hypotheses, not an accidental Kekule form | Contradiction produces `UNRESOLVED` |
| Provenance and identity | Parser, format/version, graph/state hash and source authority | Cannot enter `PRESERVE` |
| Geometry | Optional evidence only | Never substitutes for topology or $Q$ |

### 3.3 Input profiles

| Profile | Minimum input | Permitted result |
|---|---|---|
| `PRESERVE_PROFILE` | Verified state, atom mapping and matching graph/state hash | Preserve and validate exactly |
| `MAIN_GROUP_INFERENCE_PROFILE` | Known $Q$, complete H/protonation, supported main-group elements and supported 2c--2e/delocalized semantics | Full Lewis candidate search; may be unique modulo resonance |
| `HYPOTHESIS_PROFILE` | $Q$ or important bond/electron semantics missing | Candidate set or `UNRESOLVED` only |
| `COORDINATION_PROFILE` | Explicit $Q$, full coordination/haptic semantics and complete H inventory | Charge module resolves $Q$/full-graph Lewis evidence; ionic/CBC cleavage is delegated |
| `OUT_OF_DOMAIN_PROFILE` | Unsupported multicentre, metal-covalent, radical or information-lost representation | Explicit abstention or separately validated specialist model |

A normalized input can make the computational problem deterministic, bounded
and auditable. It cannot make an intrinsically unidentifiable electronic state
chemically unique.

## 4. Ownership boundary

### 4.1 Owns

The charge package owns:

- authority resolution for net-charge evidence;
- net-charge candidates when no authoritative $Q$ exists;
- original full-graph connected-component identities;
- joint Lewis bond-order, nonbonding-electron, radical-electron and formal-charge
  candidates;
- resonance/symmetry equivalence and deterministic canonical localization;
- global electron/charge constraints;
- solver completeness, ambiguity, confidence and provenance;
- explicit application of a selected full-graph Lewis state.

### 4.2 Consumes

It consumes immutable `MolecularFacts`:

- computational-boundary identity;
- atoms, elements, isotopes and stable mapping;
- complete H/protonation facts;
- connectivity and semantic bond hypotheses;
- optional explicit $Q$, atom/component charges, radicals and bond orders;
- source, parser, version and graph/state identity.

### 4.3 Must not decide

The charge package must not:

- perform ionic/donor-return/CBC cleavage or L--X--Z classification;
- infer metal oxidation states;
- choose UHF, multiplicity or magnetic coupling;
- decide protonation, tautomerization, redox products or reaction products;
- silently perceive a new aromaticity model;
- use force-field/geometry build components as charge fragments;
- round partial charges into formal or net charges;
- replace authoritative charge evidence after a serialization boundary.

The orchestration layer composes the independently typed results. No inference
module may call a sibling implementation through a hidden fallback.

## 5. Graph views are distinct

| View | Definition | Owner/use |
|---|---|---|
| Original connected components | Connected components of the intact semantic molecular graph | Charge result; salts and truly disconnected entities |
| Lewis solving blocks | Mathematically factorable subproblems with an explicit electron boundary ledger | Internal charge-solver optimization |
| Delocalized domains | Aromatic, carboxylate, nitro, azide, allylic and other coupled electron domains | Joint Lewis/resonance enumeration |
| Coordination electron-counting fragments | Metal/core/ligand partitions under a named ionic, donor-return or CBC convention | Oxidation-state `ligand_electron_counting` |
| FF/geometry components | Temporary components made by hiding bonds for build or optimization | Force fields only; forbidden as charge evidence |

`ligand`, `counterion` and `solvent` are chemical roles. They may be consumed
when authoritative, but connected-component membership alone does not prove
one of those roles.

A solver may factor across a block only when every cut-edge electron
contribution is explicit and the remaining blocks are coupled by the global
constraints. It may not select one local optimum per block before performing
the global combination.

## 6. Preserve, validate and infer

### 6.1 Preserve

`PRESERVE` accepts only an explicit user state, a resolved in-memory state, or
a verified sidecar bound to the same graph identity. A stored zero does not by
itself prove authoritative neutrality.

### 6.2 Validate

`VALIDATE` checks:

- molecular/atom identity;
- $Q=\sum_iq_i=\sum_cQ_c$;
- electron and H inventory;
- compatibility of formal charges, bond orders, nonbonding electrons and
  radicals;
- representation/provenance consistency.

It does not change the supplied state. A validation failure is not permission
to infer a replacement silently.

### 6.3 Infer

`INFER` enumerates candidates for a fixed boundary, atom/H inventory,
protonation and topology hypothesis. A supplied $Q$ is a hard constraint.

When $Q$ is missing:

- a registered and validated closed-shell main-group policy may generate
  explicit net-charge candidates;
- metal-containing, open-shell, non-innocent, multicentre or otherwise
  out-of-domain systems return candidates or `UNRESOLVED`;
- no policy silently selects neutral, minimum $|Q|$, or a common oxidation
  state.

Inference is pure with respect to `Molecule`. Applying one selected state is a
separate operation.

### 6.4 Authority order

1. explicit user state tied to the declared boundary;
2. resolved in-memory state with matching graph identity;
3. verified sidecar;
4. trustworthy format-native integer charge semantics;
5. registered-domain, warning-bearing candidate inference;
6. `UNRESOLVED`.

A round trip never promotes lower-authority format defaults over an existing
resolved state.

## 7. Representation invariance and delocalization

For the same computational boundary and chemical state, resolution must be
invariant to:

- atom ordering;
- equivalent resonance/Kekule localizations;
- explicit versus implicit H when the atom inventory is identical;
- lossless SMILES, SDF/MOL and MOL2 round trips;
- deterministic serialization bond numbering.

Changing protonation, tautomer, atom inventory, computational boundary or total
electron count is not a representation change.

The charge package does not consume one arbitrary alternating bond-order vector
as truth. It evaluates an auditable set of bond-order/delocalization hypotheses
and returns resonance-equivalence classes. Aromaticity and formal charge are
partly interdependent; when upstream hypotheses disagree, the correct result is
multiple candidates or `UNRESOLVED`, not hidden re-perception.

A canonical localized charge vector may be generated for serialization, but it
is not promoted above its resonance-equivalence class.

## 8. Current implementation and reproduced defect

The production implementation is currently
`hotpot/cheminfo/calculator/formal_charges.py`:

- `infer_charge()` is non-mutating and returns atom, component and total integer
  charges;
- `formal_charge()` applies one result;
- a clone hides metal--nonmetal bonds and separates all-metal from nonmetal
  components;
- all ligand atoms are then flattened;
- `valence` selects one lowest-penalty charge independently per supported atom;
- `valence-constrained` performs one charge-sum dynamic program across all
  flattened ligand atoms;
- metal values are preserved, supplied by a callback, or taken from the
  existing element-default path;
- bond orders, lone pairs, radicals and resonance alternatives are not jointly
  enumerated.

Thus current code is not a fragment-candidate composition solver, and its
metal default can introduce an unvalidated oxidation-state-like assumption.

Only H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, Te and I have current
main-group templates.

The current scorer is representation-dependent. On 182 retained structures,
source-SDF and MOL2-reload inference gave:

```text
matches:      7 / 182
mismatches: 175 / 182
```

| MOL2 charge minus source charge | Cases |
|---:|---:|
| 0 | 7 |
| -2 | 98 |
| -4 | 29 |
| -6 | 44 |
| -8 | 4 |

This demonstrates non-invariance across the tested boundary. Inspected cases
show changes in semantic aromatic flags, numeric Kekule orders and H/charge
representation. It does not prove either format result is chemically correct.

This defect did not cause the retained xTB benchmark failures: that benchmark
resolved $Q$ from the source/in-memory state and passed `--chrg` without
re-inferring it from MOL2.

### 8.1 Current reliability boundary

| Current behavior | Consequence |
|---|---|
| Hides metal--nonmetal edges and flattens ligand atoms | Temporary FF-like fragments are conflated with electron-bookkeeping domains; donor-boundary electrons are not represented explicitly |
| Chooses local valence penalties before a complete Lewis model exists | Bond order, lone pairs, radicals and formal charge can become mutually inconsistent |
| `valence-constrained` constrains only the final integer sum | A numerically valid total does not prove that any chemically valid joint electron allocation exists |
| Existing metal charge or element default enters the result | An oxidation-state-like prior may be mistaken for independently inferred charge |
| Limited main-group template coverage | Organometallic, radical, multicentre, hypervalent and unsupported-element results are incomplete or out-of-domain |
| One result rather than a resonance/symmetry equivalence class | Representation changes can alter the selected formal charges and sometimes $Q$ |

A normalized input can prevent parser, H-inventory, boundary and aromaticity
drift in the current path, but it cannot repair these algorithmic omissions.
Until the target joint solver exists, the reliable production policy is to
preserve an authoritative $Q$ and treat newly inferred atom/component charges
as limited-domain evidence, not as a universal chemical truth.

## 9. Target inference algorithm

### 9.1 Resolve net-charge evidence first

`resolve_net_charge()` records authoritative $Q$ or returns explicit candidates.
It never derives $Q$ from partial charges, a common metal oxidation state or a
neutrality default.

If an authoritative $Q$ exists, xTB can consume it even when atom formal
charges or oxidation states remain ambiguous.

### 9.2 Build full-graph electron-solving domains

From the intact semantic graph:

1. identify original connected components;
2. identify localized and delocalized Lewis domains;
3. create solving blocks only where the electron boundary ledger is exact;
4. retain coordination, unknown, multicentre and haptic edges as semantic
   evidence rather than deleting them;
5. reject or delegate unsupported domains.

### 9.3 Joint Lewis candidate generation

For each atom $i$ and bond/domain $e$, candidate variables include:

- localized bond electron-pair order $b_e$;
- nonbonding electron count $N_i$;
- radical electron count $r_i$;
- atom formal charge $q_i$;
- optional multicentre/delocalized electron-domain allocation.

Each localized candidate obeys:

$$
q_i=V_i-N_i-\frac{1}{2}B_i,
$$

and the full candidate obeys:

$$
\sum_iq_i=Q.
$$

The implementation also checks total valence-electron conservation, fixed H
inventory, fixed local evidence and all declared bond semantics. Formal charge
and bond order are generated together; neither is guessed independently from
the other.

A complete allocation stores:

```python
LewisElectronAllocation(
    atom_formal_charges=...,
    localized_bond_orders=...,
    nonbonding_electrons=...,
    radical_electrons=...,
    electron_domains=...,
    resonance_class=...,
    assumptions=...,
)
```

### 9.4 Global combination and factorization

For a true connected component or mathematically factorable block $f$, retain
all feasible candidates:

$$
Q_f\in\{Q_{f,1},Q_{f,2},\ldots\}.
$$

Then solve the global constraints, including:

$$
\sum_fQ_f=Q.
$$

No block selects a local winner before global composition. Branch-and-bound,
dynamic programming or ILP/CSP methods may be used, but resource-limited output
must be labelled as truncated rather than globally unique.

Equivalent resonance and symmetry solutions are quotient classes before
confidence evaluation.

### 9.5 Scores and three independent result axes

Scores are versioned ordinal penalties, not energies or calibrated
probabilities. Every term and template source is evidence.

`solver_status`:

- `PRESERVED`;
- `OPTIMAL_COMPLETE`;
- `FEASIBLE_TRUNCATED`;
- `UNSATISFIABLE`;
- `RESOURCE_LIMIT`.

`ChargeResolutionStatus`:

- `UNIQUE_VALUE` for one net-charge value;
- `UNIQUE_EQUIVALENCE_CLASS` for one Lewis resonance/symmetry class;
- `AMBIGUOUS`;
- `UNRESOLVED`;
- `OUT_OF_DOMAIN`;
- `INFORMATION_LOSS`.

`chemical_confidence`:

- `AUTHORITATIVE`;
- `HIGH_IN_DOMAIN`;
- `MEDIUM`;
- `LOW`;
- `UNRESOLVED`.

A complete solver returning one model class does not imply high chemical
confidence. Conversely, several resonance-localized vectors may still be one
unique resonance-equivalence class.

Every valence template is versioned data with its source, supported environment,
counterexamples and held-out validation. IUPAC defines formal charge; it does
not supply a universal inference template table.

## 10. Coordination electron counting is a separate contract

The charge package exports full-graph Lewis evidence. A separate coordination
electron-counting operation consumes that evidence plus semantic coordination
topology:

```text
MolecularFacts + NetChargeResult + LewisFormalChargeResult
    -> CoordinationElectronCountingResult(convention=ionic|donor-return|CBC)
    -> OxidationStateInferenceResult
```

This operation owns:

- explicit cleavage convention;
- donor/acceptor direction and boundary electron ledger;
- Green CBC/L--X--Z classification and donated-electron counts;
- haptic/multicentre handling;
- metal/core and ligand electron-counting candidates.

It must not overwrite `NetChargeResult` or the full-graph formal-charge result.
Its payload is convention-specific: an ionic assignment may contain ligand
bookkeeping charges, donor-return records boundary-electron ownership, and CBC
records L/X/Z classes and electron donation. These quantities must not be
coerced into one generic `fragment_charge` field. Different conventions may
therefore produce different but non-interchangeable bookkeeping descriptions
while remaining consistent with the same $Q$.

## 11. Typed API and module layout

`hotpot/cheminfo/calculator/contracts.py` owns only immutable
`MolecularFacts` and shared identity/provenance protocols.

```text
hotpot/cheminfo/calculator/charge/
├── __init__.py
├── contracts.py
├── net_charge.py
├── preserve.py
├── validate.py
├── components.py
├── lewis_blocks.py
├── lewis_candidates.py
├── resonance.py
├── solver.py
├── apply.py
├── settings.py
└── data/
    ├── valence_templates.json
    └── SOURCES.md
```

Proposed public contracts:

`ChargeInferenceBundle` is intentionally distinct from the production
`electronic_state.contracts.ChargeInferenceResult` described in Section 8;
the latter is the legacy flat result and must not be silently reinterpreted as
the new two-result contract during migration.

```python
class ChargeInferenceMode(Enum):
    PRESERVE = "preserve"
    VALIDATE = "validate"
    INFER = "infer"


@dataclass(frozen=True)
class NetChargeCandidate:
    identity: str
    charge: int
    evidence: tuple[ChargeEvidence, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class NetChargeResult:
    identity: str
    molecular_identity: str
    candidates: tuple[NetChargeCandidate, ...]
    selected_candidate: int | None
    solver_status: SolverStatus
    resolution_status: ChargeResolutionStatus
    chemical_confidence: ChemicalConfidence


@dataclass(frozen=True)
class ConnectedComponentCharge:
    atom_indices: tuple[int, ...]
    charge: int


@dataclass(frozen=True)
class LewisElectronAllocation:
    identity: str
    net_charge_candidate_identity: str
    atom_formal_charges: tuple[int, ...]
    localized_bond_orders: tuple[LewisBondOrder, ...]
    nonbonding_electrons: tuple[int, ...]
    radical_electrons: tuple[int, ...]
    electron_domains: tuple[ElectronDomainAllocation, ...]
    connected_components: tuple[ConnectedComponentCharge, ...]
    resonance_class: int
    score: EvidenceScore
    evidence: tuple[ChargeEvidence, ...]


@dataclass(frozen=True)
class LewisFormalChargeResult:
    molecular_identity: str
    net_charge_result_identity: str
    candidates: tuple[LewisElectronAllocation, ...]
    selected_candidate: int | None
    solver_status: SolverStatus
    resolution_status: ChargeResolutionStatus
    chemical_confidence: ChemicalConfidence


@dataclass(frozen=True)
class ChargeInferenceBundle:
    net_charge: NetChargeResult
    lewis: LewisFormalChargeResult | None


def resolve_net_charge(request: NetChargeRequest) -> NetChargeResult: ...

def infer_lewis_allocations(
    molecular_facts: MolecularFacts,
    net_charge: NetChargeResult,
    *,
    policy: LewisInferencePolicy,
) -> LewisFormalChargeResult: ...

def infer_charge(request: ChargeInferenceRequest) -> ChargeInferenceBundle: ...

def apply_lewis_allocation(
    mol: Molecule,
    allocation: LewisElectronAllocation,
) -> None: ...
```

`infer_charge()` is only an explicit facade over the two independent
operations. It may not hide a coordination-electron-counting or oxidation-state
fallback.

## 12. Pipeline transport contract

```text
MolecularRecord
├── Molecule graph and coordinates
├── NetChargeResult
├── optional LewisFormalChargeResult
├── coordination_electron_counting_results
│   └── tuple[CoordinationElectronCountingResult, ...]
├── optional OxidationStateInferenceResult
├── optional ElectronicStateInferenceResult
└── immutable graph/state boundary identity
```

GFN1/2-xTB requires a selected, sufficiently trusted net $Q$ and electronic
state. It does not require a unique atom formal-charge vector. Therefore Lewis
ambiguity alone must not block xTB once $Q$ is authoritative.

GFN-FF may expose backend-specific charge or fragment files. Those are adapter
contracts and must not be assumed equivalent to connected-component charges or
ionic/CBC ligand assignments without explicit validation.

An external-process sidecar records boundary identity, atom mapping, selected
$Q$, Lewis candidate/equivalence identity, solver status, confidence and
provenance. Native output never overwrites these facts implicitly.

## 13. Reliability envelope and limitations

The two requested guarantees have different answers. A strict normalized input
guarantees deterministic, bounded behavior; when the solver completes the
declared search, it can also prove uniqueness *inside that model*. It cannot
guarantee that the model has identified one physically privileged charge
description or that it will agree with every chemical convention or
higher-level population analysis.

### 13.1 Guarantee 1: uniqueness and non-pathological execution

| Goal | Guarantee |
|---|---|
| Deterministic execution | Yes, with frozen inputs, versions, seeds and tie ordering |
| Bounded time/memory | Yes, with explicit candidate/time/memory limits |
| No fabricated result on failure | Yes, through `UNSATISFIABLE`, `RESOURCE_LIMIT` or `UNRESOLVED` |
| Unique resonance-equivalence class | Conditional: only after complete search leaves one class |
| Unique localized atom-charge vector | Generally no; canonical localization is a serialization convention |
| Chemical plausibility | Only within a validated applicability domain |
| Equality to physical electron density | No; formal charge is a bookkeeping construct |
| Equality to a “high-level formal charge” | No universal target exists |
| Reliable partial charges | No; requires an independent electronic-structure/population method |

Normalization can make the stated model well posed. It cannot turn an
information-theoretically ambiguous chemical state into a unique one.

### 13.2 Applicability by chemical domain

| Domain | Expected behavior |
|---|---|
| Known $Q$, complete H, common closed-shell main-group 2c--2e structures | Usually tractable; may be unique modulo resonance |
| Common aromatic/resonance systems with supported templates | Joint equivalence-class result |
| Hypervalent P/S, boranes or electron-deficient systems | Specialist templates or low confidence |
| Odd-electron radicals | Requires explicit radical/electron evidence; otherwise ambiguous |
| Zwitterions with several close Lewis classes | Preserve alternatives |
| Large coupled delocalized systems | Candidate/combinatorial limits may truncate |
| Covalent organometallic, haptic or multicentre bonding | Specialist model or out-of-domain |
| Non-innocent ligands and mixed-valence metal systems | Joint downstream hypotheses; no forced ligand charge |
| Unknown protonation/H inventory, lost $Q$ or unclear system boundary | Information loss; cannot be recovered reliably |
| Geometry-only input without authoritative topology | Out-of-domain for this graph solver |

### 13.3 Controlling pathological searches

The implementation must expose `max_candidates`, `max_time` and `max_memory`,
use deterministic ordering, factor only exact blocks, quotient resonance and
symmetry early, and retain tied candidates.

When a limit is reached:

- return `FEASIBLE_TRUNCATED` or `RESOURCE_LIMIT`;
- never label the current best candidate globally unique;
- retain the search bounds and rejected/unevaluated count where possible.

### 13.4 Guarantee 2: chemical common sense and higher-level methods

Standardized input removes representation errors; it does not guarantee
chemical truth. Higher chemical confidence requires:

1. versioned templates with explicit applicability;
2. expert-curated Lewis/resonance equivalence classes;
3. element, functional-group, charge and radical holdouts;
4. active out-of-domain rejection;
5. independent structure/database cross-checks; and
6. optional QM/NBO/Wiberg or spectroscopy evidence for ranking difficult
   hypotheses.

QM cannot supply a unique universal “true formal charge.” Population charges
and bond indices depend on the method, basis and partitioning scheme. Higher-
level methods are useful to compare fixed-$Q$ electronic states, reject
implausible candidates and provide delocalization evidence; they are not an
absolute formal-charge oracle.

## 14. Validation requirements

### 14.1 Unit and scientific cases

- authoritative positive, negative and zero $Q$;
- ordinary main-group neutral/ionic structures;
- carboxylate, nitro, azide, guanidinium and allylic resonance;
- explicit radicals and electron-deficient/hypervalent fixtures;
- $\mathrm{NH_3\rightarrow BF_3}$ full-graph versus donor-return accounting;
- $\mathrm{[Fe(CN)_6]^{3-/4-}}$, ferrocenium/ferrocene and non-innocent
  examples proving that missing $Q$ is not graph-identifiable;
- proof that charge APIs never call oxidation-state, CBC or spin inference;
- resource-limit results are never reported as unique.

### 14.2 Representation-metamorphic tests

For each equivalent representation:

- permute atom order;
- enumerate valid Kekule/resonance localizations;
- toggle equivalent explicit/implicit-H representation;
- round-trip through supported SMILES, SDF/MOL and MOL2 paths;
- compare boundary, atom inventory, semantic graph, selected $Q$ and
  resonance-equivalence classes;
- compare one localized vector only after canonicalization.

The existing 182-case SDF/MOL2 corpus remains a mandatory regression set. The
target is 182/182 invariant decisions or an explicit information-loss/out-of-
domain result; silent drift is unacceptable.

### 14.3 Benchmark reporting

Report separately:

- net-charge accuracy and coverage;
- Lewis/resonance-class agreement;
- abstention and false-confident-error rates;
- representation invariance;
- candidate count and search completeness;
- p50/p95/p99 time and peak memory;
- out-of-domain detection.

Use scaffold, functional-group, element, $|Q|$, radical/closed-shell and
data-source holdouts, not only random splits. PubChem and ChEBI are useful
independent structure corpora, not atom formal-charge ground truth. Coordination
sets require expert-verified bookkeeping conventions.

## 15. References

1. IUPAC, “formal charge,” `Compendium of Chemical Terminology`,
   DOI: [10.1351/goldbook.F02430](https://doi.org/10.1351/goldbook.F02430).
2. IUPAC, “partial charge,” `Compendium of Chemical Terminology`,
   DOI: [10.1351/goldbook.P04420](https://doi.org/10.1351/goldbook.P04420).
3. IUPAC, “coordinate covalent bond,” `Compendium of Chemical Terminology`,
   DOI: [10.1351/goldbook.D01523](https://doi.org/10.1351/goldbook.D01523).
4. P. Karen et al., “Comprehensive definition of oxidation state,”
   `Pure Appl. Chem.` 88 (2016) 831--839,
   DOI: [10.1515/pac-2015-1204](https://doi.org/10.1515/pac-2015-1204).
5. I. D. Welsh and J. R. Allison, “Automated simultaneous assignment of bond
   orders and formal charges,” `J. Cheminform.` 11 (2019) 18,
   DOI: [10.1186/s13321-019-0340-0](https://doi.org/10.1186/s13321-019-0340-0).
6. M. L. H. Green, “A new approach to the formal classification of covalent
   compounds of the elements,” `J. Organomet. Chem.` 500 (1995) 127--148,
   DOI: [10.1016/0022-328X(95)00508-N](https://doi.org/10.1016/0022-328X(95)00508-N).
7. C. A. James and the OpenSMILES contributors, `OpenSMILES specification`
   version 1.0 (2016),
   [source specification](https://github.com/opensmiles/OpenSMILES).
8. Dassault Systemes BIOVIA, `CTfile Formats` (2020),
   [official specification](https://discover.3ds.com/sites/default/files/2020-08/biovia_ctfileformats_2020.pdf).
9. D. Weininger, “SMILES, a chemical language and information system. 1,”
   `J. Chem. Inf. Comput. Sci.` 28 (1988) 31--36,
   DOI: [10.1021/ci00057a005](https://doi.org/10.1021/ci00057a005).
10. RDKit, `rdDetermineBonds.DetermineBondOrders` API and source documentation;
    its required total-charge argument is software-behavior evidence, not a
    universal chemical oracle.
11. S. Kim et al., “PubChem 2023 update,” `Nucleic Acids Res.` 51 (2023)
    D1373--D1380, DOI:
    [10.1093/nar/gkac956](https://doi.org/10.1093/nar/gkac956).
12. J. Hastings et al., “ChEBI in 2016,” `Nucleic Acids Res.` 44 (2016)
    D1214--D1219, DOI:
    [10.1093/nar/gkv1031](https://doi.org/10.1093/nar/gkv1031).

No reference above provides a universal graph-to-charge oracle. Definitions,
format contracts, algorithms and validation corpora have different evidentiary
roles and remain labelled accordingly.
