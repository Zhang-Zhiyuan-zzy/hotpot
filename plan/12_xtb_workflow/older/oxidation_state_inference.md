# Oxidation-state inference: scientific and implementation specification

> **OLDER / NON-NORMATIVE.** This design record is retained for theory,
> registries and historical alternatives. The active implementation contract
> is [`scientific_state_implementation.md`](../scientific_state_implementation.md)
> and the normative specifications indexed there.

Status: design specification only; no production implementation is provided.

Related documents:

- [Charge inference](charge_inference.md)
- [Electronic-state inference](electronic_state_inference.md)
- [xTB failure root-cause and recovery plan](xtb_failure_root_cause_and_recovery_plan.md)

## 1. Purpose and ownership

This module answers one question: given an authoritative molecular graph and
an independently resolved charge result, what oxidation-state assignments are
consistent with the structure and available chemical evidence?

| Contract | Meaning |
|---|---|
| Owns | Convention-labelled coordination electron-counting assignments; per-atom oxidation-state candidates; globally consistent assignments, ambiguity, confidence and provenance |
| Consumes | Immutable molecular facts, a `NetChargeResult`, an optional `LewisFormalChargeResult`, optional independently produced coordination-counting results and explicit oxidation-state overrides |
| Must not decide | System net charge, original-component charge, intact-graph Lewis formal charge, partial charge, UHF, multiplicity, spin coupling or xTB options |
| Must not do | Call a charge or electronic-state inference implementation, mutate the graph, or fabricate one assignment when evidence is non-unique |

The planned operations are independent pure functions. The oxidation-state
resolver must not hide a call to the coordination-counting operation:

```python
def infer_coordination_electron_counting(
    molecular_facts: MolecularFacts,
    net_charge_result: NetChargeResult,
    lewis_result: LewisFormalChargeResult | None,
    *,
    convention: CoordinationCountingConvention,
    policy: CoordinationCountingPolicy,
) -> CoordinationElectronCountingResult: ...


def infer_oxidation_states(
    molecular_facts: MolecularFacts,
    net_charge_result: NetChargeResult,
    lewis_result: LewisFormalChargeResult | None,
    *,
    electron_counting_results: tuple[
        CoordinationElectronCountingResult, ...
    ] = (),
    overrides: tuple[OxidationStateOverride, ...] = (),
    policy: OxidationStatePolicy = OxidationStatePolicy(),
) -> OxidationStateInferenceResult
```

`MolecularFacts` is a read-only view of the authoritative Hotpot `Molecule`,
not a second mutable chemical graph. `infer_oxidation_states()` must never
wrap or call the current `Atom.oxidation_state`: that legacy property does not
enforce molecular charge conservation and mixes unrelated valence heuristics.

## 2. Definitions and symbols

Oxidation state (OS) is the charge an atom has after ionic approximation of
its bonds according to bond-electron allegiance. It is a formal integer
bookkeeping construct, not an atomic partial charge or a directly measured
observable.

| Symbol | Meaning |
|---|---|
| $Q$ | Net molecular charge from one `NetChargeCandidate` |
| $q_i$ | Formal charge of atom $i$ in one Lewis/resonance candidate |
| $x_i$ | Oxidation state assigned to atom $i$ |
| $b_{ij}$ | Bond order in electron pairs between atoms $i$ and $j$ |
| $\chi_i$ | Adopted Allen electronegativity of atom $i$ |
| $a_{ij}$ | Bonding electrons assigned to atom $i$ by ionic approximation |
| $D_i$ | Candidate oxidation-state domain for atom $i$ |
| $R_{ij}$ | Observed interatomic distance used by an optional bond-valence check |
| $Q_c$ | Charge of an original disconnected component in one matching full-graph Lewis allocation |
| $\kappa$ | Explicit coordination electron-counting convention |
| $Q_L^{(\kappa)}$ | Ligand bookkeeping charge when convention $\kappa$ is ionic; not a universal field for donor-return or CBC counting |

Every complete assignment must obey:

$$\sum_i x_i=Q.$$

Formal charge and oxidation state must remain distinct. Formal charge splits
shared bonding electrons equally; oxidation state assigns them according to
bond-electron allegiance. A neutral ligand may contain atoms with non-zero
oxidation states, and a metal's formal charge is not automatically its
oxidation state.

## 3. Independent data contracts

```python
@dataclass(frozen=True)
class AtomOxidationStateEvidence:
    atom_index: int
    element: str
    candidates: tuple[int, ...]
    selected: int | None
    evidence: tuple[EvidenceItem, ...]
    confidence: Confidence


@dataclass(frozen=True)
class CenterOxidationState:
    atom_index: int
    element: str
    oxidation_state: int


@dataclass(frozen=True)
class LigandRedoxEvidence:
    atom_indices: tuple[int, ...]
    electron_bookkeeping_delta: int
    source: SourceRecord
    assumptions: tuple[str, ...]


class CoordinationCountingConvention(Enum):
    IONIC = "ionic"
    DONOR_RETURN = "donor-return"
    CBC = "cbc"


@dataclass(frozen=True)
class CoordinationFragmentAssignment:
    atom_indices: tuple[int, ...]
    ionic_bookkeeping_charge: int | None
    lxz_class: LXZClass | None
    donated_electrons: int | None
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class CoordinationCountingAssignment:
    identity: str
    fragments: tuple[CoordinationFragmentAssignment, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class CoordinationElectronCountingResult:
    identity: str
    molecular_identity: str
    net_charge_candidate_identity: str
    lewis_allocation_identity: str | None
    convention: CoordinationCountingConvention
    candidates: tuple[CoordinationCountingAssignment, ...]
    selected_candidate: int | None
    solver_status: SolverStatus
    resolution_status: CoordinationCountingResolutionStatus
    chemical_confidence: ChemicalConfidence
    provenance: tuple[SourceRecord, ...]


@dataclass(frozen=True)
class OxidationStateAssignment:
    identity: str
    net_charge_candidate_identity: str
    lewis_allocation_identity: str | None
    coordination_counting_result_identity: str | None
    coordination_counting_assignment_identity: str | None
    atom_states: tuple[int, ...]
    center_states: tuple[CenterOxidationState, ...]
    ligand_redox_evidence: tuple[LigandRedoxEvidence, ...]
    total: int
    score: EvidenceScore
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class OxidationStateInferenceResult:
    molecular_identity: str
    net_charge_result_identity: str
    lewis_formal_charge_result_identity: str | None
    atom_evidence: tuple[AtomOxidationStateEvidence, ...]
    assignments: tuple[OxidationStateAssignment, ...]
    selected_assignment: int | None
    solver_status: SolverStatus
    resolution_status: OxidationResolutionStatus
    chemical_confidence: ChemicalConfidence
    provenance: tuple[SourceRecord, ...]
```

`CoordinationCountingResolutionStatus` and `OxidationResolutionStatus` are
module-specific enums rather than aliases of `ChargeResolutionStatus`. They use
`UNIQUE_ASSIGNMENT`, `AMBIGUOUS`, `UNRESOLVED` and `CONFLICT`; this keeps a
unique bookkeeping assignment distinct from a unique net-charge value or
Lewis equivalence class.

`selected_assignment=None` is a valid and necessary result. A graph,
protonation or charge change invalidates the result identity and requires the
resolver to call the module again explicitly.

## 4. Evidence hierarchy and source policy

The inference engine uses evidence in this order:

1. explicit, internally consistent user or curated-compound assignments;
2. IUPAC bond-electron allegiance on a complete Lewis/resonance description;
3. net-charge conservation, matching intact-graph connected-component charges
   and convention-labelled ligand electron-counting evidence;
4. independently validated motif evidence for redox-active ligands;
5. source-tagged occurrence evidence as ranking support, only inside its
   validated chemical domain;
6. optional geometry-dependent bond-valence evidence inside its calibrated
   chemical domain;
7. broad element candidate tables as domain-expansion priors only.

The element table below is not a list of states that may be assigned without
structural evidence. Its broad candidate column is the PubChem periodic-table
compilation retrieved through PUG REST; the secondary column is the union of
the pymatgen-core and Mendeleev registries after removing PubChem entries. The
two secondary registries trace their lists to a Wikipedia compilation and are
therefore coverage aids, not primary scientific authority. RSC is the planned
first professional cross-check, but its values are intentionally not hidden in
either displayed column. The ordered-page checksum below records acquisition
evidence only; RSC values cannot influence runtime ranking until a separate
source-tagged machine-readable snapshot is committed and reviewed.

Data snapshot used for this plan (2026-10-12):

| Data | Identity | Use |
|---|---|---|
| RSC periodic-table element pages, ordered Z=1--118 | SHA-256 `27142ce30a257d43c924f22f2dc976466361ff206f58b3d78087d6b28e4ba197` | Acquisition evidence for a future source-specific cross-check; not folded into this table |
| PubChem PUG REST periodic table CSV | SHA-256 `efcadb3dd180bd06fc0fa069a81082e86553ba2b8b5d679b7827bb8c03afd3ce` | Broad candidate coverage |
| pymatgen-core `oxidation_states.yaml` | SHA-256 `2d9f78d23b69fe1c7c9c38fdc6efad1b528c236d9a421cf15e20f4ae6f523aee` | Secondary coverage check |
| Mendeleev `elements.db` | SHA-256 `8a6312b57603e22c58ab28005bb8cacfbff721bc859eea8ddc2b784c17ffd717` | Secondary main/extended classification check |

No environment-free default oxidation state is defined. Even apparently
common defaults fail for compounds such as Fe(CO)5, peroxides, mixed-valence
solids and non-innocent ligand complexes. Oxidation state zero is introduced
when the graph and electron-allocation facts require it; it is not silently
added to every molecular candidate domain.

For $Z=104$--108, experimental evidence is sparse and mixed with prediction.
For $Z=109$--118, the listed candidates are predominantly predicted. Such
entries may expand an explicitly requested search but may not produce an
automatic high-confidence assignment.

For each row, the selectable reference domain is the source-tagged union of
the two candidate columns after policy filtering. It is a search domain, not a
default. Graph-derived IUPAC assignments remain admissible even when absent
from a compact “common states” source, but the discrepancy is recorded and
lowers confidence. ICSD/SMACT occurrence frequencies are a planned independent
ranking validation for ionic-solid-like domains only; they are not generic
priors for molecular coordination or organometallic chemistry, and no such
frequency is claimed by this frozen table. Before production, every registry
must record its package/data release and source URL in addition to the content
hash; the Mendeleev row above currently has content identity but insufficient
release metadata and therefore remains secondary planning evidence.

## 5. Per-element candidate registry

| Z | Element | Broad compiled candidates (PubChem) | Additional registry-only candidates | Evidence boundary |
|---:|---|---|---|---|
| 1 | H | +1, -1 | — | compiled; environment required |
| 2 | He | 0 | — | compiled; environment required |
| 3 | Li | +1 | — | compiled; environment required |
| 4 | Be | +2 | 0, +1 | compiled; environment required |
| 5 | B | +3 | -5, -1, 0, +1, +2 | compiled; environment required |
| 6 | C | +4, +2, -4 | -3, -2, -1, 0, +1, +3 | compiled; environment required |
| 7 | N | +5, +4, +3, +2, +1, -1, -2, -3 | 0 | compiled; environment required |
| 8 | O | -2 | -1, 0, +1, +2 | compiled; environment required |
| 9 | F | -1 | — | compiled; environment required |
| 10 | Ne | 0 | — | compiled; environment required |
| 11 | Na | +1 | -1, 0 | compiled; environment required |
| 12 | Mg | +2 | 0, +1 | compiled; environment required |
| 13 | Al | +3 | -2, -1, 0, +1, +2 | compiled; environment required |
| 14 | Si | +4, +2, -4 | -3, -2, -1, 0, +1, +3 | compiled; environment required |
| 15 | P | +5, +3, -3 | -2, -1, 0, +1, +2, +4 | compiled; environment required |
| 16 | S | +6, +4, -2 | -1, 0, +1, +2, +3, +5 | compiled; environment required |
| 17 | Cl | +7, +5, +1, -1 | +2, +3, +4, +6 | compiled; environment required |
| 18 | Ar | 0 | — | compiled; environment required |
| 19 | K | +1 | -1 | compiled; environment required |
| 20 | Ca | +2 | +1 | compiled; environment required |
| 21 | Sc | +3 | 0, +1, +2 | compiled; environment required |
| 22 | Ti | +4, +3, +2 | -2, -1, 0, +1 | compiled; environment required |
| 23 | V | +5, +4, +3, +2 | -3, -1, 0, +1 | compiled; environment required |
| 24 | Cr | +6, +3, +2 | -4, -2, -1, 0, +1, +4, +5 | compiled; environment required |
| 25 | Mn | +7, +4, +3, +2 | -3, -2, -1, 0, +1, +5, +6 | compiled; environment required |
| 26 | Fe | +3, +2 | -4, -2, -1, 0, +1, +4, +5, +6, +7 | compiled; environment required |
| 27 | Co | +3, +2 | -3, -1, 0, +1, +4, +5 | compiled; environment required |
| 28 | Ni | +3, +2 | -2, -1, 0, +1, +4 | compiled; environment required |
| 29 | Cu | +2, +1 | -2, 0, +3, +4 | compiled; environment required |
| 30 | Zn | +2 | -2, 0, +1 | compiled; environment required |
| 31 | Ga | +3 | -5, -4, -3, -2, -1, 0, +1, +2 | compiled; environment required |
| 32 | Ge | +4, +2 | -4, -3, -2, -1, 0, +1, +3 | compiled; environment required |
| 33 | As | +5, +3, -3 | -2, -1, 0, +1, +2, +4 | compiled; environment required |
| 34 | Se | +6, +4, -2 | -1, 0, +1, +2, +3, +5 | compiled; environment required |
| 35 | Br | +5, +1, -1 | +2, +3, +4, +7 | compiled; environment required |
| 36 | Kr | 0 | +1, +2 | compiled; environment required |
| 37 | Rb | +1 | -1 | compiled; environment required |
| 38 | Sr | +2 | +1 | compiled; environment required |
| 39 | Y | +3 | 0, +1, +2 | compiled; environment required |
| 40 | Zr | +4 | +1, +2, +3 | compiled; environment required |
| 41 | Nb | +5, +3 | -3, -1, 0, +1, +2, +4 | compiled; environment required |
| 42 | Mo | +6 | -4, -2, -1, 0, +1, +2, +3, +4, +5 | compiled; environment required |
| 43 | Tc | +7, +6, +4 | -3, -1, +1, +2, +3, +5 | compiled; environment required |
| 44 | Ru | +3 | -4, -2, +1, +2, +4, +5, +6, +7, +8 | compiled; environment required |
| 45 | Rh | +3 | -3, -1, +1, +2, +4, +5, +6, +7 | compiled; environment required |
| 46 | Pd | +3, +2 | +1, +4, +5 | compiled; environment required |
| 47 | Ag | +1 | -2, -1, 0, +2, +3 | compiled; environment required |
| 48 | Cd | +2 | -2, +1 | compiled; environment required |
| 49 | In | +3 | -5, -2, -1, 0, +1, +2 | compiled; environment required |
| 50 | Sn | +4, +2 | -4, -3, -2, -1, 0, +1, +3 | compiled; environment required |
| 51 | Sb | +5, +3, -3 | -2, -1, 0, +1, +2, +4 | compiled; environment required |
| 52 | Te | +6, +4, -2 | -1, 0, +1, +2, +3, +5 | compiled; environment required |
| 53 | I | +7, +5, +1, -1 | +2, +3, +4, +6 | compiled; environment required |
| 54 | Xe | 0 | +2, +4, +6, +8 | compiled; environment required |
| 55 | Cs | +1 | -1 | compiled; environment required |
| 56 | Ba | +2 | +1 | compiled; environment required |
| 57 | La | +3 | 0, +1, +2 | compiled; environment required |
| 58 | Ce | +4, +3 | +1, +2 | compiled; environment required |
| 59 | Pr | +3 | 0, +1, +2, +4, +5 | compiled; environment required |
| 60 | Nd | +3 | 0, +2, +4 | compiled; environment required |
| 61 | Pm | +3 | +2 | compiled; environment required |
| 62 | Sm | +3, +2 | 0, +1 | compiled; environment required |
| 63 | Eu | +3, +2 | 0 | compiled; environment required |
| 64 | Gd | +3 | 0, +1, +2 | compiled; environment required |
| 65 | Tb | +3 | 0, +1, +2, +4 | compiled; environment required |
| 66 | Dy | +3 | 0, +1, +2, +4 | compiled; environment required |
| 67 | Ho | +3 | 0, +1, +2 | compiled; environment required |
| 68 | Er | +3 | 0, +1, +2 | compiled; environment required |
| 69 | Tm | +3 | 0, +1, +2 | compiled; environment required |
| 70 | Yb | +3, +2 | 0, +1 | compiled; environment required |
| 71 | Lu | +3 | 0, +1, +2 | compiled; environment required |
| 72 | Hf | +4 | -2, 0, +1, +2, +3 | compiled; environment required |
| 73 | Ta | +5 | -3, -1, 0, +1, +2, +3, +4 | compiled; environment required |
| 74 | W | +6 | -4, -2, -1, 0, +1, +2, +3, +4, +5 | compiled; environment required |
| 75 | Re | +7, +6, +4 | -3, -1, 0, +1, +2, +3, +5 | compiled; environment required |
| 76 | Os | +4, +3 | -4, -2, -1, 0, +1, +2, +5, +6, +7, +8 | compiled; environment required |
| 77 | Ir | +4, +3 | -3, -2, -1, +1, +2, +5, +6, +7, +8, +9 | compiled; environment required |
| 78 | Pt | +4, +2 | -3, -2, -1, 0, +1, +3, +5, +6 | compiled; environment required |
| 79 | Au | +3, +1 | -3, -2, -1, 0, +2, +5 | compiled; environment required |
| 80 | Hg | +2, +1 | -2, +4 | compiled; environment required |
| 81 | Tl | +3, +1 | -5, -2, -1, +2 | compiled; environment required |
| 82 | Pb | +4, +2 | -4, -2, -1, 0, +1, +3 | compiled; environment required |
| 83 | Bi | +5, +3 | -3, -2, -1, 0, +1, +2, +4 | compiled; environment required |
| 84 | Po | +4, +2 | -2, +5, +6 | compiled; environment required |
| 85 | At | +7, +5, +3, +1, -1 | — | compiled; environment required |
| 86 | Rn | 0 | +2, +6 | compiled; environment required |
| 87 | Fr | +1 | — | compiled; environment required |
| 88 | Ra | +2 | — | compiled; environment required |
| 89 | Ac | +3 | — | compiled; environment required |
| 90 | Th | +4 | -1, +1, +2, +3 | compiled; environment required |
| 91 | Pa | +5, +4 | +2, +3 | compiled; environment required |
| 92 | U | +6, +5, +4, +3 | -1, +1, +2 | compiled; environment required |
| 93 | Np | +6, +5, +4, +3 | +2, +7 | compiled; environment required |
| 94 | Pu | +6, +5, +4, +3 | +2, +7, +8 | compiled; environment required |
| 95 | Am | +6, +5, +4, +3 | +2, +7 | compiled; environment required |
| 96 | Cm | +3 | +4, +5, +6 | compiled; environment required |
| 97 | Bk | +4, +3 | +2, +5 | compiled; environment required |
| 98 | Cf | +3 | +2, +4, +5 | compiled; environment required |
| 99 | Es | +3 | +2, +4 | compiled; environment required |
| 100 | Fm | +3 | +2 | compiled; environment required |
| 101 | Md | +3, +2 | — | compiled; environment required |
| 102 | No | +3, +2 | — | compiled; environment required |
| 103 | Lr | +3 | — | compiled; environment required |
| 104 | Rf | +4 | +3 | limited observations + prediction |
| 105 | Db | +5, +4, +3 | — | limited observations + prediction |
| 106 | Sg | +6, +5, +4, +3, 0 | — | limited observations + prediction |
| 107 | Bh | +7, +5, +4, +3 | — | limited observations + prediction |
| 108 | Hs | +8, +6, +5, +4, +3, +2 | — | limited observations + prediction |
| 109 | Mt | +9, +8, +6, +4, +3, +1 | — | predominantly predicted |
| 110 | Ds | +8, +6, +4, +2, 0 | — | predominantly predicted |
| 111 | Rg | +5, +3, +1, -1 | — | predominantly predicted |
| 112 | Cn | +2, +1, 0 | +4 | predominantly predicted |
| 113 | Nh | — | — | predominantly predicted |
| 114 | Fl | +6, +4, +2, +1, 0 | — | predominantly predicted |
| 115 | Mc | +3, +1 | — | predominantly predicted |
| 116 | Lv | +4, +2, -2 | — | predominantly predicted |
| 117 | Ts | +5, +3, +1, -1 | — | predominantly predicted |
| 118 | Og | +6, +4, +2, +1, 0, -1 | — | predominantly predicted |

## 6. Inference algorithm

### 6.1 Normalize facts without changing chemistry

The module receives, but does not infer:

- stable atom identities, elements and coordinates;
- covalent, aromatic, dative and metal-metal bond semantics;
- explicit and implicit hydrogen representation;
- paired Lewis/resonance bond-order and formal-charge allocations from
  `LewisFormalChargeResult`, when available;
- net-charge candidates from `NetChargeResult`;
- original connected-component charges from each matching Lewis allocation;
- optional convention-labelled `CoordinationElectronCountingResult` objects.

The last three quantities are distinct. In particular, an original disconnected
component is not a ligand fragment produced by cutting a coordination edge.

It classifies edges for electron-allocation purposes. Dative direction may
identify donor/acceptor roles but does not replace IUPAC electronegativity
allegiance. Each localized bond-order assignment remains paired with the
formal-charge vector that generated it. Equivalent aromatic/Kekule
representations must produce the same oxidation-state candidates or an
explicit ambiguity result.

### 6.2 Ionic approximation

For one paired `LewisElectronAllocation`, a covalent bond containing $b_{ij}$
electron pairs contributes:

$$
a_{ij}=\begin{cases}
2b_{ij}, & \chi_i>\chi_j,\\
b_{ij}, & \chi_i=\chi_j,\\
0, & \chi_i<\chi_j.
\end{cases}
$$

Using formal charge to eliminate the unobserved lone-pair count gives:

$$x_i=q_i+\sum_j b_{ij}-\sum_j a_{ij}.$$

This reproduces, for example, carbon $-4$ and hydrogen $+1$ in methane.
Homonuclear bonds split equally and therefore contribute no oxidation-state
separation. When the required Lewis/resonance description is unavailable or
multiple descriptions give different integer results, the module branches and
retains all consistent candidates.

Allen electronegativity is the default numerical ordering because it is based
on spectroscopic valence-state energies and is consistent with the IUPAC
ionic-approximation discussion. Ties, unavailable values and chemically
recognized exceptions lower confidence instead of being hidden behind an
arbitrary element ordering.

The solver exposes two explicit, mutually exclusive treatments for a semantic
coordination edge: whole-graph IUPAC bond-electron allegiance, or a selected
coordination-counting convention. One assignment must not consume both views
of the same boundary electron pair. Donor-return and CBC results may generate
candidates or evidence, but they are not ionic ligand charges unless a
reviewed, provenance-bearing conversion says so.

### 6.3 Candidate domains and global constraint solving

For each atom, construct $D_i$ from:

1. explicit overrides, when present;
2. oxidation states generated by every valid Lewis/resonance candidate;
3. charge-conserving alternatives for registered non-innocent motifs;
4. reference-table expansion only when requested or needed to represent an
   otherwise unexplained but structurally supported assignment.

Then solve the finite constraint problem:

$$x_i\in D_i,\qquad \sum_i x_i=Q,$$

with additional original-component totals and atom overrides. Branch-and-bound
must apply the remaining minimum/maximum sum bounds before expanding a branch.
Symmetry-equivalent assignments are canonicalized rather than reported as
artificially distinct solutions.

For a coordination complex, the independent coordination-counting operation
constructs immutable bookkeeping partitions under an explicit convention
$\kappa$; it never mutates the molecular graph. Only for an ionic bookkeeping
assignment, if $M$ is the set of metal centers, $L$ the ligands and $C$ the
original disconnected counterions, the metal-center constraint is

$$
\sum_{m\in M}x_m
=Q-\sum_{\ell\in L}Q_{\ell}^{(\kappa)}-\sum_{c\in C}Q_c.
$$

Here $Q$ comes from `NetChargeResult`; $Q_c$ comes from a matching full-graph
Lewis allocation or an authoritative original-component fact; and
$Q_{\ell}^{(\kappa)}$ comes only from an ionic
`CoordinationElectronCountingResult`. CBC L/X/Z classes and donor-return
electron ownership cannot be substituted into this equation as generic
fragment charges. For one metal the ionic balance may determine a unique
oxidation state; for several metals it often determines only their sum. Each
proposed value must still be supported by the IUPAC allocation or an explicitly
referenced ligand/redox model and by the element's source-tagged candidate
domain. The element table alone never overrides this equation.

The precedence rule is explicit:

1. select one net-charge candidate;
2. align one full-graph Lewis allocation when that evidence is available;
3. select and validate at most one treatment of each coordination boundary;
4. generate globally charge-conserving oxidation-state assignments;
5. distribute a metal-center sum only when center-specific evidence permits;
6. retain convention-dependent alternatives without double-counting the same
   coordination electrons;
7. report conflict for inconsistent authoritative facts and ambiguity for
   unresolved model alternatives.

### 6.4 Non-innocent ligands and multiple metal centers

The solver must preserve alternatives for ligands such as NO, O2,
dithiolenes, quinones and other ligand-centered redox systems. Motif handlers
may propose ligand-redox and coordination-counting alternatives with
references, but cannot overwrite `NetChargeResult`. Every alternative must
retain charge conservation and provenance.

For multiple metals, only the sum may be uniquely determined. Mixed-valence
assignments that differ only by localization on equivalent centers remain one
delocalized/equivalence-class result unless independent structural evidence
resolves the sites.

### 6.5 Optional geometry cross-check

Within a calibrated element-pair domain, the bond-valence sum is:

$$s_{ij}=\exp\left(\frac{R_{0,ij}-R_{ij}}{B_{ij}}\right),\qquad
V_i=\sum_j s_{ij}.$$

The residual between $V_i$ and a candidate oxidation magnitude is soft
evidence only. Bond-valence parameters are strongest for ionic crystals and
must not become a hard rejection for organometallic, f-element or
non-innocent-ligand chemistry.

### 6.6 Ranking and confidence

Assignments are compared with an explicit evidence vector rather than an
uncalibrated probability:

```text
hard-constraint violations
-> IUPAC allocation conflicts
-> unresolved resonance/non-innocent alternatives
-> curated compound/motif support
-> element-candidate evidence tier
-> calibrated bond-valence residual
```

| `OxidationResolutionStatus` | Required interpretation |
|---|---|
| `UNIQUE_ASSIGNMENT` | One assignment survives hard constraints and independent evidence agrees |
| `AMBIGUOUS` | Multiple assignments remain scientifically plausible |
| `UNRESOLVED` | Available facts cannot construct a defensible assignment |
| `CONFLICT` | Explicit facts or overrides violate charge/electron constraints |

`solver_status` separately records whether enumeration was complete, truncated
or unsatisfiable; `chemical_confidence` records evidentiary support. Numerical
probabilities are forbidden until calibrated against held-out, curated
compounds and reported with coverage/error statistics.

### 6.7 Pseudocode

```python
def infer_oxidation_states(
    molecular_facts,
    net_charge_result,
    lewis_result,
    *,
    electron_counting_results,
    overrides,
    policy,
):
    require_matching_graph_identity(
        molecular_facts,
        net_charge_result,
        lewis_result,
        *electron_counting_results,
    )
    explicit_constraints = compile_overrides(overrides)

    assignments = []
    for net_candidate in net_charge_result.candidates:
        lewis_allocations = matching_lewis_allocations(
            lewis_result,
            net_candidate.identity,
        )
        counting_assignments = compatible_counting_assignments(
            electron_counting_results,
            net_candidate.identity,
            lewis_allocations,
        )
        for state_path in align_evidence_paths(
            lewis_allocations,
            counting_assignments,
            explicit_constraints,
        ):
            atom_domains = initialize_empty_domains(molecular_facts)
            allocation = state_path.lewis_allocation
            if allocation is not None:
                iupac_assignment = ionic_approximation(
                    molecular_facts,
                    formal_charges=allocation.atom_formal_charges,
                    bond_orders=allocation.localized_bond_orders,
                    electronegativity=policy.electronegativity_scale,
                    excluded_coordination_edges=state_path.partitioned_edges,
                )
                add_assignment_to_domains(atom_domains, iupac_assignment)

            add_noninnocent_alternatives(
                atom_domains,
                molecular_facts,
                allocation,
                state_path.coordination_counting,
                policy.motif_registry,
            )
            if policy.allow_reference_expansion:
                expand_supported_domains(atom_domains, policy.element_registry)

            assignments.extend(
                solve_charge_constrained_assignments(
                    atom_domains,
                    total_charge=net_candidate.charge,
                    original_component_charges=(
                        allocation.connected_components
                        if allocation is not None
                        else ()
                    ),
                    coordination_counting=state_path.coordination_counting,
                    coordination_counting_result_identity=(
                        state_path.coordination_counting_result_identity
                    ),
                    coordination_counting_assignment_identity=(
                        state_path.coordination_counting_assignment_identity
                    ),
                    net_charge_candidate_identity=net_candidate.identity,
                    lewis_allocation_identity=(
                        allocation.identity if allocation is not None else None
                    ),
                    explicit_constraints=explicit_constraints,
                )
            )

        if not lewis_allocations and not counting_assignments and not overrides:
            record_unresolved_path(net_candidate, "insufficient electron evidence")

    assignments = canonicalize_equivalent_centers(assignments, molecular_facts)
    evidence = evaluate_iupac_motif_and_geometry_evidence(
        assignments,
        molecular_facts,
        net_charge_result,
        lewis_result,
        electron_counting_results,
        policy,
    )
    ranked = rank_by_evidence_vector(assignments, evidence)
    return build_result_without_forcing_unique_selection(ranked)
```

## 7. Worked examples

| Structure | Required conclusion |
|---|---|
| CH4 | C $-4$, each H $+1$ from direct bond-electron allegiance |
| H2O2 | Each O $-1$; the O-O homonuclear bond distinguishes peroxide from water |
| Fe(CO)5 | Neutral CO ligands and total charge zero support Fe $0$ despite Fe(II/III) being common |
| [Fe(CN)6]4- | Six CN- ionic bookkeeping assignments plus $Q=-4$ support Fe $+2$ |
| Eu(III) complex | Eu $+3$ only after a selected ionic coordination-counting assignment and $Q$ make that state consistent; the element symbol alone is insufficient |
| NH3->BF3 | Full-graph N+/B- formal charges and neutral donor-return fragments are different bookkeeping views and must not overwrite one another |
| Mixed-valence equivalent metal sites | Return a delocalized/equivalence-class candidate unless site-specific evidence resolves localization |
| Redox-active ligand complex | Return metal/ligand oxidation alternatives rather than forcing all redox charge onto the metal |

## 8. Planned module boundary

```text
hotpot/cheminfo/calculator/
├── contracts.py                  shared immutable MolecularFacts only
├── charge/                       independent upstream package
├── oxidation_state/
│   ├── __init__.py               public contracts and infer function
│   ├── contracts.py              immutable results, status and provenance
│   ├── registry.py               versioned candidate/electronegativity loader
│   ├── ionic_approximation.py    IUPAC bond-electron allegiance
│   ├── ligand_electron_counting.py
│   │                             ionic, donor-return and CBC contracts
│   ├── motifs.py                 referenced non-innocent alternatives
│   ├── bond_valence.py           optional calibrated geometry evidence
│   ├── inference.py              constraint solving and ranking
│   └── data/
│       ├── element_oxidation_states.json
│       └── SOURCES.md            releases, URLs, hashes and applicability
├── electronic_state/             independent downstream package
└── state_resolver.py             sole orchestration layer
```

Dependency direction:

```text
MolecularFacts + NetChargeResult + optional LewisFormalChargeResult
                 |
                 +--> ligand_electron_counting(convention)
                 |        |
                 |        v
                 |   CoordinationElectronCountingResult tuple
                 |        |
                 +--------+
                          v
              oxidation_state.inference
                          |
                          v
             OxidationStateInferenceResult
```

The module exports a C/Python-independent scientific contract. A future C++
solver may implement the hot loop, but Python and C++ entry points must share
one backend and identical result semantics.

## 9. Validation requirements

- exact charge-sum invariance for every returned assignment;
- equivalent Kekule/resonance forms produce the same candidate set;
- atom renumbering changes only mapped indices, not chemistry;
- explicit contradictory overrides return `CONFLICT`;
- peroxides, superoxides, hydrides, metal carbonyls and ordinary salts;
- neutral, cationic and anionic coordination complexes;
- Eu(II)/Eu(III), Fe(II)/Fe(III), mixed-valence and multimetal examples;
- NO/O2/dithiolene/quinone non-innocent alternatives;
- optional bond-valence evidence tested only inside its declared domain;
- ionic, donor-return and CBC views may differ without changing $Q$ or the
  intact-graph Lewis result;
- CBC L/X/Z output is never substituted for an ionic ligand charge;
- every result identity binds the molecular graph, net-charge candidate,
  optional Lewis allocation and counting convention;
- whole-graph and partitioned paths never count one coordination electron pair
  twice;
- missing Lewis evidence or a required counting convention yields ambiguity or
  `UNRESOLVED`, never an implicit cleavage default;
- all 118 element rows parse, retain source tier and never imply an automatic
  default;
- cross-check a held-out curated set against at least two independent
  assignments or expert review, reporting coverage separately from accuracy.

## 10. Known limits

- Oxidation state is model-dependent in highly delocalized, metallic and
  strongly covalent systems.
- A molecular graph without reliable bond order, hydrogens or total charge may
  be insufficient.
- Coordination partitioning, haptic/multicentre bonding, metal--ligand multiple
  bonds and non-innocent ligands remain convention- or model-dependent. A
  unique $Q$ does not imply a unique oxidation state or ligand electron count.
- Bond-valence analysis is not universally transferable from inorganic solids
  to organometallic and f-element complexes.
- Superheavy-element candidates are largely predictions.
- A resolved oxidation state does not determine spin, multiplicity or the
  physical ground electronic state.

## 11. References

1. P. Karen et al., “Toward a comprehensive definition of oxidation state,”
   *Pure Appl. Chem.* **2014**, 86, 1017-1081.
   DOI: `10.1515/pac-2013-0505`.
2. P. Karen et al., “Comprehensive definition of oxidation state (IUPAC
   Recommendations 2016),” *Pure Appl. Chem.* **2016**, 88, 831-839.
   DOI: `10.1515/pac-2015-1204`.
3. L. C. Allen, “Electronegativity is the average one-electron energy of the
   valence-shell electrons in ground-state free atoms,” *J. Am. Chem. Soc.*
   **1989**, 111, 9003-9014. DOI: `10.1021/ja00207a003`.
4. I. D. Brown and D. Altermatt, “Bond-valence parameters obtained from a
   systematic analysis of the Inorganic Crystal Structure Database,”
   *Acta Crystallogr. B* **1985**, 41, 244-247.
   DOI: `10.1107/S0108768185002063`.
5. N. E. Brese and M. O'Keeffe, “Bond-valence parameters for solids,”
   *Acta Crystallogr. B* **1991**, 47, 192-197.
   DOI: `10.1107/S0108768190011041`.
6. Royal Society of Chemistry, “Periodic Table,” element oxidation-state
   summaries, accessed 2026-10-12. <https://periodic-table.rsc.org/>.
7. D. W. Davies et al., “Computer-aided design of metal chalcohalide
   semiconductors: from chemical composition to crystal structure,”
   *Faraday Discuss.* **2018**, 211, 553-568.
   DOI: `10.1039/C8FD00032H`.
8. D. W. Davies et al., “SMACT: Semiconducting Materials by Analogy and
   Chemical Theory,” *J. Open Source Softw.* **2019**, 4, 1361.
   DOI: `10.21105/joss.01361`.
9. O. R. Luca and R. H. Crabtree, “Redox-active ligands in catalysis,”
   *Chem. Soc. Rev.* **2013**, 42, 1440-1459.
   DOI: `10.1039/C2CS35228A`.
10. S. P. Ong et al., “Python Materials Genomics (pymatgen),”
   *Comput. Mater. Sci.* **2013**, 68, 314-319.
   DOI: `10.1016/j.commatsci.2012.10.028`.
11. S. Kim et al., “PubChem 2023 update,” *Nucleic Acids Res.* **2023**, 51,
   D1373-D1380. DOI: `10.1093/nar/gkac956`.
12. Mendeleev project documentation, versioned elemental-property database,
   <https://mendeleev.readthedocs.io/>.
13. IUPAC Gold Book entries “oxidation state” and “formal charge,” current
   online edition, accessed 2026-10-12.
14. M. L. H. Green, “A new approach to the formal classification of covalent
   compounds of the elements,” *J. Organomet. Chem.* **1995**, 500, 127-148.
   DOI: `10.1016/0022-328X(95)00508-N`.

The machine-readable element registries in Section 5 are reproducibility
inputs, not substitutes for the normative IUPAC definitions or compound-level
validation. CBC is a coordination electron-counting convention, not a source
of system charge or a universal ionic ligand charge.
