# Electronic-state inference: principles and implementation contract

> **OLDER / NON-NORMATIVE.** This design record is retained for equations,
> data provenance and historical alternatives. The active implementation
> contract is
> [`scientific_state_implementation.md`](../scientific_state_implementation.md)
> and the normative specifications indexed there.

Status: implementation specification; no production-code change is made by
this document.

This document specifies molecular spin-state inference and the conversion to
the xTB `UHF` execution parameter. It is deliberately separate from:

- [charge inference](charge_inference.md), which separately owns net-charge
  candidates and optional intact-graph Lewis allocations;
- [oxidation-state inference](oxidation_state_inference.md), which owns formal
  oxidation-state assignments and their alternatives.

The combined failure/recovery policy is documented in
[the xTB root-cause report](xtb_failure_root_cause_and_recovery_plan.md).

## 1. Scope and ownership

Electronic-state inference consumes immutable molecular facts and completed
upstream results. It must not load or mutate a `Molecule`, infer charge, or
invoke oxidation-state inference internally.

| Contract | This module |
|---|---|
| Owns | Electron count derived from upstream facts; allowed molecular UHF/multiplicity candidates; local-spin coupling; parity filtering; explicit-state reconciliation; confidence, alternatives and provenance; conversion to backend parameters |
| Consumes | One immutable `MolecularFacts`; one immutable `NetChargeResult`; optional `LewisFormalChargeResult` and `OxidationStateInferenceResult`; optional explicit UHF or multiplicity |
| Must not decide | Net, connected-component, formal or partial charge; oxidation state; protonation; aromaticity; bond order; coordination topology; geometry quality; whether a converged state is the physical ground state |

The planned upstream contracts must contain enough immutable evidence for this
module to remain molecule-independent:

```text
MolecularFacts
├── molecular_identity
├── atomic_numbers
├── explicit radical/local-state evidence, when authoritative
└── nuclear_charge_sum

NetChargeResult
├── molecular_identity
├── candidates: NetChargeCandidate
│   ├── identity
│   └── charge
└── selected_candidate / statuses / confidence

optional LewisFormalChargeResult
└── candidates: LewisElectronAllocation
    ├── identity
    ├── net_charge_candidate_identity
    └── radical_electrons

optional OxidationStateInferenceResult
├── molecular_identity
├── assignments
│   ├── net_charge_candidate_identity
│   ├── optional lewis_allocation_identity
│   └── center and ligand-redox evidence
└── provenance / confidence / assumptions
```

Every supplied input must carry the same molecular identity hash; a mismatch is
an orchestration error, not an inference branch. Ionic configurations and their
spin candidates are derived here from element plus oxidation state, so the
oxidation-state module does not own a hidden spin model. Lewis evidence is
needed only when its radical/nonbonding-electron or ligand-redox alternatives
change the electronic-state path; resonance localization alone does not.

## 2. UHF and spin multiplicity

xTB's command-line parameter named `UHF` is the number of unpaired electrons
represented as the spin-occupation difference

$$
u = N_{\alpha}-N_{\beta}.
$$

It does **not** request the unrestricted Hartree--Fock method. For a pure spin
state represented by its highest spin projection, $M_S=S$,

$$
u=2S=M-1,
\qquad
M=2S+1=u+1,
$$

where $M$ is spin multiplicity. Therefore multiplicity and xTB UHF carry the
same scalar information under this assumption:

| Multiplicity $M$ | Total spin $S$ | xTB UHF $u$ |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | $1/2$ | 1 |
| 3 | 1 | 2 |
| 4 | $3/2$ | 3 |
| 5 | 2 | 4 |
| 7 | 3 | 6 |

xTB 6.7.1 accepts `--uhf INT` or `$spin INT`; it has no separate
spin-multiplicity CLI option. Hotpot may accept either human-facing form, but
must convert multiplicity to `u=M-1` at the adapter boundary. Neither form is
more accurate or faster once converted.

One scalar cannot describe broken-symmetry antiferromagnetic states, local
spin coupling in multinuclear complexes, spin contamination or strong
spin--orbit coupling. Ordinary GFN1/2-xTB convergence also does not prove that
the selected candidate is the physical ground state. Cross-state ranking
requires a method with an applicable spin-polarized energy model, such as an
explicitly validated tblite spin-polarization route, or a higher-level method.

## 3. Mathematical constraints

Given the upstream nuclear-charge sum $Z_{\Sigma}$ and total molecular charge
$Q$, the molecular electron count is

$$
N_e=Z_{\Sigma}-Q.
$$

Every candidate $u$ must satisfy

$$
0\le u\le N_e,
\qquad
(N_e-u)\bmod 2=0,
$$

so that both occupations are non-negative integers:

$$
N_{\alpha}=\frac{N_e+u}{2},
\qquad
N_{\beta}=\frac{N_e-u}{2}.
$$

For a single $d^n$ or $f^n$ open shell, the maximum number of unpaired
electrons is

$$
u_{\max}^{(d)}=\min(n,10-n),
\qquad
u_{\max}^{(f)}=\min(n,14-n).
$$

The spin-only candidates descend by two:

$$
\mathcal U_k=\{u_{\max},u_{\max}-2,\ldots,u_{\min}\},
$$

where $u_{\min}=0$ for even occupancy and $u_{\min}=1$ for odd occupancy.
This generates possible spin occupations; it does not select high spin or low
spin without additional evidence.

For two local centers $A$ and $B$, with $u_A=2S_A$ and $u_B=2S_B$, allowed
spin coupling gives

$$
\mathcal U_{AB}=\{|u_A-u_B|,|u_A-u_B|+2,\ldots,u_A+u_B\}.
$$

The same operation is applied recursively to further metal centers and ligand
radicals. Molecular parity filtering is always the final mathematical step.

### 3.1 Ionic-configuration and local-spin evidence

An oxidation-state integer does not by itself define a subshell occupation.
The implementation therefore uses a versioned, source-tagged registry keyed by
`(atomic_number, oxidation_state)`:

```text
IonicConfigurationRecord
├── atomic_number and oxidation_state
├── electron_configuration
├── evaluated ground term, when available
├── allowed spin-only UHF candidates
├── adopted execution prior or None
├── evidence tier and applicability notes
└── source, release/access date and content hash
```

Evaluated atomic/ionic levels from NIST ASD are the primary evidence where
available. Neutral records use the table in Section 5. Curated ion records
encode known electron-removal exceptions explicitly: for example, transition-
metal cations normally lose the outer $ns$ electrons before $(n-1)d$, but this
rule is not treated as a universal configuration oracle. A simple Aufbau
subtraction may propose a low-confidence candidate for an unregistered
main-group ion; an unregistered transition-metal, lanthanide, actinide or
mixed-configuration ion remains unresolved rather than receiving a fabricated
configuration.

For a registered configuration, open-shell occupancy generates all
parity-compatible local candidates by the equations above. Ranking is an
evidence vector, not a universal high-spin/low-spin switch:

1. explicit or experimentally curated local state;
2. evaluated ionic ground term in the declared environment, when transferable;
3. independently validated ligand-field/coordination evidence;
4. family-specific execution prior, such as maximum-spin $4f^n$ for many
   lanthanide ions, retained only with `medium` confidence and alternatives;
5. otherwise no selected prior.

For $d^n$ centers, geometry and ligand identity alone do not silently select
high or low spin. For $f^n$ centers, strong spin--orbit coupling means a
spin-only UHF remains an execution parameter, not a complete $J$-state model.
`rank_execution_prior()` implements only this versioned evidence ordering. If
the required registry/evidence record is absent, it returns no selection.

## 4. Inference policy

The decision order is:

```text
explicit UHF
  -> explicit multiplicity
  -> upstream authoritative local-state evidence
  -> oxidation-state-derived ionic configurations and ligand radicals
  -> parity-only execution default for a proven closed-shell main-group case
  -> unresolved candidate set
```

### 4.1 Confidence and ambiguity

Confidence describes the evidence supporting the inference; it is not a
statistical probability.

| Level | Required evidence | Selection behavior |
|---|---|---|
| `authoritative` | Consistent explicit UHF/multiplicity from the user or a trusted upstream calculation | Select the stated value; retain its provenance |
| `high` | Upstream oxidation assignment is unique and the local configuration has only one parity-compatible spin candidate | Select; retain the empty alternative set |
| `medium` | A chemically motivated execution prior ranks one candidate, but alternatives remain physically possible | Return selected prior and all alternatives; label it an execution prior |
| `low` | Oxidation state, redox localization, ligand radicals or local coupling remain ambiguous | Do not silently select a physical state; return ordered candidates and reasons |
| `unresolved` | Upstream results are incomplete/inconsistent or no candidate survives parity | Return no selected state; require explicit input or a state scan |

Rules:

1. Explicit UHF and multiplicity supplied together must satisfy $u=M-1$.
2. Different `NetChargeCandidate` values always remain distinct. Lewis
   alternatives propagate only when radical/local-spin or ligand-redox evidence
   differs; resonance-localization alternatives with identical spin evidence
   do not create artificial UHF candidates. Oxidation-state ambiguity
   propagates whenever it changes a local-spin path.
3. A neutral isolated-atom prior is never summed across atoms.
4. A high-spin lanthanide prior is an execution starting point, not a ground-
   state declaration.
5. Candidate energies from GFN1/2 calculations without a separately validated
   spin-polarization and state-ranking protocol must not be presented as a
   validated spin-state ordering.

### 4.2 Planned inference flow

```text
MolecularFacts -----------------------------+
NetChargeResult ----------------------------+
optional LewisFormalChargeResult -----------+--> verify identity and evidence
optional OxidationStateInferenceResult -----+                 |
                                                              v
                         for each aligned upstream pair
                              compute Ne = Zsum - Q
                                            |
              +-----------------------------+--------------------------+
              |                                                        |
       explicit UHF/multiplicity                                no explicit state
              |                                                        |
       reconcile and parity-check                    expand each upstream oxidation
              |                                      assignment to local spin sets
              |                                                        |
              |                                         add ligand-radical spins
              |                                                        |
              |                                         couple local spin sets
              |                                                        |
              +-------------------------------> molecular parity filter
                                                                       |
                                              +------------------------+--------+
                                              |                                 |
                                      unique supported state              alternatives remain
                                              |                                 |
                                   resolved result + provenance       unresolved/ranked candidates
                                                                                |
                                                                  explicit choice or state scan
```

### 4.3 Pseudocode

```python
def infer_electronic_states(
    molecular_facts,
    net_charge_result,
    *,
    lewis_result=None,
    oxidation_result=None,
    explicit_uhf=None,
    explicit_multiplicity=None,
):
    require_same_molecular_identity(
        molecular_facts,
        net_charge_result,
        lewis_result,
        oxidation_result,
    )
    candidate_records = []
    for state_path in align_upstream_paths(
        net_charge_result,
        lewis_result=lewis_result,
        oxidation_result=oxidation_result,
    ):
        net_charge_candidate = state_path.net_charge_candidate
        electron_count = (
            molecular_facts.nuclear_charge_sum - net_charge_candidate.charge
        )

        if explicit_uhf is not None or explicit_multiplicity is not None:
            uhf = reconcile_explicit_spin(explicit_uhf, explicit_multiplicity)
            if not electron_parity_is_valid(uhf, electron_count):
                record_incompatible_upstream_candidate(
                    state_path,
                    uhf,
                )
                continue
            candidate_records.append(
                authoritative_candidate(
                    uhf,
                    electron_count,
                    state_path,
                )
            )
            continue

        local_sets = tuple(
            ionic_configuration_spin_candidates(center)
            for center in state_path.center_states
        )
        local_sets += infer_ligand_spin_candidates(
            molecular_facts,
            state_path.radical_and_ligand_redox_evidence,
        )
        coupled = couple_local_spin_sets(local_sets)
        candidates = filter_electron_parity(coupled, electron_count)
        candidate_records.extend(
            attach_upstream_provenance(candidates, state_path)
        )

    candidates = merge_same_scientific_paths(candidate_records)

    if has_one_high_confidence_candidate(candidates):
        return resolved_result(candidates, selected_candidate=0)

    if has_justified_execution_prior(candidates):
        return ambiguous_result(
            selected_candidate=rank_execution_prior(candidates),
            candidates=candidates,
            chemical_confidence="medium",
        )

    return ambiguous_result(
        selected_candidate=None,
        candidates=candidates,
        chemical_confidence="low" if candidates else "unresolved",
        recommended_action="explicit state or spin-polarized state scan",
    )
```

The function orchestrator, not this function, decides whether an unresolved
result blocks a requested backend or expands into independent state-scan runs.
Every path carries its net-charge candidate identity and, when applicable, its
Lewis allocation and oxidation-assignment identities. Scientific candidates
are not merged merely because they share one UHF value; an execution scheduler
may separately deduplicate identical $(Q,u)$ backend jobs while preserving all
source paths. `selected_candidate` is an index into the full typed candidate
tuple and therefore retains all charge, oxidation, electron-count and evidence
provenance.

## 5. Neutral isolated-atom UHF priors

This table is a **neutral isolated-atom prior table**, not a molecular default
and not an xTB element-support table. Candidate values are spin-only UHF
values derived from adopted neutral ground configurations and open-subshell
coupling. The default is $u=2S$ from the evaluated neutral ground term when a
sufficiently stable assignment is available. Empty cells are intentionally
unresolved; predicted superheavy configurations are not presented as measured
facts.

For one open subshell $k$ with occupancy $n_k$ and capacity
$g_k=2(2l_k+1)$,

$$
u_{\max,k}=\min(n_k,g_k-n_k).
$$

If $\mathcal S_k$ contains the allowed subshell spins, multiple open subshells
are coupled recursively:

$$
\mathcal C_1=\mathcal S_1,
\qquad
\mathcal C_{k+1}=\bigcup_{S_a\in\mathcal C_k,\,S_b\in\mathcal S_{k+1}}
\{|S_a-S_b|,|S_a-S_b|+1,\ldots,S_a+S_b\},
$$

$$
\mathcal U_{\mathrm{atom}}=\{2S:S\in\mathcal C_m\}.
$$

| Z | Element | Neutral-configuration UHF candidates | Neutral-atom prior |
|---:|---|---|---:|
| 1 | H | 1 | 1 |
| 2 | He | 0 | 0 |
| 3 | Li | 1 | 1 |
| 4 | Be | 0 | 0 |
| 5 | B | 1 | 1 |
| 6 | C | 0, 2 | 2 |
| 7 | N | 1, 3 | 3 |
| 8 | O | 0, 2 | 2 |
| 9 | F | 1 | 1 |
| 10 | Ne | 0 | 0 |
| 11 | Na | 1 | 1 |
| 12 | Mg | 0 | 0 |
| 13 | Al | 1 | 1 |
| 14 | Si | 0, 2 | 2 |
| 15 | P | 1, 3 | 3 |
| 16 | S | 0, 2 | 2 |
| 17 | Cl | 1 | 1 |
| 18 | Ar | 0 | 0 |
| 19 | K | 1 | 1 |
| 20 | Ca | 0 | 0 |
| 21 | Sc | 1 | 1 |
| 22 | Ti | 0, 2 | 2 |
| 23 | V | 1, 3 | 3 |
| 24 | Cr | 0, 2, 4, 6 | 6 |
| 25 | Mn | 1, 3, 5 | 5 |
| 26 | Fe | 0, 2, 4 | 4 |
| 27 | Co | 1, 3 | 3 |
| 28 | Ni | 0, 2 | 2 |
| 29 | Cu | 1 | 1 |
| 30 | Zn | 0 | 0 |
| 31 | Ga | 1 | 1 |
| 32 | Ge | 0, 2 | 2 |
| 33 | As | 1, 3 | 3 |
| 34 | Se | 0, 2 | 2 |
| 35 | Br | 1 | 1 |
| 36 | Kr | 0 | 0 |
| 37 | Rb | 1 | 1 |
| 38 | Sr | 0 | 0 |
| 39 | Y | 1 | 1 |
| 40 | Zr | 0, 2 | 2 |
| 41 | Nb | 1, 3, 5 | 5 |
| 42 | Mo | 0, 2, 4, 6 | 6 |
| 43 | Tc | 1, 3, 5 | 5 |
| 44 | Ru | 0, 2, 4 | 4 |
| 45 | Rh | 1, 3 | 3 |
| 46 | Pd | 0 | 0 |
| 47 | Ag | 1 | 1 |
| 48 | Cd | 0 | 0 |
| 49 | In | 1 | 1 |
| 50 | Sn | 0, 2 | 2 |
| 51 | Sb | 1, 3 | 3 |
| 52 | Te | 0, 2 | 2 |
| 53 | I | 1 | 1 |
| 54 | Xe | 0 | 0 |
| 55 | Cs | 1 | 1 |
| 56 | Ba | 0 | 0 |
| 57 | La | 1 | 1 |
| 58 | Ce | 0, 2 | 0 |
| 59 | Pr | 1, 3 | 3 |
| 60 | Nd | 0, 2, 4 | 4 |
| 61 | Pm | 1, 3, 5 | 5 |
| 62 | Sm | 0, 2, 4, 6 | 6 |
| 63 | Eu | 1, 3, 5, 7 | 7 |
| 64 | Gd | 0, 2, 4, 6, 8 | 8 |
| 65 | Tb | 1, 3, 5 | 5 |
| 66 | Dy | 0, 2, 4 | 4 |
| 67 | Ho | 1, 3 | 3 |
| 68 | Er | 0, 2 | 2 |
| 69 | Tm | 1 | 1 |
| 70 | Yb | 0 | 0 |
| 71 | Lu | 1 | 1 |
| 72 | Hf | 0, 2 | 2 |
| 73 | Ta | 1, 3 | 3 |
| 74 | W | 0, 2, 4 | 4 |
| 75 | Re | 1, 3, 5 | 5 |
| 76 | Os | 0, 2, 4 | 4 |
| 77 | Ir | 1, 3 | 3 |
| 78 | Pt | 0, 2 | 2 |
| 79 | Au | 1 | 1 |
| 80 | Hg | 0 | 0 |
| 81 | Tl | 1 | 1 |
| 82 | Pb | 0, 2 |  |
| 83 | Bi | 1, 3 | 3 |
| 84 | Po | 0, 2 | 2 |
| 85 | At | 1 | 1 |
| 86 | Rn | 0 | 0 |
| 87 | Fr | 1 | 1 |
| 88 | Ra | 0 | 0 |
| 89 | Ac | 1 | 1 |
| 90 | Th | 0, 2 | 2 |
| 91 | Pa | 1, 3 | 3 |
| 92 | U | 0, 2, 4 | 4 |
| 93 | Np | 1, 3, 5 | 5 |
| 94 | Pu | 0, 2, 4, 6 | 6 |
| 95 | Am | 1, 3, 5, 7 | 7 |
| 96 | Cm | 0, 2, 4, 6, 8 | 8 |
| 97 | Bk | 1, 3, 5 | 5 |
| 98 | Cf | 0, 2, 4 | 4 |
| 99 | Es | 1, 3 | 3 |
| 100 | Fm | 0, 2 | 2 |
| 101 | Md | 1 | 1 |
| 102 | No | 0 | 0 |
| 103 | Lr | 1 | 1 |
| 104 | Rf | 0, 2 | 2 |
| 105 | Db | 1, 3 | 3 |
| 106 | Sg | 0, 2, 4 |  |
| 107 | Bh | 1, 3, 5 |  |
| 108 | Hs | 0, 2, 4 |  |
| 109 | Mt |  |  |
| 110 | Ds |  |  |
| 111 | Rg |  |  |
| 112 | Cn |  |  |
| 113 | Nh |  |  |
| 114 | Fl |  |  |
| 115 | Mc |  |  |
| 116 | Lv |  |  |
| 117 | Ts |  |  |
| 118 | Og |  |  |

Pb has no default here because its strongly mixed neutral ground state is not
represented by a single robust spin-only LS prior. Sg--Hs retain configuration
candidates but no adopted default; Mt--Og remain empty. The table likewise
does not imply that a selected xTB backend supports all listed elements.

The Eu counterexample shows why neutral priors are not molecular defaults:

```text
neutral Eu atom: 4f7 6s2 -> neutral prior UHF=7
Eu(III):          4f6     -> candidates 0, 2, 4, 6; execution prior UHF=6
```

## 6. Planned typed API and module boundary

The current `infer_lowest_spin(mol, charge)` reads `Molecule` directly and
returns `electron_count % 2`. It remains useful only as an explicitly named
lowest-spin parity execution policy. It is not the target molecular-state
inference API and must not silently stand in for it in metal complexes.

The three scientific quantities are sibling packages. The orchestration layer
is the only code allowed to call them in sequence:

```text
hotpot/cheminfo/calculator/
├── contracts.py              # shared immutable MolecularFacts only
├── charge/                   # independent charge implementation
├── oxidation_state/          # independent oxidation-state implementation
├── electronic_state/
│   ├── __init__.py           # stable re-exports only
│   ├── contracts.py          # immutable electronic-state contracts
│   ├── spin.py               # UHF/multiplicity inference
│   ├── coupling.py           # local-spin coupling
│   ├── ionic_configuration.py
│   └── data/
│       ├── neutral_atom_spin_priors.py
│       └── ionic_configurations.json
└── state_resolver.py         # sole orchestration layer
```

Planned types and signatures:

```python
@dataclass(frozen=True)
class ElectronicStateCandidate:
    net_charge_candidate_identity: str
    lewis_allocation_identity: str | None
    oxidation_assignment_identity: str | None
    electron_count: int
    uhf: int
    multiplicity: int
    chemical_confidence: InferenceConfidence
    provenance: tuple[str, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ElectronicStateInferenceResult:
    candidates: tuple[ElectronicStateCandidate, ...]
    selected_candidate: Optional[int]
    solver_status: SolverStatus
    resolution_status: ElectronicResolutionStatus
    chemical_confidence: InferenceConfidence
    ambiguities: tuple[str, ...]
    molecular_identity: str


def infer_electronic_states(
    molecular_facts: MolecularFacts,
    net_charge_result: NetChargeResult,
    *,
    lewis_result: LewisFormalChargeResult | None = None,
    oxidation_result: OxidationStateInferenceResult | None = None,
    explicit_uhf: Optional[int] = None,
    explicit_multiplicity: Optional[int] = None,
) -> ElectronicStateInferenceResult: ...


def uhf_from_multiplicity(multiplicity: int) -> int: ...


def multiplicity_from_uhf(uhf: int) -> int: ...
```

`ElectronicResolutionStatus` is module-specific and uses
`UNIQUE_ASSIGNMENT`, `AMBIGUOUS`, `UNRESOLVED` and `CONFLICT`; it is not the
charge module's value/equivalence-class status.

The sibling inference packages may depend on shared immutable fact/provenance
protocols, but never on each other's implementation modules. `spin.py` may
import only the upstream result contracts it consumes, `MolecularFacts`, and
its own prior data. The pipeline or `state_resolver.py` calls the three modules
in order and owns all control flow. This prevents hidden circular inference
and makes every intermediate result independently testable.

## 7. Worked examples

### 7.1 Closed-shell water

Upstream results provide $Z_{\Sigma}=10$ and $Q=0$, so $N_e=10$. There is no
open-shell or redox ambiguity. The only candidate is `UHF=0`, multiplicity 1,
with high confidence.

### 7.2 Nitric oxide radical

Upstream charge is zero and radical bookkeeping identifies one unpaired
electron. $N_e=15$ is odd, so `UHF=1`, multiplicity 2 is parity-compatible.
The electronic-state module consumes the radical fact; it does not rediscover
it from bond orders.

### 7.3 Eu(III) coordination complex

The oxidation-state result assigns Eu(III), hence an $4f^6$ ionic
configuration. Spin-only candidates are `UHF = 0, 2, 4, 6`. `UHF=6` may be
returned as a medium-confidence lanthanide execution prior while retaining
`0, 2, 4` as alternatives. The retained benchmark showed that `UHF=6`
recovered 21/22 formerly failing direct-GFN2 executions, but that causal
stability evidence is not proof of a universal Eu(III) ground state.

### 7.4 Fe(II) $d^6$

The candidate set is `UHF = 0, 2, 4`. Without upstream ligand-field or
authoritative state evidence, no unique physical state is selected. A silent
lowest-spin default would incorrectly erase the high-spin alternative.

### 7.5 Two local doublets

Two local `UHF=1` centers can couple to molecular `UHF=0` or `UHF=2`.
Returning only `UHF=2` by summing local unpaired electrons would discard the
antiparallel coupling state.

## 8. Validation requirements

### 8.1 Unit tests

1. Verify `multiplicity == uhf + 1` for every accepted explicit state.
2. Verify electron parity and integral $N_\alpha,N_\beta$ for every candidate.
3. Verify the neutral-atom table contains exactly Z=1--118 in order; blanks
   remain explicit blanks and are never coerced to zero.
4. Verify representative shell candidates: C, Cr, Fe, Eu, Gd, U and Am.
5. Verify Eu(III) produces `0, 2, 4, 6`, not the neutral-Eu default 7.
6. Verify Fe(II) $d^6$ remains ambiguous without external state evidence.
7. Verify local doublet coupling produces singlet and triplet candidates.
8. Verify explicit UHF and multiplicity conflicts are rejected.
9. Verify ambiguous upstream charge/oxidation evidence remains visible.
10. Verify `spin.py` can be tested using constructed result objects and cannot
    call charge or oxidation estimators.
11. Verify authoritative $Q$ plus explicit UHF succeeds without Lewis or
    oxidation-state input.
12. Verify $N_e=Z_{\Sigma}-Q$ independently for every net-charge candidate and
    preserve distinct net-charge identities even when UHF is equal.
13. Verify resonance-localization-only Lewis alternatives do not multiply UHF
    paths, while alternatives with different radical counts do.
14. Verify missing Lewis evidence does not block electron counting, and verify
    mismatched molecular or candidate identities are rejected by orchestration.
15. Verify execution deduplication by $(Q,u)$ never deletes the scientific
    Lewis/oxidation provenance paths.

### 8.2 Integration and scientific tests

1. Freeze `NetChargeResult`, optional `LewisFormalChargeResult`, optional
   `OxidationStateInferenceResult` and electronic-state provenance in every xTB
   workspace manifest.
2. Confirm GFN1/2 receives the selected UHF unchanged at the CLI boundary.
3. Rerun closed-shell main-group, organic-radical, transition-metal,
   lanthanide and multinuclear fixtures.
4. For ambiguous cases, confirm that one run is never reported as a validated
   ground-state ranking; optional state scans retain one independent record per
   candidate.
5. Reproduce the Eu diagnostic while distinguishing execution recovery,
   geometric acceptance and physical-state assignment.

Acceptance requires mathematical correctness, module independence and honest
uncertainty reporting. Convergence rate alone is not an acceptance criterion.

## 9. References and evidence boundary

1. C. Bannwarth, S. Ehlert, S. Grimme, “GFN2-xTB—An Accurate and Broadly
   Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with
   Multipole Electrostatics and Density-Dependent Dispersion Contributions,”
   *J. Chem. Theory Comput.* **2019**, 15, 1652–1671.
   [DOI: 10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176).
2. S. Grimme, C. Bannwarth, P. Shushkov, “A Robust and Accurate Tight-Binding Quantum
   Chemical Method for Structures, Vibrational Frequencies, and Noncovalent
   Interactions of Large Molecular Systems Parametrized for All spd-Block
   Elements (Z=1–86),” *J. Chem. Theory Comput.* **2017**, 13, 1989–2009.
   [DOI: 10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118).
3. xTB documentation, “Commandline Arguments,” definitions of `--uhf` and
   `$spin`.
   [xtb-docs.readthedocs.io](https://xtb-docs.readthedocs.io/en/latest/commandline.html).
4. tblite project documentation, spin-polarization interface and method
   limitations.
   [tblite.readthedocs.io](https://tblite.readthedocs.io/).
5. A. Kramida, Yu. Ralchenko, J. Reader, and NIST ASD Team, NIST Atomic
   Spectra Database, evaluated atomic ground levels and configurations.
   [NIST ASD](https://physics.nist.gov/asd), accessed 2026-10-12.
6. NIST, *Handbook of Basic Atomic Spectroscopic Data*, atomic electronic
   configurations and ground terms.
   [NIST Handbook](https://physics.nist.gov/PhysRefData/Handbook/).
7. IUPAC, “multiplicity (spin multiplicity),” *Compendium of Chemical
   Terminology (Gold Book)*.
   [IUPAC Gold Book](https://goldbook.iupac.org/terms/view/M03988).

References 5 and 6 are the primary evaluated sources for the neutral-atom
defaults. The candidate columns are deterministic spin-only constructions from
adopted configurations, not separately measured lists of every molecular spin
state. Values for ions and molecules require the independently inferred charge
and oxidation-state results; the neutral table must never substitute for them.
