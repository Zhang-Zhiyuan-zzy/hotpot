# Scientific-state implementation status

Date: 2026-10-17

Normative plan:
[`scientific_state_implementation.md`](scientific_state_implementation.md)

Measured pre-activation baseline:
[`scientific_state_pre_activation_baseline.md`](scientific_state_pre_activation_baseline.md)

Branch: `feature/scientific-state-resolution`

## 1. Outcome

The canonical scientific-state system is implemented through Phase 9. One
immutable molecular-facts record now feeds attributable charge, oxidation-state
and electronic-state candidates; hard validators compose them into coupled
paths; a decision layer selects or explicitly abstains; and backend projection
converts only a selected scientific state into method-specific execution
values.

The xTB CLI, pipeline and coordination benchmark consume that same resolver and
projector. GFN-FF receives topology-preserving SDF plus charge, whereas
GFN1/2-xTB receives XYZ plus charge and UHF. Reports and molecular streams bind
the scientific-state and molecular-facts identities instead of reconstructing
state provenance from raw scalars.

Two boundaries remain intentionally open:

1. `COORDINATION_SCREENING_V1` is an implemented but inactive profile artifact.
   Its optional scientific rules are not a production default until individual
   expert approval and held-out activation criteria are satisfied.
2. Phase 10 is incomplete: the stage-resolved coordination benchmark contract
   and a 2280-record pre-activation baseline are implemented, but the
   topology-preserving held-out profile-activation report has not been
   produced.

No neutral-charge, low-spin or alternate-state fallback was introduced to hide
either boundary.

## 2. Phase status

| Phase | Status | Delivered evidence |
|---:|---|---|
| 0 | complete | Frozen normative plan, individually indexed rule/provider/execution/recovery specifications |
| 1 | complete | Immutable facts, identity, status and serializable result contracts |
| 2 | complete | Charge, oxidation-state, molecular-spin, coupled-state and backend-electron hard legality |
| 3 | complete | Explicit evidence, independent providers, strict policy and preserved ambiguity |
| 4 | complete | Rule abstraction, registry, dependency layers, trace and deterministic execution |
| 5 | complete | Factorized candidates, bounded lazy expansion, frontier evidence, coupled composition and certification |
| 6 | implemented; activation pending | Versioned screening artifact, eligibility, decision order and attributable optional rules; production activation remains blocked by scientific review |
| 7 | complete | Backend execution decisions, exact xTB contexts, projections, GFN-FF SDF and GFN-xTB XYZ adapters |
| 8 | complete for safe continuation | Intermediate finite geometry is a warning; final acceptance remains authoritative; typed recovery decisions exist, while unapproved alternate-state retry stops explicitly |
| 9 | complete for xTB consumers | CLI, pipeline, stream and benchmark share one resolver; obsolete flat estimators removed; public exports and runtime matrix tested |
| 10 | partial | Explicit-state coordination benchmark, stage-resolved failure schema and a 2280-record pre-activation baseline delivered; topology-preserving held-out profile activation remains outstanding |

## 3. Execution and failure semantics

The coordination benchmark uses explicit, attributable states until the
screening profile is approved:

| Target | Net charge | UHF |
|---|---:|---:|
| Ligand | 0 | 0 |
| Eu complex | +3 | 6 |

GFN-FF consumes only the selected charge. A following GFN2-xTB stage receives
the carried scientific state and its explicit UHF projection. Intermediate
GFN-FF geometry quality is recorded as a warning and does not by itself block
GFN2-xTB. Final geometry remains a terminal acceptance gate.

Benchmark schema version 3 separates:

- unavailable and invalid source structures;
- scientific-state decision failure;
- backend projection or method-applicability failure;
- other input/provenance failure;
- backend execution/result/timeout failure;
- intermediate geometry warnings; and
- final geometry failure.

This separation avoids interpreting a poor intermediate geometry as an xTB
state failure or treating a scientific abstention as a backend crash.

## 4. Validation performed

| Scope | Result |
|---|---:|
| Calculator, pipeline and xTB focused suite after consumer migration | 564 passed, 8 skipped |
| Installed/runtime compatibility target covering the calculator package | 411 passed |
| xTB plugin plus stage-resolved benchmark contracts | 102 passed, 8 skipped |
| Explicit benchmark-state contract tests | 15 passed |
| Pre-activation structure registry | 2280 / 2280 resolved, 0 replay mismatches |
| Current calculator, xTB and state-benchmark integration suite | 519 passed, 8 skipped |

The maintained full-suite script also reached collection and execution. In the
restricted validation environment its remaining failures were external to this
change: read-only model-cache paths prevented MCA/CBond downloads, and the
local extractant corpus was absent. These conditions are not recorded as a
scientific-state pass.

The current integration suite also closes three previously implicit hard-rule
acceptance points:

- backend projection uses the backend effective-electron count for ECP UHF
  bounds and parity, independently of molecular all-electron legality;
- positive and negative ions receive the same explicit electron-count, $2S$
  bound and parity checks; and
- graph, protonation or component-boundary changes invalidate previously bound
  scientific evidence with `HARD-001.FACTS_IDENTITY_MISMATCH`.

The topology-manifest benchmark path additionally proves that only explicit,
digest-bound atom pairs are promoted to `DATIVE`. Unlisted unknown metal bonds
remain unknown.

## 5. Commit groups

The implementation begins at `8d5708e` and is intentionally split into small,
revertible commits. Its principal groups are:

| Area | Representative commits |
|---|---|
| Facts and scientific contracts | `833fb85`--`5a8ea9b` |
| Hard validators and explicit providers | `1a4a811`--`3102077` |
| Candidate composition and strict resolution | `af10aa0`--`4bceede` |
| Rule engine, bounded search and certification | `804955e`--`a72c874` |
| Screening artifact and optional rule implementations | `d51eb0c`--`c2f81c9` |
| Backend projection and xTB migration | `704c6aa`--`d060337` |
| Pipeline gates, recovery contracts and cleanup | `04073f7`--`19d1fd6` |
| Runtime and benchmark closure | `3cfe48f`--`69382fd` |

## 6. Required next validation

Profile activation is a separate scientific decision, not a code-completion
shortcut. Before changing `COORDINATION_SCREENING_V1` from planned to active:

1. approve each enabled optional rule independently;
2. freeze the effective profile and representative held-out corpora;
3. run the dry resolver at corpus scale and record throughput, peak memory,
   expanded-path statistics, truncation, abstention and replay determinism;
4. report automatic-selection coverage and conflict, out-of-domain and
   uncertified-resource rates against preregistered limits; and
5. rerun the 187/182 execution benchmark with the approved automatic profile,
   retaining the stage-resolved outcome schema.
