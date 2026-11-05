# Automatic coordination-state integration contract

Status: bounded owner-authorized design, 2026-11-05; profile1.1.0 activated by [root decision](activation_decision.json), with actual/runtime delivery evidence pending. Base a2799dba45790a4385fc4d30f8a99cbb869c33a4, branch feat/automatic-coordination-state-workflow.

This maintained contract unblocks parallel implementation. Method/activation detail follows in the adjacent method companion.

| Boundary | Exact bounded contract |
|---|---|
| CBond publication (root) | New predicted metal–donor edges use BondKind.DATIVE and existing order1. Preserve original endpoints, existing source bonds and trained features. On actual new-metal composition only, clear inherited molecular charge/spin declarations via the existing atomic declaration commit. Keep atom formal-charge facts. |
| Source request orchestration (root) | A private `_state_request_from_molecule(mol, facts: MolecularFacts, request: StateRequest) -> StateRequest` belongs in existing `operations/calculation/scientific_state/facts.py`. Read actual present/current declarations and source formal-charge presence without using inferred public getters as facts; explicit invocation values take precedence. Existing `extract_molecular_facts()` remains unchanged in public signature. Source evidence uses existing StateRequest and CandidateProvenance/AtomFormalChargeEvidence/ComponentChargeEvidence. |
| Scientific profile (root/scientific implementation owner) | Reuse StatePolicy.COORDINATION_SCREENING_V1, its manifest and single resolver. Add the smallest generic attributable component/periodic fallback rule only where current rules lack charge/local-spin coverage. Existing specific rules precede common defaults. Freeze finite domains and explicit ranks; retain candidate ambiguity and trace. No second solver/report. Activation requires frozen independent held-out and stress evidence. |
| Backend context (xtb_effective_spin) | Correct effective valence-electron contributions and backend identity by exact xTB version/method convention. Optional hook: `project_scientific_spin(facts: MolecularFacts, two_s: int | None, multiplicity: int | None) -> tuple[int, int] | None`. Preserve existing generic BackendContext implementations: absent hook retains existing highest-MS behavior. For current ordinary GFN1/GFN2 with any f-in-core center, hook returns None (no safe physical-to-active inference); otherwise return the supplied pure-spin pair. GFN-FF stays charge-only. |
| Execution projection (xtb_effective_spin) | Use the safely projected pair for highest-MS comparison/projection only. Keep original physical ScientificStateBundle unchanged. Choose hard-valid explicit UHF before compatible exact-context history, safely projected scientific spin, then lowest legal active-electron UHF (DEFAULT-004). No sum-of-f-spins subtraction. With f-in-core and no safe projection, BackendStateProjection.multiplicity remains None; physical multiplicity remains in scientific state. Correct active parity/range and backend identity are hard gates. |
| Periodic data (root/scientific owner) | Extend the existing oxidation registry with possible/common/default metadata; preserve its current unranked possible-domain API. Distinguish model defaults/predictions from observed states. Neutral isolated atomic spin is a reference, never a table of molecular multiplicities. All118 elements have registry coverage or explicit unavailable metadata; unsupported backends remain unsupported. |
| Consumer wiring (root) | CLI, direct workflow, pipeline, carried state and stream provenance use the same source request/profile manifest/resolver and method-specific projection. Coordinate-only changes preserve scientific identity; H/representation/composition changes require new facts and resolution. Existing serializable bundle, rule trace, execution decision and stream schema carry provenance. |

This contract approves no global mutation change, new public envelope, new scientific result schema, public registry or trained-model change. Exact scientific defaults, fixtures and activation gates must be recorded before their outcome is inspected.

## Immediate connected implementation decisions

Root's 2026-11-05 periodic-data decision is adopted: append defaulted `common_states`, `important_states`, `default_state`, `default_basis` fields to the existing `ElementOxidationStateDomain`; no new public function/type. Preserve the existing `.states` possible-domain meaning. Add the five directly RSC-attributed possible states (Ru0, Rh0, Pd0, Ir0, AuIV) without deleting old entries. A unique source-important state supplies the named general default; multiple important states use the explicitly modeled least-absolute nonzero important state, or common state when importance is absent (Sn/Pt2, Ir1, Xe2). Source-blank/unknown rows use elemental0 as an explicitly modeled reference default, not an observed common oxidation state; Nh's possible domain remains empty. Existing scoped Ln/Am/Cm/Th rules outrank this general table. Elemental0 is not a claim that an unknown coordination center is chemically established in oxidation state0.

The minimal source bridge is constructor-bound immutable evidence, following the existing TrustedHistoryRule precedent; do not make a second charge/OS/spin solver in the source-request helper:

```python
# Existing scientific-state operation owner; facts are captured exactly once.
_state_request_from_molecule(mol, facts: MolecularFacts, request: StateRequest) -> StateRequest

# Existing manifest owner; private additions, existing manifest result reused.
_coordination_screening_manifest(
    trusted_records=(), *,
    source_atom_formal_charges: AtomFormalChargeEvidence | None = None,
    source_component_charges: ComponentChargeEvidence | None = None,
    source_radical_electrons: tuple[int | None, ...] = (),
) -> _ScientificStateRuntimeManifest
```

The operation only binds actual declarations/presence/source, source atom charge evidence and current radical counts to the captured facts. It computes a component charge as a source fact only for complete component coverage. It does not choose metal OS, a modeled total charge, or molecular spin. The manifest binds that immutable evidence into the new pure rule object(s); full evidence/facts digests must enter the enabled rule toggle configuration and candidate/proposal identities so changing source input changes effective configuration and state identity. Source-bound values and their assumptions remain inspectable through existing RuleOutcome/RuleTrace and candidate provenance/serialization. No new RuleContext, StateRequest, MolecularFacts or report field is required.

Use one new provider module with at most two small rules because source binding must precede donor templates while missing-component completion must follow them:

1. A SOURCE_EVIDENCE rule publishes existing typed per-component ionic-counting candidates for complete actual component-charge evidence. It never interprets a raw zero as presence. The existing L/X rules omit components already covered by authoritative source charge, including source0, to prevent duplicate per-component accounting.
2. A REDOX_CANDIDATE rule (the existing phase between coordination counting and OS generation) completes uncovered ligand-component bookkeeping, proposes common/default center OS only where no applicable scoped provider owns that center, and supplies COMPONENT spin candidates for components without a coordinated metal center, including disconnected spectator ions. It reuses existing candidate types, immutable periodic/ionic data and the one composition/coupling engine. It binds parent charge/OS identities so charge, spin and oxidation choices cannot cross independently. Existing ionic-configuration and f-Hund rules continue to provide and rank registered center spin. Fallback center spin is limited to ion paths without a registered ionic model and is explicitly a parity-compatible screening assumption; no neutral-atom spin is substituted for ionic/molecular spin.

For incomplete ligand charge information, the common default is the sum of present supplied site charges plus neutral unspecified sites, under a named representation-default assumption. This preserves charged sites while acknowledging missing charge allocation; it is not complete verified source evidence. When even that accounting boundary is unavailable, the defined neutral-component screening default is attributable and lower priority than an applicable whole-component domain rule. Never infer the entire ligand's charge0 merely because each coordinating donor has neutral valence. Narrow OPTIONAL-003's broad donor-only whole-component neutrality claim accordingly; source-charged components must not retain an incompatible neutral-L0 alternative. The X-type rule similarly cannot override actual source component charges.

For ligand COMPONENT spin, use actual supplied radical information where available; retain spin-coupling alternatives with their explicit parent charge identities. With no stronger spin information, the named component default is the lowest legal electron-parity spin. A supplied nonzero radical count must not be erased into a closed-shell source assertion. Existing coupling enumerates the allowed total spins. After specific local-spin ranks, the screening decision may apply a separately documented lowest-total-two_s execution preference to coupled paths; keep scientific ambiguity and the selected preference trace. This is a declared default decision, not a physical ground-state assertion or an arbitrary atomic-spin table.

Root may implement the connected conversion correction: new CBond DATIVE edges receive donor-to-metal direction; registered Python/Open Babel metal–nonmetal DATIVE projection uses numeric SINGLE while the original Hotpot facts remain authoritative through a Hotpot-initiated roundtrip. Cover `operations/conversion.py`, `adapters/openbabel/conversion.py` and their focused tests; the strict standalone native-wrapper contract is unchanged. Record the lossy external bond-kind projection in this method companion; no hidden label is required. Root may also fix the selected-frame spin projection locally so an attachment marked invalidated_by cannot supply a current molecular spin. The source-request operation always reads actual current declarations, never a public getter's inferred/stale result. No blanket frame deletion or global topology-mutation change is authorized.

## Connected consumer boundary decision (2026-11-05)

Root's reproduced stream failures are in scope. Use the existing closed current-factual codec (`structure/_cif_facts.py::_facts_to_data/_facts_from_data`) over `mol._capture_factual_record(include_frames=False)` and the existing `MoleculeRecord` installer. A private reserved SDF facts property may carry that existing codec data plus an exact structure-block SHA binding. Restore it before xTB metadata facts validation. Write it for Hotpot stream records even when xTB result metadata is absent, so CBond/FF output retains declaration absence, per-atom source/presence, canonical DATIVE kind/direction and supplied radical transport. The external SDF bond representation may be lossy; the authenticated Hotpot facts are authoritative for a Hotpot-written record. A plain external SDF without that property retains its ordinary imported semantics. Never infer DATIVE from arbitrary metal SINGLE bonds. Reuse the codec rather than pickle, a second molecule record, or a full live-session loader. Keep current xTB metadata DTOs; a separate private reserved facts namespace avoids conflating facts-only transport with the requirement for complete xTB result tags. Reject malformed or mismatched bindings. Current frame history and a full persisted `ScientificStateBundle` restart remain outside this correction.

For stream electron parity/range, factor the existing ordinary-GFN effective nuclear-charge arithmetic into one private pure helper at the xTB backend context owner and have both `XTBBackendContext.effective_nuclear_charge` and stream validation call it with the same method/convention. No executable probe belongs in SDF validation. Preserve explicit unsupported-element rejection before evaluating an unavailable active-electron contribution. GFN-FF remains charge-only; a preserved prior GFN UHF must be checked under its recorded GFN spin convention, not all-electron parity. Physical molecular 2S is never replaced by this active count.

For pipeline payload identity, reuse the new operation-owned `_molecule_state_source_identity(mol, facts)` as the raw supplied-state identity coordinate. It covers molecular declaration presence/value/source, atom formal-charge presence/value/source, and radical occurrences/namespaces while excluding geometry. Append that coordinate to the existing payload document; retain existing topology/geometry/bundle coordinates. This is a digest correction in the existing pipeline envelope, not another cache or result type.

The implemented default policy route is contextual: CLI/stage absent `--state-policy` passes `None` to the shared resolver; semantic metal coordination selects `COORDINATION_SCREENING_V1`, ordinary inputs select existing `STRICT`. An explicit policy flag is unchanged. Current source evidence is bound into the two new rule toggle configurations and provenance. Reused in-memory scientific bundles require the same raw supplied-state identity retained as `SOURCE.MOLECULE_STATE_BINDING` in the existing net-charge findings. Coordinate-only changes can reuse the same bundle; changes in source charge, spin, formal-charge provenance or radical presence require resolution.

Stream reconstruction preserves the original source/request authority. CLI now resolves restored source facts first; recorded total charge reconstructs an earlier explicit CLI request only when the exact expected scientific identity matches. Six connected automatic/explicit-Q restart cases pass and invoke one public workflow each. Inherited GFN UHF is not promoted into a GFN-FF request. Source/evidence digests normalize absent backing values/sources toNone; current-factual transport can canonicalize unused numeric storage without changing semantic absence. No scientific identity comparison is relaxed. Full custom request/provider replay is outside this bounded correction.

Pre-activation implementation checkpoint: the source request bridge, OPTIONAL-015/016, source-aware OPTIONAL-002/003/004 v1.1.0, single resolver wiring and contextual CLI/stage defaults are present. All12 preregistered positive holdouts pass under isolated candidate activation; two-/three-center truncation abstains under the unchanged budget, and new outcomes cross existing decision-snapshot JSON exactly. Final connected143 tests pass in6.91s after source-absence, CLI and bound corrections. The earlier broad core run remains675passed/5old frozen-contract failures. Production remained `planned` at this partition's handoff. The [method companion](method_and_activation.md) and [validation record](validation.md) contain exact defaults, source/API effects, artifact identities, timings and remaining root activation/actual/delivery gates.

The source-binding finding now retains `(raw_source_identity, profile_artifact_digest)` for screening, or `(raw_source_identity, "strict")` for strict. This prevents a carried bundle from bypassing a changed profile artifact while allowing a method-only projection change. Existing candidate-space serialization and result envelopes are reused. The two donor rules accept an optional immutable source atom-charge evidence argument; their previous no-argument construction is retained.


Accepted root lifecycle decision2026-11-05: any periodic/prior/default data change used by this profile MUST accompany a profile artifact/version revision, invalidating carried bundles through the existing digest binding. Data are fixed within an invocation/install; no hot-edit API or aggregate data hashing is added. The changed-profile rebind regression covers that boundary. A data-only manual edit without profile revision violates this maintenance contract.


## Observed native convergence correction (2026-11-05)

The frozen actual187 run exposed five zero-exit native optimizations whose
stdout explicitly reports `FAILED TO CONVERGE GEOMETRY OPTIMIZATION` while all
required artifacts, including `.xtboptok`, exist: GFN-FF59/74/135/136 and GFN2
23. The report's artifact/exit-only `converged=True` is incorrect. Root accepts
the bounded repair at the existing xTB runner/adapter owners: report the actual
terminal native optimization decision; an explicit terminal nonconvergence
cannot be overridden by file presence. Keep current required-artifact,
nonzero-exit, finite-metric, atom-order and commit-integrity contracts.

For a zero-exit, complete, finite, atom-consistent result, `converged=False`
remains a usable observation. The artifact parser must not reject it solely for
that flag. Existing stage code emits `XTB:convergence` and respects configured
sample/batch breaks; no new result type, broad catch, retry, optimizer setting or
claim of convergence is introduced. Missing required artifacts and SCC/native
execution failures retain their precise local failure behavior and observations.
Retain the original five reports unchanged and reparse their actual native
outputs under the corrected code. Logger/final-coordinate differences are
separate observations; authoritative final files must still exactly match the
coordinates committed by the workflow.
