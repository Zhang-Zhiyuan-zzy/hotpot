# Scientific-state rule specifications

Status: normative companion to
[`scientific_state_implementation.md`](../scientific_state_implementation.md).

This directory contains one review unit per rule. A rule document must contain
its scientific basis, implementation boundary and acceptance points. A rule is
not implemented merely because its document exists.

## 1. Rule classes

| Prefix | Meaning | Toggleable | May select a value |
|---|---|:---:|:---:|
| `HARD` | Identity, conservation or backend-legality invariant | no | no; it validates or rejects |
| `DEFAULT` | Named deterministic decision/control policy | by choosing a policy | yes; it decides among hard-admissible options or emits a typed terminal action |
| `OPTIONAL` | Replaceable empirical or AI proposal/ranking rule | yes | no directly; it proposes evidence/candidates/ranks |

Hard rules are never toggleable and always run when their declared lifecycle
boundary has the required inputs. Scientific hard rules run during scientific
resolution; backend-only HARD-008/HARD-013 run during projection after
`BackendContext` exists. Missing future backend context is not itself a
scientific-state failure. Defaults are selected explicitly through a named
profile. Optional rules run only through `InferenceRuleRegistry` and are hard-
validated after every action.

Default policies are the workflow's decision drive. They do not generate
scientific evidence. After providers and optional rules propose candidates and
the controller composes hard-valid joint paths, defaults decide which path or
terminal action the declared workflow takes. `STRICT` may decide
`NO_AUTOMATIC_STATE`; a screening profile may select one execution path while
leaving scientific resolution `AMBIGUOUS`.

`DEFAULT` is a specification namespace, not a claim that all decisions share
one registry. Scientific-state decisions, backend execution decisions and
pipeline recovery decisions are registered independently; their documents
state the owning registry and lifecycle.

## 2. Review states

Every optional rule passes through:

```text
DRAFT -> DOMAIN_REVIEWED -> IMPLEMENTED_DISABLED
      -> VALIDATED -> PROFILE_ENABLED
```

`PROFILE_ENABLED` requires owner approval. A checked profile default in the
plan expresses the intended configuration after validation; it is not evidence
that production code already enables the rule.

Every `RuleSpecification` declares `phase`, `requires`, `consumes`, `produces`
and an applicability-schema version. The registry rejects missing dependencies
and cycles. The resolver runs topological layers; all rules in one layer consume
one immutable context, and their merged proposals are hard-validated before the
next context is built.

Every optional document also records rule version, review state, owner approval
state, primary evidence and the applicability artifact/digest. A missing digest
is an activation blocker even when the planned profile table contains a check.

## 3. Shared rule outcome

The proposal contracts below are owned and exported by
`hotpot.cheminfo.calculator.state_composition`; optional rules import those
same objects. They are shown here only to make the rule-output schema readable,
not as second `state_rules` definitions or re-exports.

```python
@dataclass(frozen=True)
class ProposedCandidateValue:
    candidate: StateCandidate
    branch_assignments: tuple[BranchAssignment, ...] = ()


@dataclass(frozen=True)
class ProposedCandidateFactor:
    factor_key: FactorKey
    domain_definition_id: str
    materialized_candidates: tuple[ProposedCandidateValue, ...]
    domain_scope: CandidateDomainScope
    materialization_status: MaterializationStatus
    unexpanded_domain_summary: CandidateDomainSummary | None


@dataclass(frozen=True)
class ConstraintGuard:
    all_candidate_ids: tuple[str, ...] = ()
    branch_assignments: tuple[BranchAssignment, ...] = ()


@dataclass(frozen=True)
class GuardedCandidateConstraint:
    guard: ConstraintGuard
    constraint: CandidateConstraint


@dataclass(frozen=True)
class CandidateProposal:
    proposal_id: str
    factors: tuple[ProposedCandidateFactor, ...]
    constraints: tuple[GuardedCandidateConstraint, ...]


@dataclass(frozen=True)
class RuleOutcome:
    rule_id: str
    rule_version: str
    applicability: RuleApplicability
    candidate_proposal: CandidateProposal | None
    evidence: tuple[EvidenceItem, ...]
    rank_adjustments: tuple[RankAdjustment, ...]
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
```

Only candidate-producing outcomes carry `CandidateProposal`. Each proposed
factor's `domain_scope` describes its scientific domain (`EXHAUSTIVE`,
`DOMAIN_BOUNDED` or `OPEN`), while `materialization_status` independently says
whether that page/domain is complete or resource-truncated. A truncated factor
must provide a remainder summary with enough continuation and optimistic-bound
information for factor construction and decision certification. Correlated
states are expressed by multiple factors plus explicitly guarded constraints;
they are never flattened into independent candidate tuples. Ranking-only or
finding-only outcomes set `candidate_proposal=None`; they do not manufacture
`N/A` enum values.

Proposal merge algebra is deterministic:

1. Compatible proposals co-activate by default, allowing ligand-charge, metal-
   OS and spin proposals from different rules to form one path. Mutual
   exclusivity exists only when candidate values carry different options of the
   same explicit `BranchAssignment.group_id`.
2. Every merged candidate value retains candidate-level source proposal IDs and
   branch assignments. A guard activates exactly when all its candidate IDs and
   branch assignments occur on the current path; an inactive guarded constraint
   has no effect.
3. Factors merge only when `factor_key` and `domain_definition_id` match.
   Candidate values are unioned by scientific identity and retain every source
   proposal ID. Explicit/request constraints are global intersections.
4. For one matching domain, a complete `EXHAUSTIVE` proposal proves the merged
   scope exhaustive if every other value is inside that domain. Otherwise
   `OPEN` dominates `DOMAIN_BOUNDED`; all-bounded proposals remain bounded.
5. Merged materialization is `COMPLETE` only when every contributing remainder
   is closed or a complete exhaustive superset demonstrably covers it;
   otherwise it is `RESOURCE_TRUNCATED`. `FactorDomainContribution` retains
   branch-local scope/status/remainders, so the aggregate never erases which
   producer was truncated.

The base rule's final `evaluate(toggle, context)` method maps an immutable
toggle to `DISABLED`, `ABSTAINED` or the subclass's pure `action()` result.
An action does not mutate `Molecule`, invoke another inference module, launch a
backend, decide a retry or suppress a conflict.

## 4. Rule catalogue

### Hard

- [HARD-001: computational-boundary identity](HARD-001-boundary-identity.md)
- [HARD-002: charge conservation](HARD-002-charge-conservation.md)
- [HARD-003: charge-quantity separation](HARD-003-charge-quantity-separation.md)
- [HARD-004: oxidation-state conservation](HARD-004-oxidation-state-conservation.md)
- [HARD-005: counting-convention exclusivity](HARD-005-counting-convention-exclusivity.md)
- [HARD-006: molecular electron/spin legality](HARD-006-electron-spin-legality.md)
- [HARD-007: explicit-state consistency](HARD-007-explicit-state-consistency.md)
- [HARD-008: backend state projection](HARD-008-backend-projection.md)
- [HARD-009: state invalidation](HARD-009-state-invalidation.md)
- [HARD-010: module independence](HARD-010-module-independence.md)
- [HARD-011: method-specific transport](HARD-011-method-specific-transport.md)
- [HARD-012: failure and gate semantics](HARD-012-failure-and-gate-semantics.md)
- [HARD-013: backend electron/UHF legality](HARD-013-backend-electron-uhf-legality.md)

### Defaults

- [DEFAULT-001: strict policy](DEFAULT-001-strict.md)
- [DEFAULT-002: coordination screening V1](DEFAULT-002-coordination-screening-v1.md)
- [DEFAULT-003: source authority](DEFAULT-003-source-authority.md)
- [DEFAULT-004: lowest legal xTB UHF](DEFAULT-004-lowest-legal-xtb-uhf.md)
- [DEFAULT-005: oxidation-state ambiguity](DEFAULT-005-oxidation-ambiguity.md)
- [DEFAULT-006: candidate preservation](DEFAULT-006-candidate-preservation.md)
- [DEFAULT-007: intermediate recovery](DEFAULT-007-intermediate-recovery.md)

### Optional empirical and AI rules

- [OPTIONAL-001: main-group closed shell](OPTIONAL-001-main-group-closed-shell.md)
- [OPTIONAL-002: monoatomic ions](OPTIONAL-002-monoatomic-ions.md)
- [OPTIONAL-003: neutral L donors](OPTIONAL-003-neutral-l-donors.md)
- [OPTIONAL-004: anionic X ligands](OPTIONAL-004-anionic-x-ligands.md)
- [OPTIONAL-005: lanthanide(III)](OPTIONAL-005-lanthanide-iii.md)
- [OPTIONAL-006: americium/curium(III)](OPTIONAL-006-americium-curium-iii.md)
- [OPTIONAL-007: thorium(IV)](OPTIONAL-007-thorium-iv.md)
- [OPTIONAL-008: actinyl states](OPTIONAL-008-actinyl-states.md)
- [OPTIONAL-009: ionic-configuration spin](OPTIONAL-009-ionic-configuration-spin.md)
- [OPTIONAL-010: transition-metal ligand field](OPTIONAL-010-transition-metal-ligand-field.md)
- [OPTIONAL-011: f-block Hund prior](OPTIONAL-011-f-block-hund-prior.md)
- [OPTIONAL-012: non-innocent ligands](OPTIONAL-012-noninnocent-ligands.md)
- [OPTIONAL-013: bond-valence ranking](OPTIONAL-013-bond-valence-ranking.md)
- [OPTIONAL-014: trusted scientific history](OPTIONAL-014-trusted-history.md)
- [OPTIONAL-015: AI state prior](OPTIONAL-015-ai-state-prior.md)
- [OPTIONAL-016: oxidation-state occurrence ranking](OPTIONAL-016-oxidation-occurrence-ranking.md)

## 5. Required review checklist

Each reviewer answers:

- Is the scientific quantity owned by the correct module?
- Is every prerequisite explicit and identity-bound?
- Is the applicability domain narrower than the evidence supports?
- Can the rule abstain without manufacturing a value?
- Does it preserve alternatives and provenance?
- Does every candidate-producing outcome separately declare scientific domain
  scope and materialization status without treating a common-state prior as
  exhaustive or hiding a truncated remainder?
- Can its proposal always pass through the common hard validators?
- Are counterexamples and out-of-domain cases tested?
- Are source versions and data hashes reproducible?
