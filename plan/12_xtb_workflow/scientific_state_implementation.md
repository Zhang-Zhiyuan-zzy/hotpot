# Scientific-state resolution: normative implementation plan

Status: **the sole normative implementation reference** for charge,
oxidation-state and electronic-state resolution in Stage 12.

Rule specifications under [`rules/`](rules/) are normative parts of this
plan. Deterministic [`providers/`](providers/) and optional pipeline
[`execution/`](execution/) and [`recovery/`](recovery/) policies are also
normative supporting specifications.
The five source documents listed in Section 2 are retained under
[`older/`](older/) as historical evidence only. If an Older document conflicts
with this normative set, the normative set wins.

This document specifies target behavior. It does not claim that the target
production implementation already exists.

## 1. Owner decision

Hotpot addresses two mandatory problems:

1. efficient automated preparation of charge, oxidation-state and electronic-
   state inputs; and
2. basic legality: no state may violate a hard conservation, identity,
   electron-count or target-backend constraint.

Legality does **not** imply universal chemical reasonableness, a physical
ground state or an optimal calculation setting. The implementation therefore
separates three governance layers:

```text
hard legality constraints
optional, replaceable empirical or AI rules
named default decision policies
```

The layers have different jobs. Hard rules define the non-negotiable feasible
set. Optional rules add attributable chemical candidates or preferences inside
that set. Default policies provide the **decision drive**: they deterministically
choose what the declared workflow should do among the remaining legal options.
A default may choose one execution state, continue despite a non-blocking
ambiguity, abstain outside its declared domain or stop one affected route. It
may not manufacture evidence or make an illegal state legal.

The invariant is **admissibility before preference, and preference before
decision**. The runtime order is therefore:

```text
bind and hard-validate explicit/source evidence
    -> generate independent local candidates
    -> hard-validate every proposal layer
    -> compose compatible charge/OS/electronic-state paths
    -> hard-validate every composed path
    -> apply optional, attributable preference evidence
    -> let the named default policy select exactly one execution action
```

The named policy is selected before providers/rules so that activation and
budgets are reproducible, but it performs its decision only after joint paths
have been composed and validated. Charge, oxidation state and spin are never
selected independently and then assumed to be mutually compatible.

For unattended screening, completion means a machine-actionable result rather
than an interactive question. A statically applicable record with a certified
decision receives one hard-valid execution state. Static out-of-domain inputs
and legal-but-uncertified choices receive typed `NO_AUTOMATIC_STATE`; illegal or
contradictory authoritative inputs receive their typed terminal actions, and
the batch proceeds to the next record. A separately named, opt-in
execution-only fallback may be defined later; it must not be hidden inside the
routine profile.

The API must permit explicit settings and candidate sets. A valid explicit
choice has higher authority than a default. A conflicting explicit choice is
reported as a conflict; Hotpot must not silently rewrite it.

“Do not let the program fail” applies to expected scientific uncertainty and
backend outcomes: missing evidence, ambiguity, hard-state conflict, unsupported
methods and nonconvergence receive typed results with diagnostics. Malformed API
use, corrupt contracts and programming defects remain exceptions; they are not
converted into a fake scientific result or silent fallback.

## 2. Normative scope and Older source map

The normative set is exactly:

- this document;
- every rule document indexed by [`rules/README.md`](rules/README.md);
- every provider indexed by [`providers/README.md`](providers/README.md);
- every execution-selection policy indexed by
  [`execution/README.md`](execution/README.md); and
- every recovery policy indexed by [`recovery/README.md`](recovery/README.md).

The following documents supplied source facts and design alternatives. They
are archived, superseded and non-normative:

| Older document | Facts retained by this plan | Superseded material |
|---|---|---|
| `older/xtb_failure_root_cause_and_recovery_plan.md` | Version-bound 187/182 failure audit; Eu UHF diagnostic; GFN-FF SDF adjacency diagnostic; method-specific transport; intermediate/final gate distinction | Its Phase A--E ordering and mixed inference policy |
| `older/charge_inference.md` | Separation of net, Lewis/formal, component and partial charge; graph non-identifiability; representation invariance | A universal joint Lewis solver as a prerequisite for xTB |
| `older/oxidation_state_inference.md` | IUPAC ionic approximation; global conservation; convention-labelled counting; source registry | Element occurrence as a default and oxidation-state resolution as a backend prerequisite |
| `older/electronic_state_inference.md` | Electron/UHF equations; local-spin candidate construction; scientific limitations | Backend projection mixed into scientific inference and restrictive fallback ordering |
| `older/admissibility_before_preference.md` | Admissibility-before-preference; candidate and provenance preservation; requested/executed evidence distinction | One combined admissibility axis and unqualified multiplicity/UHF equivalence |

Historical numbers remain evidence bound to their original code, backend and
artifact identities. They are not acceptance constants unless a new tracked
test reproduces them.

## 3. In scope and out of scope

### 3.1 In scope

- immutable facts and identity binding;
- independent net-charge, Lewis/formal-charge, oxidation-state and electronic-
  state results;
- hard scientific validation;
- named automatic selection policies;
- optional empirical/AI rule registration;
- xTB/GFN-FF projection and executability validation;
- method-specific topology/state transport;
- typed failure, ambiguity and provenance;
- migration from the current combined resolver; and
- unit, integration, representation and 187/182 workflow validation.

### 3.2 Out of scope

- proving a universal ground electronic state;
- treating formal charge, oxidation state or a population charge as the same
  quantity;
- a universal graph-to-charge oracle;
- silently selecting protonation, tautomerization or reaction products;
- generic high-/low-spin selection without domain evidence;
- using ordinary xTB convergence as physical-state validation;
- rewriting raw GJF/LOG evidence; and
- making the complete Lewis solver or oxidation-state solver a prerequisite
  when a backend only needs a legal authoritative total charge and UHF.

Gaussian archive reconciliation belongs to dataset intake and backend
projection. It may consume these contracts but is not part of the first xTB
implementation milestone.

## 4. Required separation of concerns

### 4.1 Scientific quantities

| Quantity | Owner | May consume | Must not decide |
|---|---|---|---|
| molecular facts | `calculator.contracts` | authoritative `Molecule` and source metadata | charge, oxidation state or spin |
| net charge | `calculator.charge` | molecular facts and attributable charge evidence | oxidation state, spin or coordination cleavage |
| Lewis/formal charge | `calculator.charge` | molecular facts plus one net-charge candidate | partial charge, oxidation state or spin |
| coordination electron counting | `calculator.oxidation_state` | facts, charge/Lewis evidence and one named convention | intact-system charge or spin |
| oxidation state | `calculator.oxidation_state` | facts, local counting evidence and explicit constraints | net charge, UHF or multiplicity |
| scientific electronic state | `calculator.electronic_state` | facts, net charge and optional radical/oxidation evidence | charge, topology or backend support |
| coupled state composition | `calculator.state_composition` | independently produced charge, component, counting, OS and electronic-state candidates | unconstrained scientific values, empirical ranking or backend execution |
| default state decision | `calculator.state_selection` | hard-valid coupled paths, named profile and attributable preference evidence | new evidence, backend-independent scientific truth or ground-state claims |
| backend execution selection | `calculator.execution_selection` plus backend plugin | one default-selected scientific path, backend context and exact-context execution evidence | backend-independent scientific truth |
| backend projection | backend plugin | one selected legal execution state plus backend context | scientific truth or ground-state ranking |
| orchestration | `calculator.state_resolver` | all immutable results and policy | hidden sibling fallback or graph mutation |

Inference modules are independently callable. They do not invoke sibling
inference implementations. `state_composition` performs a pure compatibility
join and constraint propagation over their immutable outputs; it does not
invent, rank or select candidates. Only the resolver controls the sequence,
and only a named default policy makes the final workflow decision.

### 4.2 Independent result axes

The implementation must not compress the following into one status:

| Axis | Examples |
|---|---|
| scientific admissibility | `ADMISSIBLE`, `INADMISSIBLE`, `NOT_EVALUATED` |
| scientific resolution | `UNIQUE`, `AMBIGUOUS`, `UNRESOLVED`, `OUT_OF_DOMAIN`, `CONFLICT` |
| workflow decision | explicit, source-preserved, profile default, optional-rule prior, no automatic state |
| chemical confidence | authoritative, in-domain high, medium, low, unresolved |
| backend projection | projected, incompatible, missing context |
| backend executability | supported, unsupported method/element, invalid effective electron count |
| runtime execution | not run, converged, nonconverged, failed |
| geometry quality | pass, warning, uncertain, fail |

`None`, zero, convergence and geometry success must not stand in for any other
axis.

## 5. Core contracts

The following field names, order and types are the approved target contract.
Implementation may change them only through an explicit revision of this plan;
Section 11.1 presents the same contracts as a closed public reachability tree.

```python
class ScientificQuantity(Enum):
    NET_CHARGE = "net_charge"
    LEWIS_FORMAL_CHARGE = "lewis_formal_charge"
    COORDINATION_COUNTING = "coordination_counting"
    OXIDATION_STATE = "oxidation_state"
    ELECTRONIC_STATE = "electronic_state"


class ScientificAdmissibility(Enum):
    ADMISSIBLE = "admissible"
    INADMISSIBLE = "inadmissible"
    NOT_EVALUATED = "not_evaluated"


class ResolutionStatus(Enum):
    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    OUT_OF_DOMAIN = "out_of_domain"
    CONFLICT = "conflict"


class ChemicalConfidence(Enum):
    AUTHORITATIVE = "authoritative"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"


class CoverageStatus(Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class StateValueAuthority(Enum):
    VERIFIED_EXPLICIT = "verified_explicit"
    VERIFIED_SCIENTIFIC_SOURCE = "verified_scientific_source"
    EXACT_IDENTITY_TRUSTED_SOURCE = "exact_identity_trusted_source"
    DOMAIN_RULE = "domain_rule"
    DERIVED_CONSTRAINT = "derived_constraint"
    UNRESOLVED = "unresolved"


class StatePolicy(Enum):
    STRICT = "strict"
    COORDINATION_SCREENING_V1 = "coordination_screening_v1"


@dataclass(frozen=True)
class MolecularFactsIdentity:
    chemical_identity_digest: str


@dataclass(frozen=True)
class BackendIdentity:
    digest: str


@dataclass(frozen=True)
class ComponentIdentity:
    identity: str
    atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class HydrogenInventory:
    explicit_counts: tuple[int | None, ...]
    implicit_counts: tuple[int | None, ...]


SemanticBondKind: TypeAlias = Literal["covalent", "coordination", "unknown"]


@dataclass(frozen=True)
class SemanticBond:
    atom_indices: tuple[int, int]
    order: float | None
    aromatic: bool
    kind: SemanticBondKind


@dataclass(frozen=True)
class ScientificFinding:
    code: str
    message: str
    related_identities: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateProvenance:
    identity: str
    authority: StateValueAuthority
    source_identity: str
    evidence_ids: tuple[str, ...] = ()
    assumption_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MolecularFacts:
    identity: MolecularFactsIdentity
    geometry_digest: str | None
    atomic_numbers: tuple[int, ...]
    mass_numbers: tuple[int | None, ...]
    atom_map: tuple[int, ...]
    source_components: tuple[ComponentIdentity, ...]
    covalent_components: tuple[ComponentIdentity, ...]
    hydrogen_inventory: HydrogenInventory
    semantic_bonds: tuple[SemanticBond, ...]
    coordinates: tuple[tuple[float, float, float], ...] | None
    nuclear_charge_sum: int


ProviderValue: TypeAlias = (
    Union[None, bool, int, float, str, tuple["ProviderValue", ...]]
)


@dataclass(frozen=True)
class StateRequest:
    net_charge: int | None = None
    atom_formal_charges: AtomFormalChargeEvidence | None = None
    component_charges: ComponentChargeEvidence | None = None
    oxidation_overrides: tuple[OxidationStateOverride, ...] = ()
    two_s: int | None = None
    multiplicity: int | None = None
    charge_candidates: tuple[NetChargeCandidateEvidence, ...] = ()
    oxidation_candidates: tuple[OxidationStateCandidateEvidence, ...] = ()
    electronic_candidates: tuple[ElectronicStateCandidateEvidence, ...] = ()


@dataclass(frozen=True)
class ScientificOperationRequirements:
    operation_id: str
    required_resolved_quantities: frozenset[ScientificQuantity]
    allowed_unresolved_quantities: frozenset[ScientificQuantity]


@dataclass(frozen=True)
class ProvidedScientificResults:
    net_charge_results: tuple[NetChargeResult, ...] = ()
    lewis_results: tuple[LewisFormalChargeResult, ...] = ()
    coordination_counting_results: tuple[CoordinationElectronCountingResult, ...] = ()
    oxidation_state_results: tuple[OxidationStateInferenceResult, ...] = ()
    electronic_state_results: tuple[ElectronicStateInferenceResult, ...] = ()


@dataclass(frozen=True)
class ProviderToggle:
    provider_id: str
    enabled: bool
    configuration: tuple[tuple[str, ProviderValue], ...] = ()


@dataclass(frozen=True)
class ProviderSet:
    toggles: tuple[ProviderToggle, ...] = ()


@dataclass(frozen=True)
class CandidateBudget:
    max_materialized_values_per_factor: int
    max_expanded_paths: int
    max_frontier_paths: int


DEFAULT_CANDIDATE_BUDGET: CandidateBudget


class BackendContext(Protocol):
    @property
    def identity(self) -> BackendIdentity: ...

    @property
    def scientific_operation_requirements(
        self,
    ) -> ScientificOperationRequirements: ...

    @property
    def projection_convention(self) -> str: ...

    def effective_nuclear_charge(self, atomic_number: int) -> int: ...


@dataclass(frozen=True)
class NetChargeCandidateEvidence:
    provenance: CandidateProvenance
    charge: int


@dataclass(frozen=True)
class AtomFormalChargeEvidence:
    provenance: CandidateProvenance
    charges: tuple[int | None, ...]
    coverage: CoverageStatus


@dataclass(frozen=True)
class ComponentChargeEvidence:
    provenance: CandidateProvenance
    component_charges: tuple[tuple[str, int], ...]
    coverage: CoverageStatus


@dataclass(frozen=True)
class NetChargeResult:
    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: tuple[NetChargeCandidateEvidence, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: tuple[ScientificFinding, ...]


@dataclass(frozen=True)
class LewisFormalChargeCandidate:
    provenance: CandidateProvenance
    net_charge_candidate_identity: str
    atom_formal_charges: tuple[int, ...]
    bond_orders: tuple[float, ...]
    nonbonding_electrons: tuple[int, ...]
    radical_electrons: tuple[int, ...]
    component_charges: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class LewisFormalChargeResult:
    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: tuple[LewisFormalChargeCandidate, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: tuple[ScientificFinding, ...]


class CountingConvention(Enum):
    IONIC = "ionic"
    DONOR_RETURN = "donor_return"
    CBC = "cbc"


@dataclass(frozen=True)
class OxidationStateOverride:
    atom_index: int
    oxidation_state: int


@dataclass(frozen=True)
class CoordinationFragmentAssignment:
    component_identity: str
    ionic_bookkeeping_charge: int | None
    lxz_class: str | None
    donated_electrons: int | None


@dataclass(frozen=True)
class CoordinationCountingCandidate:
    provenance: CandidateProvenance
    convention: CountingConvention
    fragments: tuple[CoordinationFragmentAssignment, ...]


@dataclass(frozen=True)
class CoordinationElectronCountingResult:
    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: tuple[CoordinationCountingCandidate, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: tuple[ScientificFinding, ...]


@dataclass(frozen=True)
class OxidationStateCandidateEvidence:
    provenance: CandidateProvenance
    convention: CountingConvention
    atom_states: tuple[int | None, ...]
    coverage: CoverageStatus
    coordination_counting_candidate_identity: str | None = None


@dataclass(frozen=True)
class OxidationStateInferenceResult:
    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: tuple[OxidationStateCandidateEvidence, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: tuple[ScientificFinding, ...]


@dataclass(frozen=True)
class ElementOxidationStateDomain:
    atomic_number: int
    states: tuple[int, ...]
    source_identity: str
    registry_version: str
    content_digest: str


@dataclass(frozen=True)
class ElectronicStateCandidateEvidence:
    provenance: CandidateProvenance
    two_s: int
    multiplicity: int | None
    pure_spin: bool


class SpinContributionScope(Enum):
    MOLECULAR = "molecular"
    COMPONENT = "component"
    ATOM = "atom"


@dataclass(frozen=True)
class SpinContributionCandidate:
    provenance: CandidateProvenance
    scope: SpinContributionScope
    subject_identity: str
    parent_candidate_ids: tuple[str, ...]
    two_s: int
    pure_spin: bool


@dataclass(frozen=True)
class ElectronicStateCandidate:
    provenance: CandidateProvenance
    net_charge_candidate_identity: str
    oxidation_state_candidate_identity: str | None
    electron_count: int
    two_s: int
    multiplicity: int | None
    pure_spin: bool


@dataclass(frozen=True)
class ElectronicStateInferenceResult:
    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: tuple[ElectronicStateCandidate, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: tuple[ScientificFinding, ...]


@dataclass(frozen=True)
class NeutralAtomReference:
    atomic_number: int
    electron_count: int
    ground_configuration: str
    two_s: int
    source_identity: str
    registry_version: str
    content_digest: str


class CandidateDomainScope(Enum):
    EXHAUSTIVE = "exhaustive"
    DOMAIN_BOUNDED = "domain_bounded"
    OPEN = "open"


class MaterializationStatus(Enum):
    COMPLETE = "complete"
    RESOURCE_TRUNCATED = "resource_truncated"


class CandidateSearchStatus(Enum):
    NOT_EVALUATED = "not_evaluated"
    DECISION_CERTIFIED = "decision_certified"
    TRUNCATED_BUT_DECISION_CERTIFIED = "truncated_but_decision_certified"
    NO_CERTIFIED_DECISION = "no_certified_decision"
    RESOURCE_LIMIT_NO_CERTIFIED_DECISION = "resource_limit_no_certified_decision"


class StateValueOrigin(Enum):
    RESULT = "result"
    DERIVED_CONSTRAINT = "derived_constraint"


@dataclass(frozen=True)
class ScientificStateIdentity:
    digest: str


@dataclass(frozen=True)
class FactorKey:
    quantity: ScientificQuantity
    subject_identity: str


ConstraintValue: TypeAlias = (
    Union[None, bool, int, float, str, tuple["ConstraintValue", ...]]
)


@dataclass(frozen=True)
class CandidateConstraint:
    identity: str
    constraint_type: str
    schema_version: str
    factor_keys: tuple[FactorKey, ...]
    parameters: tuple[tuple[str, ConstraintValue], ...]


@dataclass(frozen=True)
class FrontierBound:
    branch_identity: str
    schema_version: str
    optimistic_key: tuple[str, ...]


@dataclass(frozen=True)
class CandidateDomainSummary:
    remaining_count_upper_bound: int | None
    continuation_identity: str | None
    frontier_bound: FrontierBound | None


@dataclass(frozen=True)
class RejectedPathSummary:
    candidate_identities: tuple[str, ...]
    constraint_identity: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class StateValueRef:
    quantity: ScientificQuantity
    value: int
    origin: StateValueOrigin
    origin_identity: str


@dataclass(frozen=True)
class DerivedStateValue:
    provenance: CandidateProvenance
    quantity: ScientificQuantity
    value: int
    equation_id: str
    parent_candidate_ids: tuple[str, ...]
    counting_convention: CountingConvention


StateCandidate: TypeAlias = (
    Union[
        NetChargeCandidateEvidence,
        LewisFormalChargeCandidate,
        CoordinationCountingCandidate,
        OxidationStateCandidateEvidence,
        SpinContributionCandidate,
        ElectronicStateCandidate,
        DerivedStateValue,
    ]
)


@dataclass(frozen=True)
class BranchAssignment:
    group_id: str
    option_id: str


@dataclass(frozen=True)
class FactorCandidateValue:
    candidate: StateCandidate
    source_proposal_ids: tuple[str, ...]
    branch_assignments: tuple[BranchAssignment, ...]


@dataclass(frozen=True)
class FactorDomainContribution:
    proposal_id: str
    domain_scope: CandidateDomainScope
    materialization_status: MaterializationStatus
    unexpanded_domain_summary: CandidateDomainSummary | None


@dataclass(frozen=True)
class CandidateFactor:
    identity: str
    factor_key: FactorKey
    domain_definition_id: str
    values: tuple[FactorCandidateValue, ...]
    domain_contributions: tuple[FactorDomainContribution, ...]
    unexpanded_domain_summary: CandidateDomainSummary | None
    domain_scope: CandidateDomainScope
    materialization_status: MaterializationStatus


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
class CoupledScientificStateCandidate:
    identity: ScientificStateIdentity
    net_charge: StateValueRef
    component_assignment_ids: tuple[str, ...]
    coordination_assignment_ids: tuple[str, ...]
    oxidation_state_assignment_id: str | None
    electronic_state_result_id: str
    electronic_state_candidate_identity: str | None
    derived_values: tuple[DerivedStateValue, ...]
    evidence_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]


@dataclass(frozen=True)
class ScientificStateCandidateSpace:
    factors: tuple[CandidateFactor, ...]
    constraints: tuple[CandidateConstraint, ...]
    candidate_budget: CandidateBudget
    candidate_budget_artifact_id: str
    frontier_bound_schema_version: str
    expanded_paths: tuple[CoupledScientificStateCandidate, ...]
    rejected_path_summaries: tuple[RejectedPathSummary, ...]
    frontier: tuple[FrontierBound, ...]
    search_status: CandidateSearchStatus


@dataclass(frozen=True)
class ScientificResolutionResult:
    status: ResolutionStatus
    exhaustive: bool
    equivalent_state_identities: tuple[ScientificStateIdentity, ...]


class StateDecisionSource(Enum):
    NOT_SELECTED = "not_selected"
    EXPLICIT = "explicit"
    SCIENTIFIC_SOURCE = "scientific_source"
    EXECUTION_DEFAULT = "execution_default"
    EXECUTION_PRIOR = "execution_prior"
    CANONICAL_EQUIVALENT_TIE = "canonical_equivalent_tie"


@dataclass(frozen=True)
class StaticEligibilityFinding:
    code: str
    message: str


@dataclass(frozen=True)
class DecisionReason:
    code: str
    message: str
    related_identities: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticProfileEligibility:
    profile_id: str
    effective_profile_identity: str
    in_domain: bool
    findings: tuple[StaticEligibilityFinding, ...]


class DecisionCertificationStatus(Enum):
    NOT_EVALUATED = "not_evaluated"
    CERTIFIED = "certified"
    NON_EQUIVALENT_TIE = "non_equivalent_tie"
    OPEN_OR_UNBOUNDABLE_DOMAIN = "open_or_unboundable_domain"
    RESOURCE_LIMIT_UNCERTIFIED = "resource_limit_uncertified"


@dataclass(frozen=True)
class DecisionCertifiability:
    status: DecisionCertificationStatus
    findings: tuple[DecisionReason, ...]


class StateDecisionAction(Enum):
    NOT_REQUESTED = "not_requested"
    SELECT_STATE = "select_state"
    NO_AUTOMATIC_STATE = "no_automatic_state"
    CONFLICT = "conflict"
    INADMISSIBLE = "inadmissible"


@dataclass(frozen=True)
class ExecutionStateDecision:
    selected_state_identity: ScientificStateIdentity | None
    action: StateDecisionAction
    source: StateDecisionSource
    static_profile_eligibility: StaticProfileEligibility
    decision_certifiability: DecisionCertifiability
    rationale: tuple[DecisionReason, ...]


@dataclass(frozen=True)
class BackendStateValues:
    charge: int
    uhf: int | None
    multiplicity: int | None


class BackendDecisionSource(Enum):
    EXPLICIT = "explicit"
    EXACT_CONTEXT_HISTORY = "exact_context_history"
    EXECUTION_DEFAULT = "execution_default"


class BackendExecutability(Enum):
    NOT_EVALUATED = "not_evaluated"
    SUPPORTED = "supported"
    UNSUPPORTED_METHOD = "unsupported_method"
    UNSUPPORTED_ELEMENT = "unsupported_element"
    INVALID_EFFECTIVE_ELECTRON_COUNT = "invalid_effective_electron_count"


@dataclass(frozen=True)
class BackendProjectionFinding:
    code: str
    message: str
    related_identities: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendExecutionDecision:
    backend_identity: BackendIdentity
    requested_state: BackendStateValues
    selected_state: BackendStateValues
    source: BackendDecisionSource
    rationale: tuple[DecisionReason, ...]


@dataclass(frozen=True)
class BackendExecutionEvidence:
    scientific_state_identity: ScientificStateIdentity
    backend_identity: BackendIdentity
    requested_state: BackendStateValues
    executed_state: BackendStateValues
    artifact_digests: tuple[str, ...]


@dataclass(frozen=True)
class BackendStateRequest:
    uhf: int | None = None


@dataclass(frozen=True)
class ExecutionPolicySet:
    policy_ids: tuple[str, ...] = ("EXECUTION-001", "DEFAULT-004")


@dataclass(frozen=True)
class ScientificStateBundle:
    facts: MolecularFacts
    net_charge_results: tuple[NetChargeResult, ...]
    lewis_results: tuple[LewisFormalChargeResult, ...]
    coordination_counting_results: tuple[CoordinationElectronCountingResult, ...]
    oxidation_state_results: tuple[OxidationStateInferenceResult, ...]
    electronic_state_results: tuple[ElectronicStateInferenceResult, ...]
    candidate_space: ScientificStateCandidateSpace
    scientific_resolution: ScientificResolutionResult
    execution_decision: ExecutionStateDecision
    rule_trace: tuple[RuleTrace, ...]


@dataclass(frozen=True)
class BackendStateProjection:
    scientific_state_identity: ScientificStateIdentity
    facts_identity: MolecularFactsIdentity
    backend_identity: BackendIdentity
    effective_electron_count: int
    charge: int
    uhf: int | None
    multiplicity: int | None
    projection_convention: str
    execution_decision: BackendExecutionDecision
    executability: BackendExecutability


class BackendProjectionStatus(Enum):
    NOT_PROJECTED = "not_projected"
    PROJECTED = "projected"
    PROJECTION_CONFLICT = "projection_conflict"
    BACKEND_UNSUPPORTED = "backend_unsupported"


@dataclass(frozen=True)
class BackendProjectionResult:
    status: BackendProjectionStatus
    backend_identity: BackendIdentity
    request: BackendStateRequest
    projection: BackendStateProjection | None
    findings: tuple[BackendProjectionFinding, ...]


@dataclass(frozen=True)
class ScientificStateWorkflowResult:
    scientific_state: ScientificStateBundle
    backend_projection: BackendProjectionResult
```

Every candidate has a stable identity, evidence, assumptions and source.
Scientific resolution and execution-path decision are separate fields. A
default execution choice must not make a scientific candidate set appear
unique or populate a scientific selection that the evidence did not establish.

`ScientificStateCandidateSpace` is factorized. Charge, component, OS and
electronic-state domains plus their compatibility constraints are stored once;
the implementation must not construct their full Cartesian product. Explicit
user values constrain the factors before expansion. Constraint propagation and
a deterministic best-first/lazy join expand only the paths needed for the
selected decision. Every factor separately declares whether its scientific
domain is exhaustive, bounded by a reviewed applicability domain or open, and
whether that domain was completely materialized or resource-truncated. One
remaining generated state implies scientific `UNIQUE` only when every relevant
factor has `domain_scope=EXHAUSTIVE` and
`materialization_status=COMPLETE`; a default-selected singleton from a
bounded/open or truncated domain remains scientifically `AMBIGUOUS` or
`UNRESOLVED`.

Candidate generation is budgeted before joint-path expansion. A provider or
rule may emit one or more correlated factors plus guarded compatibility
constraints as a compact domain or lazy page, but it may not materialize more
than `max_materialized_values_per_factor`. Any remainder is represented by
`unexpanded_domain_summary`; exhausting this limit sets
`materialization_status=RESOURCE_TRUNCATED` without erasing the factor's
original `domain_scope`, and enters the same decision-certification logic as
path or frontier truncation. Provider-internal search limits remain
independently versioned in the provider configuration and must produce this
typed status rather than silently dropping candidates.

In V1, `StateValueRef` gives the scalar net-charge value, which may be either
source-produced or constraint-derived, one uniform reference. `RESULT` points
to an attributable module/provider result;
`DERIVED_CONSTRAINT` points to a `DerivedStateValue`. A derived value is
therefore never represented by a fictitious provider-result ID.

Structured OS assignments and electronic-state results continue to use their
own result identities. If a later phase permits a structured quantity to be
constraint-derived, it must first define an equivalent typed reference rather
than overloading a source result ID.

`electronic_state_result_id` always points to an attributable scientific
result envelope. `electronic_state_candidate_identity` identifies the exact
candidate when one is selected and is `None` only when that result is
`UNRESOLVED`. This preserves a positive identity for the unresolved scientific
axis without conflating sibling candidates, while allowing a later backend
execution policy to choose a legal projection.

The search frontier stores an optimistic decision-key bound for every
unexpanded branch. A decision is certified only when the selected path's key
cannot be surpassed by any frontier bound. Budget exhaustion therefore yields
either `TRUNCATED_BUT_DECISION_CERTIFIED` or
`RESOURCE_LIMIT_NO_CERTIFIED_DECISION`; only the latter forces
`NO_AUTOMATIC_STATE`. A completed search with a non-equivalent unresolved tie
uses `NO_CERTIFIED_DECISION`; an early hard terminal action uses
`NOT_EVALUATED`. Every actually expanded path or compact rejection summary is
retained, so no separate smaller retention budget can discard audit data.

`DEFAULT_CANDIDATE_BUDGET` is a versioned profile artifact, not an incidental
loop limit. Its numerical values and frontier-bound semantics are frozen during
Phase 5 from deterministic stress fixtures and reported with every decision.
Both the resolved numerical budget and its immutable artifact identity are
stored in `ScientificStateCandidateSpace`, so a serialized result is replayable
without consulting the then-current profile defaults. The serialized frontier-
bound schema version fixes how the same budget and frontier are interpreted.
Retry counts belong to a separate `RecoveryBudget`; they are not part of the
scientific candidate-search budget. No screening profile may be marked active
while either required artifact is unspecified.

`ScientificResolutionResult` and `ExecutionStateDecision` are different
objects. A default may select one hard-valid execution path while the former
remains `AMBIGUOUS`. Static out-of-domain scope or failure to certify among
legal alternatives produces `NO_AUTOMATIC_STATE`. An intrinsically illegal
authoritative value produces `INADMISSIBLE`; incompatible authoritative
constraints produce `CONFLICT`. None becomes an interactive prompt or a
fabricated neutral/low-spin value.

`ExecutionStateDecision` has a strict serialization invariant:

```text
action == SELECT_STATE  <=>  selected_state_identity is not None
action != SELECT_STATE  <=>  selected_state_identity is None
```

`NOT_REQUESTED`, `CONFLICT`, `INADMISSIBLE` and `NO_AUTOMATIC_STATE` carry
diagnostics but never a selected identity.

Scientific terminal actions have one shared meaning across all scientific-
state rules and entry points:

| Action | Exact condition |
|---|---|
| `NOT_REQUESTED` | Hard validation found no `INADMISSIBLE`/`CONFLICT` terminal, and the registered operation is analysis-only with no execution-state decision requested. |
| `INADMISSIBLE` | A mandatory scalar/asserted singleton is intrinsically illegal, or an authoritative allowed domain has no hard-admissible member. |
| `CONFLICT` | Individually admissible authoritative constraints bound to the same identity have an empty intersection. |
| `NO_AUTOMATIC_STATE` | No hard illegality/conflict is present, but the profile is statically out of scope, lacks attributable candidates or cannot certify one allowed decision. |
| `SELECT_STATE` | Exactly one execution decision is certified and its identity is present. |

An impossible value such as negative $2S$ is `INADMISSIBLE`. A legal scalar
$Q$ whose simultaneous legal candidate-domain constraint excludes that value,
or a legal partial OS override that cannot extend the same path's legal $Q$,
is `CONFLICT`.

An illegal member of a multi-value allowed domain is recorded as a rejected
candidate; it does not invalidate legal siblings. If every member of an
authoritative request/source domain is rejected, the scientific decision is
`INADMISSIBLE`. If an optional provider/rule merely proposes no legal member,
that outcome contributes rejections/abstention and other evidence may continue;
an otherwise legal record with no attributable final candidate becomes
`NO_AUTOMATIC_STATE`.

This action table governs scientific-state resolution only. Backend projection
and execution use their own typed results such as `PROJECTION_CONFLICT` and
`BACKEND_UNSUPPORTED`; they do not rewrite the scientific action.

`StaticProfileEligibility` records the versioned profile and the exact reason
an input is in or out of its static domain. It is computed only from immutable
molecular facts, requested operation, configured component versions and
versioned static applicability artifacts. Candidate-dependent rule matching is
recorded later as dynamic `RuleApplicability` in the rule trace and contributes
to `DecisionCertifiability`; it is not part of static eligibility. Successful
selection or backend convergence cannot retroactively change eligibility, and
a non-equivalent tie or resource limit does not rewrite a statically in-domain
record as out of domain.

The profile artifact freezes scientific-operation requirements, provider/rule
versions and digests, static eligibility predicates, decision order,
assumption-severity table, candidate budget and canonical tie rule. Caller
toggle/configuration overrides create a distinct
`EffectiveProfileIdentity`; they do not inherit the original profile's coverage
claim unless that exact effective identity has been validated.

`MolecularFactsIdentity` excludes coordinates, inferred/selected charge,
oxidation state, spin and backend context so that coordinate-only optimization
and alternative state candidates can preserve one chemical identity.
`MolecularFacts.geometry_digest` separately binds geometry-dependent evidence
and rules to the coordinates they consumed; such evidence must be recomputed
after a coordinate change without invalidating topology-only state evidence.
`ScientificStateIdentity` combines the facts identity with one state path.
`BackendStateProjection` then binds that state path to method/ECP/runtime
context.

`AtomFormalChargeEvidence` and `ComponentChargeEvidence` include identity,
source, values, covered atom/component IDs and a `COMPLETE`/`PARTIAL` marker.
No validator infers completeness from tuple length or a zero value.

Scalar fields in `StateRequest` are authoritative single-value constraints.
The corresponding `*_candidates` fields are attributable allowed domains: all
listed alternatives are intentionally admissible until other hard constraints
or preferences remove/rank them. Supplying multiple candidates is not a pair
of mutually exclusive authoritative facts and must not itself produce
`CONFLICT`.

| Scalar constraint | Candidate domain | Resolver meaning |
|---|---|---|
| absent | absent | quantity remains unresolved until evidence/rules act |
| absent | non-empty | use the attributable allowed domain |
| present | absent | constrain to the scalar |
| present | contains scalar | intersect to the scalar and retain both provenance paths |
| present | excludes scalar | `CONFLICT`: authoritative request constraints have empty intersection |

`source_components` preserve the input record boundary. `covalent_components`
are a non-mutating view obtained by excluding only bonds already classified as
semantic coordination edges. The facts extractor records explicit, implicit,
zero and unknown hydrogen states separately in `HydrogenInventory`; an
uninitialized H count is never interpreted as explicit zero. Ambiguous
metal--ligand bond semantics remain an attributable uncertainty and may place a
record outside an automatic profile rather than being silently rewritten.

Path completion terms are precise:

- **scientifically complete**: every quantity the result claims to resolve has
  a complete assignment and provenance;
- **operation-complete**: every quantity required by the selected scientific
  workflow action is present; an unconsumed quantity such as OS may remain
  explicitly unresolved; and
- **backend-projectable**: after `BackendContext` and any backend-specific
  request are supplied, every input required by that backend can be produced
  and hard-validated.

No function may use the unqualified word `complete` in a status or public
contract.

`ScientificOperationRequirements` makes operation completeness independent of
backend representation. The profile artifact supplies its reviewed
requirements. `requirements=None` selects those profile requirements
(`STRICT_ANALYSIS` for `STRICT`). An explicit caller may request another
registered operation, but that creates a distinct `EffectiveProfileIdentity`,
cannot weaken mandatory profile quantities and does not inherit validated
coverage automatically. `COORDINATION_SCREENING_V1` requires resolved net charge and an
attributable metal-OS assignment; it may carry an unresolved molecular $2S$
to backend projection. `STRICT_ANALYSIS` requires no execution selection. A
high-level backend workflow maps the requested calculation to one registered
scientific-operation requirement before resolution; the backend itself never
silently changes that requirement afterward.

Effective requirements are normalized deterministically:

$$
R_{\mathrm{effective}}=R_{\mathrm{profile}}\cup
R_{\mathrm{caller}}\cup R_{\mathrm{backend}},
$$

$$
U_{\mathrm{effective}}=
(U_{\mathrm{profile}}\cup U_{\mathrm{caller}}\cup U_{\mathrm{backend}})
\setminus R_{\mathrm{effective}}.
$$

Thus a required quantity always overrides permission to leave it unresolved,
and the two sets are disjoint after normalization. `STRICT_ANALYSIS` has no
required execution quantity. It still runs hard validation first: illegal or
conflicting authoritative input returns `INADMISSIBLE`/`CONFLICT`; only a
hard-valid analysis request returns `action=NOT_REQUESTED`, together with every
scientific result and candidate trace it can establish.

Public orchestration is flexible but explicit:

```python
def extract_molecular_facts(mol: Molecule) -> MolecularFacts: ...


def resolve_scientific_state(
    facts: MolecularFacts,
    request: StateRequest = StateRequest(),
    *,
    policy: StatePolicy = StatePolicy.STRICT,
    requirements: ScientificOperationRequirements | None = None,
    provider_set: ProviderSet = ProviderSet(),
    provided_results: ProvidedScientificResults = ProvidedScientificResults(),
    rule_set: RuleSet = RuleSet(),
    candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
) -> ScientificStateBundle: ...


def project_state(
    state: ScientificStateBundle,
    backend: BackendContext,
    *,
    request: BackendStateRequest = BackendStateRequest(),
    execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
    execution_history: tuple[BackendExecutionEvidence, ...] = (),
) -> BackendProjectionResult: ...


def resolve_and_project_state(
    facts: MolecularFacts,
    backend: BackendContext,
    request: StateRequest = StateRequest(),
    *,
    policy: StatePolicy = StatePolicy.STRICT,
    requirements: ScientificOperationRequirements | None = None,
    provider_set: ProviderSet = ProviderSet(),
    provided_results: ProvidedScientificResults = ProvidedScientificResults(),
    rule_set: RuleSet = RuleSet(),
    candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
    backend_request: BackendStateRequest = BackendStateRequest(),
    execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
    execution_history: tuple[BackendExecutionEvidence, ...] = (),
) -> ScientificStateWorkflowResult: ...
```

`BackendProjectionResult` is a tagged result. `PROJECTED` holds if and only if
`projection` is present. If the scientific decision is anything other than
`SELECT_STATE`, projection returns `NOT_PROJECTED`, `projection=None`, and a
finding that cites the unchanged scientific action. `PROJECTION_CONFLICT` and
`BACKEND_UNSUPPORTED` likewise carry findings but no selected
`BackendStateProjection` or `BackendExecutionDecision`; adapters must not
fabricate executable values to populate a non-projected result. Malformed API
contracts remain exceptions rather than projection statuses.

A high-level `resolve_and_project_state(..., backend=...)` canonical API must
map the requested calculation to registered backend-minimal
`ScientificOperationRequirements`, union them with the named profile's
mandatory quantities, then call these two functions in sequence and return both
immutable results. It must not reimplement inference, selection or projection,
and its report keeps the scientific resolution, state decision and backend
execution decision separate.

`STRICT` is the low-level library default. A CLI or controlled pipeline may
explicitly request `COORDINATION_SCREENING_V1`.

Every supplied `ProvidedScientificResults` object is identity- and hard-
validated but is not re-inferred. A `ProviderSet` is explicit and independent
from `RuleSet`; named policies may propose a profile configuration, and caller
toggles override it visibly. `STRICT` enables no provider implicitly.
`COORDINATION_SCREENING_V1` may conditionally enable a provider only after its
acceptance specification and applicability artifact pass review; provider use
and abstention remain visible in the result.

## 6. Hard legality

Hard rules are non-toggleable. Their individual specifications are
`rules/HARD-*.md`.

### 6.1 Charge

For the exact computational boundary $\Omega$, the net charge $Q$ is an
integer. For any result claiming complete atom and component assignments:

$$
Q=\sum_i q_i=\sum_c Q_c.
$$

An explicit zero is authoritative only when its source says it is explicit;
an uninitialized field containing zero is not evidence. Coordination cleavage,
CBC bookkeeping, oxidation-state assignment and partial charges never alter
the intact-system $Q$.

### 6.2 Oxidation state

For a complete oxidation-state assignment:

$$
x_i\in\mathbb Z,\qquad \sum_i x_i=Q.
$$

A partial override is legal only if at least one complete assignment can extend
it under the declared convention. Ionic, donor-return and CBC conventions are
not interchangeable. Multimetal conservation is solved globally.

Oxidation state may remain unresolved when the requested backend does not
consume it.

### 6.3 Molecular electron count and spin

The scientific electronic-state layer represents molecular spin as $2S$ (and,
where meaningful, multiplicity $2S+1$). Local metal/ligand spin candidates and
their coupling therefore remain backend-independent.

For an all-electron molecular scientific state:

$$
N_e=\sum_i Z_i-Q.
$$

A molecular spin-only value $2S$ is legal only when

$$
0\le 2S\le N_e,\qquad (N_e-2S)\bmod 2=0.
$$

Where a pure-spin multiplicity is claimed, $M=2S+1$. Strong spin--orbit or
otherwise non-pure-spin limitations remain explicit in the scientific result;
they are not erased by later backend projection.

### 6.4 Coupled-path consistency

Every selectable path must bind all quantities to one `MolecularFactsIdentity`
and one declared counting convention. The composite validator applies
HARD-001/002/004/005/006/007 together and rejects a path when any local result
is individually legal but incompatible with another result on that path.

Under a declared ionic coordination-counting convention, a scientifically
complete ionic-accounting path may use

$$
Q_{\mathrm{system}}=
\sum_m x_m+
\sum_l q_l+
\sum_c Q_c,
$$

where $x_m$ are metal oxidation-state assignments, $q_l$ are complete ionic
ligand bookkeeping charges and $Q_c$ are explicit spectator/counterion
component charges. This equation is valid only for that named convention; it
does not identify oxidation state with atom formal charge and does not alter
the intact graph's stored formal charges.

Composition may remove incompatible combinations and may create a
`DerivedStateValue` only when one declared hard equation uniquely determines
it from complete parent candidates. For example, complete ionic metal, ligand
and counterion assignments may derive $Q_{\mathrm{system}}$. The derived value
records its equation, convention and every parent candidate. Composition may
not introduce a free charge, OS or electronic-state value lacking such a proof,
and derived values carry no preference rank of their own.

### 6.5 Backend electron count and xTB UHF

xTB's UHF input is $u=N_\alpha-N_\beta=2M_S$ and belongs to
`BackendStateProjection`, not the backend-independent scientific state. For an
all-electron xTB projection, $N_e=\sum_i Z_i-Q$ and

$$
0\le u\le N_e,\qquad (N_e-u)\bmod 2=0.
$$

Consequently,

$$
N_\alpha=\frac{N_e+u}{2},\qquad
N_\beta=\frac{N_e-u}{2}
$$

must be non-negative integers. Only a named pure-spin highest-$M_S$
projection may set $u=2S$. An ECP calculation uses its declared effective-
electron definition rather than the all-nuclear count. HARD-013 owns these
backend checks; HARD-006 owns only molecular $N_e$/$2S$ legality.

An explicit backend UHF is hard-validated here before execution-policy
precedence is applied. Under a named highest-$M_S$ projection it must agree
with any selected scientific $2S$/multiplicity; disagreement is
`PROJECTION_CONFLICT`. Only a compatible explicit UHF, or one supplied while
scientific spin remains unresolved, can outrank DEFAULT-004.

## 7. Named defaults

Defaults are explicit **decision policies**, not hidden fallback values. They
consume hard-valid choices and attributable preferences, then decide which
action the declared workflow takes. They never act as scientific evidence.

### 7.1 `STRICT`

- preserve and validate attributable explicit/source facts;
- return candidates, ambiguity or unresolved when facts are insufficient;
- do not activate empirical rules;
- do not invent a numerical net charge or oxidation state; and
- decide `NO_AUTOMATIC_STATE` when a backend needs a unique value that no
  attributable input selected.

### 7.2 `COORDINATION_SCREENING_V1`

- preserve valid explicit/source settings;
- activate only the optional rules checked for this profile in Section 15;
- treat explicit user values as path constraints before candidate expansion;
- compose charge, component, counting, OS and electronic-state candidates
  before selecting any of them;
- compare operation-complete coupled paths by the source-authority order in
  DEFAULT-003, followed by enabled domain-rule preferences and a stable
  canonical tie breaker limited to decision-equivalent paths;
- for a statically applicable record, select exactly one hard-valid operation-
  complete scientific path whenever `DecisionCertifiability` is `CERTIFIED`,
  even when scientific resolution remains `AMBIGUOUS`;
- certify automatic selection only when decision-relevant factors
  are `EXHAUSTIVE` or finite `DOMAIN_BOUNDED` and expose a provable optimistic
  `FrontierBound`; an `OPEN`, unboundable or uncertified branch produces
  `NO_AUTOMATIC_STATE` rather than an arbitrary truncation decision;
- require approved profile preferences to strictly order any surviving paths
  that differ in operation-relevant $Q$, OS-equivalence class or molecular
  $2S$; an unresolved non-equivalent tie fails decision certification and
  cannot be settled by identity order;
- require an attributable metal-OS assignment for in-domain coordination-
  complex decisions; unresolved OS is permitted only for a nonmetal or another
  explicitly declared operation profile that does not request it;
- apply a validated f-block Hund/free-ion prior to molecular $2S$ candidates
  when its narrow applicability test passes;
- when scientific spin evidence remains insufficient, preserve that unresolved
  status; a target-backend execution policy may later apply its own explicitly
  labelled legal fallback;
- label profile choices `EXECUTION_DEFAULT` or `EXECUTION_PRIOR`, never
  `GROUND_STATE`;
- return `NO_AUTOMATIC_STATE` rather than asking the user when the system is out
  of profile domain, has legal paths but no certified choice, or reaches
  `RESOURCE_LIMIT_NO_CERTIFIED_DECISION`; a truncated search may select only
  when its frontier bound certifies the same decision; and
- retain the selected path, all actually expanded alternatives, factorized
  unexpanded domains, rejection summaries and the complete rule trace.

The profile targets routine Hotpot coordination-chemistry screening. It is not
a universal policy for redox-active, mixed-valence, strongly covalent,
multicentre or deliberately excited-state systems.

### 7.3 Deterministic path decision

The screening decision is path-wise and ordered:

1. remove every path failing a hard rule or explicit user constraint; mutually
   exclusive authoritative facts produce `CONFLICT` and no default decision;
2. remove any path dominated by another path with equal-or-higher evidence
   authority for every requested quantity and strictly higher authority for at
   least one;
3. apply enabled, versioned domain-rule rank adjustments to the surviving
   operation-complete coupled paths;
4. prefer fewer/weaker recorded assumptions according to the profile's
   versioned severity table;
5. if only decision-equivalent paths remain, choose the smallest stable
   canonical `ScientificStateIdentity` for execution only.

Decision equivalence requires the same operation-relevant $Q$, OS-equivalence
class and molecular $2S$ (or the same explicit unresolved marker), with any
remaining difference limited to symmetry/resonance localization or provenance
that does not change the requested operation. If non-equivalent paths remain
after Step 4, static profile applicability is unchanged but decision
certification fails and the policy returns `NO_AUTOMATIC_STATE`; canonical
identity must never replace missing chemical preference evidence. Step 5 is not
chemical evidence: the result retains
scientific `AMBIGUOUS`, all tie identities and the canonical-tie reason. File
order, rule-registration order, hash randomization and parallel scheduling
never affect this decision.
The decision implementation returns the selected identity and rationale; it
does not copy selected values into unrelated candidate records.

A constraint-derived value has authority category `DERIVED_CONSTRAINT`. It is
not an independent evidence coordinate and does not inherit the maximum or
minimum authority of its parents. Instead, its parent evidence remains as
separate coordinates on the coupled path. An explicit or verified source value
for the same quantity constrains composition: a compatible derivation may
remain as corroborating lineage, while an incompatible derivation rejects the
path. Thus charge conservation can complete a path without falsely promoting a
derived $Q$ above its actual parent evidence.

The versioned profile compiles these steps into a stable `DecisionKey` and an
optimistic `FrontierBound`. Every lazily unexpanded branch must expose the best
key it could still achieve. A rule preference may participate in lazy search
only when its bound is monotone or otherwise provable; an unboundable rule
requires full expansion of its finite declared domain. Search stops with a
certified decision only when no frontier branch can beat the selected key or
produce an equal decision-equivalence key with a smaller canonical identity.
An optimistic bound that permits a chemically non-equivalent tie cannot
certify selection. Budget exhaustion before that proof returns
`RESOURCE_LIMIT_NO_CERTIFIED_DECISION`; budget exhaustion after that proof is
`TRUNCATED_BUT_DECISION_CERTIFIED`.

### 7.4 Minimal decision example

After the listed rules have passed review and profile activation,
`c1ncccc1O[Eu]` illustrates the intended decision drive when no authoritative
explicit state is supplied. The exact computational boundary contains one Eu
and one ligand, with no counterion. The raw metal-SMILES supplies connectivity
and a fixed zero-H oxygen inventory; for this recognized coordination encoding,
the profile classifies unannotated Eu/O formal zeros as representation-level
defaults rather than complete authoritative atom-formal-charge evidence.

```text
fixed-zero-H pyridinolato/heteroaryloxide-like ligand
    + semantic Eu--O coordination boundary
    -> OPTIONAL-004 proposes whole-ligand ionic-bookkeeping charge -1
versioned profile admits this minimal bookkeeping fixture
    -> OPTIONAL-005 ranks Eu(III)
hard ionic conservation
    -> derive intact-system Q = (+3) + (-1) = +2
Eu(III) 4f6 configuration
    -> OPTIONAL-009 proposes legal local Eu two_s candidates
complete zero-radical ligand Lewis result
    -> OPTIONAL-001 proposes ligand-component two_s = 0
electronic-state coupling
    -> compose legal molecular two_s candidates
reviewed localized-f-ion prior
    -> OPTIONAL-011 ranks local Eu two_s = 6
single open-shell center + closed-shell ligand coupling
    -> ranks the molecular two_s = 6 path for this fixture
DEFAULT-002
    -> select the coupled (Q=+2, Eu OS=+3, two_s=6) execution path
```

HARD-001/002/004/005/006 validate the coupled path; an xTB highest-$M_S$
projection then maps the selected $2S=6$ to `UHF=6` and applies HARD-013.
Explicit/source constraints can remove or replace this profile path before
ranking. If the user instead asserts a complete intact atom-formal-charge sum
of zero, the derived $Q=+2$ path conflicts and cannot be selected. Any missing
hydrogen/coordination semantics, exclusion motif, non-equivalent unresolved tie
or uncertified search yields a typed terminal action instead of silently
reusing these values. This fixture demonstrates policy mechanics; it is not a
stability, ground-state or universal Eu claim and does not define the profile
domain.

## 8. Optional-rule architecture

### 8.1 Abstract A: registry and immutable activation

The registry owns discovery and configuration, not chemical logic:

```python
class RuleKind(Enum):
    EMPIRICAL = "empirical"
    AI = "ai"


class RulePhase(Enum):
    SOURCE_EVIDENCE = "source_evidence"
    CHARGE_EVIDENCE = "charge_evidence"
    COORDINATION_COUNTING = "coordination_counting"
    REDOX_CANDIDATE = "redox_candidate"
    OXIDATION_STATE_CANDIDATE = "oxidation_state_candidate"
    OXIDATION_STATE_RANKING = "oxidation_state_ranking"
    ELECTRONIC_STATE_CANDIDATE = "electronic_state_candidate"
    ELECTRONIC_STATE_RANKING = "electronic_state_ranking"


class ResultType(Enum):
    NET_CHARGE_RESULT = "net_charge_result"
    LEWIS_FORMAL_CHARGE_RESULT = "lewis_formal_charge_result"
    COORDINATION_COUNTING_RESULT = "coordination_counting_result"
    OXIDATION_STATE_RESULT = "oxidation_state_result"
    ELECTRONIC_STATE_RESULT = "electronic_state_result"
    SPIN_CONTRIBUTION = "spin_contribution"
    CANDIDATE_PROPOSAL = "candidate_proposal"
    EVIDENCE = "evidence"
    RANK_ADJUSTMENT = "rank_adjustment"


RuleValue: TypeAlias = ProviderValue


@dataclass(frozen=True)
class RuleDependency:
    rule_id: str
    version_specifier: str


@dataclass(frozen=True)
class RuleSpecification:
    rule_id: str
    rule_version: str
    kind: RuleKind
    phase: RulePhase
    requires: tuple[RuleDependency, ...]
    consumes: tuple[ResultType, ...]
    produces: tuple[ResultType, ...]
    applicability_schema: str


@dataclass(frozen=True)
class RuleToggle:
    rule_id: str
    enabled: bool
    configuration: tuple[tuple[str, RuleValue], ...] = ()


@dataclass(frozen=True)
class EvidenceItem:
    identity: str
    authority: StateValueAuthority
    source_identity: str
    claim: str
    locator: str | None = None


@dataclass(frozen=True)
class RankAdjustment:
    target_identity: str
    rank_dimension: str
    value: float
    reason: str


@dataclass(frozen=True)
class RuleApplicability:
    matches: bool
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleContext:
    facts: MolecularFacts
    phase: RulePhase
    requirements: ScientificOperationRequirements
    candidate_space: ScientificStateCandidateSpace | None
    prior_outcomes: tuple["RuleOutcome", ...]
    candidate_budget: CandidateBudget


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


@dataclass(frozen=True)
class RuleTrace:
    phase: RulePhase
    outcome: RuleOutcome


@dataclass(frozen=True)
class ActivatedRule:
    rule: "InferenceRule"
    toggle: RuleToggle


@dataclass(frozen=True)
class RuleSet:
    layers: tuple[tuple[ActivatedRule, ...], ...] = ()


class InferenceRuleRegistry:
    def register(self, rule: "InferenceRule") -> None: ...
    def available(self) -> tuple[RuleSpecification, ...]: ...
    def activate(
        self,
        profile: StatePolicy,
        toggles: tuple[RuleToggle, ...],
    ) -> "RuleSet": ...
```

Registration rejects duplicate IDs and incompatible schema versions.
Activation is deterministic. The registry rejects missing dependencies and
cycles, then returns topological rule layers. File name, registration order and
ranking priority never define scientific dependency order.

`RuleSet()` means that no optional rule is enabled. Only registry activation
constructs a non-empty, dependency-validated and topologically layered set;
the resolver consumes `layers` in order and never reorders them.
`InferenceRule` implementations contain no mutable per-run state. All runtime
configuration is carried by the immutable `RuleToggle`.

The initial `RulePhase` order is:

```text
SOURCE_EVIDENCE
  -> CHARGE_EVIDENCE
  -> COORDINATION_COUNTING
  -> REDOX_CANDIDATE
  -> OXIDATION_STATE_CANDIDATE
  -> [controller: CHARGE_OS_COMPOSITION + hard validation]
  -> OXIDATION_STATE_RANKING
  -> ELECTRONIC_STATE_CANDIDATE
  -> ELECTRONIC_STATE_RANKING
  -> [controller: COUPLED_STATE_COMPOSITION + hard validation]
  -> [default policy: DECISION]
```

The bracketed operations are resolver-owned barriers, not optional-rule phases.
OS rules may propose local OS candidates before a total charge is selected;
the composition barrier joins them with compatible ligand, counterion and
charge evidence under one named counting convention. Electronic-state rules
then operate only on surviving partial paths. This prevents cyclic
`selected Q -> OS -> Q` inference and avoids expanding impossible spin paths.

`requires` refines ordering inside or across rule phases. A registered AI rule
must select one existing phase in its immutable model specification; it cannot
invent an unreviewed phase or a decision policy at runtime.

### 8.2 Abstract B: toggle to action

`ProposedCandidateValue`, `ProposedCandidateFactor`, `ConstraintGuard`,
`GuardedCandidateConstraint` and `CandidateProposal` are owned once by
`state_composition.contracts`. Optional rules consume and return those shared
contracts; `state_rules` does not redefine or re-export them. This keeps the
composer below the optional-rule layer and prevents a package dependency cycle.

```python
class InferenceRule(ABC):
    specification: RuleSpecification

    @final
    def evaluate(
        self,
        toggle: RuleToggle,
        context: RuleContext,
    ) -> RuleOutcome:
        if not toggle.enabled:
            return _disabled_rule_outcome(self.specification)
        applicability = self.applicable(context)
        if not applicability.matches:
            return _abstained_rule_outcome(self.specification, applicability)
        return self.action(context, toggle.configuration)

    @abstractmethod
    def applicable(self, context: RuleContext) -> RuleApplicability: ...

    @abstractmethod
    def action(
        self,
        context: RuleContext,
        configuration: tuple[tuple[str, RuleValue], ...],
    ) -> RuleOutcome: ...
```

The controller performs:

```text
immutable RuleToggle
    -> registry resolves enabled rule
    -> rule.evaluate(toggle, immutable context)
         -> rule.applicable(context)
         -> rule.action(context, immutable configuration)
    -> hard revalidation of every proposal
    -> candidate merge/ranking with provenance
```

`RuleOutcome` may propose candidates, evidence, rank adjustments or an
out-of-domain finding. It may not mutate `Molecule`, launch a backend, suppress
a conflict, control retries or directly select a final state. The resolver
owns control flow; named default policies own selection.

A candidate-producing outcome wraps one or more proposed factors, their
`CandidateDomainScope`, `MaterializationStatus`, any unexpanded-domain summary
and explicitly guarded compatibility constraints in `CandidateProposal`.
Ranking-only and finding-only outcomes carry no proposal; they do not claim
fictitious candidate-domain metadata.

Candidate producers receive the applicable per-factor materialization budget
and return compact/lazy domains. The registry rejects an outcome that exceeds
the budget or omits a remainder summary after truncation. This prevents Lewis
or other local enumeration from exhausting memory before the factorized
composer can enforce its path/frontier limits.

Every candidate-producing provider/rule declares both domain scope and
materialization status. A common-state prior is normally `DOMAIN_BOUNDED`, not
`EXHAUSTIVE`; producing only one favored value can never by itself prove
scientific uniqueness. Resource truncation is orthogonal and never erases that
scope.

After each topological layer, the resolver merges all outcomes from that layer,
hard-validates the proposals, and constructs a new immutable `RuleContext` for
the next layer. A disabled or abstaining prerequisite causes the dependent rule
to return `DEPENDENCY_UNAVAILABLE`; it does not trigger an implicit fallback.

Required dependencies include:

| Consumer rule | Required immutable input |
|---|---|
| OPTIONAL-001 | one hard-valid partial charge path plus a complete supported `LewisFormalChargeResult` from PROVIDER-001 or authoritative source |
| OPTIONAL-004 | fixed protonation/coordination-boundary facts plus compatible Lewis or motif evidence |
| OPTIONAL-005/006/007 | metal identity/environment plus convention-labelled local ligand/counting candidates; selected $Q$ is not required |
| OPTIONAL-009 | a hard-valid partial charge/OS path plus ionic-configuration registry |
| OPTIONAL-010/011 | candidates from OPTIONAL-009 |

An AI rule implements the same interface and additionally records model ID,
artifact SHA-256, feature schema, applicability domain, calibration version
and uncertainty. Missing or out-of-domain AI output is abstention, not a
fallback value.

### 8.3 Providers, execution and recovery policies are not optional rules

Deterministic solvers and data registries live under [`providers/`](providers/).
They produce immutable facts/candidates but do not rank by empirical preference.
Exact backend-context selection lives under [`execution/`](execution/), and
pipeline retry lives under [`recovery/`](recovery/). Both run only after
scientific resolution; recovery additionally requires a legal projection. None
of these categories is registered as an `InferenceRule`.

Default policies are likewise not registered as optional rules. A default
policy is the controller's declared decision function over an already
hard-valid candidate space. Its inputs, ordering, tie handling, applicability
and terminal action must be deterministic and serializable.

## 9. End-to-end control flow

```text
Molecule + source metadata
        |
        v
extract immutable MolecularFacts + identity
        |
        v
bind explicit values and source candidates
        |
        v
hard scientific validation ----------------------> INADMISSIBLE / CONFLICT
        |
        v
select named policy, providers and RuleToggles
        |
        v
run providers and dependency-layered optional rules
        |
        v
hard revalidate after every layer; retain RuleOutcome trace
        |
        v
factorized charge/OS composition; reject incompatible paths
        |
        v
generate/rank electronic states only for surviving paths
        |
        v
compose and hard-validate complete scientific-state paths
        |
        v
named default policy makes one deterministic action decision
        |                              |
        |                              +--> NO_AUTOMATIC_STATE; record and continue batch
        |
        v
ScientificStateBundle + selected state identity
        |
        v
exact-context execution evidence/policies (optional)
        |
        v
backend-specific state projection
        |-------------------------------------------> PROJECTION_CONFLICT
        |
        v
backend executability validation ----------------> BACKEND_UNSUPPORTED
        |
        v
run backend -> runtime status + geometry status + artifacts
```

Scientific ambiguity stops only an operation that requires a unique value.
For example, a legal authoritative $Q$ and execution UHF can launch GFN2 even
when Lewis allocation or oxidation state is unresolved.

The selected execution path may therefore be unique while the scientific
resolution remains `AMBIGUOUS`. This is deliberate: the default policy drives
the calculation without claiming that its choice is the unique physical state.

For the automated xTB workflow, `resolve_scientific_state()` first chooses the
operation-complete scientific path. `project_state(..., backend=xtb)` then
maps selected molecular $2S$ to UHF or, when scientific spin remains unresolved,
invokes backend DEFAULT-004. The combined workflow can therefore deliver one
backend-ready $(Q,\mathrm{UHF})$ without placing xTB policy inside scientific
inference.

`COORDINATION_SCREENING_V1` deliberately requires an attributable OS-bearing
path because its operation is coordination-state preparation, even though xTB
itself does not consume OS. DEFAULT-005 remains available to a separately
specified operation profile whose required quantities are only $Q$ and a
backend projection; no generic xTB-only automatic profile is activated by this
plan.

## 10. xTB transport and recovery boundaries

These are implementation invariants derived from the historical failure
audit, not empirical state-selection rules.

| Node | Native input | State consumed | Authoritative topology behavior |
|---|---|---|---|
| GFN-FF | one-record SDF/MOL with explicit endpoints | selected total charge | backend topology is input evidence; output commits coordinates only |
| GFN1/2-xTB | XYZ | selected total charge and projected UHF | no input bonds required by Hamiltonian; Hotpot graph remains authoritative |

All adapters verify atom identity/order before coordinate commit. A native
output must not silently replace the Hotpot graph, formal charges, aromaticity
or coordination semantics.

Preflight executability, intermediate geometry observation and final product
acceptance are distinct. A finite but poor intermediate geometry normally
emits a warning and may proceed. Non-finite coordinates, invalid atom mapping,
unsupported backend state or illegal electron count stop the affected route.
Final acceptance decides whether the delivered structure passes.

State retry remains a separate pipeline-control API; neither the scientific
resolver nor a single backend node starts another backend attempt:

```python
@dataclass(frozen=True)
class RecoveryBudget:
    max_backend_attempts: int


DEFAULT_RECOVERY_BUDGET = RecoveryBudget(max_backend_attempts=1)


class RecoveryAction(Enum):
    STOP = "stop"
    RETRY = "retry"


@dataclass(frozen=True)
class RecoveryPolicySet:
    policy_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    projection: BackendStateProjection | None
    rationale: tuple[DecisionReason, ...]


def decide_state_recovery(
    state: ScientificStateBundle,
    backend: BackendContext,
    *,
    request: BackendStateRequest = BackendStateRequest(),
    execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
    execution_history: tuple[BackendExecutionEvidence, ...],
    recovery_policies: RecoveryPolicySet = RecoveryPolicySet(),
    recovery_budget: RecoveryBudget = DEFAULT_RECOVERY_BUDGET,
) -> RecoveryDecision: ...
```

`RecoveryPolicySet()` and `DEFAULT_RECOVERY_BUDGET` permit only the original
attempt and therefore return `STOP`; only explicit RECOVERY-001 activation can
request another legal projection. `RETRY` holds if and only if `projection` is
present. The function selects the next state but never launches a process.

## 11. Target package layout

```text
hotpot/cheminfo/calculator/
├── __init__.py
├── base.py                  # retained Calculator API
├── mca_inference.py         # retained MCA facade
├── contracts.py
├── charge/
│   ├── __init__.py
│   ├── contracts.py
│   ├── net_charge.py
│   ├── lewis.py
│   ├── validate.py
│   └── data/
├── oxidation_state/
│   ├── __init__.py
│   ├── contracts.py
│   ├── ionic_approximation.py
│   ├── coordination_counting.py
│   ├── inference.py
│   └── data/
├── electronic_state/
│   ├── __init__.py
│   ├── contracts.py
│   ├── spin.py
│   ├── coupling.py
│   └── data/
├── state_composition/
│   ├── __init__.py
│   ├── contracts.py
│   ├── constraints.py
│   ├── candidate_space.py
│   └── compose.py
├── state_rules/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── profiles.py
│   ├── charge.py
│   ├── oxidation_state.py
│   ├── electronic_state.py
│   ├── ai.py
│   └── data/
│       └── profiles/
│           └── coordination_screening_v1.json
├── legality.py
├── operation_requirements.py
├── profile_eligibility.py
├── defaults.py
├── state_selection.py
├── execution_selection.py
└── state_resolver.py

hotpot/plugins/xtb/
├── state_projection.py
├── adapter.py
├── workflow.py
└── ...

hotpot/pipeline/
└── state_recovery.py        # typed, opt-in state-retry decision only
```

Versioned element, ligand and ionic-configuration registries are data files
with source URL/release, content digest, applicability and counterexamples.
They are not embedded as scattered Python branches.

`state_composition` performs no chemical inference. It joins independently
produced candidates, propagates hard constraints, maintains the factorized
candidate space and expands paths within `CandidateBudget`. `state_selection`
applies named scientific-path decision policies; `execution_selection` handles
only backend-context decisions. Neither duplicates composition or hard
validators.

### 11.1 Complete public API signature tree

This is the closed **target** public API owned by this refactor, not a claim
about the current production checkout. In every shown namespace, names appear
in exact `__all__` order. Returned object types listed in the tree are public
contracts even when they are not first-level re-exports. Constructor/field
order shown here and in Section 5 is normative; the two views must remain
identical. Helpers outside this tree are private.

The first-level namespace is deliberately small and preserves the unrelated
`Calculator` and `mca` APIs:

```python
hotpot.cheminfo.calculator.__all__ = [
    "Calculator",
    "mca",
    "extract_molecular_facts",
    "resolve_scientific_state",
    "project_state",
    "resolve_and_project_state",
    "MolecularFacts",
    "StateRequest",
    "StatePolicy",
    "ScientificOperationRequirements",
    "ScientificStateBundle",
    "BackendStateRequest",
    "BackendProjectionResult",
    "ScientificStateWorkflowResult",
]
```

The target submodule exports are likewise exact and ordered:

```python
hotpot.cheminfo.calculator.contracts.__all__ = [
    "ScientificQuantity", "ScientificAdmissibility", "ResolutionStatus",
    "ChemicalConfidence", "CoverageStatus", "StateValueAuthority",
    "StatePolicy", "MolecularFactsIdentity", "BackendIdentity",
    "ComponentIdentity", "HydrogenInventory", "SemanticBondKind",
    "SemanticBond", "ScientificFinding", "CandidateProvenance",
    "MolecularFacts", "ProviderValue", "ProviderToggle", "ProviderSet",
    "CandidateBudget", "DEFAULT_CANDIDATE_BUDGET",
    "ScientificOperationRequirements", "BackendContext",
]

hotpot.cheminfo.calculator.state_resolver.__all__ = [
    "extract_molecular_facts", "resolve_scientific_state", "StateRequest",
    "ProvidedScientificResults", "ScientificStateBundle",
]

hotpot.cheminfo.calculator.charge.__all__ = [
    "resolve_net_charge", "solve_main_group_lewis",
    "NetChargeCandidateEvidence", "AtomFormalChargeEvidence",
    "ComponentChargeEvidence", "NetChargeResult",
    "LewisFormalChargeCandidate", "LewisFormalChargeResult",
]

hotpot.cheminfo.calculator.oxidation_state.__all__ = [
    "infer_coordination_electron_counting", "infer_oxidation_states",
    "element_oxidation_state_domain", "CountingConvention",
    "OxidationStateOverride",
    "CoordinationFragmentAssignment", "CoordinationCountingCandidate",
    "CoordinationElectronCountingResult", "OxidationStateCandidateEvidence",
    "OxidationStateInferenceResult", "ElementOxidationStateDomain",
]

hotpot.cheminfo.calculator.electronic_state.__all__ = [
    "infer_electronic_states", "neutral_atom_reference",
    "ElectronicStateCandidateEvidence", "SpinContributionScope",
    "SpinContributionCandidate", "ElectronicStateCandidate",
    "ElectronicStateInferenceResult", "NeutralAtomReference",
]

hotpot.cheminfo.calculator.state_composition.__all__ = [
    "compose_scientific_state_space", "CandidateDomainScope",
    "MaterializationStatus", "CandidateSearchStatus", "StateValueOrigin",
    "ScientificStateIdentity", "FactorKey", "ConstraintValue",
    "CandidateConstraint", "FrontierBound", "CandidateDomainSummary",
    "RejectedPathSummary", "BranchAssignment", "StateValueRef",
    "DerivedStateValue", "StateCandidate", "FactorCandidateValue",
    "FactorDomainContribution",
    "CandidateFactor", "ProposedCandidateValue", "ProposedCandidateFactor",
    "ConstraintGuard", "GuardedCandidateConstraint", "CandidateProposal",
    "CoupledScientificStateCandidate", "ScientificStateCandidateSpace",
    "ScientificResolutionResult",
]

hotpot.cheminfo.calculator.state_rules.__all__ = [
    "RuleKind", "RulePhase", "ResultType", "RuleValue", "RuleDependency",
    "RuleSpecification", "RuleToggle", "EvidenceItem", "RankAdjustment",
    "RuleApplicability", "RuleContext", "RuleOutcome", "RuleTrace",
    "ActivatedRule", "RuleSet", "InferenceRule", "InferenceRuleRegistry",
]

hotpot.cheminfo.calculator.state_selection.__all__ = [
    "StateDecisionAction", "StateDecisionSource",
    "DecisionCertificationStatus", "StaticEligibilityFinding",
    "DecisionReason", "StaticProfileEligibility", "DecisionCertifiability",
    "ExecutionStateDecision",
]

hotpot.cheminfo.calculator.execution_selection.__all__ = [
    "project_state", "resolve_and_project_state", "BackendStateValues",
    "BackendStateRequest", "BackendDecisionSource", "BackendExecutability",
    "BackendProjectionFinding", "BackendExecutionDecision",
    "BackendExecutionEvidence", "ExecutionPolicySet", "BackendStateProjection",
    "BackendProjectionStatus", "BackendProjectionResult",
    "ScientificStateWorkflowResult",
]

hotpot.pipeline.state_recovery.__all__ = [
    "decide_state_recovery", "RecoveryAction", "RecoveryPolicySet",
    "RecoveryBudget", "DEFAULT_RECOVERY_BUDGET", "RecoveryDecision",
]
```

The tree uses `[constructible]`, `[extension]`, `[returned]`, `[enum]`,
`[protocol]`, `[ABC]`, `[alias]`, `[constant]` and `[exception]` to distinguish
stable public roles. A type reachable beneath another result remains a public
contract even when it is not re-exported from the calculator facade.

`MolChargeCalculator`, `formal_charge`, `infer_charge`, `infer_lowest_spin`,
`resolve_electronic_state` and the old flat `ElectronicState` are migration
sources, not target compatibility exports. They are removed after all consumers
move to this API; no parallel legacy state resolver remains.

MCA result types keep their canonical
`hotpot.cheminfo.AImodels.mca` ownership and are only reachable from `mca()`;
they are not re-exported by `calculator`. The implementation uses a forward
annotation under `TYPE_CHECKING`, preserving the existing guarantee that
importing `hotpot.cheminfo.calculator` does not load the MCA runtime.

```text
hotpot.cheminfo.calculator
|
|- Calculator()                                    # retained empty calculator base
|  `- no declared public methods or attributes
|- mca(
|      mol: Molecule,
|      *,
|      device: str | None = None,
|      allow_charged: bool = False,
|      model_source: str | None = None,
|      model_dir: str | None = None,
|  ) -> hotpot.cheminfo.AImodels.mca.MoleculePrediction
|  |- also stores prediction at mol.properties["mca"]
|  |- each mol.atoms[index].mca -> float            # read-only, kJ/mol
|  |- mol.mca_sites -> dict[Atom, float]             # defensive copy
|  `- MoleculePrediction(                            # canonical MCA namespace
|         smiles: str,
|         formal_charge: int,
|         sites: tuple[SitePrediction, ...],
|         model_variant: str,
|         atom_predictions: tuple[AtomPrediction, ...] = (),
|     )
|     |- prediction.smiles -> str
|     |- prediction.formal_charge -> int
|     |- prediction.sites -> tuple[SitePrediction, ...]
|     |- prediction.model_variant -> str
|     |- prediction.atom_predictions -> tuple[AtomPrediction, ...]
|     |- prediction.to_dict() -> dict[str, object]
|     |- SitePrediction(atom_index: int, element: str,
|     |                 site_type: str, mca_kj_mol: float)
|     |  `- site.to_dict() -> dict[str, object]
|     `- AtomPrediction(atom_index: int, element: str, mca_kj_mol: float)
|        `- atom_prediction.to_dict() -> dict[str, object]
|
|- extract_molecular_facts(mol: Molecule) -> MolecularFacts
|
|- resolve_scientific_state(
|      facts: MolecularFacts,
|      request: StateRequest = StateRequest(),
|      *,
|      policy: StatePolicy = StatePolicy.STRICT,
|      requirements: ScientificOperationRequirements | None = None,
|      provider_set: ProviderSet = ProviderSet(),
|      provided_results: ProvidedScientificResults = ProvidedScientificResults(),
|      rule_set: RuleSet = RuleSet(),
|      candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
|  ) -> ScientificStateBundle
|
|- project_state(
|      state: ScientificStateBundle,
|      backend: BackendContext,
|      *,
|      request: BackendStateRequest = BackendStateRequest(),
|      execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
|      execution_history: tuple[BackendExecutionEvidence, ...] = (),
|  ) -> BackendProjectionResult
|
|- resolve_and_project_state(
|      facts: MolecularFacts,
|      backend: BackendContext,
|      request: StateRequest = StateRequest(),
|      *,
|      policy: StatePolicy = StatePolicy.STRICT,
|      requirements: ScientificOperationRequirements | None = None,
|      provider_set: ProviderSet = ProviderSet(),
|      provided_results: ProvidedScientificResults = ProvidedScientificResults(),
|      rule_set: RuleSet = RuleSet(),
|      candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
|      backend_request: BackendStateRequest = BackendStateRequest(),
|      execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
|      execution_history: tuple[BackendExecutionEvidence, ...] = (),
|  ) -> ScientificStateWorkflowResult
|
|- MolecularFacts                                  # immutable returned facts
|- StateRequest(...)                               # user-constructible input
|- StatePolicy                                     # Enum: STRICT, COORDINATION_SCREENING_V1
|- ScientificOperationRequirements(...)            # user-constructible input
|- ScientificStateBundle                           # immutable result; first-level export
|- BackendStateRequest(...)                        # user-constructible input
|- BackendProjectionResult                         # immutable tagged result
`- ScientificStateWorkflowResult
   |- result.scientific_state -> ScientificStateBundle
   `- result.backend_projection -> BackendProjectionResult

hotpot.cheminfo.calculator.contracts
|
|- ScientificQuantity                              # [enum]
|  |- NET_CHARGE
|  |- LEWIS_FORMAL_CHARGE
|  |- COORDINATION_COUNTING
|  |- OXIDATION_STATE
|  `- ELECTRONIC_STATE
|- ScientificAdmissibility                         # [enum]
|  |- ADMISSIBLE
|  |- INADMISSIBLE
|  `- NOT_EVALUATED
|- ResolutionStatus                                # [enum]
|  |- UNIQUE
|  |- AMBIGUOUS
|  |- UNRESOLVED
|  |- OUT_OF_DOMAIN
|  `- CONFLICT
|- ChemicalConfidence                              # [enum]
|  |- AUTHORITATIVE
|  |- HIGH
|  |- MEDIUM
|  |- LOW
|  `- UNRESOLVED
|- CoverageStatus                                  # [enum]: COMPLETE, PARTIAL
|- StateValueAuthority                             # [enum]
|  |- VERIFIED_EXPLICIT
|  |- VERIFIED_SCIENTIFIC_SOURCE
|  |- EXACT_IDENTITY_TRUSTED_SOURCE
|  |- DOMAIN_RULE
|  |- DERIVED_CONSTRAINT
|  `- UNRESOLVED
|- StatePolicy                                     # [enum]
|  |- STRICT
|  `- COORDINATION_SCREENING_V1
|- MolecularFactsIdentity(chemical_identity_digest: str) # [returned], frozen
|- BackendIdentity(digest: str)                    # [returned], frozen
|- ComponentIdentity(identity: str,
|                     atom_indices: tuple[int, ...])      # [returned], frozen
|- HydrogenInventory(                              # [returned], frozen
|      explicit_counts: tuple[int | None, ...],
|      implicit_counts: tuple[int | None, ...],
|  )
|- SemanticBondKind = Literal["covalent", "coordination", "unknown"]
|                                                    # [alias]
|- SemanticBond(                                    # [returned], frozen
|      atom_indices: tuple[int, int],
|      order: float | None,
|      aromatic: bool,
|      kind: SemanticBondKind,
|  )
|- ScientificFinding(code: str, message: str,
|                    related_identities: tuple[str, ...] = ())
|                                                    # [returned], frozen
|- CandidateProvenance(                             # [extension], frozen
|      identity: str,
|      authority: StateValueAuthority,
|      source_identity: str,
|      evidence_ids: tuple[str, ...] = (),
|      assumption_ids: tuple[str, ...] = (),
|  )
|- MolecularFacts                                  # [returned], frozen
|  |- identity: MolecularFactsIdentity
|  |- geometry_digest: str | None
|  |- atomic_numbers: tuple[int, ...]
|  |- mass_numbers: tuple[int | None, ...]
|  |- atom_map: tuple[int, ...]
|  |- source_components: tuple[ComponentIdentity, ...]
|  |- covalent_components: tuple[ComponentIdentity, ...]
|  |- hydrogen_inventory: HydrogenInventory
|  |- semantic_bonds: tuple[SemanticBond, ...]
|  |- coordinates: tuple[tuple[float, float, float], ...] | None
|  `- nuclear_charge_sum: int
|- ProviderValue = bool | int | float | str |
|                    tuple[ProviderValue, ...] | None  # [alias]
|- ProviderToggle(provider_id: str, enabled: bool,
|                  configuration: tuple[tuple[str, ProviderValue], ...] = ())
|                                                    # [constructible], frozen
|- ProviderSet(toggles: tuple[ProviderToggle, ...] = ())
|                                                    # [constructible], frozen
|- CandidateBudget(                                # [constructible], frozen
|      max_materialized_values_per_factor: int,
|      max_expanded_paths: int,
|      max_frontier_paths: int,
|  )
|- DEFAULT_CANDIDATE_BUDGET: CandidateBudget        # [constant]
|- ScientificOperationRequirements(                # [constructible], frozen
|      operation_id: str,
|      required_resolved_quantities: frozenset[ScientificQuantity],
|      allowed_unresolved_quantities: frozenset[ScientificQuantity],
|  )
`- BackendContext                                  # [protocol]
   |- context.identity -> BackendIdentity
   |- context.scientific_operation_requirements
   |      -> ScientificOperationRequirements
   |- context.projection_convention -> str
   `- context.effective_nuclear_charge(atomic_number: int) -> int

hotpot.cheminfo.calculator.state_resolver
|
|- extract_molecular_facts(mol: Molecule) -> MolecularFacts
|- resolve_scientific_state(
|      facts: MolecularFacts,
|      request: StateRequest = StateRequest(),
|      *,
|      policy: StatePolicy = StatePolicy.STRICT,
|      requirements: ScientificOperationRequirements | None = None,
|      provider_set: ProviderSet = ProviderSet(),
|      provided_results: ProvidedScientificResults = ProvidedScientificResults(),
|      rule_set: RuleSet = RuleSet(),
|      candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
|  ) -> ScientificStateBundle
|- StateRequest                                    # [constructible], frozen
|  |- net_charge: int | None = None
|  |- atom_formal_charges: AtomFormalChargeEvidence | None = None
|  |- component_charges: ComponentChargeEvidence | None = None
|  |- oxidation_overrides: tuple[OxidationStateOverride, ...] = ()
|  |- two_s: int | None = None
|  |- multiplicity: int | None = None
|  |- charge_candidates: tuple[NetChargeCandidateEvidence, ...] = ()
|  |- oxidation_candidates: tuple[OxidationStateCandidateEvidence, ...] = ()
|  `- electronic_candidates: tuple[ElectronicStateCandidateEvidence, ...] = ()
|- ProvidedScientificResults                       # [constructible], frozen
|  |- net_charge_results: tuple[NetChargeResult, ...] = ()
|  |- lewis_results: tuple[LewisFormalChargeResult, ...] = ()
|  |- coordination_counting_results:
|  |      tuple[CoordinationElectronCountingResult, ...] = ()
|  |- oxidation_state_results: tuple[OxidationStateInferenceResult, ...] = ()
|  `- electronic_state_results: tuple[ElectronicStateInferenceResult, ...] = ()
`- ScientificStateBundle                           # [returned], frozen
   |- facts: MolecularFacts
   |- net_charge_results: tuple[NetChargeResult, ...]
   |- lewis_results: tuple[LewisFormalChargeResult, ...]
   |- coordination_counting_results:
   |      tuple[CoordinationElectronCountingResult, ...]
   |- oxidation_state_results: tuple[OxidationStateInferenceResult, ...]
   |- electronic_state_results: tuple[ElectronicStateInferenceResult, ...]
   |- candidate_space: ScientificStateCandidateSpace
   |- scientific_resolution: ScientificResolutionResult
   |- execution_decision: ExecutionStateDecision
   `- rule_trace: tuple[RuleTrace, ...]

hotpot.cheminfo.calculator.charge
|
|- resolve_net_charge(
|      facts: MolecularFacts,
|      request: StateRequest = StateRequest(),
|      *,
|      provided: tuple[NetChargeResult, ...] = (),
|  ) -> tuple[NetChargeResult, ...]
|- solve_main_group_lewis(
|      facts: MolecularFacts,
|      net_charge: NetChargeResult,
|      *,
|      candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
|  ) -> tuple[LewisFormalChargeResult, ...]
|- NetChargeCandidateEvidence(provenance: CandidateProvenance, charge: int)
|                                                    # [extension], frozen
|- AtomFormalChargeEvidence(                        # [extension], frozen
|      provenance: CandidateProvenance,
|      charges: tuple[int | None, ...],
|      coverage: CoverageStatus,
|  )
|- ComponentChargeEvidence(                         # [extension], frozen
|      provenance: CandidateProvenance,
|      component_charges: tuple[tuple[str, int], ...],
|      coverage: CoverageStatus,
|  )
|- NetChargeResult                                  # [returned], frozen
|  |- identity: str
|  |- facts_identity: MolecularFactsIdentity
|  |- candidates: tuple[NetChargeCandidateEvidence, ...]
|  |- admissibility: ScientificAdmissibility
|  |- resolution: ResolutionStatus
|  |- confidence: ChemicalConfidence
|  `- findings: tuple[ScientificFinding, ...]
|- LewisFormalChargeCandidate                       # [returned], frozen
|  |- provenance: CandidateProvenance
|  |- net_charge_candidate_identity: str
|  |- atom_formal_charges: tuple[int, ...]
|  |- bond_orders: tuple[float, ...]
|  |- nonbonding_electrons: tuple[int, ...]
|  |- radical_electrons: tuple[int, ...]
|  `- component_charges: tuple[tuple[str, int], ...]
`- LewisFormalChargeResult                          # [returned], frozen
   |- identity: str
   |- facts_identity: MolecularFactsIdentity
   |- candidates: tuple[LewisFormalChargeCandidate, ...]
   |- admissibility: ScientificAdmissibility
   |- resolution: ResolutionStatus
   |- confidence: ChemicalConfidence
   `- findings: tuple[ScientificFinding, ...]

hotpot.cheminfo.calculator.oxidation_state
|
|- infer_coordination_electron_counting(
|      facts: MolecularFacts,
|      *,
|      lewis_results: tuple[LewisFormalChargeResult, ...] = (),
|      conventions: tuple[CountingConvention, ...] = (),
|  ) -> tuple[CoordinationElectronCountingResult, ...]
|- infer_oxidation_states(
|      facts: MolecularFacts,
|      request: StateRequest = StateRequest(),
|      *,
|      counting_results: tuple[CoordinationElectronCountingResult, ...] = (),
|      provided: tuple[OxidationStateInferenceResult, ...] = (),
|  ) -> tuple[OxidationStateInferenceResult, ...]
|- element_oxidation_state_domain(
|      atomic_number: int,
|  ) -> ElementOxidationStateDomain
|- CountingConvention                               # [enum]: IONIC, DONOR_RETURN, CBC
|- OxidationStateOverride(atom_index: int, oxidation_state: int)
|                                                    # [constructible], frozen
|- CoordinationFragmentAssignment                   # [returned], frozen
|  |- component_identity: str
|  |- ionic_bookkeeping_charge: int | None
|  |- lxz_class: str | None
|  `- donated_electrons: int | None
|- CoordinationCountingCandidate                    # [returned], frozen
|  |- provenance: CandidateProvenance
|  |- convention: CountingConvention
|  `- fragments: tuple[CoordinationFragmentAssignment, ...]
|- CoordinationElectronCountingResult               # [returned], frozen
|  |- identity: str
|  |- facts_identity: MolecularFactsIdentity
|  |- candidates: tuple[CoordinationCountingCandidate, ...]
|  |- admissibility: ScientificAdmissibility
|  |- resolution: ResolutionStatus
|  |- confidence: ChemicalConfidence
|  `- findings: tuple[ScientificFinding, ...]
|- OxidationStateCandidateEvidence                  # [extension], frozen
|  |- provenance: CandidateProvenance
|  |- convention: CountingConvention
|  |- atom_states: tuple[int | None, ...]
|  |- coverage: CoverageStatus
|  `- coordination_counting_candidate_identity: str | None = None
|- OxidationStateInferenceResult                     # [returned], frozen
|  |- identity: str
|  |- facts_identity: MolecularFactsIdentity
|  |- candidates: tuple[OxidationStateCandidateEvidence, ...]
|  |- admissibility: ScientificAdmissibility
|  |- resolution: ResolutionStatus
|  |- confidence: ChemicalConfidence
|  `- findings: tuple[ScientificFinding, ...]
`- ElementOxidationStateDomain                       # [returned], frozen
   |- atomic_number: int
   |- states: tuple[int, ...]
   |- source_identity: str
   |- registry_version: str
   `- content_digest: str

hotpot.cheminfo.calculator.electronic_state
|
|- infer_electronic_states(
|      facts: MolecularFacts,
|      request: StateRequest = StateRequest(),
|      *,
|      net_charge_results: tuple[NetChargeResult, ...],
|      oxidation_state_results: tuple[OxidationStateInferenceResult, ...] = (),
|      provided: tuple[ElectronicStateInferenceResult, ...] = (),
|  ) -> tuple[ElectronicStateInferenceResult, ...]
|- neutral_atom_reference(
|      atomic_number: int,
|  ) -> NeutralAtomReference | None
|- ElectronicStateCandidateEvidence                # [extension], frozen
|  |- provenance: CandidateProvenance
|  |- two_s: int
|  |- multiplicity: int | None
|  `- pure_spin: bool
|- SpinContributionScope                            # [enum]
|  |- MOLECULAR
|  |- COMPONENT
|  `- ATOM
|- SpinContributionCandidate                        # [extension], frozen
|  |- provenance: CandidateProvenance
|  |- scope: SpinContributionScope
|  |- subject_identity: str
|  |- parent_candidate_ids: tuple[str, ...]
|  |- two_s: int
|  `- pure_spin: bool
|- ElectronicStateCandidate                        # [returned], frozen
|  |- provenance: CandidateProvenance
|  |- net_charge_candidate_identity: str
|  |- oxidation_state_candidate_identity: str | None
|  |- electron_count: int
|  |- two_s: int
|  |- multiplicity: int | None
|  `- pure_spin: bool
|- ElectronicStateInferenceResult                  # [returned], frozen
|  |- identity: str
|  |- facts_identity: MolecularFactsIdentity
|  |- candidates: tuple[ElectronicStateCandidate, ...]
|  |- admissibility: ScientificAdmissibility
|  |- resolution: ResolutionStatus
|  |- confidence: ChemicalConfidence
|  `- findings: tuple[ScientificFinding, ...]
`- NeutralAtomReference                            # [returned], frozen
   |- atomic_number: int
   |- electron_count: int
   |- ground_configuration: str
   |- two_s: int
   |- source_identity: str
   |- registry_version: str
   `- content_digest: str

hotpot.cheminfo.calculator.state_composition
|
|- compose_scientific_state_space(
|      facts: MolecularFacts,
|      *,
|      net_charge_results: tuple[NetChargeResult, ...],
|      lewis_results: tuple[LewisFormalChargeResult, ...] = (),
|      coordination_counting_results:
|          tuple[CoordinationElectronCountingResult, ...] = (),
|      oxidation_state_results: tuple[OxidationStateInferenceResult, ...] = (),
|      electronic_state_results: tuple[ElectronicStateInferenceResult, ...] = (),
|      candidate_proposals: tuple[CandidateProposal, ...] = (),
|      constraints: tuple[CandidateConstraint, ...] = (),
|      candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
|  ) -> ScientificStateCandidateSpace
|- CandidateDomainScope                             # [enum]
|  |- EXHAUSTIVE
|  |- DOMAIN_BOUNDED
|  `- OPEN
|- MaterializationStatus                            # [enum]
|  |- COMPLETE
|  `- RESOURCE_TRUNCATED
|- CandidateSearchStatus                            # [enum]
|  |- NOT_EVALUATED
|  |- DECISION_CERTIFIED
|  |- TRUNCATED_BUT_DECISION_CERTIFIED
|  |- NO_CERTIFIED_DECISION
|  `- RESOURCE_LIMIT_NO_CERTIFIED_DECISION
|- StateValueOrigin                                 # [enum]: RESULT, DERIVED_CONSTRAINT
|- ScientificStateIdentity(digest: str)             # [returned], frozen
|- FactorKey(quantity: ScientificQuantity, subject_identity: str)
|                                                    # [extension], frozen
|- ConstraintValue = bool | int | float | str |
|                    tuple[ConstraintValue, ...] | None  # [alias]
|- CandidateConstraint                              # [extension], frozen
|  |- identity: str
|  |- constraint_type: str
|  |- schema_version: str
|  |- factor_keys: tuple[FactorKey, ...]
|  `- parameters: tuple[tuple[str, ConstraintValue], ...]
|- FrontierBound                                    # [returned], frozen
|  |- branch_identity: str
|  |- schema_version: str
|  `- optimistic_key: tuple[str, ...]
|- CandidateDomainSummary                           # [returned], frozen
|  |- remaining_count_upper_bound: int | None
|  |- continuation_identity: str | None
|  `- frontier_bound: FrontierBound | None
|- RejectedPathSummary                              # [returned], frozen
|  |- candidate_identities: tuple[str, ...]
|  |- constraint_identity: str
|  `- reason_codes: tuple[str, ...]
|- BranchAssignment(group_id: str, option_id: str)  # [extension], frozen
|- StateValueRef                                    # [returned], frozen
|  |- quantity: ScientificQuantity
|  |- value: int
|  |- origin: StateValueOrigin
|  `- origin_identity: str
|- DerivedStateValue                                # [returned], frozen
|  |- provenance: CandidateProvenance
|  |- quantity: ScientificQuantity
|  |- value: int
|  |- equation_id: str
|  |- parent_candidate_ids: tuple[str, ...]
|  `- counting_convention: CountingConvention
|- StateCandidate = NetChargeCandidateEvidence |
|      LewisFormalChargeCandidate | CoordinationCountingCandidate |
|      OxidationStateCandidateEvidence | SpinContributionCandidate |
|      ElectronicStateCandidate | DerivedStateValue # [alias]
|- FactorCandidateValue                             # [returned], frozen
|  |- candidate: StateCandidate
|  |- source_proposal_ids: tuple[str, ...]
|  `- branch_assignments: tuple[BranchAssignment, ...]
|- FactorDomainContribution                         # [returned], frozen
|  |- proposal_id: str
|  |- domain_scope: CandidateDomainScope
|  |- materialization_status: MaterializationStatus
|  `- unexpanded_domain_summary: CandidateDomainSummary | None
|- CandidateFactor                                  # [returned], frozen
|  |- identity: str
|  |- factor_key: FactorKey
|  |- domain_definition_id: str
|  |- values: tuple[FactorCandidateValue, ...]
|  |- domain_contributions: tuple[FactorDomainContribution, ...]
|  |- unexpanded_domain_summary: CandidateDomainSummary | None
|  |- domain_scope: CandidateDomainScope
|  `- materialization_status: MaterializationStatus
|- ProposedCandidateValue(candidate: StateCandidate,
|                          branch_assignments: tuple[BranchAssignment, ...] = ())
|                                                    # [extension], frozen
|- ProposedCandidateFactor                          # [extension], frozen
|  |- factor_key: FactorKey
|  |- domain_definition_id: str
|  |- materialized_candidates: tuple[ProposedCandidateValue, ...]
|  |- domain_scope: CandidateDomainScope
|  |- materialization_status: MaterializationStatus
|  `- unexpanded_domain_summary: CandidateDomainSummary | None
|- ConstraintGuard(all_candidate_ids: tuple[str, ...] = (),
|                   branch_assignments: tuple[BranchAssignment, ...] = ())
|                                                    # [extension], frozen
|- GuardedCandidateConstraint(guard: ConstraintGuard,
|                             constraint: CandidateConstraint)
|                                                    # [extension], frozen
|- CandidateProposal                                # [extension], frozen
|  |- proposal_id: str
|  |- factors: tuple[ProposedCandidateFactor, ...]
|  `- constraints: tuple[GuardedCandidateConstraint, ...]
|- CoupledScientificStateCandidate                  # [returned], frozen
|  |- identity: ScientificStateIdentity
|  |- net_charge: StateValueRef
|  |- component_assignment_ids: tuple[str, ...]
|  |- coordination_assignment_ids: tuple[str, ...]
|  |- oxidation_state_assignment_id: str | None
|  |- electronic_state_result_id: str
|  |- electronic_state_candidate_identity: str | None
|  |- derived_values: tuple[DerivedStateValue, ...]
|  |- evidence_ids: tuple[str, ...]
|  `- assumption_ids: tuple[str, ...]
|- ScientificStateCandidateSpace                    # [returned], frozen
|  |- factors: tuple[CandidateFactor, ...]
|  |- constraints: tuple[CandidateConstraint, ...]
|  |- candidate_budget: CandidateBudget
|  |- candidate_budget_artifact_id: str
|  |- frontier_bound_schema_version: str
|  |- expanded_paths: tuple[CoupledScientificStateCandidate, ...]
|  |- rejected_path_summaries: tuple[RejectedPathSummary, ...]
|  |- frontier: tuple[FrontierBound, ...]
|  `- search_status: CandidateSearchStatus
`- ScientificResolutionResult                       # [returned], frozen
   |- status: ResolutionStatus
   |- exhaustive: bool
   `- equivalent_state_identities: tuple[ScientificStateIdentity, ...]

hotpot.cheminfo.calculator.state_rules
|
|- RuleKind                                         # [enum]: EMPIRICAL, AI
|- RulePhase                                        # [enum]
|  |- SOURCE_EVIDENCE
|  |- CHARGE_EVIDENCE
|  |- COORDINATION_COUNTING
|  |- REDOX_CANDIDATE
|  |- OXIDATION_STATE_CANDIDATE
|  |- OXIDATION_STATE_RANKING
|  |- ELECTRONIC_STATE_CANDIDATE
|  `- ELECTRONIC_STATE_RANKING
|- ResultType                                       # [enum]
|  |- NET_CHARGE_RESULT
|  |- LEWIS_FORMAL_CHARGE_RESULT
|  |- COORDINATION_COUNTING_RESULT
|  |- OXIDATION_STATE_RESULT
|  |- ELECTRONIC_STATE_RESULT
|  |- SPIN_CONTRIBUTION
|  |- CANDIDATE_PROPOSAL
|  |- EVIDENCE
|  `- RANK_ADJUSTMENT
|- RuleValue = ProviderValue                        # [alias]
|- RuleDependency(rule_id: str, version_specifier: str)
|                                                    # [extension], frozen
|- RuleSpecification                                # [extension], frozen
|  |- rule_id: str
|  |- rule_version: str
|  |- kind: RuleKind
|  |- phase: RulePhase
|  |- requires: tuple[RuleDependency, ...]
|  |- consumes: tuple[ResultType, ...]
|  |- produces: tuple[ResultType, ...]
|  `- applicability_schema: str
|- RuleToggle(rule_id: str, enabled: bool,
|             configuration: tuple[tuple[str, RuleValue], ...] = ())
|                                                    # [constructible], frozen
|- EvidenceItem                                     # [extension], frozen
|  |- identity: str
|  |- authority: StateValueAuthority
|  |- source_identity: str
|  |- claim: str
|  `- locator: str | None = None
|- RankAdjustment                                   # [extension], frozen
|  |- target_identity: str
|  |- rank_dimension: str
|  |- value: float
|  `- reason: str
|- RuleApplicability(matches: bool,
|                     reason_codes: tuple[str, ...] = ())
|                                                    # [returned], frozen
|- RuleContext                                      # [returned], frozen
|  |- facts: MolecularFacts
|  |- phase: RulePhase
|  |- requirements: ScientificOperationRequirements
|  |- candidate_space: ScientificStateCandidateSpace | None
|  |- prior_outcomes: tuple[RuleOutcome, ...]
|  `- candidate_budget: CandidateBudget
|- RuleOutcome                                      # [extension], frozen
|  |- rule_id: str
|  |- rule_version: str
|  |- applicability: RuleApplicability
|  |- candidate_proposal: CandidateProposal | None
|  |- evidence: tuple[EvidenceItem, ...]
|  |- rank_adjustments: tuple[RankAdjustment, ...]
|  |- assumptions: tuple[str, ...]
|  `- warnings: tuple[str, ...]
|- RuleTrace(phase: RulePhase, outcome: RuleOutcome) # [returned], frozen
|- ActivatedRule(rule: InferenceRule, toggle: RuleToggle)
|                                                    # [returned], frozen
|- RuleSet(layers: tuple[tuple[ActivatedRule, ...], ...] = ())
|                                                    # [constructible-empty], frozen
|  `- rule_set.layers -> tuple[tuple[ActivatedRule, ...], ...]
|- InferenceRule                                    # [ABC]
|  |- rule.specification -> RuleSpecification
|  |- rule.evaluate(
|  |      toggle: RuleToggle,
|  |      context: RuleContext,
|  |  ) -> RuleOutcome
|  |- rule.applicable(context: RuleContext) -> RuleApplicability
|  `- rule.action(
|         context: RuleContext,
|         configuration: tuple[tuple[str, RuleValue], ...],
|     ) -> RuleOutcome
`- InferenceRuleRegistry()                          # [constructible extension registry]
   |- registry.register(rule: InferenceRule) -> None
   |- registry.available() -> tuple[RuleSpecification, ...]
   `- registry.activate(
          profile: StatePolicy,
          toggles: tuple[RuleToggle, ...],
      ) -> RuleSet

hotpot.cheminfo.calculator.state_selection
|
|- StateDecisionAction                              # [enum]
|  |- NOT_REQUESTED
|  |- SELECT_STATE
|  |- NO_AUTOMATIC_STATE
|  |- CONFLICT
|  `- INADMISSIBLE
|- StateDecisionSource                              # [enum]
|  |- NOT_SELECTED
|  |- EXPLICIT
|  |- SCIENTIFIC_SOURCE
|  |- EXECUTION_DEFAULT
|  |- EXECUTION_PRIOR
|  `- CANONICAL_EQUIVALENT_TIE
|- DecisionCertificationStatus                      # [enum]
|  |- NOT_EVALUATED
|  |- CERTIFIED
|  |- NON_EQUIVALENT_TIE
|  |- OPEN_OR_UNBOUNDABLE_DOMAIN
|  `- RESOURCE_LIMIT_UNCERTIFIED
|- StaticEligibilityFinding(code: str, message: str)
|                                                    # [returned], frozen
|- DecisionReason(code: str, message: str,
|                 related_identities: tuple[str, ...] = ())
|                                                    # [returned], frozen
|- StaticProfileEligibility                         # [returned], frozen
|  |- profile_id: str
|  |- effective_profile_identity: str
|  |- in_domain: bool
|  `- findings: tuple[StaticEligibilityFinding, ...]
|- DecisionCertifiability                           # [returned], frozen
|  |- status: DecisionCertificationStatus
|  `- findings: tuple[DecisionReason, ...]
`- ExecutionStateDecision                           # [returned], frozen
   |- selected_state_identity: ScientificStateIdentity | None
   |- action: StateDecisionAction
   |- source: StateDecisionSource
   |- static_profile_eligibility: StaticProfileEligibility
   |- decision_certifiability: DecisionCertifiability
   `- rationale: tuple[DecisionReason, ...]

hotpot.cheminfo.calculator.execution_selection
|
|- project_state(
|      state: ScientificStateBundle,
|      backend: BackendContext,
|      *,
|      request: BackendStateRequest = BackendStateRequest(),
|      execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
|      execution_history: tuple[BackendExecutionEvidence, ...] = (),
|  ) -> BackendProjectionResult
|- resolve_and_project_state(
|      facts: MolecularFacts,
|      backend: BackendContext,
|      request: StateRequest = StateRequest(),
|      *,
|      policy: StatePolicy = StatePolicy.STRICT,
|      requirements: ScientificOperationRequirements | None = None,
|      provider_set: ProviderSet = ProviderSet(),
|      provided_results: ProvidedScientificResults = ProvidedScientificResults(),
|      rule_set: RuleSet = RuleSet(),
|      candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
|      backend_request: BackendStateRequest = BackendStateRequest(),
|      execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
|      execution_history: tuple[BackendExecutionEvidence, ...] = (),
|  ) -> ScientificStateWorkflowResult
|- BackendStateValues(charge: int, uhf: int | None,
|                     multiplicity: int | None)     # [returned], frozen
|- BackendStateRequest(uhf: int | None = None)      # [constructible], frozen
|- BackendDecisionSource                            # [enum]
|  |- EXPLICIT
|  |- EXACT_CONTEXT_HISTORY
|  `- EXECUTION_DEFAULT
|- BackendExecutability                             # [enum]
|  |- NOT_EVALUATED
|  |- SUPPORTED
|  |- UNSUPPORTED_METHOD
|  |- UNSUPPORTED_ELEMENT
|  `- INVALID_EFFECTIVE_ELECTRON_COUNT
|- BackendProjectionFinding(code: str, message: str,
|                           related_identities: tuple[str, ...] = ())
|                                                    # [returned], frozen
|- BackendExecutionDecision                         # [returned], frozen
|  |- backend_identity: BackendIdentity
|  |- requested_state: BackendStateValues
|  |- selected_state: BackendStateValues
|  |- source: BackendDecisionSource
|  `- rationale: tuple[DecisionReason, ...]
|- BackendExecutionEvidence                         # [constructible], frozen
|  |- scientific_state_identity: ScientificStateIdentity
|  |- backend_identity: BackendIdentity
|  |- requested_state: BackendStateValues
|  |- executed_state: BackendStateValues
|  `- artifact_digests: tuple[str, ...]
|- ExecutionPolicySet(policy_ids: tuple[str, ...] =
|                      ("EXECUTION-001", "DEFAULT-004"))
|                                                    # [constructible], frozen
|- BackendStateProjection                           # [returned], frozen
|  |- scientific_state_identity: ScientificStateIdentity
|  |- facts_identity: MolecularFactsIdentity
|  |- backend_identity: BackendIdentity
|  |- effective_electron_count: int
|  |- charge: int
|  |- uhf: int | None
|  |- multiplicity: int | None
|  |- projection_convention: str
|  |- execution_decision: BackendExecutionDecision
|  `- executability: BackendExecutability
|- BackendProjectionStatus                          # [enum]
|  |- NOT_PROJECTED
|  |- PROJECTED
|  |- PROJECTION_CONFLICT
|  `- BACKEND_UNSUPPORTED
|- BackendProjectionResult                          # [returned], frozen
|  |- status: BackendProjectionStatus
|  |- backend_identity: BackendIdentity
|  |- request: BackendStateRequest
|  |- projection: BackendStateProjection | None
|  `- findings: tuple[BackendProjectionFinding, ...]
`- ScientificStateWorkflowResult                    # [returned], frozen
   |- scientific_state: ScientificStateBundle
   `- backend_projection: BackendProjectionResult

hotpot.plugins.xtb
|
|- GFNXTBMethod(str, Enum)                         # [enum]
|  |- GFN0_XTB = "gfn0"
|  |- GFN1_XTB = "gfn1"
|  `- GFN2_XTB = "gfn2"
|- XTBApplicabilityError(XTBInputError)             # [exception]
|- XTBArtifact                                      # [returned], frozen
|  |- name: str
|  |- path: Path
|  |- sha256: str
|  `- size_bytes: int
|- XTBBackendInfo                                   # [returned], frozen
|  |- executable: Path
|  |- version: str
|  |- revision: str | None
|  |- executable_sha256: str
|  |- probe_result: NativeProcessResult
|  |- gfn_xtb_max_atomic_number: int
|  `- gfnff_max_atomic_number: int | None
|- XTBError(RuntimeError)                           # [exception]
|- XTBExecutableError(XTBError)                     # [exception]
|  |- __init__(message: str,
|  |           process_result: NativeProcessResult | None = None) -> None
|  `- process_result: NativeProcessResult | None
|- XTBGeometry                                      # [constructible], frozen
|  |- symbols: tuple[str, ...]
|  `- coordinates: tuple[tuple[float, float, float], ...]
|- XTBExecutionError(XTBError)                      # [exception]
|  |- __init__(message: str, report: XTBRunReport) -> None
|  `- report: XTBRunReport
|- XTBInputError(XTBError)                          # [exception]
|- XTBMethod(str, Enum)                             # [enum]
|  |- GFNFF = "gfnff"
|  |- GFN0_XTB = "gfn0"
|  |- GFN1_XTB = "gfn1"
|  `- GFN2_XTB = "gfn2"
|- XTBParsedResult                                  # [returned], frozen
|  |- energy_hartree: float
|  |- gradient_norm: float | None
|  |- partial_charges: tuple[float, ...] | None
|  |- optimized_geometry: XTBGeometry | None
|  `- atom_order_verified: bool
|- XTBRequest                                       # [constructible], frozen
|  |- backend_info: XTBBackendInfo
|  |- method: XTBMethod
|  |- task: XTBTask
|  |- input_path: Path
|  |- work_directory: Path
|  |- charge: int
|  |- unpaired_electrons: int | None
|  |- environment: Mapping[str, str]
|  `- timeout_seconds: float | None = None
|- XTBResultError(XTBError)                         # [exception]
|  |- __init__(message: str, report: XTBRunReport) -> None
|  `- report: XTBRunReport
|- XTBRunReport                                     # [returned], frozen
|  |- backend_info: XTBBackendInfo
|  |- requested_method: XTBMethod
|  |- effective_method: XTBMethod
|  |- task: XTBTask
|  |- charge: int
|  |- unpaired_electrons: int | None
|  |- work_directory: Path
|  |- argv: tuple[str, ...]
|  |- return_code: int
|  |- stdout: str
|  |- stderr: str
|  |- elapsed_seconds: float
|  |- process_succeeded: bool
|  |- converged: bool
|  |- artifacts: Mapping[str, XTBArtifact]
|  |- provenance: ProcessProvenance
|  |- energy_hartree: float | None = None
|  |- energy_unit: str = "hartree"
|  |- gradient_norm: float | None = None
|  |- atom_order_verified: bool = False
|  |- coordinates_committed: bool = False
|  |- scientific_state_identity: ScientificStateIdentity | None = None
|  |- facts_identity: MolecularFactsIdentity | None = None
|  |- projection_convention: str | None = None
|  `- workspace_retained: bool = False
|- XTBStreamError(ValueError)                       # [exception]
|- XTBStreamMetadata                                # [constructible], frozen
|  |- total_charge: int
|  |- unpaired_electrons: int | None
|  |- method: XTBMethod
|  |- energy_hartree: float
|  `- provenance: XTBStreamProvenance
|- XTBStreamProvenance                              # [constructible], frozen
|  |- backend_version: str
|  |- backend_revision: str | None
|  |- executable_sha256: str
|  |- scientific_state_identity: ScientificStateIdentity | None
|  |- facts_identity: MolecularFactsIdentity | None
|  `- projection_convention: str | None
|- XTBStreamRecord                                  # [constructible], frozen
|  |- mol: Molecule
|  `- metadata: XTBStreamMetadata | None = None
|- XTBTask(str, Enum)                               # [enum]
|  |- SINGLEPOINT = "singlepoint"
|  `- OPTIMIZE = "optimize"
|- commit_xtb_coordinates(
|      mol: Molecule,
|      result: XTBParsedResult,
|  ) -> None
|- molecule_to_xtb_geometry(mol: Molecule) -> XTBGeometry
|- metadata_from_report(
|      report: XTBRunReport,
|      previous: XTBStreamMetadata | None = None,
|  ) -> XTBStreamMetadata
|- parse_xtb_artifacts(
|      report: XTBRunReport,
|      expected_geometry: XTBGeometry,
|  ) -> XTBParsedResult
|- probe_xtb_backend(
|      executable: str | os.PathLike[str] | None = None,
|      *,
|      environment: Mapping[str, str] | None = None,
|      timeout_seconds: float | None = 30.0,
|  ) -> XTBBackendInfo
|- prepare_xtb_input(mol: Molecule, input_path: Path) -> XTBGeometry
|- run_xtb(request: XTBRequest) -> XTBRunReport
|- run_gfn_xtb(
|      mol: Molecule,
|      *,
|      state: BackendStateProjection,
|      method: GFNXTBMethod = GFNXTBMethod.GFN2_XTB,
|      task: XTBTask = XTBTask.OPTIMIZE,
|      executable: str | Path | None = None,
|      environment: Mapping[str, str] | None = None,
|      timeout_seconds: float | None = None,
|      work_directory: Path | None = None,
|      keep_work_directory: bool = False,
|  ) -> XTBRunReport
|- run_gfnff(
|      mol: Molecule,
|      *,
|      state: BackendStateProjection,
|      task: XTBTask = XTBTask.OPTIMIZE,
|      executable: str | Path | None = None,
|      environment: Mapping[str, str] | None = None,
|      timeout_seconds: float | None = None,
|      work_directory: Path | None = None,
|      keep_work_directory: bool = False,
|  ) -> XTBRunReport
|- read_sdf_records(text: str) -> tuple[XTBStreamRecord, ...]
|- validate_element_support(
|      backend_info: XTBBackendInfo,
|      method: XTBMethod,
|      atomic_numbers: Iterable[int],
|  ) -> None
|- write_sdf_records(records: Iterable[XTBStreamRecord]) -> str
`- XTBBackendContext                                # [constructible], frozen
   |- backend_info: XTBBackendInfo
   |- method: XTBMethod
   |- context.identity -> BackendIdentity
   |- context.scientific_operation_requirements
   |      -> ScientificOperationRequirements
   |- context.projection_convention -> str
   |- context.electron_count_definition -> str
   `- context.effective_nuclear_charge(atomic_number: int) -> int

xTB reachable contracts in canonical hotpot.plugins._harness namespace
|                                                    # not re-exported by xTB
|- NativeProcessResult                              # [returned], frozen
|  |- argv: tuple[str, ...]
|  |- cwd: Path
|  |- return_code: int
|  |- stdout: str
|  |- stderr: str
|  `- elapsed_seconds: float
|- ArtifactProvenance(path: Path, sha256: str)       # [returned], frozen
`- ProcessProvenance                                # [returned], frozen
   |- executable: Path
   |- executable_sha256: str
   |- argv: tuple[str, ...]
   |- cwd: Path
   |- return_code: int
   |- stdout: str
   |- stderr: str
   |- elapsed_seconds: float
   `- artifacts: tuple[ArtifactProvenance, ...]

hotpot.pipeline.state_recovery
|
|- decide_state_recovery(
|      state: ScientificStateBundle,
|      backend: BackendContext,
|      *,
|      request: BackendStateRequest = BackendStateRequest(),
|      execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
|      execution_history: tuple[BackendExecutionEvidence, ...],
|      recovery_policies: RecoveryPolicySet = RecoveryPolicySet(),
|      recovery_budget: RecoveryBudget = DEFAULT_RECOVERY_BUDGET,
|  ) -> RecoveryDecision
|- RecoveryAction                                   # [enum]: STOP, RETRY
|- RecoveryPolicySet(policy_ids: tuple[str, ...] = ())
|                                                    # [constructible], frozen
|- RecoveryBudget(max_backend_attempts: int)         # [constructible], frozen
|- DEFAULT_RECOVERY_BUDGET: RecoveryBudget           # [constant]
`- RecoveryDecision                                 # [returned], frozen
   |- decision.action -> RecoveryAction
   |- decision.projection -> BackendStateProjection | None
   `- decision.rationale -> tuple[DecisionReason, ...]
```

The complete target xTB export list is the existing production list with one
new final name:

```python
hotpot.plugins.xtb.__all__ = [
    "GFNXTBMethod",
    "XTBApplicabilityError",
    "XTBArtifact",
    "XTBBackendInfo",
    "XTBError",
    "XTBExecutableError",
    "XTBGeometry",
    "XTBExecutionError",
    "XTBInputError",
    "XTBMethod",
    "XTBParsedResult",
    "XTBRequest",
    "XTBResultError",
    "XTBRunReport",
    "XTBStreamError",
    "XTBStreamMetadata",
    "XTBStreamProvenance",
    "XTBStreamRecord",
    "XTBTask",
    "commit_xtb_coordinates",
    "molecule_to_xtb_geometry",
    "metadata_from_report",
    "parse_xtb_artifacts",
    "probe_xtb_backend",
    "prepare_xtb_input",
    "run_xtb",
    "run_gfn_xtb",
    "run_gfnff",
    "read_sdf_records",
    "validate_element_support",
    "write_sdf_records",
    "XTBBackendContext",
]
```

The low-level `run_xtb(XTBRequest)` invocation, native process, parser and
artifact contracts remain intact. The two high-level node functions migrate to
the signatures above: they consume one already validated
`BackendStateProjection` and never repeat scientific inference. Explicit
scientific $Q$/$2S$ enters through `StateRequest`; explicit execution UHF enters
through `BackendStateRequest`. Bare backend-expert calls remain available
through `run_xtb(XTBRequest)`. Each high-level node verifies that the
projection's backend identity, method and scientific-state identity match its
invocation and the exact molecule facts before launch; no compatibility
overload remains.

`XTBBackendContext.effective_nuclear_charge(atomic_number)` returns the atomic
number for the current all-electron xTB adapter. The protocol remains explicit
so a future ECP backend can supply its own effective nuclear charge without
changing HARD-013 or the scientific-state contracts.

The backend report/stream provenance also migrates away from legacy
`ChargeInferenceSource` and `SpinInferenceSource`. High-level nodes populate
`scientific_state_identity`, `facts_identity` and `projection_convention`;
low-level expert runs leave all three as `None`, and the fields must otherwise
be present together.
`metadata_from_report()` transports them unchanged. It may carry a prior UHF
across a GFN-FF node only when both records have the same non-`None`
`ScientificStateIdentity` and `MolecularFactsIdentity`, the same total charge
and a compatible projection convention; anonymous low-level runs never inherit
spin implicitly. Executed
charge/UHF remain in their existing numeric fields, while scientific evidence,
assumptions and rule traces remain in `ScientificStateBundle`; the backend
report no longer duplicates `charge_source`, `spin_source`,
`state_assumptions` or `fragment_charges`.

All exposed scientific-state data classes are immutable. Returned objects
remain public contracts even when they are not first-level re-exports. Public
scientific-state inference, composition and projection functions never mutate
`Molecule`; the separately retained MCA facade keeps its documented property
updates. Expected scientific, projection and backend outcomes use the typed
results above;
malformed calls and programming defects remain exceptions. `legality.py`,
profile internals, frontier joins, validators and transport helpers are not
additional public APIs.

## 12. Migration from current production code

Current implementation facts must be rechecked at the implementation commit.
At the preparation baseline:

- `electronic_state/resolver.py` combines charge and spin selection;
- `spin.py` provides a lowest-parity implementation;
- `formal_charges.py` mixes limited-domain atom formal-charge assignment with
  molecular-charge output;
- `molecular_charge.py` is a legacy estimator;
- xTB CLI, stage and workflow each contain state-resolution wiring; and
- the xTB adapter writes XYZ for every method.

The migration must explicitly close these observed defects rather than merely
rename the current classes:

| Current location/behavior | Problem to remove | Target owner |
|---|---|---|
| `electronic_state/resolver.py`: explicit $Q$ invokes a constrained atom-formal-charge assignment | Legal total charge is incorrectly coupled to finding one Lewis allocation | `charge.net_charge` validates $Q$ independently; Lewis inference is optional |
| `electronic_state/resolver.py`: explicit UHF passes through `infer_lowest_spin()` | Backend-specific UHF and hard electron counting are coupled to scientific resolution | scientific layer resolves $2S$; xTB projection validates explicit UHF or applies DEFAULT-004 after `BackendContext` is known |
| preconstructed `ElectronicState` accepted by workflow | Some entry paths can bypass molecular-spin or backend-UHF validation | scientific composition applies HARD-006; backend projection independently applies HARD-013 without re-inferring the scientific state |
| `multiplicity = UHF + 1` stored as a general relation | xTB projection convention is presented as universal science | relation exists only in a named highest-$M_S$ projection |
| `formal_charges.py` metal default calls `Atom.get_formal_charge()` | element default valence, atom formal charge and OS are conflated | optional metal/OS rules with distinct result types |
| flat charge source in the current result | mixed per-component sources and rule evidence are lost | candidate- and component-level provenance |
| charge and spin are resolved directly into one flat `ElectronicState` | no jointly attributable charge/OS/electronic-state path exists | factorized candidate space plus coupled-state composition |
| lowest-parity spin runs whenever no estimator is supplied | a generic legal fallback preempts stronger domain preferences | named default decision applies reviewed preferences before DEFAULT-004 |
| `cli.py` and `stage.py` repeat carried/explicit/estimator merging | entry points can drift | one resolver plus backend projector |
| `run_gfnff()` accepts `ChargeInferenceResult`/raw charge and `run_gfn_xtb()` accepts old `ElectronicState`/raw charge/UHF | high-level backend nodes can bypass the canonical resolver/projector and depend on types removed by this migration | both high-level nodes require one validated `BackendStateProjection`; low-level experts use `run_xtb(XTBRequest)` |
| xTB reports/stream provenance expose legacy charge/spin source enums and duplicate state assumptions | removed resolver types leak into transport and scientific evidence can drift from its bundle | bind reports to optional `scientific_state_identity` plus `projection_convention`; keep evidence only in `ScientificStateBundle` |
| GFN-FF path relabels carried charge as explicit | state provenance is degraded at a method transition | transport the original selected candidate identity/source |
| every xTB method receives XYZ | GFN-FF reconstructs topology from coordinates | method-specific SDF/MOL versus XYZ adapters |

Legacy Gaussian helpers such as `Molecule.calc_mol_default_charge()` and
`default_spin_mult` must be inventoried before claiming project-wide
migration. They may temporarily remain behind a documented adapter, but they
must not become a hidden second scientific-state implementation.

Migration rules:

1. introduce immutable contracts and hard validators before behavior changes;
2. keep the existing public facade only when it has a real published contract;
3. route all state control through one resolver;
4. split scientific inference from xTB projection;
5. implement factorized joint composition and coupled hard validation;
6. implement the two named decision policies;
7. add optional rules one independently revertible commit at a time;
8. switch GFN-FF to its native topology-preserving adapter;
9. update pipeline gate semantics;
10. migrate consumers and tests; and
11. delete replaced private estimators, duplicated CLI/stage wiring and
    compatibility branches after the new path owns all consumers.

No broad `try/except`, silent neutral/low-spin fallback or parallel hidden
implementation is permitted.

## 13. Implementation phases

| Phase | Deliverable | Required independent acceptance |
|---:|---|---|
| 0 | freeze this plan, rule specs and source digests | docs link check; all rule IDs unique |
| 1 | common facts, identity, result/status contracts | serialization and identity tests |
| 2 | hard charge/OS/molecular-spin legality | formula, conflict and conservation tests |
| 3 | explicit input API, providers and `STRICT` | provider independence; no empirical rule activated; ambiguity preserved |
| 4 | registry, dependency DAG, rule abstraction and trace | toggle/action isolation; cycle rejection; deterministic layered execution |
| 5 | factorized candidate space, budgeted proposal generation, joint composition and coupled hard validation | derived-value lineage, domain scope, materialization status, frontier certification, partial constraints, multimetal paths and budget tests |
| 6 | `COORDINATION_SCREENING_V1` decision policy and approved optional rules | one selected domain-valid path; typed abstention; one commit and focused tests per rule |
| 7 | backend execution decisions, xTB projection and method-specific adapters | scientific $2S$ versus UHF separation; DEFAULT-004 lifecycle; SDF GFN-FF and XYZ GFN1/2 native parity |
| 8 | pipeline recovery/gate separation | finite warning continuation and final acceptance tests |
| 9 | consumer migration and obsolete-code cleanup | full unit/CLI/package tests |
| 10 | scientific, throughput and performance validation | 187/182 plus large-corpus state-resolution report with stage-resolved outcomes |

Each phase is independently revertible. Optional rules are not bundled into a
single commit.

## 14. Acceptance matrix

### 14.1 Hard legality

- explicit positive, negative and zero charge;
- atom/component sums both matching and conflicting with $Q$;
- molecular electron-count/$2S$ bounds and parity across positive and negative ions;
- backend effective-electron/UHF bounds and parity under a named projection;
- explicit UHF versus molecular-$2S$/multiplicity agreement and conflict under
  a named projection;
- complete and partial oxidation-state assignments, including impossible
  extensions;
- multimetal global conservation;
- coupled-path consistency when individually legal Q, OS and spin candidates
  are mutually incompatible;
- graph/protonation/component identity invalidation; and
- ECP effective-electron projection independent of xTB all-electron parity.

### 14.2 Policy and rule system

- `STRICT` never enables optional rules;
- profile defaults remain explicitly attributable;
- `COORDINATION_SCREENING_V1` selects exactly one hard-valid path for every
  statically in-domain, decision-certifiable fixture without changing its
  scientific-resolution status; an in-domain non-equivalent tie remains
  in-domain and returns `NO_AUTOMATIC_STATE`;
- static profile eligibility, dynamic rule applicability, decision
  certifiability and effective-profile identity are separately machine-readable;
  toggle/configuration overrides cannot inherit an unvalidated coverage claim;
- out-of-domain, empty and uncertified resource-limited spaces return
  `NO_AUTOMATIC_STATE` without an interactive pause;
- disabling a rule removes only that rule's outcomes;
- no optional result bypasses hard revalidation;
- missing dependencies and dependency cycles are explicit;
- rules in one layer consume the same immutable context;
- factorized candidate domains, expanded paths, rejections, selected identity
  and truncation state survive serialization;
- singleton paths from non-exhaustive domains never report scientific
  `UNIQUE`;
- frontier-bound fixtures distinguish `DECISION_CERTIFIED`,
  `TRUNCATED_BUT_DECISION_CERTIFIED` and
  `NO_CERTIFIED_DECISION`, `RESOURCE_LIMIT_NO_CERTIFIED_DECISION` and
  `NOT_EVALUATED`;
- every automatically selected in-domain path is composed only from finite
  `EXHAUSTIVE`/`DOMAIN_BOUNDED` decision-relevant factors with certified
  frontier bounds; `OPEN` or unboundable factors never receive a canonical
  truncation decision;
- explicit partial Q/OS/$2S$ constraints prune before scientific path
  expansion, while explicit backend UHF is validated during projection;
- atom order and rule registration order do not change the selected path;
- scalar requests constrain one value, while multi-value request candidates
  define an allowed domain and do not create a false source conflict;
- canonical identity resolves only decision-equivalent paths; a surviving
  operation-relevant $Q$/OS/$2S$ tie returns `NO_AUTOMATIC_STATE`;
- `SELECT_STATE` is the only action carrying a selected identity;
- analysis-only requirements return `NOT_REQUESTED`, while effective required
  and allowed-unresolved quantity sets are disjoint;
- rule order is deterministic and every action is traceable;
- AI rule abstention is not converted into a value; and
- no rule mutates `Molecule` or calls a backend.

### 14.3 Chemistry and representations

- ordinary closed-shell main-group molecules and radicals;
- salts, counterions, neutral L donors and anionic X ligands;
- Eu(II/III), Am(III), Cm(III), Th(IV), actinyl and exceptions;
- Fe(II/III), high-/low-spin alternatives and multinuclear coupling;
- peroxide/superoxide, NO/O2, dithiolene and quinone alternatives;
- atom-order, resonance/Kekule, explicit-H and SDF/MOL2 metamorphic tests;
- representation information loss is explicit, never silently repaired.

The chemistry matrix includes complete-path fixtures for neutral L donors,
anionic X ligands, explicit counterions, Eu(II/III), Ce(III/IV), ordinary
f-block Hund selection, redox-active abstention, mixed-valence/multimetal
systems and partial user overrides. `c1ncccc1O[Eu]` is one minimal fixture, not
the design basis.

### 14.4 Backend and pipeline

- GFN-FF receives explicit adjacency and charge;
- GFN1/2 receives charge and legal UHF;
- method/element unsupported status is distinct from scientific conflict;
- native output commits coordinates only after atom-map validation;
- intermediate geometry warning does not automatically stop downstream work;
- non-finite and fatal preflight states stop only the affected route; and
- the complete 187/182 corpus reports source-invalid, inference conflict,
  backend failure, intermediate warning and final geometry separately;
- a large-corpus dry resolution benchmark records throughput, peak memory,
  mean/max expanded paths, truncated fraction, automatic-selection coverage
  and deterministic replay; and
- profile activation requires a preregistered minimum automatic-selection
  coverage and maximum rates for conflict, out-of-domain and uncertified-
  resource outcomes on representative held-out coordination corpora;
- the default profile launches one backend state per record unless a separately
  enabled bounded recovery policy requests another.

## 15. Complete rule list

Checkboxes in the `COORDINATION_SCREENING_V1` column are the **planned profile
defaults**, not claims that the rules are implemented. Every optional rule
still requires its individual expert review before activation.

`Applied stage` is normative. OPTIONAL entries use the exact `RulePhase` from
their specifications. DEFAULT entries use the closed resolver/backend lifecycle
vocabulary below. HARD and supporting entries use that vocabulary and may cite
an exact `RulePhase.*` when the invariant/provider is applied inside that rule
phase. The two explicitly defined non-runtime scopes are reserved for
cross-cutting invariants. A semicolon means the same rule is applied at every
listed boundary.

### Closed lifecycle-stage vocabulary

The following is the closed ordered runtime vocabulary. OPTIONAL rows use the
separately closed `RulePhase` enum from Section 8.1; HARD/supporting rows may
reference either closed vocabulary under the rule above.

| Lifecycle stage | Exact boundary |
|---|---|
| `FACT_EXTRACTION` | immutable molecular facts and chemical identity are created |
| `EXPLICIT_AND_SOURCE_BINDING` | caller/source constraints are bound to those facts |
| `SCIENTIFIC_PREVALIDATION` | all currently complete explicit/source quantities pass hard validation |
| `PROFILE_ACTIVATION` | effective profile, provider and rule configuration is frozen |
| `POST_RULE_HARD_VALIDATION` | one dependency-layered optional-rule result set has been merged |
| `CHARGE_OS_COMPOSITION` | charge, component/counting and OS factors are joined |
| `COUPLED_STATE_COMPOSITION` | complete charge/OS/electronic-state paths are joined |
| `CANDIDATE_SEARCH` | bounded lazy/best-first path expansion and certification run |
| `DECISION_AUTHORITY_FILTER` | operation-complete paths are filtered by evidence authority |
| `DECISION_OPERATION_COMPLETENESS` | unresolved quantities are compared with operation requirements |
| `DECISION` | one named state policy emits its terminal action |
| `POST_DECISION_RETENTION` | candidate evidence/frontier and the selected path are serialized |
| `BACKEND_EXECUTION_SELECTION` | exact-context history and execution-only UHF policies run |
| `BACKEND_PROJECTION_VALIDATION` | scientific state is projected and HARD-008/HARD-013 are validated |
| `BACKEND_PREFLIGHT` | executable, method, element and projected-state support are checked before launch |
| `BACKEND_INPUT_TRANSPORT` | method-native input is serialized |
| `BACKEND_RUNTIME_RECORDING` | process return, convergence and artifacts are recorded without reinterpretation |
| `BACKEND_OUTPUT_COORDINATE_COMMIT` | atom identity/order is verified before coordinate commit |
| `INTERMEDIATE_QUALITY_GATE` | finite intermediate geometry is classified |
| `INTERMEDIATE_RECOVERY_DECISION` | continuation/rollback/stop is selected for that intermediate |
| `POST_BACKEND_FAILURE_RECOVERY_DECISION` | an opt-in bounded retry policy selects the next legal projection or stops |
| `FINAL_ACCEPTANCE` | final product geometry and route outcome are accepted/rejected |
| `POST_IDENTITY_CHANGING_OPERATION` | chemical identity is recomputed after an operation capable of changing it |

Two non-runtime scopes are also closed vocabulary, not pseudo-stages:

| Application scope | Exact meaning |
|---|---|
| `ALL_STATE_DATA_BOUNDARIES` | construction, validation, composition, serialization and transport of every state quantity |
| `ARCHITECTURE_REVIEW` | module-dependency/static API tests that enforce separation and one orchestrator |

### 15.1 Hard rules

| Rule | Type | Applied stage / scope | Meaning |
|---|---|---|---|
| [HARD-001](rules/HARD-001-boundary-identity.md) | hard | `FACT_EXTRACTION`; `CHARGE_OS_COMPOSITION`; `COUPLED_STATE_COMPOSITION`; `BACKEND_PROJECTION_VALIDATION`; `POST_IDENTITY_CHANGING_OPERATION` | bind facts and states to one exact chemical boundary |
| [HARD-002](rules/HARD-002-charge-conservation.md) | hard | `EXPLICIT_AND_SOURCE_BINDING`; `POST_RULE_HARD_VALIDATION`; `CHARGE_OS_COMPOSITION`; `COUPLED_STATE_COMPOSITION` | integer net charge and atom/component conservation |
| [HARD-003](rules/HARD-003-charge-quantity-separation.md) | hard | `ALL_STATE_DATA_BOUNDARIES` | separate net, formal, partial, OS and counting quantities |
| [HARD-004](rules/HARD-004-oxidation-state-conservation.md) | hard | `CHARGE_OS_COMPOSITION`; `COUPLED_STATE_COMPOSITION` | integer, globally charge-conserving OS assignments |
| [HARD-005](rules/HARD-005-counting-convention-exclusivity.md) | hard | `RulePhase.COORDINATION_COUNTING`; `CHARGE_OS_COMPOSITION` | do not mix ionic, donor-return and CBC bookkeeping |
| [HARD-006](rules/HARD-006-electron-spin-legality.md) | hard | `SCIENTIFIC_PREVALIDATION`; `COUPLED_STATE_COMPOSITION` | validate molecular electron count, $2S$ bounds and parity whenever charge/spin inputs are complete |
| [HARD-007](rules/HARD-007-explicit-state-consistency.md) | hard | `EXPLICIT_AND_SOURCE_BINDING`; `SCIENTIFIC_PREVALIDATION`; `CHARGE_OS_COMPOSITION`; `COUPLED_STATE_COMPOSITION` | detect conflicts among explicit/source scientific settings and constrain every compatible joint path; explicit UHF is deferred to HARD-013 |
| [HARD-008](rules/HARD-008-backend-projection.md) | hard | `BACKEND_PROJECTION_VALIDATION` | separate scientific state from backend representation |
| [HARD-009](rules/HARD-009-state-invalidation.md) | hard | `BACKEND_OUTPUT_COORDINATE_COMMIT`; `POST_IDENTITY_CHANGING_OPERATION` | preserve state after verified coordinate-only updates and invalidate it after identity changes |
| [HARD-010](rules/HARD-010-module-independence.md) | hard | `ARCHITECTURE_REVIEW` | independent inference modules and one orchestrator |
| [HARD-011](rules/HARD-011-method-specific-transport.md) | hard | `BACKEND_INPUT_TRANSPORT`; `BACKEND_OUTPUT_COORDINATE_COMMIT` | preserve authoritative topology across xTB adapters |
| [HARD-012](rules/HARD-012-failure-and-gate-semantics.md) | hard | `SCIENTIFIC_PREVALIDATION`; `POST_RULE_HARD_VALIDATION`; `CHARGE_OS_COMPOSITION`; `COUPLED_STATE_COMPOSITION`; `BACKEND_PROJECTION_VALIDATION`; `BACKEND_PREFLIGHT`; `BACKEND_RUNTIME_RECORDING`; `INTERMEDIATE_QUALITY_GATE`; `FINAL_ACCEPTANCE` | separate conflict, projection, support, runtime and geometry states |
| [HARD-013](rules/HARD-013-backend-electron-uhf-legality.md) | hard | `BACKEND_PROJECTION_VALIDATION`; `BACKEND_PREFLIGHT` | backend effective-electron count and xTB UHF legality |

### 15.2 Default policies

| Rule | Type | Registry | Applied stage | Policy |
|---|---|---|---|---|
| [DEFAULT-001](rules/DEFAULT-001-strict.md) | default decision | `StatePolicyRegistry` | `DECISION` | after complete hard validation, preserve uncertainty and decline execution without an attributable state |
| [DEFAULT-002](rules/DEFAULT-002-coordination-screening-v1.md) | default decision | `StatePolicyRegistry` | `PROFILE_ACTIVATION`; `DECISION` | select one hard-valid coupled path for routine statically applicable, decision-certifiable screening |
| [DEFAULT-003](rules/DEFAULT-003-source-authority.md) | default decision | `StatePolicyRegistry` | `DECISION_AUTHORITY_FILTER` before domain ranks/tie handling | compare operation-complete coupled paths by validated evidence authority |
| [DEFAULT-004](rules/DEFAULT-004-lowest-legal-xtb-uhf.md) | default decision | `ExecutionPolicyRegistry` | `BACKEND_EXECUTION_SELECTION` | after explicit-UHF prevalidation and before final HARD-013 projection validation, choose the lowest legal UHF only as the final execution-only spin fallback |
| [DEFAULT-005](rules/DEFAULT-005-oxidation-ambiguity.md) | default decision | `StatePolicyRegistry` | `DECISION_OPERATION_COMPLETENESS` | decide whether unresolved OS blocks the requested operation |
| [DEFAULT-006](rules/DEFAULT-006-candidate-preservation.md) | default decision | `StatePolicyRegistry` | `CANDIDATE_SEARCH`; `POST_DECISION_RETENTION` | decide what candidate evidence is retained after one path is selected |
| [DEFAULT-007](rules/DEFAULT-007-intermediate-recovery.md) | default decision | `PipelineRecoveryPolicyRegistry` | `INTERMEDIATE_RECOVERY_DECISION` before next stage/final acceptance | decide continuation versus stop for finite intermediate geometry |

`DEFAULT` is a document namespace for decisions, not one runtime registry.
DEFAULT-001/002/003/005/006 belong to `StatePolicyRegistry`; DEFAULT-004 belongs
to backend `ExecutionPolicyRegistry`; DEFAULT-007 belongs to
`PipelineRecoveryPolicyRegistry`. None is registered as an optional scientific
inference rule.

### 15.3 Optional empirical and AI rules

All OPTIONAL entries are owned by `InferenceRuleRegistry`.

| Rule | Type | Applied `RulePhase` | `COORDINATION_SCREENING_V1` | Recommended scene |
|---|---|---|:---:|---|
| [OPTIONAL-001](rules/OPTIONAL-001-main-group-closed-shell.md) | empirical | `ELECTRONIC_STATE_CANDIDATE` | [x]* | ordinary supported closed-shell main-group molecules or coordination-separated components after Lewis provider/source review |
| [OPTIONAL-002](rules/OPTIONAL-002-monoatomic-ions.md) | empirical | `CHARGE_EVIDENCE` | [x] | explicit disconnected common counterions |
| [OPTIONAL-003](rules/OPTIONAL-003-neutral-l-donors.md) | empirical | `COORDINATION_COUNTING` | [x] | ordinary L-type donor ligands with fixed protonation |
| [OPTIONAL-004](rules/OPTIONAL-004-anionic-x-ligands.md) | empirical | `COORDINATION_COUNTING` | [x] | ordinary X-type ligands with unambiguous protonation |
| [OPTIONAL-005](rules/OPTIONAL-005-lanthanide-iii.md) | empirical | `OXIDATION_STATE_CANDIDATE` | [x] | non-redox lanthanide coordination screening; exceptions retained |
| [OPTIONAL-006](rules/OPTIONAL-006-americium-curium-iii.md) | empirical | `OXIDATION_STATE_CANDIDATE` | [x] | ordinary Am/Cm donor complexes |
| [OPTIONAL-007](rules/OPTIONAL-007-thorium-iv.md) | empirical | `OXIDATION_STATE_CANDIDATE` | [x] | ordinary molecular Th coordination complexes |
| [OPTIONAL-008](rules/OPTIONAL-008-actinyl-states.md) | empirical | `OXIDATION_STATE_CANDIDATE` | [ ] | recognized actinyl motifs; specialist review required |
| [OPTIONAL-009](rules/OPTIONAL-009-ionic-configuration-spin.md) | empirical | `ELECTRONIC_STATE_CANDIDATE` | [x] | hard-valid partial charge/OS paths with registered ionic configurations |
| [OPTIONAL-010](rules/OPTIONAL-010-transition-metal-ligand-field.md) | empirical | `ELECTRONIC_STATE_RANKING` | [ ] | calibrated ligand-field/geometry domains only |
| [OPTIONAL-011](rules/OPTIONAL-011-f-block-hund-prior.md) | empirical | `ELECTRONIC_STATE_RANKING` | [x]* | localized f-block ions inside the reviewed element/OS/domain registry |
| [OPTIONAL-012](rules/OPTIONAL-012-noninnocent-ligands.md) | empirical | `REDOX_CANDIDATE` | [ ] | curated NO/O2/dithiolene/quinone motifs |
| [OPTIONAL-013](rules/OPTIONAL-013-bond-valence-ranking.md) | empirical | `OXIDATION_STATE_RANKING` | [ ] | calibrated ionic/inorganic domains |
| [OPTIONAL-014](rules/OPTIONAL-014-trusted-history.md) | empirical | `SOURCE_EVIDENCE` | [x] | exact-identity backend-independent scientific source records |
| [OPTIONAL-015](rules/OPTIONAL-015-ai-state-prior.md) | AI | model specification selects exactly one existing `RulePhase` | [ ] | versioned calibrated model within its applicability domain |
| [OPTIONAL-016](rules/OPTIONAL-016-oxidation-occurrence-ranking.md) | empirical | `OXIDATION_STATE_RANKING` | [ ] | domain-matched occurrence corpora with held-out validation |

### 15.4 Supporting providers, execution and recovery policies (not rules)

| Component | Type | Applied stage | Default | Recommended scene |
|---|---|---|:---:|---|
| [PROVIDER-001](providers/PROVIDER-001-main-group-lewis-solver.md) | scientific solver | `RulePhase.CHARGE_EVIDENCE` | [x]* | Lewis candidate generation for coordination-separated ligand components inside its reviewed main-group domain |
| [PROVIDER-002](providers/PROVIDER-002-element-oxidation-registry.md) | data provider | `RulePhase.OXIDATION_STATE_CANDIDATE` | [ ] | explicit OS-domain expansion |
| [PROVIDER-003](providers/PROVIDER-003-neutral-atom-reference.md) | data provider | `RulePhase.SOURCE_EVIDENCE`; `RulePhase.ELECTRONIC_STATE_CANDIDATE` | [ ] | isolated-atom reference lookup followed by consumer projection; the provider does not choose the final electronic state |
| [EXECUTION-001](execution/EXECUTION-001-exact-context-history.md) | execution selection | `BACKEND_EXECUTION_SELECTION` | [x] when supplied | exact scientific state plus backend/method/ECP context |
| [RECOVERY-001](recovery/RECOVERY-001-legal-state-retry.md) | pipeline recovery | `POST_BACKEND_FAILURE_RECOVERY_DECISION` | [ ] | bounded explicit state scan/retry |

`[x]*` means conditionally enabled only when the component satisfies the
versioned applicability artifact. Failing that test is visible abstention and
does not activate a broader fallback interpretation of the same rule/provider.

### 15.5 Crosswalk from the approved Overview discussion

| Earlier label | Normative destination |
|---|---|
| R01 closed-shell main group | OPTIONAL-001 |
| R02 lowest legal UHF | DEFAULT-004 |
| R03 common monoatomic ions | OPTIONAL-002 |
| R04 neutral L donors | OPTIONAL-003 |
| R05 anionic X ligands | OPTIONAL-004 |
| R06 lanthanide(III) | OPTIONAL-005 |
| R07 Am/Cm(III) | OPTIONAL-006 |
| R08 Th(IV) | OPTIONAL-007 |
| R09 actinyl candidates | OPTIONAL-008 |
| R10 ionic configuration | OPTIONAL-009 |
| R11 transition-metal ligand field | OPTIONAL-010 |
| R12 f-block Hund/free-ion prior | OPTIONAL-011 |
| R13 non-innocent alternatives | OPTIONAL-012 |
| R14 bond-valence ranking | OPTIONAL-013 |
| R15 trusted history | OPTIONAL-014 for scientific sources; EXECUTION-001 for backend-specific history |
| R16 legal-state alternatives | RECOVERY-001, reclassified because retry is controller-owned |

OPTIONAL-015 (AI) and OPTIONAL-016 (occurrence ranking) are additional disabled
extension points extracted from the archived designs; neither is an approved
profile default.
