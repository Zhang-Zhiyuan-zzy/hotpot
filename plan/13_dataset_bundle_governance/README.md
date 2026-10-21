# Stage 13: Bundle-centered dataset governance

Status: complete. The 252,665-Case managed store was rebuilt, parity-checked,
certified, ingested, and atomically activated on 2026-10-15. Activated-store
API/CLI smoke tests and the final 259-test dataset suite are green; exact
evidence is recorded in the implementation log and report.

This stage moves admission assessment, evidence grading, quality grading, and
benchmark selection into an interactive, folder-backed `Bundle`. It preserves
weakly documented but valuable data without allowing an unsupported claim to
become a reference-grade result.

Hotpot assists the user's judgment rather than replacing it. Ordinary and
explicitly forced-incomplete material can be preserved without adopting
Hotpot's benchmark classification. Canonical construction is explicit once;
later loads are validation-only. The workflow stays short:

```text
build once (or load) -> inspect status -> synchronize -> certify -> ingest
```

Revisions, digests, identity grouping, conflict comparison, locking, and
assessment generations remain internal implementation details.

The design has three independent outcomes:

1. whether a case can be archived safely;
2. what evidence and quality levels the available material supports;
3. whether the case passes the owner-controlled benchmark gate.

`Bundle.cases` contains every Case. Incomplete cases may be archived only
through an explicit opt-in and are kept in separate derived indexes and backend
storage. Normal persisted-dataset reads exclude them.

Every Bundle is bound to a required dataset name and target store. New datasets
may define their assessment policy; existing datasets must use their persisted
Gates, identity schema, and property comparison rules and cannot override them
through Bundle. Custom checker source is persisted by
content hash under a fixed JSON protocol and runs only in a required isolated
worker. Case or Gate edits make the Bundle unsynchronized;
`Bundle.synchronize()` is the only assessment operation. Certificate writing
and ingestion reject unsynchronized state and never invoke an earlier step
implicitly.

Sample identity is governed by a persisted dataset-specific identity schema,
not inferred from coordinate/file equality or generic `core_info`. The Bundle
manifest supplies one closed components list; canonical build materializes its
typed exact-comparison protocol into persisted policy. Cross-Case matching is
three-valued, while Case reflexivity ensures conflicting observations inside
one Case are not hidden by missing cross-Case identity. Scientific value
conflicts are evaluated only for same-sample observations with comparable
property context. Governance disagreement remains a separate reason category.

`Bundle.ingest()` requires an already current root certificate. The certificate
binds the complete immutable assessment-generation manifest but excludes
self-invalidating local receipt/projection files. The target store writes the
authoritative ingestion receipt transactionally, so exact retries are
idempotent and a missing local receipt can be recovered. CLI `build`,
`synchronize`, `certify`, and `ingest` remain separate commands.

Each Bundle represents exactly one release. Ingestion streams materialized
scientific assets into a target-owned SHA-256 object area, so a sealed release
does not depend on the continued location or lifetime of the submitted folder;
byte deduplication never implies shared scientific identity.

`Hotpot` benchmark authority is accepted only through a verified governance
signature. Public callers may record attributable `User` or `Other`
attestations, but cannot turn a string into Hotpot authority.

The complete assessment runtime is part of dataset policy. Policy,
assessment generation, certificate, and ingestion receipt bind the same
self-digesting runtime-policy document, including parser bindings. Historical
sealed releases verify their frozen binding; new assessment, certification,
and ingestion require the one current runtime. No historical implementation
is loaded as a compatibility backend.

The oxidation, spin, and CSD electronic-state releases received their one-time
owner/admin rebuild into this layout during the 2026-10-15 cutover. Scientific
source facts and asset hashes remained unchanged. Legacy SHA-derived
entity/structure identifiers were audited rather than treated automatically as
sample identity; every correction is recorded in an identity migration map.
Reassessment changed only derived incomplete/ordinary/benchmark organization.

- [Complete implementation plan](bundle_gate_certificate_implementation.md)
- [Complete first-level API signature tree](bundle_gate_certificate_implementation.md#41-complete-first-level-public-api-signature-tree)
- [Implementation log](implementation_log.md)
- [Implementation report](implementation_report.md)
- [Long-term workflow efficiency audit](efficiency_audit_report.md)
- [Long-term and temporary code separation audit](long_term_temporary_code_separation_audit.md)

These two uncommitted audit drafts are post-Stage-13 follow-up proposals, not
Stage 13 release criteria. Their pre-cutover observations must be rebaselined
against the activated store before any recommendation is implemented. They do
not reopen the completed cutover or the already completed schema-1 migration
path retirement.

This is a current-version replacement plan. Git owns implementation history;
the implementation must not preserve the superseded certification workflow as
a compatibility API.
