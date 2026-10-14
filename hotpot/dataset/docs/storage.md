# Dataset storage, Bundle ownership, and cutover

This document defines the current physical organization of Hotpot dataset data.
It does not define a compatibility layer. Git retains historical
implementations; each checkout reads and writes only its current schema.

## Local payload boundary

All scientific payloads are local data under `hotpot/dataset/data/` and are
excluded from Git:

```text
data/
|-- store/
|   |-- generations/
|   |   `-- <generation-id>/
|   |       |-- datasets.sqlite3
|   |       `-- datasets.sqlite3.assets/
|   |           `-- objects/sha256/<prefix>/<sha256>
|   `-- active -> generations/<generation-id>
|-- bundles/
|   `-- <dataset>/<release>/
|-- sources/
`-- audits/
```

Tracked schemas, manifests, and explanatory documentation remain outside
`data/`. They must not contain licensed coordinates, source records, private
review notes, or normalized scientific payloads.

## Canonical Bundle root

One Bundle directory represents exactly one dataset release. Before its
one-time build, a concise intake contains:

```text
BUNDLE_ROOT/
|-- bundle.json
|-- cases.jsonl                 # or declared intake.sqlite3, not both
|-- structures/
|   `-- ...
`-- evidence/
    `-- ...
```

`bundle.json` is the authoritative release declaration. It includes the
current Bundle schema, dataset and release identities, persistent release
serial, version, descriptive metadata, `core_labels`, `core_info`, required
structure roles, identity schema, property comparison rules, and the active
Case source. A Bundle cannot mix releases.

The JSON Lines form stores one current-schema Case object per line. The SQLite
form uses the current `hotpot/ds/schemas/bundle/sqlite-intake.sql` tables. A
bare `.sqlite3` file is not accepted as a Bundle; place it in a Bundle root and
declare it. Both forms represent the same explicit relationships among Cases,
structures, observations, claims, evidence, verifications, attestations,
identifiers, and entities.

Local scientific paths must be safe relative POSIX paths inside the root. A
missing remote or local asset is a declared negative fact. It is not silently
downloaded or replaced with fabricated bytes.

The one-time build publishes the canonical control and fact store in the same
root. Subsequent opens are validation-only. Implementations may maintain
regenerable projections, but the root has one authoritative canonical Case fact
copy; incomplete, ordinary, and benchmark are derived indexes, not competing
fact stores.

Synchronization adds immutable assessment generations:

```text
BUNDLE_ROOT/
|-- bundle.json
|-- cases/
|   `-- facts.sqlite3
|-- policy/
|   |-- policy.json
|   `-- checkers/<source-sha256>.py
|-- audit/
|   `-- events.jsonl
|-- assessments/
|   `-- generations/<generation-id>/
|       |-- generation.json
|       |-- assessment.sqlite3
|       |-- summary.json
|       |-- governance-registry.json
|       |-- incomplete.ids
|       |-- ordinary.ids
|       `-- benchmark.ids
|-- certificate.json             # only after certification
|-- ingestion-receipt.json       # only after successful ingestion
|-- structures/
`-- evidence/
```

The authoritative current-generation ID and digest are stored transactionally
in `cases/facts.sqlite3`; there is no editable filesystem pointer. Users must
not edit generated controls, assessment databases, indexes, policy
projections, certificates, or receipts.

## Revisions and publication

Case or new-dataset policy edits commit facts and audit events only. They make
the Bundle unsynchronized and any certificate stale. They do not recalculate
quality, evidence, conflicts, or archive class.

`Bundle.synchronize()` freezes facts, assets, policy, and the trusted
governance registry under a lock, reassesses the entire release, and atomically
publishes a complete generation. A generation becomes current only after every
manifest-listed file and digest is complete. Derived values are unavailable
through normal public reads while facts and the current generation disagree.

`Bundle.write_certificate()` accepts only a synchronized Bundle. The root
certificate binds source and policy revisions/digests, the complete generation
manifest, ordered assessment digest, positive and negative scientific-asset
inventory, trusted governance snapshot, target precondition, and aggregate
summary. It does not copy a per-Case assessment array into the certificate.

## Target store

The default target is:

```text
hotpot/dataset/data/store/active/datasets.sqlite3
```

`store/active` is a directory symlink to one complete immutable directory
below `store/generations`. The database and its adjacent asset directory are
therefore selected by one atomic pointer replacement; they are never switched
independently.

The SQLite target owns normalized dataset/release metadata, one canonical Case
fact copy, assessment headers, persistent dataset policy, Gate/checker
artifacts, class indexes, and authoritative ingestion receipts. Ordinary,
benchmark, and incomplete indexes refer to the same canonical facts.

Every positively materialized scientific asset is streamed into target-owned
content-addressed storage before the target transaction:

```text
<database>.assets/
`-- objects/
    `-- sha256/
        `-- <first-two-hex>/
            `-- <full-sha256>
```

Existing objects are accepted only after size and digest verification. Separate
Case and Structure references may point to one object, but byte equality never
establishes sample, entity, or Structure identity. Certified missing,
invalid, hash-mismatched, or remote-not-materialized declarations do not create
objects.

Ingestion requires a current root certificate and rechecks the source, policy,
generation, certificate, target identity, release identity, and selected Case
set inside one write transaction. The authoritative receipt is unique for one
store, dataset, and release. An exact retry is idempotent; different content
under an occupied release identity is rejected.

Successful ingestion seals the whole release, including certified incomplete
Cases that were excluded from storage. Incomplete data can be force-archived
only with explicit opt-in and a reason, and remains in its own derived
partition. A later correction creates a successor release; published facts are
not repaired in place.

## Policy ownership

For an absent dataset, a Bundle may propose EvidenceGate, QualityGate,
BenchmarkGate, identity, structure-requirement, and property-comparison policy.
First ingestion persists the exact policy and custom checker source.

For an established dataset, every Bundle loads the stored policy. Caller Gate
arguments, Gate replacement, checker replacement, identity changes, and
comparator changes are rejected. Owner/admin policy migration is a separate,
audited operation and produces reassessment and new releases where required.

Custom checker source is stored as UTF-8 source addressed by SHA-256, never as
pickle, bytecode, or a Python-version compatibility object. A checker/protocol
change requires an explicit current-policy migration; the loader does not adapt
old code silently.

## Query visibility

Normal `hotpot.ds.load` includes `ordinary` and `benchmark` records and excludes
`incomplete`. Explicit `archive_class="incomplete"` and mixed-class selections
are the only public opt-ins. Filtering cannot reintroduce classes excluded at
load time. Persistent public indices and Hotpot IDs never change because a
class is hidden.

## Current-only migration and cutover

Existing oxidation, spin, CSD electronic-state, and other stores require a
one-time rebuild. Never migrate the only live database in place and never query
the old database as a fallback.

The required cutover is:

1. Snapshot the current store and record database/file digests.
2. Freeze the exact legacy identity and relationship rows in a
   certificate-bound migration inventory.
3. Export each dataset release to one current-format Bundle root.
4. Verify source assets and preserve structure bytes, stored representations,
   labels, values, units, conditions, citations, and source relationships.
5. Emit an explicit identity migration map for every retained, split,
   replaced, or unresolved legacy relation.
6. Bootstrap one owner-approved current assessment policy per dataset.
7. Synchronize every release and generate a new root certificate.
8. Ingest into a new temporary current-schema target and its asset area.
9. Compare source-fact counts/digests, persistent indices, Hotpot IDs, assets,
   labels, sources, identities, relationships, and archive-class counts.
10. Atomically switch the configured store only after the parity report passes.

Legacy asset-derived entity or Structure IDs remain provenance aliases; they
are not reused as scientific sample identity without independent support.
Shared bytes remain deduplicated without merging subjects. Any changed
operative relationship must be explained by the certificate-bound migration
map.

Classification changes are expected: invalid/missing required structures,
missing core data, and significant unresolved conflicts move to incomplete;
complete Cases are ordinary unless BenchmarkGate admits them. Such changes are
reported separately from source-content parity. An unexplained source-fact or
asset difference blocks cutover.

The previous database remains an immutable rollback artifact until acceptance
finishes. It is never a compatibility backend for the new runtime.

## Distribution and licensing

Do not add payload exceptions to `.gitignore`. Distribute datasets through an
authorized artifact channel and verify declared digests after installation.
CSD-derived structures or text remain restricted unless the recipient has the
required license. A metadata field that names a license is not itself legal
authorization.
