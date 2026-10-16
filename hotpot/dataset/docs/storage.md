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

## Evidence and verification facts

Evidence classification is explicit. When an Evidence record identifies a
scientific source, its `source_kind` must use exactly one current value:

- `primary_literature`;
- `secondary_literature`;
- `database_record`;
- `expert_statement`;
- `other`.

A DOI, scientific-work identifier, or HTTP URL identifies a source; none of
them implicitly makes that source primary literature. An identified source
without `source_kind` remains preservable but cannot count as primary support.
An out-of-vocabulary value is an integrity defect and makes the Case
incomplete. Evidence also declares its target, relation, and, when it claims an
explicit report, a usable `locator`, `quote`, or `matched_text`.

File-backed Evidence participates in the same scientific-asset inventory as
Structure representations. Its asset assessment is usable for EvidenceGate
only when the declared bytes are present and no declared SHA-256 mismatch is
found. Missing bytes and hash mismatch produce `evidence_asset_missing` and
`evidence_asset_hash_mismatch` reasons and remove that Evidence row from the
support calculation. These negative asset facts are certificate-bound. They do
not alone force the Case into `incomplete`; if no other usable support remains,
the affected target receives E0 under the default EvidenceGate.

Human verification requires an attributable non-empty verifier. AI
verification additionally requires `tool`, `tool_version`,
`procedure_version`, and a timezone-aware `verified_at`. E3 still requires the
linked Evidence to identify `primary_literature` and to carry the relevant
locator, quote, or matched text. E4 additionally requires two independently
identified scientific works with disjoint primary result-lineage roots.
Verification facts never assign E0-E4 directly.

The persisted `identity_schema` is the sole cross-Case sample-identity policy.
It is a closed object containing an array of typed, exactly compared
components. Ordinary components address `core_info` or `identifiers`;
structure components select exactly one active declared role and then a
semantic structure field such as molecular graph, stereochemistry,
isotope/protonation state, formal charge, or crystal context. Raw coordinates,
representation paths, and asset-byte digests are not identity components.
Equal coordinate bytes permit asset deduplication only. Missing, invalid,
ambiguous, or non-decisive components produce an indeterminate comparison
instead of guessed equality.

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
|   |-- leases/<generation-id>.lock
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

Generation retention is private maintenance. A reader holds a shared OS lock
for the complete read, certificate, or ingestion operation. Collection obtains
a non-blocking exclusive lock and removes a generation only after confirming
that it is not current, root-certificate-bound, target-receipt-bound, or
leased. Process failure releases a lease automatically; no timeout can expire a
live reader. The small lock files are runtime state, are not certificate
content, and may remain after a process exits. An unreadable durable reference
or an unverifiable lock always causes conservative retention.

## Revisions and publication

Case or new-dataset policy edits commit facts and audit events only. They make
the Bundle unsynchronized and any certificate stale. They do not recalculate
quality, evidence, conflicts, or archive class.

`Bundle.synchronize()` freezes facts, assets, policy, and the trusted
governance registry under a lock, reassesses the entire release, and atomically
publishes a complete generation. A generation becomes current only after every
manifest-listed file and digest is complete. Derived values are unavailable
through normal public reads while facts and the current generation disagree.

Every generation summary has one fixed top-level shape: `cases`, `integrity`,
`completeness`, `evidence`, `quality`, `benchmark`, `attestation`, `conflict`,
and `reasons`. Archive-class counts are under
`completeness.archive_class`; the other sections report their own status,
level, or reason-code counts. This aggregate is derived from the authoritative
Case assessments and is bound by the generation manifest and certificate.

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

The SQLite target owns normalized dataset/release metadata, one compact Case
fact copy, final assessment labels, persistent dataset policy with embedded
custom-checker source, identifiers, asset references, and authoritative
ingestion receipts. Canonical Case facts are stored as deterministic
zlib-compressed JSON and verified against the SHA-256 digest of their
uncompressed canonical value. Archive-class selection uses the final
assessment index; the target does not duplicate Cases into class tables.

The permanent fact projection retains original observations, claims, evidence,
verifications, attestations, entities, source classifications, `core_info`, and
dataset-owned metadata. Storage identities and identifiers are normalized and
reconstructed by the read API. Migration row mirrors and migration
implementation metadata are not permanent scientific facts.

Only final `quality_level`, `evidence_level`, `archive_class`, benchmark
decision/status, and concise reason records are stored as assessments. Full
Gate traces, parser working assessments, conflict-pair products, generation
databases, class-ID files, and summaries are reproducible Bundle products and
are not copied into the formal target.

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

Target CAS contains irreplaceable scientific structure/evidence bytes,
explicit original source/provenance control assets, and the compact
certificate. It does not contain copies of Bundle `facts.sqlite3`, assessment
generations, or generated migration identity/relationship databases. For a
Stage 13 migrated release, `legacy-source-inventory` and
`migration-provenance` are retained as provenance controls; the large
`identity-migration` and `legacy-identity-relationships` projections are not.

Stored Case reads expose each materialized Evidence file in
`Data.info["evidence_assets"]`. These references resolve only into the formal
target CAS and include the Evidence ID, asset index, role, status, SHA-256,
size, and path. They never retain the Bundle source path as the readable byte
location. The unmodified source declaration remains in
`Data.info["facts"]["evidence"]` for provenance.

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

After successful ingestion, the Bundle root can be pruned with:

```bash
hotpot ds bundle rm --all
```

The short form uses the target already bound by the Bundle; `--root`,
`--dataset`, and `--db` may be supplied as matching assertions. Cleanup first
verifies the authoritative receipt, certificate, target database, every
compressed Case fact, and every target-owned asset. A resumed cleanup repeats
those checks before deleting another byte. It then deletes only a closed
allowlist of reproducible facts/generation/cache products and the two exactly
named generated migration maps, and writes `bundle-cleanup.json`. Intake data,
original scientific assets, policy and checker source, certificate, receipt,
audit history, source inventory, and migration provenance remain. The cleaned
Bundle is an archived intake record and is no longer editable or reassessable
without rebuilding its derived products.

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

The persisted policy additionally contains one complete, self-digesting
`runtime_policy`. Its closed document binds the assessment source and store
protocol, checker protocol, parser selection and parser
source/protocol/distribution identities, Python and SQLite identities, and the
relevant installed distributions. The assessment generation, root certificate,
authoritative ingestion receipt, and parser-backed asset assessments bind that
same frozen document or its selected parser entry.

Frozen validation verifies the recorded document's closed structure and nested
digests without comparing it to the ambient process. It therefore authenticates
what a sealed release used without loading a compatibility implementation.
Synchronization, certificate issuance, and ingestion additionally require the
frozen document to equal the one current ambient `runtime_policy`. Drift makes
an editable Bundle stale and requires an explicit owner policy migration and
reassessment; a sealed release remains readable and verifiable against its
frozen policy, but any scientific correction is a successor release. Custom
checker execution requires the OS-isolated `bubblewrap` worker; missing
isolation support is an error, never an in-process fallback.

## Query visibility

Normal `hotpot.ds.load` includes `ordinary` and `benchmark` records and excludes
`incomplete`. Explicit `archive_class="incomplete"` and mixed-class selections
are the only public opt-ins. Filtering cannot reintroduce classes excluded at
load time. Persistent public indices and Hotpot IDs never change because a
class is hidden.

## Administrative store replacement

The public runtime has no migration or compatibility path. It opens only the
current schema selected by `store/active`; historical conversion programs are
not public data-management interfaces and are not used during query, Bundle
assessment, certification, or ingestion. Conversion inventories, identity
maps, parity reports, and owner decisions remain release audit artifacts.

An owner-directed schema replacement or recovery is built beside the active
store. The operation must preserve canonical facts and scientific assets,
record any identity transformation explicitly, synchronize and certify every
affected release, ingest into a complete new target and adjacent asset store,
and compare source facts, assets, identifiers, relationships, stable public
indices, and assessment-class counts. Any unexplained difference blocks
activation. `store/active` changes only after integrity and parity checks pass.

Identifiers historically derived from asset bytes remain provenance aliases;
they are not reused as scientific sample or Structure identity without
independent support. Shared bytes may remain deduplicated without merging
Cases, subjects, or Structures. The replaced store remains an immutable audit
or rollback artifact for the retention period and is never queried as a
fallback backend.

## Distribution and licensing

Do not add payload exceptions to `.gitignore`. Distribute datasets through an
authorized artifact channel and verify declared digests after installation.
CSD-derived structures or text remain restricted unless the recipient has the
required license. A metadata field that names a license is not itself legal
authorization.
