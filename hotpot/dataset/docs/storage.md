# Local dataset storage

The materialized Hotpot store is local data, not package source. Its canonical path is
`hotpot/dataset/data/store/datasets.sqlite3`. The full `hotpot/dataset/data/` tree is excluded
from Git, including databases, normalized bundles, certificates, supplier files, coordinates,
and generated audits.

## Canonical data directories

| Directory | Content |
| --- | --- |
| `data/store` | Current SQLite query store and its generated assessment artifacts |
| `data/sources` | Source-native labels, evidence, structures, and build records |
| `data/structures` | Shared structure objects and relation indexes |
| `data/audits` | Supplier adjudications and locally generated quality reports |

Tracked schemas and explanatory documents remain outside `data/`. They must contain no
licensed coordinates, individual source records, or normalized scientific payloads.

## Current-only rule

The active checkout has one storage schema. A store with any other fingerprint is rejected.
Rebuild or re-ingest the source under the current code and schema, then validate it before
replacing the active store.

Writes follow the public quality gate:

```text
submission -> ds.load_bundle -> ds.write_certificate -> ds.store.ingest
```

The resulting certificate binds the assessed content and policy. It is an integrity and
quality artifact, not proof of an external expert's identity or a data redistribution right.

## Distribution

Do not add payload exceptions to `.gitignore`. Distribute a dataset through an authorized
artifact channel and verify its declared digest after installation. CSD-derived data must
remain restricted to environments whose users hold the required license.
