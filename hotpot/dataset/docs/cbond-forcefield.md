# CBond to force-field benchmark dataset

This governed dataset records the selected final structure and terminal
validation evidence from Plan 17's `O0`, `F1`, `F2`, `C1`, and `C2` routes.
It is a regression and route-comparison corpus, not a coordinate ground truth.
New structures may differ while satisfying the same scientific acceptance
contract.

## Population and identity

The proposal source is the Git-tracked 187-line
`molecules/extractant/extractants.smi` file with SHA-256
`e67ae3167a1a96d3c6a4c6df3ce12b70abfaf479bdd89bb156bfaefa771795a7`.
Current CBond reconstruction yields 182 eligible complex graphs; source indices
139, 182, 185, 186, and 187 have no CBond and are not route failures. The
release has exactly 910 Case identities, ordered first by source line and then
by the fixed route order `O0`, `F1`, `F2`, `C1`, `C2`. A Case ID such as
`cbond-ff-001-f1` permanently identifies that source-line/route result. An F1
repeat is control provenance for the same Case and never creates another Case.

Each Case retains the source SMILES and digest, current-run semantic graph
digest, CBond metal/donor mapping, run and seed, dependency identity,
scientific settings, A01-A10 decisions, recovery diagnostics, selected-frame
numerics, and manual-review result. IEEE NaN and infinities are never emitted
as nonstandard JSON tokens: they are represented as
`{"nonfinite_float":"nan"}`, `+inf`, or `-inf` diagnostic objects.
The persisted sample identity is the source-line index, route, and semantic
graph digest; coordinate bytes and energies never participate in identity.

## Structure and failure semantics

Only a route's actually selected, accepted final force-field structure can
have the required `selected-forcefield` role. Its declared digest and copied
bytes must agree. A failed route-case remains a Case with its complete terminal
diagnostics but without that structure role. No O0 structure, rejected frame,
or result from another route is borrowed to fill the missing asset. Therefore
the difference between 910 and the coordinate-asset count is scientifically
meaningful and must remain visible.

The required structure role makes failed results `incomplete` under the
existing `hotpot.ds` assessment contract. This is deliberate preservation, not
missing publication work. Publication always calls
`Bundle.ingest(include_incomplete=True, reason=...)` so those negative results
remain isolated and queryable rather than disappearing.

## Authority and provenance

The raw route output does not assert Hotpot authority. Preparation requires an
explicit `User` authorization record with actor, reason, time, license, access
policy, and authorization ID. That record becomes both a bound Bundle control
asset and a `User` attestation on every route-case. The publisher rejects a raw
`Hotpot` authority assertion; package governance may independently derive
trust but is not supplied by this importer.

The Bundle also binds copies and digests of the 187-source corpus, normalized
910-result matrix, result schema, authorization, and matrix summary as control
assets. Coordinate payloads and all generated Bundle/store products remain
under ignored `hotpot/dataset/data/**`. This tracked document and
[`cbond-forcefield-benchmark.json`](../schemas/cbond-forcefield-benchmark.json)
define the release input contract.

## Two-stage publication

Preparation never opens or changes the formal target store:

```bash
python -m tests.benchmarks.coordination_complexes.validation_dataset prepare \
  --results RUN/route-results.jsonl \
  --result-root RUN \
  --source molecules/extractant/extractants.smi \
  --authorization RUN/owner-authorization.json \
  --bundle-root hotpot/dataset/data/bundles/cbond-forcefield.v001
```

After the 910 records, structure bytes, authorization, and generated Bundle
have been reviewed, publication is a separate explicit action:

```bash
python -m tests.benchmarks.coordination_complexes.validation_dataset publish \
  --bundle-root hotpot/dataset/data/bundles/cbond-forcefield.v001 \
  --database hotpot/dataset/data/store/active/datasets.sqlite3 \
  --reason "Preserve failed Plan-17 route results as governed negative evidence" \
  --inventory-output hotpot/dataset/data/audits/cbond-forcefield.v001-inventory.json \
  --confirm-ingest
```

The publish command performs, in order, synchronization, certificate writing,
and ingestion with `include_incomplete=True`. The confirmation flag and reason
are mandatory. It emits an ignored inventory candidate whose `release` object
matches the tracked `current-inventory.json` release-summary fields. No tool
automatically edits the tracked current inventory; owner review and store
verification precede that separate repository update.
