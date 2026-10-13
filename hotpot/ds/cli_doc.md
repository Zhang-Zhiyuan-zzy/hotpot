# `hotpot ds` command-line reference

The installed command is `hotpot ds`; no `python -m` prefix is required:

```bash
hotpot ds --help
```

If the console entry point is not on `PATH`, `python -m hotpot ds` is an equivalent
fallback.

The public commands follow the dataset workflow in this order:

```text
hotpot ds list
hotpot ds info
hotpot ds get
hotpot ds labels
hotpot ds subset
hotpot ds search
hotpot ds export
hotpot ds certify
hotpot ds ingest
```

Successful public commands emit one compact JSON envelope to standard output. Human
presentation and diagnostics never change that machine schema. Errors go to standard error
and return a non-zero exit status. `detail` is an advanced diagnostics group rather than
part of the normal user workflow.

The implementation lives in `hotpot/ds/`. Dataset documentation, schemas, manifests, and
local payloads share the `hotpot/dataset/` namespace. Git ignores
`hotpot/dataset/data/`, including the default local store at
`hotpot/dataset/data/store/datasets.sqlite3`. A clean clone therefore contains the rules
needed to assess and ingest data but does not claim to redistribute licensed or generated
payloads. Pass `--db` when using another store.

Benchmark eligibility is a governed `quality_level` state of data or a subset, not a
parallel database or directory. The current Git revision defines the only supported schema
and CLI protocol.

## Machine output and pipelines

### The exact JSON envelope

Every `hotpot ds` machine result has exactly four top-level members:

```json
{
  "key": "hotpot.ds.subset",
  "channel": "default",
  "protocol_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "payload": {}
}
```

- `key` identifies the command that produced the result.
- `channel` identifies that command's semantic result variant.
- `protocol_sha256` is the SHA-256 fingerprint of the canonical current communication map.
- `payload` contains the command result.

The communication map is shipped as `hotpot/ds/cli_communication.json`. Git owns its
history: a checkout or installation supports exactly its one current map.
`protocol_sha256` is an exact-revision guard, not a version. A consumer accepts only the
fingerprint of that map.

The public producer keys and channels are:

| Command | `key` | `channel` | Can feed `get`/`export` |
| --- | --- | --- | --- |
| `list` | `hotpot.ds.list` | `default` | No |
| `info` | `hotpot.ds.info` | `default` | No |
| `get` | `hotpot.ds.get` | `default`, or `structure` with `--smi`/`--xyz` | Yes |
| `labels` | `hotpot.ds.labels` | `default` | No |
| `subset` | `hotpot.ds.subset` | `default` | Yes |
| `search --scope dataset` | `hotpot.ds.search` | `dataset` | No |
| `search --scope data` | `hotpot.ds.search` | `data` | Yes |
| `search --scope smarts` | `hotpot.ds.search` | `smarts` | Yes |
| `export` | `hotpot.ds.export` | `default` | No |
| `certify` | `hotpot.ds.certify` | `default` | No |
| `ingest` | `hotpot.ds.ingest` | `default` | No |

Advanced commands use `hotpot.ds.detail` or a `hotpot.ds.detail.<command>` key. Their
results are terminal unless the shipped communication map explicitly declares otherwise.

Selection-producing payloads carry immutable dataset context and ordered items:

```json
{
  "dataset": {
    "name": "spin",
    "release_id": "immutable-release-id",
    "version": "source-version",
    "version_serial": "001",
    "database": "/absolute/path/to/datasets.sqlite3"
  },
  "items": [
    {"id": "hp-spin[001]-0"}
  ],
  "count": 1
}
```

`subset` and `search` put summaries in `items`; `get` puts complete data-information
objects there. Downstream commands use the declarative map to project `items[].id` and the
dataset locator. They do not scrape text or reconstruct context from an ID.

### Explicit stdin activation

Only a documented positional `-` activates protocol input:

```text
hotpot ds get DATASET - [GET OPTIONS]
hotpot ds export DATASET OUTPUT - [EXPORT OPTIONS]
```

A pipe, redirected stdin, or non-interactive shell does not by itself make a command read
stdin. The input must be one JSON envelope from the current communication map, not raw
identifiers, a JSON array, or JSON Lines.

Direct arguments remain authoritative controls and constraints:

- positional `DATASET` must equal the dataset carried by stdin;
- an omitted `--version` and `--db` inherit the immutable release and store carried by
  stdin;
- an explicit `--version` or `--db` must identify that same release or store;
- `--index`, `--id`, and `--identifier` cannot accompany the `-` selector;
- `--smi`, `--xyz`, attached-structure selectors, export `--format`, and export `OUTPUT`
  remain direct downstream options.

The following producers can feed either consumer:

```bash
# subset -> get; this is a batch get in subset order
hotpot ds subset spin -q 1 -e 2 | hotpot ds get spin -

# data search -> get
hotpot ds search spin --scope data \
  --where 'labels.spin_multiplicity=5' --limit 2 \
  | hotpot ds get spin - --xyz --structure-index 0

# one get -> another get operation
hotpot ds get spin 7 | hotpot ds get spin - --smi --structure-index 0

# subset -> export
hotpot ds subset spin -q 3 | hotpot ds export spin export/reference -

# SMARTS search -> export
hotpot ds search spin --scope smarts --smarts '[N]-[C]' \
  | hotpot ds export spin export/nc-complexes - --format mol2

# one get -> export
hotpot ds get spin 7 | hotpot ds export spin export/record-7 -
```

Pipelines may continue because either `get` channel retains the normalized data-selection
contract:

```bash
hotpot ds subset spin -q 1 \
  | hotpot ds get spin - --xyz --structure-index 0 \
  | hotpot ds export spin export/selected - --format mol2
```

Batch `get` accepts zero or more selected records, preserves their projected order, and
emits one envelope. An empty input selection succeeds with `payload.count` equal to `0`.
Protocol-driven `export` requires at least one record; an empty selection is an error.
Every batch is resolved, and every requested structure conversion is completed, before the
first stdout byte. A malformed envelope, unknown key/channel, fingerprint mismatch, context
conflict, missing record, or conversion failure produces a diagnostic on stderr, a non-zero
exit status, and no partial stdout. Export also retains its atomic file-writing guarantee.

### Reading payloads

Because stdout has a stable envelope, ordinary JSON tools can extract the desired value:

```bash
# Dataset names
hotpot ds list | jq -r '.payload.datasets[]'

# Complete information for one item
hotpot ds get oxidation 0 | jq '.payload.items[0]'

# IDs from a quality/evidence-controlled subset
hotpot ds subset spin -q 1 -e 2 | jq -r '.payload.items[].id'

# Exact XYZ bytes represented by the JSON string, without jq adding a newline
hotpot ds get spin 0 --xyz --structure-index 0 \
  | jq -rj '.payload.representations[0].content'
```

Do not parse stderr as data. It may contain diagnostics and the optional interactive
structure presentation described under `get`. When a structure-producing command is piped
but stderr remains attached to a terminal, that convenience mirror remains visible; redirect
stderr separately when it should be suppressed.

## Discovery and retrieval

### `hotpot ds list`

```text
hotpot ds list [--db PATH]
```

Places dataset names in `payload.datasets`.

```bash
hotpot ds list --db hotpot/dataset/data/store/datasets.sqlite3
```

### `hotpot ds info`

```text
hotpot ds info DATASET [--version VERSION] [--db PATH]
```

`VERSION` may be the source version, internal release ID, integer public release serial, or
three-digit serial such as `001`.

```bash
hotpot ds info oxidation --version 001
```

The selected immutable locator is in `payload.dataset`; the complete dataset properties are
in `payload.info`.

### `hotpot ds get`

```text
hotpot ds get DATASET SELECTOR
              [--index | --id | --identifier [NAMESPACE]]
              [--smi | --xyz]
              [--structure-index N | --structure-role ROLE | --structure-id ID]
              [--version VERSION] [--db PATH]
```

`SELECTOR` is interpreted as a zero-based index by default. `--index` makes that default
explicit, `--id` interprets it as a Hotpot data ID, and `--identifier` interprets it as an
external identifier. The special selector `-` instead consumes a protocol envelope and can
therefore retrieve an ordered batch.

```bash
hotpot ds get oxidation 0
hotpot ds get oxidation 0 --index
hotpot ds get oxidation 'hp-oxidation[001]-0' --id
hotpot ds get oxidation AACANI11 --identifier CSD
```

The namespace argument to `--identifier` is optional only when the selected dataset exposes
exactly one external identifier namespace:

```bash
hotpot ds get oxidation AACANI11 --identifier
```

If a dataset has several namespaces, provide one explicitly. Ambiguous identifiers return an
error rather than selecting an arbitrary record.

By default, `get` uses channel `default` and places complete data-information objects in
`payload.items`. `--smi` and `--xyz` are mutually exclusive; either changes the channel to
`structure` and adds one machine-readable entry per item to `payload.representations`:

```json
{
  "data_id": "hp-spin[001]-0",
  "structure_id": "stored-structure-id",
  "structure_roles": ["primary"],
  "selection": {
    "mode": "primary",
    "structure_index": null,
    "structure_role": null,
    "structure_id": null
  },
  "format": "xyz",
  "source_format": "mol2",
  "converted": true,
  "content": "3\n..."
}
```

The full envelope remains the only stdout output. When stderr is an interactive terminal,
Hotpot may also mirror the requested SMI or XYZ text there for convenient viewing; batch
presentation includes deterministic data/structure identity headings. This mirror is not a
data interface and is absent when stderr is redirected or captured. Hotpot uses the stored
representation when available and otherwise asks `hotpot.Molecule` to convert it; a missing
structure or failed conversion is an error.

```bash
hotpot ds get oxidation 0 --smi --structure-index 0
hotpot ds get oxidation 0 --xyz --structure-index 0
```

Extract raw text from the JSON payload when it is needed as a file or input to a non-Hotpot
tool:

```bash
hotpot ds get oxidation 0 --smi --structure-index 0 \
  | jq -rj '.payload.representations[0].content'
hotpot ds get oxidation 0 --xyz --structure-index 0 \
  | jq -rj '.payload.representations[0].content' \
  > oxidation-0.xyz
```

If a data item has several attached structures, select one explicitly when requesting an
SMI or XYZ representation:

```bash
hotpot ds get spin 0 --smi --structure-role multiplicity_6
hotpot ds get spin 0 --xyz --structure-index 1
hotpot ds get spin 0 --xyz --structure-id STRUCTURE_ID
```

`--structure-index` follows the zero-based order of the `structures` array in each object in
`payload.items`. `--structure-role` requires exactly one structure carrying that role;
duplicate roles are an error and must be resolved with an index or exact ID. `--structure-id`
matches the stored `structure_id` exactly. The three selectors are mutually exclusive and
are accepted only together with `--smi` or `--xyz`. In a batch, the same selector is applied
to every item and must resolve successfully for all of them. Without one of these selectors,
Hotpot requires each item to have either one structure or exactly one declared primary
structure.

### `hotpot ds labels`

```text
hotpot ds labels DATASET [--version VERSION] [--db PATH]
```

Places core labels and the number of data items containing each active label in the
envelope payload.

## Filtering and search

### `hotpot ds subset`

```text
hotpot ds subset DATASET [--where FIELD=JSON ...]
                         [-q LEVEL] [-e LEVEL]
                         [--limit N] [--version VERSION] [--db PATH]
```

Repeated `--where` expressions are combined with AND. A JSON array means membership in the
listed values; a scalar means equality. Nested paths use dots or `__`.

`-q/--quality` and `-e/--evidence` are inclusive minimum-level gates. Each accepts an
integer from 0 through 4 and defaults to 0, so the default excludes nothing. Quality values
map to B0-B4 and evidence values map to E0-E4. Both gates are applied together and are also
combined with every `--where` expression.

```bash
hotpot ds subset spin \
  --quality 1 \
  --evidence 2 \
  --where 'labels.spin_multiplicity=[1,5]'
```

### `hotpot ds search`

```text
hotpot ds search [DATASET] [--scope data|dataset|smarts]
                            [--where FIELD=JSON ...] [--smarts PATTERN]
                            [--limit N] [--version VERSION] [--db PATH]
```

Dataset-metadata search does not take a dataset name:

```bash
hotpot ds search --scope dataset --where 'core_labels=["oxidation_state"]'
```

Data-field and SMARTS searches require the positional dataset name:

```bash
hotpot ds search spin \
  --scope data \
  --where 'labels.spin_multiplicity=[1,5]'

hotpot ds search oxidation \
  --scope smarts \
  --smarts '[N]-[C]'
```

Logical composition is richer in the Python API, which supports `&`, `|`, `^`, and `~`.
Python cannot overload `!`.

## Export

### `hotpot ds export`

```text
hotpot ds export DATASET OUTPUT [ITEM]
                [--index | --id | --identifier [NAMESPACE]]
                [--range START STOP | --where FIELD=JSON ...]
                [--format FORMAT]
                [--version VERSION] [--db PATH]
```

With no `ITEM`, `--range`, or `--where`, the complete dataset is exported. An `ITEM` is a
zero-based index by default; `--index` makes the mode explicit, `--id` selects by Hotpot ID,
and `--identifier [NAMESPACE]` selects by external identifier. The special `ITEM` `-`
exports the non-empty ordered selection from a protocol envelope. Namespace inference
follows the same rule as `get`: it is allowed only when the dataset has exactly one external
identifier namespace.

```bash
# Complete dataset
hotpot ds export oxidation export/oxidation

# One record by index, Hotpot ID, or external identifier
hotpot ds export oxidation export/record-0 0
hotpot ds export oxidation export/record-0 'hp-oxidation[001]-0' --id
hotpot ds export oxidation export/AACANI11 AACANI11 --identifier CSD

# Half-open index range: indices 100 through 199
hotpot ds export oxidation export/oxidation-100-199 --range 100 200

# Subset; repeated predicates are combined with AND
hotpot ds export oxidation export/oxidation-iii \
  --where 'labels.oxidation_state=3' \
  --where 'quality_level="B1_candidate"'
```

Every structure attached to the selected records is exported. Without `--format`, each
structure is copied in its default stored format. With `--format`, every selected structure
is converted by `hotpot.Molecule`:

```bash
hotpot ds export oxidation export/oxidation-mol2 --format mol2
```

Conversion reserializes the molecular representation and is not guaranteed to be lossless
for stereochemistry, inferred bonds, or format-specific metadata. Each `info*.json` records
the source format, output format, and whether conversion occurred. Omit `--format` when an
exact copy of the governed structure asset is required.

A missing structure, unsupported format, or failed conversion aborts the whole export. The
operation is atomic and never overwrites an existing `OUTPUT` path, so a failed export does
not leave a partially populated destination. A successful command emits a
`hotpot.ds.export` envelope whose payload contains the output path and record, structure,
and format counts.

The compact layout is used only when the selection resolves to exactly one record and that
record has exactly one stored structure:

```text
OUTPUT/
|-- struct.<format>
`-- info.json
```

All other results use numbered, root-level file pairs. This includes a single record with
multiple stored structures. Structure and information files have aligned stems;
`summary.json` describes the source dataset, selection, record and structure counts, output
format, and file mapping:

```text
OUTPUT/
|-- struct-000000.<format>
|-- info-000000.json
|-- struct-000001.<format>
|-- info-000001.json
`-- summary.json
```

The six-digit stem is the record index. If one record has several structures, an additional
structure ordinal keeps each pair distinct, for example `struct-000000-00.mol2` and
`info-000000-00.json`.

## Certification

### `hotpot ds certify`

```text
hotpot ds certify SOURCE DATASET --certificate PATH
                                  [--quality-level LEVEL]
                                  [--core-label NAME ...]
                                  [--core-info PATH ...]
                                  [--no-core-fields]
                                  [--configure PATH]
                                  [--asset-root DIR]
                                  [--existing --db PATH]
```

For a new dataset, specify at least one of:

- `--quality-level`;
- one or more `--core-label`;
- one or more `--core-info`;
- `--no-core-fields`, which explicitly declares both lists empty.

```bash
hotpot ds certify csd-submission csd-example \
  --core-label oxidation_state \
  --configure csd-policy.json \
  --certificate csd-example.certificate.json
```

An established dataset must use its stored policy:

```bash
hotpot ds certify next-release csd-example \
  --existing \
  --db work/datasets.sqlite3 \
  --certificate next-release.certificate.json
```

`--configure` and the new-dataset declaration flags (`--quality-level`, `--core-label`,
`--core-info`, and `--no-core-fields`) cannot be combined with `--existing`. Established
dataset assessment always loads the persisted policy. The written assessment is returned in
`payload.certificate`, with its absolute certificate and submission paths alongside it. A
completed certificate whose assessment is negative still emits this result and returns exit
code 2; an execution failure emits no envelope.

## Ingestion

### `hotpot ds ingest`

```text
hotpot ds ingest SOURCE --certificate PATH --db PATH [--asset-root DIR]
```

```bash
hotpot ds ingest csd-submission \
  --certificate csd-example.certificate.json \
  --db work/datasets.sqlite3
```

The certificate is mandatory and must match the exact normalized submission and persisted
configuration. A new release is written atomically. An established release is never modified
in place. `payload.result` contains the dataset name, submitted versions, and public count
summary; it does not expose ingestion or certificate table identifiers. `payload.database`
records the absolute target store path.

## Advanced detail commands

`hotpot ds detail` contains normalized-backend diagnostics that are intentionally absent
from the normal user surface.

```bash
hotpot ds detail status --db work/datasets.sqlite3
hotpot ds detail audit --db work/datasets.sqlite3
hotpot ds detail query claims \
  --db work/datasets.sqlite3 \
  --evidence-level E1_extracted \
  --limit 20
```

## Submission organization

Folder submissions use a root `dataset.json` and safe relative structure paths. Compact
SQLite submissions use the `dataset`, `data`, `structures`, and `labels` exchange tables.
See [the Python API reference](README.md#assessment-input-fences) for mandatory fields,
multi-structure roles, certificate configuration, and complete examples.
