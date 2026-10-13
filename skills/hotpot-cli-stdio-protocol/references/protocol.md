# Hotpot CLI stdin/stdout communication specification

## Scope and authority

This document defines the reusable communication rules for Hotpot CLI commands. The
current executable profile covers only `hotpot ds`; future command groups must adopt these
same rules rather than inventing a parallel transport.

Git is the sole protocol history. Every checkout or installation contains exactly one
current communication map and one accepted wire shape.

Normative terms such as MUST, MUST NOT, SHOULD, and MAY have their usual requirements
meaning.

## Stream model

Hotpot separates three output planes:

- stdout is the machine-result plane. A successful participating command MUST write JSON
  envelopes only. One invocation writes exactly one envelope document followed by LF.
- stderr is the diagnostic plane and, when explicitly requested, the interactive
  presentation plane. Presentation MUST NOT be required to recover machine data.
- files named by command arguments are artifact outputs. Their schemas are independent of
  the CLI transport envelope.

Help text is an argparse user interface and is exempt from the machine envelope. Parser
errors are diagnostics and go to stderr.

## Exact JSON envelope

Every machine result has exactly these four top-level members:

```json
{
  "key": "hotpot.ds.subset",
  "channel": "default",
  "protocol_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "payload": {}
}
```

The members mean:

- `key`: the command that produced this result, never the intended consumer.
- `channel`: the semantic output variant of that command.
- `protocol_sha256`: the lowercase SHA-256 digest of the canonical current communication
  map. It is an exact-revision guard, not a protocol version.
- `payload`: command data in the shape declared for the producer key and channel.

Consumers MUST reject missing or extra envelope members, malformed JSON, non-string keys
or channels, unknown producers, and unequal digests. They MUST NOT guess a producer from
payload contents. Map and input parsers MUST reject duplicate object keys and the non-JSON
numeric tokens `NaN`, `Infinity`, and `-Infinity`.

The digest input is the parsed map serialized with `ensure_ascii=false`, keys sorted by the
JSON member names, compact `,` and `:` separators, non-finite numbers forbidden, and the
result encoded as UTF-8. Formatting-only changes to the source file therefore do not
change the digest; semantic map changes do.

## Communication map

The dedicated JSON map is executable configuration only in the declarative sense. It MUST
contain:

- the exact envelope member list;
- every participating producer identified by `key + channel`;
- normalized semantic contracts and their extraction bindings;
- each consumer's accepted contracts and cardinality.

Routing is:

```text
producer key + channel -> semantic contract -> consumer bindings
```

Bindings belong to a semantic contract, not to an individual consumer. This indirection
prevents a producer/consumer cross-product. The map MAY project values
through dotted object paths and `[]` list traversal, and MAY assign JSON constants. It MUST
NOT contain Python expressions, imports, shell fragments, templates with execution, or
general-purpose query extensions.

The current map schema is closed and has this form:

```json
{
  "envelope_keys": ["key", "channel", "protocol_sha256", "payload"],
  "contracts": {
    "hotpot.example.collection": {
      "bindings": {
        "context": {
          "path": "payload.context",
          "type": "object",
          "properties": {"name": "string"}
        },
        "selectors": {
          "path": "payload.items[].id",
          "type": "array",
          "items_type": "string"
        },
        "selection_kind": {"constant": "identifier_collection", "type": "string"}
      }
    }
  },
  "producers": {
    "hotpot.example.select": {"default": "hotpot.example.collection"},
    "hotpot.example.report": {"default": null}
  },
  "consumers": {
    "hotpot.example.get": {
      "accepts": {
        "hotpot.example.collection": {
          "cardinality": {
            "field": "selectors",
            "minimum": 0,
            "maximum": null
          },
          "duplicates": "preserve"
        }
      }
    }
  }
}
```

`null` marks terminal, non-consumable producer output. Every binding MUST declare `type`;
its values are JSON
`object`, `array`, `string`, `integer`, `number`, `boolean`, or `null`. `items_type`
constrains every array item. `properties` is an exact object-property/type fence: missing
and additional properties both fail. Cardinality limits are inclusive nonnegative integers;
`maximum: null` is unbounded. Duplicate behavior is either `preserve` or `reject` and is
applied to the cardinality field. A future need for transformation or validation beyond
these operations requires replacing this schema and runtime together, not adding ad hoc
consumer code.

An implementation MUST validate map uniqueness and all referenced contracts before using
the map. A key/channel absent from the map is not a participating producer.

While only `hotpot ds` implements the protocol, the single canonical map is
`hotpot/ds/cli_communication.json` and its runtime is private to that package. The first
change that adds a non-`ds` command group MUST atomically move the complete map to
`hotpot/cli_communication.json` and the shared runtime to `hotpot/_communication.py`, then
rewire every existing `ds` producer and consumer in that same change. Hotpot MUST NOT
compose per-group maps, calculate per-group digests, keep both locations, or accept both
wire forms. Thus the repository still has exactly one current map and one digest after the
generalization.

## stdin activation and argument precedence

A command reads stdin only when its documented input positional is exactly `-`. In every
protocol-participating command, positional `-` is reserved for one protocol envelope; raw
SDF, text, or other byte streams require a different explicit option. The
presence of a pipe, redirected stdin, or a non-TTY stdin MUST NOT activate protocol input.
This avoids accidental blocking and makes command behavior reproducible.

Direct arguments remain command controls. For data locators carried by stdin:

- a required positional dataset name is a constraint and MUST equal the projected name;
- an omitted release selector inherits the projected immutable release identity;
- an explicit release selector MUST resolve to that same release;
- an omitted store path MAY inherit the projected store locator;
- an explicit store path MUST identify the same store;
- selector-mode flags are invalid when the map supplies normalized selectors;
- formatting, structure selection, thresholds, and artifact destinations remain direct
  command options unless their consumer contract explicitly says otherwise.

Unknown producer contracts, dataset/store/release conflicts, and invalid cardinality are
hard errors.

## Batches, order, and atomicity

Contracts state whether zero, one, or many values are accepted. A consumer MUST preserve
projected item order. Duplicate handling must be explicit in the consumer semantics.

The command MUST parse and validate the full envelope, resolve all selected records, and
compute all requested conversions before its first stdout byte. If any item fails before
emission, stdout remains empty. The encoded envelope is prepared completely before the
write, although an operating-system write failure can necessarily leave a partial byte
stream. Artifact-writing commands MUST retain their own atomic-write guarantees.

An empty selection is valid only when the consumer contract permits zero items. In the
current dataset profile, `get` accepts zero or more items and `export` requires one or
more.

## Results, presentation, and failures

Machine content MUST remain in `payload` even when a flag requests a different human
display. For example, `get --smi` and `get --xyz` place structure text and identity in
`payload.representations`. They may mirror it to stderr only when stderr is an interactive
terminal. Multiple displayed records require deterministic identity headings.

TTY-dependent stdout formats and direct writes to `/dev/tty` are forbidden. A caller can
always redirect stdout without changing its schema.

Protocol, usage, and execution failures write no result envelope, write a concise diagnostic
to stderr, and return exit code 2. A completed domain-negative result may also return 2
while emitting a valid envelope—for example, a fully generated quality certificate whose
scientific assessment is `passed: false`. Callers distinguish them by the presence of a
valid envelope, not by parsing diagnostic text. Exit code 0 means a completed nonnegative
result. Interactive presentation is best-effort: a presentation-stream write failure MUST
NOT change a completed machine result or its exit status.

## Command keys and channels

Keys use lowercase dotted ownership:

```text
hotpot.<command-group>.<command>[.<subcommand>...]
```

Keys identify behavior, not Python module names. Renaming or splitting behavior changes
the key in the same Git change. A channel distinguishes stable
semantic variants such as dataset results, data results, or structure representations.
Channel meaning is declared in the map. Multi-document stdout, JSONL, and incremental
streaming are not supported by the current protocol; a channel name alone cannot introduce
them. Supporting streaming later requires replacing the framing, atomicity, map, runtime,
documentation, and tests together in one Git change.

## Current `hotpot ds` profile

All successful `hotpot ds` commands emit keyed envelopes. Selection-producing payloads use
this common locator:

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
  ]
}
```

The initial consumers are activated as follows:

```bash
hotpot ds subset spin -q 1 | hotpot ds get spin -
hotpot ds search spin --scope data --where 'labels.spin_multiplicity=5' \
  | hotpot ds get spin - --xyz
hotpot ds subset spin -q 1 | hotpot ds export spin output - --format mol2
```

`subset`, data/SMARTS `search`, and `get` produce the normalized data-selection contract.
`get` consumes zero or more ordered Hotpot IDs. `export` consumes one or more. Dataset
search, labels, receipts, certificates, ingestion reports, and diagnostics are currently
terminal contracts unless the communication map explicitly declares otherwise.

The structure result shape is:

```json
{
  "data_id": "hp-spin[001]-0",
  "structure_id": "stored-structure-id",
  "format": "xyz",
  "source_format": "mol2",
  "converted": true,
  "content": "3\n..."
}
```

Consumers use the map projection and do not scrape structure content or parse Hotpot IDs to
reconstruct dataset context.

## Extension procedure

To add a command or connect an existing command:

1. Define one stable payload shape and enumerate its key/channel producer entry.
2. Reuse an existing semantic contract when its meaning and cardinality are identical;
   otherwise add one narrowly named contract.
3. Add consumer acceptance and declarative bindings.
4. Implement explicit `-` activation and strict direct-argument conflict checks.
5. Include the communication map in source distributions and wheels.
6. Update English CLI documentation and examples.
7. Test envelope exactness, digest rejection, unknown keys/channels, malformed input,
   empty and many-item behavior, order, context conflicts, no partial stdout, and display
   separation.
8. Replace the map and runtime atomically in the same Git change.

## Compliance checklist

A change is compliant only if:

- stdout contains envelopes and no logs, progress text, warnings, or raw structures;
- every envelope has the exact current digest and exact member set;
- stdin is untouched unless a positional `-` was supplied;
- all projections originate in the dedicated map;
- the complete batch succeeds before output begins;
- screen-oriented content is also available as machine payload;
- stderr presentation cannot change stdout schema;
- source and wheel artifacts contain the map;
- tests and documentation change with the implementation;
- the repository contains one communication map and one matching runtime.

