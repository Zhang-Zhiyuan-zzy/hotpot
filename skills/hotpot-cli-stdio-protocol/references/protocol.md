# Hotpot CLI stdin/stdout communication specification

## Scope and authority

This document defines the reusable machine-communication rules for Hotpot CLI
commands. The current executable profile is limited to `hotpot ds`. A future
command group must extend this one protocol rather than create a parallel
transport.

Git is the protocol history. Each checkout contains exactly one current map,
one accepted envelope, and one matching runtime. There is no wire-version
negotiation or compatibility mode.

The terms MUST, MUST NOT, SHOULD, and MAY are normative.

## Output planes

Hotpot keeps three output planes separate:

- stdout is the machine-result plane. A successful command writes exactly one
  JSON envelope followed by LF.
- stderr is the diagnostic plane and, when explicitly requested, the
  interactive presentation plane. It never carries information required to
  reconstruct the machine result.
- paths supplied as command arguments are artifact outputs. Their file formats
  are independent of the stdout envelope.

Help output is an interactive argparse interface and is exempt from the JSON
envelope. Parser and execution failures write diagnostics to stderr.

## Exact envelope

Every machine result has exactly four top-level members:

```json
{
  "key": "hotpot.ds.subset",
  "channel": "default",
  "protocol_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "payload": {}
}
```

- `key` identifies the producing command, not a future consumer.
- `channel` identifies the producer's semantic output variant.
- `protocol_sha256` is the lowercase SHA-256 of the canonical current map.
- `payload` conforms to the named contract for that key and channel.

Consumers MUST reject missing or extra envelope fields, duplicate JSON object
keys, malformed JSON, unknown keys or channels, an unequal protocol digest,
and the non-JSON numbers `NaN`, `Infinity`, and `-Infinity`. They MUST NOT infer
a producer from payload contents.

The map digest is computed from the parsed map serialized as canonical JSON:
UTF-8, `ensure_ascii=false`, sorted object keys, compact separators, and no
non-finite numbers. Because the map embeds every referenced schema digest, the
protocol fingerprint transitively binds all payload schemas.

## Communication map and payload contracts

For the current profile, `hotpot/ds/cli_communication.json` is the only
executable communication map. Routing is:

```text
producer key + channel -> named contract -> optional consumer bindings
```

The closed map has four sections:

```json
{
  "envelope_keys": ["key", "channel", "protocol_sha256", "payload"],
  "contracts": {
    "data_collection": {
      "schema": {
        "path": "schemas/cli/contracts.schema.json",
        "pointer": "#/$defs/data_collection",
        "sha256": "<lowercase-file-sha256>"
      },
      "bindings": {
        "dataset": {"path": "payload.dataset"},
        "selectors": {"path": "payload.items[].id"}
      }
    }
  },
  "producers": {
    "hotpot.ds.subset": {"default": "data_collection"}
  },
  "consumers": {
    "hotpot.ds.get": {
      "accepts": {
        "data_collection": {
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

Every producer/channel MUST name a contract. A `null` contract is forbidden.
Every contract MUST reference a closed payload JSON Schema under the current
package schema root and record the SHA-256 of the complete schema file. The
runtime verifies that digest before accepting or emitting a message and then
validates the selected local JSON pointer.

Bindings use only dotted object paths and `[]` traversal of one array field.
They MUST NOT contain Python expressions, executable JSONPath, imports, shell
fragments, or templates. A need that cannot be represented by the closed map
requires an atomic map-and-runtime change.

Consumer contracts declare inclusive cardinality and whether duplicates are
preserved or rejected. The runtime validates the complete payload before
extracting bindings.

## stdin activation and direct arguments

A command reads protocol stdin only when its documented input positional is
exactly `-`. A pipe, redirected stdin, or non-TTY stdin MUST NOT activate input
implicitly. Raw JSON, JSON Lines, and structure text are not accepted in place
of an envelope.

Direct arguments remain downstream controls. For the current dataset
selection pipeline:

- the required positional dataset is a constraint and must match upstream;
- an omitted release selector inherits the upstream immutable release;
- an explicit release selector must resolve to the same release;
- an omitted store may inherit the upstream store locator;
- an explicit store must resolve to the same store;
- index, ID, and external-identifier modes are invalid with protocol input;
- archive-class visibility remains a downstream direct control; the consumer
  must be able to load every selected ID and still emits only upstream IDs;
- structure selection, display format, conversion format, and output path
  remain downstream command arguments.

Unknown routes, locator conflicts, and invalid cardinality fail closed.

## Batches, ordering, and atomicity

Consumers preserve the projected item order. Duplicate behavior is exactly the
policy in the map. In the current profile, `get` accepts zero or more selected
IDs and preserves duplicates; `export` requires one or more and rejects
duplicates.

A command parses and validates the entire envelope, resolves every selected
record, and completes every requested conversion before writing its first
stdout byte. A pre-emission failure therefore leaves stdout empty. An
artifact-producing command must also preserve its own atomic-publication
guarantee.

The current framing is one JSON document, not JSON Lines or an incremental
stream. Adding streaming requires a coordinated replacement of framing, map,
runtime, documentation, and tests.

## Presentation and failures

Flags such as `get --smi` and `get --xyz` keep the complete representation in
the machine payload. They may mirror a human-readable view to interactive
stderr. TTY-dependent stdout and direct `/dev/tty` writes are forbidden.

Exit status `0` means a completed nonnegative result. Status `2` covers usage,
protocol, execution, or governance failure. A completed domain-negative result
may emit a valid envelope and return `2`; a failure before a complete result
emits no stdout. Callers distinguish those cases by the presence of a valid
envelope, never by parsing diagnostic prose.

## Current `hotpot ds` producers

The current map is the machine authority. Its producer inventory is:

| Producer key | Channels and named contracts |
| --- | --- |
| `hotpot.ds.list` | `default -> dataset_list_result` |
| `hotpot.ds.info` | `default -> dataset_info_result` |
| `hotpot.ds.labels` | `default -> dataset_labels_result` |
| `hotpot.ds.get` | `default, structure -> data_collection` |
| `hotpot.ds.subset` | `default -> data_collection` |
| `hotpot.ds.search` | `data, smarts -> data_collection`; `dataset -> dataset_collection` |
| `hotpot.ds.export` | `default -> export_result` |
| `hotpot.ds.detail` | `default -> detail_result` |
| `hotpot.ds.detail.audit` | `default -> audit_result` |
| `hotpot.ds.detail.query.claims` | `default -> claims_result` |
| `hotpot.ds.detail.status` | `default -> detail_status_result` |
| `hotpot.ds.bundle.build` | `default -> bundle_build_result` |
| `hotpot.ds.bundle.get` | `info -> bundle_info_result`; `summary -> bundle_summary_result`; `cases -> bundle_cases_result`; `case_info -> case_info_result`; `case_assessment -> case_assessment_result` |
| `hotpot.ds.bundle.set` | `default -> bundle_mutation_result` |
| `hotpot.ds.bundle.gate` | `default -> bundle_gate_result` |
| `hotpot.ds.bundle.status` | `default -> bundle_status_result` |
| `hotpot.ds.bundle.synchronize` | `default -> bundle_synchronize_result` |
| `hotpot.ds.certify` | `default -> certificate_result` |
| `hotpot.ds.ingest` | `default -> ingestion_result` |

Only `data_collection` is consumable in this profile:

```bash
hotpot ds subset spin -q 1 | hotpot ds get spin -
hotpot ds search spin --scope data --where 'labels.spin_multiplicity=5' \
  | hotpot ds get spin - --xyz --structure-index 0
hotpot ds subset spin -q 3 \
  | hotpot ds export spin export/reference - --format mol2
```

Bundle lifecycle and mutation commands do not consume stdin. Their JSON files
are explicit artifacts, not transport messages.

## Extension procedure

To connect or add a command:

1. Define one stable payload shape and a named closed JSON Schema definition.
2. Add the schema file digest and producer key/channel to the one map.
3. Reuse a semantic contract only when meaning and cardinality are identical.
4. Add declarative bindings and consumer cardinality only when a pipe is
   intentionally supported.
5. Implement explicit positional-`-` activation and direct-argument conflict
   checks.
6. Validate the full result before emission and keep presentation on stderr.
7. Include the map and schemas in source and wheel package data.
8. Update English CLI documentation and tests in the same change.

The focused tests must cover exact envelope fields, schema and map digest
rejection, duplicate JSON keys, unknown routes, malformed input, empty and
multi-item batches, duplicate policy, context conflicts, no partial stdout,
and stdout/stderr separation.

When a future non-`ds` group adopts the protocol, atomically move the complete
map and runtime to their shared package location and rewire every producer and
consumer. Do not keep both paths or compose independent protocol digests.

## Compliance checklist

A change is compliant only when:

- stdout contains one validated envelope and no logs, warnings, or raw
  structures;
- every producer/channel maps to a named closed payload contract;
- the map fingerprint binds every referenced schema digest;
- stdin remains untouched unless a documented positional `-` is present;
- all consumer projections come from declarative bindings;
- the complete batch succeeds before emission;
- requested screen content remains present in the machine payload;
- stderr presentation cannot alter stdout or exit status;
- package artifacts include the current map and every referenced schema; and
- implementation, map, schemas, English docs, and tests change atomically.
