---
name: hotpot-cli-stdio-protocol
description: Design, implement, review, or extend Hotpot CLI JSON stdin/stdout pipelines, including producer keys, declarative field projection, exact-current-protocol checks, batching, and separation of machine results from terminal presentation. Use for any Hotpot command that emits pipeable data or consumes another Hotpot command; the current runtime profile is limited to `hotpot ds`.
---

# Hotpot CLI stdio protocol

Use the repository's one current communication protocol. Git owns protocol history; keep
one map, one envelope shape, and one matching runtime in each revision.

Before changing a participating command, read
[`references/protocol.md`](references/protocol.md) completely. Then inspect the current
declarative communication map and its tests. For the implemented dataset profile, the map
is `hotpot/ds/cli_communication.json`.

Apply this workflow:

1. Classify each output as a machine result, terminal presentation, or diagnostic.
2. Keep machine stdout exclusively in the exact JSON envelope defined by the reference.
3. Give every producer an exact `hotpot.<group>.<command>` key and a declared channel.
4. Add producer-to-contract and consumer acceptance entries to the communication map.
5. Activate stdin only through a documented positional `-`; never infer it from pipe state.
6. Project fields with the map's non-executable path/constant operations. Never use `eval`,
   executable JSONPath extensions, or command-specific fallback parsing.
7. Validate the complete input and compute the complete output before emitting anything.
8. Keep human presentation off stdout. Put equivalent machine content in the envelope.
9. Update implementation, map, English command documentation, package data, and tests in
   one atomic change.

Run the focused command tests and validate this skill with:

```bash
python /home/zhangzhiyuan/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/hotpot-cli-stdio-protocol
```

