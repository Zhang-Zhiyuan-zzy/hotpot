# `hotpot xtb`

Run one independent GFN-FF or GFN-xTB calculation with the official xTB
executable. The command accepts complete explicit-atom 3D structures; it does
not build a bare SMILES implicitly.

## Command synopsis

```bash
$ hotpot xtb 3d-input.sdf --method gfn2 --task optimize --charge 0 \
    > optimized.sdf
```

Use `-` to read molecular records from standard input. The input format must be
stated because standard input has no file suffix:

```bash
$ hotpot ff ligand.smi --route organic --output-format sdf \
    | hotpot xtb - --input-format sdf --method gfn2 --task optimize --charge 0 \
    > optimized.sdf
```

Standard output contains molecular records only. Native xTB output is captured
and can be written separately with `--native-log`.

## Independent methods

GFN-FF and GFN-xTB are separate nodes. GFN-FF never invokes GFN-xTB, and the
GFN-xTB node never invokes GFN-FF.

```bash
$ hotpot xtb input.sdf --method gfnff --task optimize \
    --charge 0 \
    -o gfnff.sdf --report gfnff.json --native-log gfnff.log

$ hotpot xtb gfnff.sdf --method gfn2 --task optimize \
    -o gfn2.sdf --report gfn2.json --native-log gfn2.log
```

The SDF output carries a narrow, versioned set of `HOTPOT_XTB_*` properties:
total charge, unpaired-electron count when applicable, method, energy and
backend provenance. These properties are bound to the molecular record and are
validated before reuse by a later `hotpot xtb` command. Current Hotpot facts
also preserve coordination bond kinds, supplied charge/spin declarations and
radicals across this transport. Reuse verifies the original scientific identity
before running the next method.

## Electronic state

Explicit values are authoritative:

```bash
$ hotpot xtb radical.sdf --method gfn2 --charge 0 \
    --unpaired-electrons 1 -o radical-opt.sdf
```

With no `--state-policy` flag, a molecule containing semantic metal
coordination uses `coordination-screening-v1`; other molecules use `strict`.
Current supplied molecular and atom charge facts are read in both routes.
Explicit invocation values take priority over stored declarations. The
screening policy applies specific rules before its named common defaults and
requires an active profile plus a certified legal state. Use
`--state-policy strict` to require the strict route, or
`--state-policy coordination-screening-v1` to request screening explicitly.

`--unpaired-electrons` is the backend's active UHF count. Physical molecular
spin remains in the scientific state. For ordinary GFN calculations with
f electrons in the core, physical f spin is not projected directly into active
UHF. When no safe projection or stronger execution evidence exists, xTB uses
the lowest legal active-electron parity value. These screening conventions do
not establish a physical ground state. GFN-FF accepts charge but has no spin
option; passing `--unpaired-electrons` with `--method gfnff` is an error.

## Multiple records and resources

Records in an SDF file can be processed concurrently. Output order always
matches input order:

```bash
$ hotpot xtb conformers.sdf --jobs 4 --threads 8 \
    --method gfn2 --task singlepoint --charge 0 -o energies.sdf
```

`--work-directory` selects the parent of isolated native run directories.
Native files are removed by default; use `--keep-work-directory` to retain
them. `--xtb-executable` has precedence over `HOTPOT_XTB_EXECUTABLE` and `PATH`.

## Optional geometry gate

The xTB node does not repeat Hotpot force-field validation by default. Request
one explicit post-calculation gate when needed:

```bash
$ hotpot xtb input.sdf --charge 0 --post-check standard --report checked.json \
    -o checked.sdf
```

Available profiles are `off`, `basic`, `standard`, and `strict`. These checks
reuse Hotpot's geometry rules, without requiring an FF report or inventing FF
diagnostics for xTB. Finite non-hard findings retain the inspectable molecular
structure and the original `quality-failed` diagnostic report, with exit status
0 when every input remains usable. Failed required Cartesian-frame checks emit
no molecule for that sample, retain the native and quality reports with the exact
`XTB:<check.name>` code, and make the command exit 1.

## Failure and exit behavior

Known input/state/domain, native-process, and required-result unavailability
remains local to its sample. Later independent samples run, output order follows
input order, and the ordered JSON report retains every input. The existing
schema version 2 adds an exact `error.code`; native reports and logs retain
available process/artifact evidence. The command exits 1 if any known sample
result is unavailable, 0 if all inputs yield usable results including non-hard
findings, and 2 for invalid invocation or input paths.

Unknown execution errors, molecular stream/provenance mismatches, and commit
integrity errors propagate rather than becoming local sample outcomes. Existing
parallel scheduling does not promise to cancel already-running calculations.

Per-code `--sample-break-on CODE` and `--batch-break-on CODE` belong to controlled
`hotpot run` FF/xTB stages. Standalone `hotpot xtb` does not accept those flags;
see the [controlled pipeline guide](../../pipeline/cli_doc.md).

## Method applicability

Hotpot validates the installed official backend and element domain before the
calculation. Unsupported elements fail explicitly; methods are never silently
substituted. In particular, stable xTB 6.7.1 does not support actinides in
GFN0/1/2-xTB, and its bundled GFN-FF data also ends at radon.
