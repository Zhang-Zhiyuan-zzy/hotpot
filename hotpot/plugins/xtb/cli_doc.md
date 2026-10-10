# `hotpot xtb`

Run one independent GFN-FF or GFN-xTB calculation with the official xTB
executable. The command accepts complete explicit-atom 3D structures; it does
not build a bare SMILES implicitly.

## Command synopsis

```bash
$ hotpot xtb 3d-input.sdf --method gfn2 --task optimize > optimized.sdf
```

Use `-` to read molecular records from standard input. The input format must be
stated because standard input has no file suffix:

```bash
$ hotpot ff ligand.smi --route organic --output-format sdf \
    | hotpot xtb - --input-format sdf --method gfn2 --task optimize \
    > optimized.sdf
```

Standard output contains molecular records only. Native xTB output is captured
and can be written separately with `--native-log`.

## Independent methods

GFN-FF and GFN-xTB are separate nodes. GFN-FF never invokes GFN-xTB, and the
GFN-xTB node never invokes GFN-FF.

```bash
$ hotpot xtb input.sdf --method gfnff --task optimize \
    -o gfnff.sdf --report gfnff.json --native-log gfnff.log

$ hotpot xtb gfnff.sdf --method gfn2 --task optimize \
    -o gfn2.sdf --report gfn2.json --native-log gfn2.log
```

The SDF output carries a narrow, versioned set of `HOTPOT_XTB_*` properties:
total charge, unpaired-electron count when applicable, method, energy and
backend provenance. These properties are bound to the molecular record and are
validated before reuse by a later `hotpot xtb` command.

## Electronic state

Explicit values are authoritative:

```bash
$ hotpot xtb radical.sdf --method gfn2 --charge 0 \
    --unpaired-electrons 1 -o radical-opt.sdf
```

Without explicit values, charge is inferred with `--charge-model` and spin uses
the documented lowest-spin parity policy. GFN-FF accepts charge but has no spin
option; passing `--unpaired-electrons` with `--method gfnff` is an error.

## Multiple records and resources

Records in an SDF file can be processed concurrently. Output order always
matches input order:

```bash
$ hotpot xtb conformers.sdf --jobs 4 --threads 8 \
    --method gfn2 --task singlepoint -o energies.sdf
```

`--work-directory` selects the parent of isolated native run directories.
Native files are removed by default; use `--keep-work-directory` to retain
them. `--xtb-executable` has precedence over `HOTPOT_XTB_EXECUTABLE` and `PATH`.

## Optional geometry gate

The xTB node does not repeat Hotpot force-field validation by default. Request
one explicit post-calculation gate when needed:

```bash
$ hotpot xtb input.sdf --post-check standard --report checked.json \
    -o checked.sdf
```

Available profiles are `off`, `basic`, `standard`, and `strict`. A failed gate
returns a nonzero status but still emits the inspectable molecular structure.

## Method applicability

Hotpot validates the installed official backend and element domain before the
calculation. Unsupported elements fail explicitly; methods are never silently
substituted. In particular, stable xTB 6.7.1 does not support actinides in
GFN0/1/2-xTB, and its bundled GFN-FF data also ends at radon.
