# `hotpot run`

Run a controlled sequence of registered molecular stages without evaluating a
shell command. The controller passes Hotpot molecular objects in memory and
writes an ordered, content-addressed evidence tree.

## Inline workflow

Use an exact `::` argv token between stages. Put `--` before the first stage so
stage options cannot be consumed as controller options:

```bash
$ hotpot run --results-dir results/eu-001 -- \
    cbond Eu 'O=C(O)C' \
    :: ff --rebuild --route complex --forcefield uff \
    :: xtb --method gfnff --task optimize \
    :: xtb --method gfn2 --task optimize
```

Use `--rebuild` after CBond: its payload defines the coordination topology but
retains 2D coordinates, which must not be treated as an optimized 3D starting
structure.

`::` is the only separator. The controller does not apply shell expansion,
globbing, command substitution, or `shlex` parsing. A ligand SMILES containing
shell metacharacters therefore remains one literal argument when correctly
quoted by the invoking shell.

## JSON workflow

Long workflows can use a strict JSON file:

```json
{
  "stages": [
    {"name": "cbond", "argv": ["Eu", "O=C(O)C"]},
    {"name": "ff", "argv": ["--rebuild", "--route", "complex", "--forcefield", "uff"]},
    {"name": "xtb", "argv": ["--method", "gfn2", "--task", "optimize"]}
  ]
}
```

```bash
$ hotpot run workflow.json --results-dir results/eu-001
```

Unknown fields, duplicate JSON keys, non-string argv values, empty workflows,
and unknown stages are errors.

## Results and failure semantics

`--results-dir` is the exact new run root. The command refuses an existing path
rather than selecting another name. Each completed stage appears atomically
under `stages/`; a failed execution stage uses a `.failed` suffix and prevents
all downstream stages from running.

```text
results/eu-001/
├── manifest.json
├── stages/
│   ├── 00-cbond/
│   ├── 01-ff/
│   └── 02-xtb/
└── final.sdf
```

The manifest records stage argv, relative artifact paths, SHA-256 digests and
the input/output molecular-payload lineage. FF/xTB reports preserve complete
per-input counts and ordered `sample_outcomes` rows, including local failures
and unattempted rows. Finite non-hard findings keep the selected output and
method diagnostics by default; known unavailability removes only that sample.
A nonempty surviving FF/xTB cohort succeeds even with those findings or local
failures. An empty failed cohort stops before a later generating stage.

FF/xTB stage arguments accept repeatable exact-code `--sample-break-on CODE`
and `--batch-break-on CODE`, both empty by default. For example:

```bash
$ hotpot run --results-dir results/controlled -- \
    cbond Eu 'O=C(O)C' \
    :: ff --rebuild --sample-break-on FF:atom_too_close \
    :: xtb --method gfn2 --batch-break-on XTB-NATIVE-EXECUTION
```

A sample break saves its finite observation at the row's `observation_path`
(`observations/NNNN.sdf`) and omits that sample downstream. A batch break wins
when both scopes match, retains the current defined output, stops later samples
and stages, and records remaining rows as `not-attempted`. An unavailable result
remains unavailable under either scope. A batch break, an empty failed output,
or a final quality failure returns nonzero after persistence and does not create
`final.sdf`. Other stages' intermediate nonempty quality failures may continue.
These break flags belong to controlled stages, not standalone `hotpot ff` or
`hotpot xtb`.

The controller returns `0` only when the complete workflow succeeds, `1` for an
execution or scientific-quality failure, and `2` for an invalid definition or
command request. Diagnostics go to stderr; normal execution does not place a
molecular stream on stdout because the ordered result tree is the output.
