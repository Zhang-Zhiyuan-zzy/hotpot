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
the input/output molecular-payload lineage. `final.sdf` is created only after
every stage succeeds. A scientific quality failure retains its inspectable
stage structure but returns nonzero and does not run downstream stages.

The controller returns `0` only when the complete workflow succeeds, `1` for an
execution or scientific-quality failure, and `2` for an invalid definition or
command request. Diagnostics go to stderr; normal execution does not place a
molecular stream on stdout because the ordered result tree is the output.
