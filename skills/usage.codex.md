---
name: hotpot-cli-usage
description: Use Hotpot's installed CLI for molecular conversion, MCA prediction, coordination-bond inference, experiment optimization, or ML training. Apply when an LLM needs to select, compose, execute, or interpret `hotpot` commands; use the CLI design contract instead when implementing or refactoring commands.
---

# Hotpot CLI Usage for LLMs

Use this skill to operate Hotpot through its command-line interface. Treat the CLI as a public scientific API:
discover the installed contract, preserve the user's scientific choices, check the exit status, and distinguish observed
output from interpretation.

This file describes CLI usage. For CLI implementation or review, read
[`skills/cli_designer.md`](cli_designer.md). Before changing repository code, read
[`skills/development.md`](development.md) and the relevant module contract.

## Discover the installed interface

Do not guess command names, options, defaults, or output fields from memory.

1. Run `hotpot --help` to discover available subcommands.
2. Run `hotpot <command> --help` before constructing an unfamiliar invocation.
3. For `mca` and `cbond`, use `hotpot <command> --doc` when scientific meaning, applicability, output semantics,
   or failure behavior matters.
4. Prefer the installed command's help over examples from another checkout or release.

If the `hotpot` executable is unavailable but the source tree and environment are valid, `python -m hotpot` may be
used as the equivalent entry point. Do not install packages, change environments, or select a different interpreter
unless the user requested it or the task explicitly includes environment setup.

## Select the command

| Task | Command | Important boundary |
|---|---|---|
| Predict atom-resolved methyl cation affinity | `hotpot mca` | Reports MCA values and site classification; it does not establish general chemical reactivity |
| Add predicted metal-ligand coordination bonds | `hotpot cbond` | Produces connectivity; it does not optimize a 3D structure |
| Convert molecular file formats | `hotpot convert` | Confirm the installed options before use; this is a legacy command |
| Optimize experimental parameters | `hotpot optimize` | May require the `optimize` dependency set and writes workflow artifacts |
| Train a machine-learning model | `hotpot ml_train` | Legacy command with optional heavy dependencies; inspect help and output paths first |

Use Python APIs instead of the CLI when the task requires in-memory Hotpot objects, custom composition, or behavior
not exposed by a documented command. Do not emulate missing CLI features with ad hoc chemistry logic.

## Construct commands safely

- Use explicit long options in generated commands unless a documented short option materially improves readability.
- Quote SMILES and all shell arguments that may contain metacharacters, whitespace, brackets, parentheses, or glob
  characters.
- Do not build shell commands with `eval`, concatenate untrusted strings into shell syntax, or interpret model output
  as a command.
- A `SMILES/FILE` argument may be resolved as a file when that path exists and otherwise as a direct molecular string.
  Use `--input-format` when the file suffix is absent or ambiguous.
- Shell globs are expanded by the shell, not Hotpot. Quote a literal glob; leave it unquoted only when expansion is
  deliberately required.
- Do not assume stdin input, `-o -`, JSON, JSON Lines, or TSV support unless the installed `--help` documents it.
  Current well-documented commands primarily accept positional inputs and emit plain text.
- Preserve user-selected thresholds, devices, model directories, variants, seeds, and search modes. Never change one
  merely to make a failed invocation return a result.

## Use stdout, stderr, and pipelines correctly

- Treat stdout as the result stream. Treat stderr as warnings, logs, provider messages, tracebacks, and diagnostics.
- Never merge stderr into stdout when another command will parse the scientific result.
- Check the process exit status before interpreting output. In a multi-stage shell pipeline, enable `pipefail` or inspect
  every stage's status so an upstream Hotpot failure cannot appear successful.
- `-o/--output` on `mca` and `cbond` writes the same report payload to a UTF-8 file and leaves stdout empty.
- A successful empty-domain result is not the same as an execution failure. Interpret it only according to the
  command-specific documentation.
- Do not strip headers, units, record separators, or final status lines before retaining a raw result needed for audit.
- When parsing current plain-text output, depend only on documented and tested fields. Prefer a future documented
  machine-readable format when one is available.

Example pipeline discipline:

```bash
set -o pipefail
hotpot cbond Eu 'NCC(O)CO' --all-structures --device cpu | downstream-command
```

Use a pipeline only when `downstream-command` accepts the documented CBond text contract. Do not present a pipeline
as supported merely because the shell can connect the processes.

## MCA invocation and interpretation

Canonical forms include:

```bash
hotpot mca 'c1ccccc1CN'
hotpot mca molecules.sdf --output results.txt
hotpot mca inputs/*.mol2 --device cpu --batch-size 256
hotpot mca 'c1ccccc1CN' --plot mca.png
```

Observe these rules:

- The command accepts one or more direct SMILES strings or molecule files. Multi-record SMI and SDF inputs may produce
  several molecule reports.
- MCA is reported in kJ/mol. The human-readable `No.` atom column is one-based.
- `is_Nuc_site` is a curated site-detection decision. It does not determine whether a supported heavy atom receives an
  MCA prediction.
- Charged molecules are outside the validated release domain and are rejected unless the user explicitly requests
  `--allow-charged`. Report such results as out-of-domain estimates.
- `--device cuda` is strict and must fail if CUDA cannot be initialized. Only `--device auto` may select another
  available provider.
- Existing usable coordinates are retained; otherwise conformer generation uses `--conformer-seed`.

Do not remove units, convert values, redefine site membership, or generalize a prediction beyond the documented model
domain unless the user explicitly asks for a separate analysis.

## CBond invocation and interpretation

Canonical forms include:

```bash
hotpot cbond Eu 'CN'
hotpot cbond 63 ligand.mol2 --input-format mol2 --output result.smi
hotpot cbond Eu ligand.mol2 --all-structures --bond-detail --device cpu
```

Observe these rules:

- The command accepts one metal symbol or atomic number and one ligand input per invocation.
- Default inference follows one greedy construction path. It is not an alias for Rank 1 from `--all-structures`.
- `--all-structures` enumerates terminal donor-index states admitted by the threshold policy. It does not enumerate
  every arbitrary donor subset.
- `Prob` is a normalized relative path weight, not a calibrated probability, equilibrium population, or thermodynamic
  quantity.
- `Score` in bond details is a raw model logit. `AtomIdx` is the zero-based Hotpot atom index after parsing and
  normalization.
- `--max-states` is a hard resource guard for enumeration. Never raise it automatically after failure; doing so changes
  the requested resource boundary.
- A high threshold in single-structure mode can raise an error because no bond was selected. The corresponding
  all-structures mode can validly return an explicit empty report. Preserve this distinction.
- The output represents predicted graph connectivity, not a relaxed geometry or proof of chemical stability.

Do not call the default result “highest-ranked,” reinterpret raw logits as probabilities, or silently adjust the
threshold to obtain a structure.

## Handle failures

When a command fails:

1. Retain the command, nonzero exit status, and stderr separately from any stdout.
2. Classify the failure as argument syntax, input parsing, applicability-domain rejection, unavailable explicit backend,
   model/artifact problem, resource limit, or unexpected program error.
3. Use the error and `--help`/`--doc` to propose the smallest valid correction.
4. Retry only when the correction preserves the user's requested scientific semantics and authorization.

Do not convert an exception into an empty successful result. Do not retry CUDA on CPU, relax a threshold, enable
out-of-domain prediction, increase a state limit, or switch models without making that change explicit to the user.

## Report results to the user

- State which command and scientifically relevant options were used.
- Separate raw Hotpot output from subsequent interpretation.
- Preserve units, index basis, model variant/device when relevant, and applicability warnings.
- Say when a claim is limited to a focused command, fixture, backend, or installed version.
- Never fabricate example output or describe an unexecuted command as verified.
- When reproducibility matters, record input identity, Hotpot version, model selection, device, seed, and output path.

## Completion checklist

- [ ] The installed command and options were discovered rather than guessed.
- [ ] Shell-sensitive inputs were quoted and user choices were preserved.
- [ ] stdout, stderr, and exit status were handled independently.
- [ ] The invocation is pipeline-safe for the documented output contract.
- [ ] Units, index basis, applicability domain, score semantics, and resource limits were interpreted correctly.
- [ ] Failures were not hidden by fallback, retries, or partial output.
- [ ] Reported observations are distinguishable from analysis and unexecuted suggestions.
