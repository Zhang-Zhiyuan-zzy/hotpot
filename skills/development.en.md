# Hotpot Development and Code-Change Principles

[中文版](development.md)

This document defines the fundamental principles that must govern code development, refactoring, model integration, testing, and delivery in Hotpot. It applies to both human developers and automated coding agents.

The terms **MUST**, **SHOULD**, and **MAY** are normative. Public contracts and test documentation for individual modules may impose additional constraints, but they must not weaken the requirements in this document.

## 1. Overall Goals

Hotpot's core goal is to provide unified, inspectable chemical objects and computational infrastructure suitable for coordination chemistry. Code changes must prioritize:

1. explicit chemical semantics without silent guesses that manufacture apparently successful results;
2. stable existing abstractions, with new capabilities implemented by reusing and extending established interfaces;
3. user-visible scientific applicability domains, units, errors, and backend boundaries;
4. consistent behavior from the source tree, installed packages, and supported Python versions;
5. logical changes that are testable, reviewable, and reversible.

## 2. Workflow Before Making Changes

Before implementation begins, developers MUST:

1. search Core, I/O, conversion, search, calculator, and model directories for existing identical or related implementations;
2. identify public entry points, data ownership, index conventions, exception types, and downstream callers;
3. distinguish defect fixes, compatibility extensions, scientific-semantic changes, and pure performance optimizations;
4. add the smallest regression test before fixing a defect;
5. define the contract, applicability domain, and migration impact before changing public behavior.

Do not copy a converter, parser, site detector, or model invocation path to bypass an existing implementation. If a small adjustment can make an existing implementation reusable, it SHOULD be improved directly and covered by compatibility tests.

## 3. Hotpot Native Objects Are the Sole Internal Source of Truth

### 3.1 Input Normalization

- External inputs MUST be normalized at the system boundary through `hotpot.cheminfo.convert.to_hotpot_mol()`.
- Internal chemical semantics, atom indices, site detection, and result attachment MUST use Hotpot `Molecule`, `Atom`, and `Bond` objects.
- Conversion logic for RDKit, Open Babel, Pybel, or third-party graphs must not be reimplemented independently in each model.
- Conversion MUST preserve atom order. Hotpot uses 0-based internal indices. Only human-facing CLIs or reports MAY display 1-based indices, and that conversion must be explicit.
- An existing `Molecule` input should not be copied, reparsed, or round-tripped through SMILES without cause, because doing so may discard coordinates, bond metadata, or coordination information.

### 3.2 Backend Responsibilities

- Open Babel is responsible for reading supported files and supplying the perception information it can provide.
- NetworkX is the graph backend for Hotpot substructure search and SMARTS matching.
- RDKit MAY be used for model features, conformers, format bridging, and drawing, but it must not replace Hotpot's production search backend.
- A model backend must not become an implicit source of truth for Core chemical semantics.

## 4. Preserve Existing Abstractions and Public Contracts

### 4.1 Search Abstractions

The following object structures are stable abstractions and MUST be preserved unless a justified and approved architectural change requires otherwise:

- `Query`, `QueryAtom`, and `QueryBond`
- `Substructure`
- `Searcher`
- `Hit` and `Hits`

The active SMARTS compiler entry point is `hotpot.cheminfo.search.smarts.substructure_from_smarts()`. `Molecule.search_substructure()` is the convenience entry point. Do not create a parallel search API.

Search results MUST satisfy the following contract:

- query-to-target mappings are read-only to callers;
- query automorphisms that cover the same target atom set are grouped into one `Hit`;
- `Hit.bonds` contains only target bonds corresponding to query edges;
- additional target-induced edges are exposed through `Hit.induced_bonds`;
- existence checks use `has_match()`;
- large or highly symmetric queries use `iter_mappings()` / `max_matches` for bounded enumeration;
- truncation is explicitly exposed through `Hits.truncated` and is never silent.

### 4.2 Compatibility First

- New capabilities SHOULD be implemented through explicit parameters, enumerations, profiles, or methods.
- Public return types, object relationships, and read-only properties must not be changed merely for internal convenience.
- Fixes to legacy behavior MUST include regression coverage for existing entry points.
- Before deleting an apparently legacy module, confirm production references and the impact on possible external deep imports.

## 5. Chemical Semantics Must Be Explicit and Non-Destructive

### 5.1 Named Semantic Profiles

Different chemical interpretations MUST use named profiles rather than ambiguous Boolean switches or hidden branches.

The current SMARTS target-semantics profiles are:

- `FULL_GRAPH`: the default semantics, preserving full molecular-graph behavior;
- `LIGAND_SKELETON`: a ligand-skeleton descriptor view that excludes the effect of metal-ligand edges on non-metal-side `D/X/v/R/r` values.

A descriptor view MUST operate on a copy or read-only view. It must not temporarily delete, add, or restore bonds in the original molecule. Recursive SMARTS MUST inherit the parent query's semantics. If organometallic covalent semantics are needed in the future, add a separately named profile rather than silently changing an existing one.

### 5.2 Separate Bond Semantics from Numeric Bond Order

`BondKind` is the source of truth for bond semantics and must preserve:

- `SINGLE`
- `DOUBLE`
- `TRIPLE`
- `AROMATIC`
- `ZERO`
- `DATIVE`
- `UNKNOWN`

Bond direction, source, and source metadata MUST be retained when the upstream representation provides them. Do not infer `UNKNOWN`, `ZERO`, or `DATIVE` solely from numeric bond order. A conversion that cannot represent the source without loss SHOULD fail explicitly instead of silently converting a bond to single.

### 5.3 Respect Input Perception

- A semantic profile must not silently recompute implicit hydrogens, aromaticity, or bond types.
- Information lost by Open Babel or another reader must be represented explicitly through `UNKNOWN`, exceptions, fixtures, and documentation.
- Results from external toolkits are comparative evidence, not automatically the normative Hotpot truth.
- Hotpot extensions such as `M`, `Ln`, `An`, `NP`, and `NG` must not be sent untranslated to another SMARTS engine as an oracle.

## 6. Error Handling and Control Flow

- Unknown exceptions MUST propagate. A failure must not be converted into a successful result, an empty result, or a zero value.
- Broad `try/except`, layered `if/else`, or default values must not be used as unconditional fallbacks.
- A fallback may exist only when explicitly allowed by the product contract, and it MUST have a name, documentation, a log or result marker, and dedicated tests.
- Syntax errors, recognized unsupported functionality, input-perception errors, and model-domain errors must remain distinguishable.
- Malformed SMARTS input uses `SmartsSyntaxError`; recognized but unimplemented syntax uses `UnsupportedSmartsError`.
- A read-only scientific property that has not been calculated should raise `AttributeError` with instructions for invoking the calculation. It must not return `0`, `None`, or a fabricated value.

## 7. AI Models and Scientific Results

### 7.1 Layering

Model integration SHOULD keep the following layers separate:

1. input conversion and domain checks;
2. feature and conformer construction;
3. ONNX runtime;
4. raw model output;
5. site detection or other scientific filtering;
6. attachment to Core objects;
7. Python API and CLI.

Do not reimplement model inference or site rules in a CLI, Core property, or drawing routine.

### 7.2 Separate Raw Predictions from Reliable Sites

The two MCA result layers must not be conflated:

- `Atom.mca`: the model's MCA prediction for each supported atom;
- `Molecule.mca_sites`: important, more reliable sites selected through site detection and applicability-domain rules.

An all-atom prediction does not mean that the atom is a reaction site in the model's validated domain. Applicability rules for metal centers, directly coordinated atoms, and similar cases must remain separate, explicit, and tested.

### 7.3 Scientific Boundaries

- Physical units MUST appear in field names, table headers, documentation, and plot legends, for example `mca_kj_mol` and `MCA(kJ/mol)`.
- Inputs outside the training domain must be rejected by default or explicitly marked; silent extrapolation is prohibited.
- MCA currently rejects explicit-hydrogen targets, graphs whose molecular total charge disagrees with the sum of atom formal charges, and unvalidated charged molecules by default.
- An out-of-domain option such as `allow_charged=True` must be explicitly selected by the user and must not be described as validated output.
- MCA must not be conflated with the Mayr `N` or `s_N N` quantities.

### 7.4 Inference-Only Releases

- Production releases SHOULD use ONNX Runtime and must not contain training loops, optimizers, private checkpoints, or private training data.
- Model artifacts MUST include a manifest, hashes, external-weight integrity checks, a model card, a license, applicability-domain documentation, and numerical parity results.
- An explicit `device="cuda"` request MUST fail if the CUDA provider is unavailable. Only `device="auto"` may automatically fall back to CPU.
- Dynamic shapes are preferred over one ONNX model for every input size. Models must define explicit size limits and raise errors when they are exceeded.
- Pruning, FP16, INT8, or any other compression must undergo quantitative parity testing against the originating checkpoint before release. Candidates outside the scientific tolerance must not be released.

## 8. CLI Principles

A CLI is a thin wrapper around the existing Python API:

- it MUST reuse the reader, predictor, site detection, and drawing backend;
- stdout contains only stable, redirectable data and must not include `Done`, debug text, or progress messages;
- `-o` and stdout redirection must produce the same data content;
- logs and warnings go to stderr;
- a CLI must not catch and hide underlying parsing, domain, or runtime errors;
- indices, units, and site meanings in tables, JSON, or images must remain stable and tested;
- visualization is presentation only and must not alter prediction or site selection.

## 9. Tests and Scientific Evidence

### 9.1 Test Location

All test modules and test-only fixtures MUST live under `tests/`. Test scripts, temporary data, and benchmark output must not be mixed into the production `hotpot/` package.

### 9.2 Layered Testing

Every change must include tests proportional to its risk:

1. unit tests for pure functions or object contracts;
2. integration tests across modules;
3. real reader, model, or CLI end-to-end tests;
4. wheel-outside-source-tree smoke tests when installed contents are affected.

Coordination-chemistry and SMARTS semantics SHOULD cover three target layers:

- pure Hotpot graphs independent of a perception backend;
- MOL2/SDF fixtures read through Open Babel;
- real CIF or other representative structure files.

Metal-free organic molecules alone are insufficient evidence that coordination semantics are correct.

### 9.3 SMARTS Conformance

SMARTS changes MUST include the relevant checks for:

- focused parser and matcher regressions;
- the strict contract in `tests/smarts_conformance`;
- corpus schema, case IDs, classifications, feature tags, evidence, and licenses;
- the coordination fixture manifest;
- affected differential, fuzz, and benchmark evidence.

Golden expectations must be manually reviewed as a diff. No tool may automatically overwrite golden data from the current implementation and thereby disguise a regression as a new standard.

### 9.4 Compatibility Claims

After changing Core, conversion, search, MCA, or CBond, the following matrix MUST pass before integration:

```bash
bash tests/run_inference_compatibility.sh 3.9 3.10 3.11 3.12 3.13 3.14
```

This matrix proves only the inference, conversion, and search paths it covers. It does not establish that the entire legacy repository is compatible with every version. Reports must not expand a scoped green result into a repository-wide green claim.

## 10. Performance, Caching, and Determinism

- Performance optimizations must not change chemical semantics or result-order contracts.
- A cache key or signature MUST cover every atom, bond, connectivity, `BondKind`, aromaticity, and semantics state consumed by the calculation.
- A cache must not depend on modifying the original graph, calculating, and then restoring it.
- Search paths with combinatorial-explosion risk should provide an existence fast path, streaming iteration, and an explicit bound instead of silently discarding results.
- Tests, conformer generation, fuzzing, and benchmarks should use fixed seeds. Sources of nondeterminism must be recorded.

## 11. Packaging and Dependencies

- When runtime dependencies, entry points, or package data change, `pyproject.toml`, `setup.py`, and `MANIFEST.in` MUST all be checked and kept consistent.
- Model graphs, external shards, manifests, rule files, and required resources must be included in the actual wheel.
- Before release, build a wheel, install it outside the source tree, and run a real inference or target-feature smoke test.
- An inference package should not import a training framework because of type annotations or import side effects.
- Optional heavy dependencies should be imported lazily so users who do not use that feature are not exposed to unrelated import failures.

## 12. Git and Workspace Discipline

- Development takes place on a dedicated feature branch.
- Each logical milestone corresponds to one independently understandable and reversible commit.
- Commit messages use Conventional Commit style, such as `test:`, `feat:`, `fix:`, `refactor:`, `docs:`, `ci:`, `build:`, and `perf:`.
- Test fences, implementation, documentation, and build changes may be committed as separate logical milestones, but the final commit sequence must explain why behavior changed.
- Do not commit unrelated user-owned changes, untracked directories, generated caches, temporary images, or local environment files.
- Do not use reset, checkout, or whole-file replacement to destroy another developer's uncommitted work. Inspect and reconcile intent when changes overlap.

## 13. Prohibited Anti-Patterns

The following practices are prohibited in principle:

- copying a molecule converter or file reader for one model;
- replacing NetworkX-backed Hotpot search with RDKit SMARTS;
- changing the `Searcher`, `Hits`, `Hit`, or `Query*` object structure without necessity;
- using a Boolean or hidden branch instead of named chemical semantics;
- temporarily deleting and restoring graph bonds to calculate ligand rings;
- inferring missing `BondKind` values from numeric bond order;
- broadly catching exceptions and returning an empty list, default value, CPU result, or success status;
- silently truncating search results;
- presenting every per-atom model prediction as a reliable reaction site;
- releasing quantization, pruning, or a new model without parity validation;
- restoring a matrix of fixed-shape ONNX models;
- mixing test data, training code, or checkpoints into the inference runtime;
- testing only from the source tree without validating the installed wheel;
- claiming repository-wide compatibility from scoped test results.

## 14. Definition of Done

A code change is complete only when all applicable conditions below are satisfied:

- [ ] Existing abstractions were inspected and reused; no unnecessary parallel implementation was created.
- [ ] Public APIs, chemical semantics, units, indices, and exception contracts are explicit.
- [ ] There is no silent fallback, swallowed exception, or unconditional recovery path.
- [ ] Regression tests live under `tests/` and cover both success and failure paths.
- [ ] Real file, model, or CLI paths have been validated end to end in proportion to the risk.
- [ ] Relevant focused tests, strict contracts, and the compatibility matrix pass.
- [ ] Packaging changes pass a wheel installation test outside the source tree.
- [ ] Documentation explains the applicability domain, known limitations, and user-visible behavior.
- [ ] Lint, formatting, and `git diff --check` pass.
- [ ] Commits are atomic and contain no unrelated files belonging to users or other developers.

## 15. Related Normative References

- `tests/README.md`
- `tests/smarts_conformance/SMARTS_CONFORMANCE.md`
- `tests/smarts_conformance/README.md`
- `hotpot/cheminfo/AImodels/INFERENCE_COMPATIBILITY.md`
- `hotpot/cheminfo/AImodels/mca/MODEL_CARD.md`
- `hotpot/cheminfo/AImodels/mca/README.md`
- `hotpot/cheminfo/AImodels/cbond/README.md`
- `.github/workflows/inference_compatibility.yml`

If this document appears to conflict with a more specific module contract, the developer must first identify the reason for the conflict and update the documentation or design. Do not unilaterally select the less restrictive interpretation.
