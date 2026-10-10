# Hotpot xTB plugin

`hotpot.plugins.xtb` connects Hotpot molecular objects to an official xTB
executable. It exposes GFN-FF and GFN0/1/2-xTB as independent calculation
nodes, records backend identity and artifacts, and commits optimized
coordinates only after the native result has been validated.

The plugin does not contain an xTB numerical implementation. Install xTB
separately and select it with `--xtb-executable`,
`HOTPOT_XTB_EXECUTABLE`, or `PATH`, in that order.

## Command-line use

Run a GFN2-xTB optimization:

```bash
$ hotpot xtb input.sdf --method gfn2 --task optimize \
    --output optimized.sdf --report optimized.json
```

Compose independent GFN-FF and GFN2-xTB nodes through a pure SDF stream:

```bash
$ hotpot xtb input.sdf --method gfnff --task optimize \
    | hotpot xtb - --input-format sdf --method gfn2 --task optimize \
    > refined.sdf
```

Native stdout and stderr are not mixed with the molecular stream. Use
`--native-log` when the complete native transcript is required. See
[`cli_doc.md`](cli_doc.md) for every option and additional examples.

## Python use

GFN-FF requires a complete explicit-atom 3D structure and a total charge. The
charge may be explicit or inferred by Hotpot:

<!-- Verified by tests/test_plugin/test_xtb/test_official_integration.py::test_official_gfnff_wrapper_matches_direct_cli_energy -->

```python
import hotpot
from hotpot.plugins.xtb import XTBTask, run_gfnff

mol = hotpot.read_mol("input.sdf")
report = run_gfnff(
    mol,
    task=XTBTask.OPTIMIZE,
    charge=0,
)
print(report.energy_hartree)
```

GFN-xTB additionally requires an unpaired-electron count. Explicit values are
authoritative; otherwise Hotpot resolves charge and applies its documented
lowest-spin parity policy:

<!-- Verified by tests/test_plugin/test_xtb/test_official_integration.py::test_official_gfn2_wrapper_matches_direct_cli_energy -->

```python
import hotpot
from hotpot.plugins.xtb import GFNXTBMethod, XTBTask, run_gfn_xtb

mol = hotpot.read_mol("input.sdf")
report = run_gfn_xtb(
    mol,
    method=GFNXTBMethod.GFN2_XTB,
    task=XTBTask.SINGLEPOINT,
    charge=0,
    unpaired_electrons=0,
)
```

Both functions mutate `mol.coordinates` only for a successful optimization
whose parsed atom order matches the input. A single-point calculation never
commits coordinates.

## Public API

| API | Purpose |
|---|---|
| `run_gfnff(mol, *, ...) -> XTBRunReport` | Execute one independent GFN-FF node with charge inference or an explicit charge. |
| `run_gfn_xtb(mol, *, ...) -> XTBRunReport` | Execute one independent GFN0/1/2-xTB node with a resolved electronic state. |
| `probe_xtb_backend(executable=None, ...) -> XTBBackendInfo` | Resolve and identify an official executable and its verified capability facts. |
| `run_xtb(request) -> XTBRunReport` | Perform one low-level native invocation in an existing workspace. It does not parse numerical results or mutate a molecule. |
| `molecule_to_xtb_geometry(mol) -> XTBGeometry` | Extract immutable symbols and Cartesian coordinates. |
| `prepare_xtb_input(mol, input_path) -> XTBGeometry` | Validate and write an XYZ input while preserving atom order. |
| `parse_xtb_artifacts(report, expected_geometry) -> XTBParsedResult` | Parse finite numerical and geometry evidence from a completed native run. |
| `commit_xtb_coordinates(mol, result) -> None` | Commit validated optimized coordinates to the original molecule. |
| `read_sdf_records(text)` / `write_sdf_records(records)` | Decode or encode the narrow SDF stream contract used between xTB nodes. |
| `validate_element_support(backend_info, method, atomic_numbers)` | Reject unsupported method/element combinations before execution. |

`XTBRunReport` separates process success, xTB convergence, parsed energy,
coordinate commit, inferred-state provenance, retained artifacts, and backend
identity. Failures are explicit subclasses of `XTBError`; methods are never
silently substituted.

## Internal ownership

```text
hotpot/plugins/xtb/
├── contracts.py       # Immutable requests, results, enums, and errors
├── backend.py         # Executable resolution and identity probe
├── capabilities.py    # Versioned method/element applicability
├── adapter.py         # Molecule/XYZ conversion and result parsing
├── runner.py          # One low-level native invocation
├── workflow.py        # Independent GFN-FF and GFN-xTB Python nodes
├── stream.py          # Versioned SDF metadata and provenance codec
├── stage.py           # Controlled-pipeline adapter
├── cli.py             # Standalone CLI adapter
├── cli_doc.md         # Extended CLI guide
└── README.md          # This API and extension guide
```

The private `hotpot.plugins._harness` package owns only generic process facts:
absolute executable resolution, argv/cwd/environment/timeout execution,
temporary workspaces, hashes, and provenance. xTB method semantics, electronic
state, input files, convergence, parsing, applicability, and coordinate commit
remain in this plugin.

## Reference layout for another external program

The xTB package is a reference, not a compulsory base class. A future external
calculator should begin with the same responsibility split and retain only the
files its backend needs:

```text
hotpot/plugins/<program>/
├── __init__.py
├── contracts.py
├── backend.py
├── capabilities.py
├── adapter.py
├── runner.py
├── workflow.py
├── stream.py
├── stage.py
├── cli.py
├── cli_doc.md
└── README.md
```

Do not move program-specific chemistry into `_harness`, and do not introduce a
universal plugin superclass until at least two independent integrations prove
the same abstraction.

## Scientific and operational limits

- Input must be a complete explicit-atom structure with finite 3D coordinates.
  The xTB node does not build a bare SMILES.
- GFN-FF accepts total charge but no spin option. GFN0/1/2-xTB use both charge
  and unpaired-electron count.
- Automatic spin resolution is only a lowest-spin parity assumption. It is not
  oxidation-state or ligand-field inference.
- The stable official xTB 6.7.1 backend supports GFN0/1/2-xTB and its bundled
  GFN-FF parameters only through radon (`Z <= 86`). An unsupported element is
  rejected before launch.
- xTB coordinates and numerical properties do not replace Hotpot bond kinds or
  coordination topology.
- Parsed partial charges are report evidence; they are not assigned to atom
  formal charges or written back to `Molecule`.
- Atom-order validation checks the atom count and element sequence. It cannot
  distinguish an exchange of two atoms of the same element.
- Execution success, native convergence, and an optional Hotpot geometry gate
  are separate facts.

Parity and composition evidence is recorded in
[`plan/12_xtb_workflow/xtb_workflow_validation_report.md`](../../../plan/12_xtb_workflow/xtb_workflow_validation_report.md).
