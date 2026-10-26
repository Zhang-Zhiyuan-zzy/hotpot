# Hotpot validation evidence

This document carries the detailed evidence summarized in the project
[README](../README.md). Geometry acceptance establishes that the configured
structural checks passed; it does not establish a unique global minimum or a
chemically definitive coordination environment.

## Ligand and Am–ligand benchmark

Four independently launched workflows were evaluated on the same 187 isolated
ligands and the same 182 CBond-eligible Am–ligand complexes. Hotpot uses its
default `FAST` convergence policy. Times are median construction-plus-optimize
durations among completed workflows in a 16-worker run; CBond inference,
serialization, rendering, and any external post-validation are excluded.
Hotpot `optimize_complex` uses internal acceptance checks to select its route;
their cost therefore remains inside its reported duration.

| Workflow | Ligand geometry pass | Ligand median time | Am–ligand geometry pass | Am–ligand median time |
|---|---:|---:|---:|---:|
| RDKit | 186/187 (99.5%) | 0.314 s | 30/182 (16.5%) | 0.457 s |
| Open Babel | 181/187 (96.8%) | 1.159 s | 162/182 (89.0%) | 2.134 s |
| Hotpot `obWrappers` (`FAST`) | 183/187 (97.9%) | 1.759 s | 164/182 (90.1%) | 2.919 s |
| Hotpot `optimize_complex` | 182/187 (97.3%) | 3.372 s | 182/182 (100.0%) | 3.732 s |

![Ligand and Am coordination-complex geometry and efficiency comparison](../assets/readme/coordination_complex_backend_comparison.png)

The independent launch commands and evidence files are documented with the
benchmark:

- [benchmark commands](../tests/benchmarks/coordination_complexes/README.md)
- [aggregate CSV](../assets/readme/coordination_complex_backend_comparison.csv)
- [protocol and aggregate JSON](../assets/readme/coordination_complex_backend_comparison.json)
- [per-case CSV](../assets/readme/coordination_complex_backend_comparison_cases.csv)

For Am–ligand complexes, Hotpot `optimize_complex` accepted 164 native `FAST`
results directly and sent 18 rejected candidates through the complete complex
workflow. All four convergence policies remain selectable through the Python
API and `hotpot ff --convergence-level`; automatic fallback does not silently
upgrade the caller's selected policy to `STRICT`.

## Am–ligand construction gallery

The same 187 extractant ligands were submitted to Hotpot's automatic complex
workflow with Am: native `FAST` optimization first, followed by the complete
complex workflow only when the preliminary structure was rejected. Hotpot
inferred at least one Am–donor coordination bond for **182/187** inputs;
**5/187** inputs produced no Am CBond. All **182/182** optimized outputs
passed the current generic geometry gate; this does not establish that every
coordination environment is chemically definitive.

The galleries use explicit hydrogens and a white-background, orthographic,
Materials Studio-inspired ball-and-stick style. Each structure is viewed along
the maximum principal moment axis of its mass-weighted inertia tensor.

![Am coordination complexes constructed from CBond-positive extractants](../assets/readme/am_extractant_cbond_complexes.png)

*CBond-positive inputs. Each panel shows the workflow-selected final
structure.*

![Extractant ligands without an inferred Am coordination bond](../assets/readme/am_extractant_no_cbond_ligands.png)

*Inputs for which no Am CBond was inferred. These are visualization-only 3D
ligand structures, not Am complexes.*

The path-free [gallery evidence](../assets/readme/am_extractant_gallery_evidence.json)
records the exhaustive case partition, structure and report hashes, explicit-H
and Am counts, rendering status, principal moments, and image hashes.

## xTB workflow validation

The wrapper passed five direct-parity checks against official xTB 6.7.1:
neutral GFN2 and GFN-FF energies, GFN2 optimized energy and coordinates, a
chloride anion, and a hydrogen radical. Absolute tolerances were `1e-12 Eh`
for energy and `1e-8 Å` for optimized coordinates. Three controlled
coordination pipelines also passed: two Zn/NCCO routes and the exact
Eu/acetate workflow shown in the project README.

The four-route benchmark used the same 187 ligands and the 182 Eu complexes
for which CBond supplied an input structure. Times are all-record medians from
a 16-worker run with four xTB threads per case.

| xTB route | Ligand geometry pass | Ligand median time | Eu–ligand geometry pass | Eu–ligand median time |
|---|---:|---:|---:|---:|
| Direct GFN2 | 182/187 (97.3%) | 5.637 s | 151/182 (83.0%) | 7.561 s |
| GFN-FF → GFN2 | 182/187 (97.3%) | 3.936 s | 132/182 (72.5%) | 6.647 s |

For this Eu corpus, direct GFN2 was the more reliable complex route. The lower
all-record median of the GFN-FF chain is confounded by more early failures and
does not demonstrate faster successful paths; GFN-FF therefore remains an
optional node. See the
[complete validation report](../plan/12_xtb_workflow/xtb_workflow_validation_report.md)
and its path-free
[benchmark evidence](../assets/readme/xtb_coordination_benchmark.json).
