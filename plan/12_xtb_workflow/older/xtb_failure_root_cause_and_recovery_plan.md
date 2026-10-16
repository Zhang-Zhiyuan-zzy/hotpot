# xTB failure root-cause audit and recovery plan

> **OLDER / NON-NORMATIVE.** This document is retained as historical audit
> evidence. It is superseded by
> [`scientific_state_implementation.md`](../scientific_state_implementation.md)
> and the normative specifications indexed there.

Status: audited proposal; no production-code change is made by this document.

Scope: the retained 187-ligand / 182-Eu-complex benchmark executed with xTB
6.7.1 revision `edcfbbe`, plus focused diagnostic reruns described below.

## 1. Executive conclusions

The observed non-passing results are not one failure class:

| Target and route | Native execution failure | Final geometry failure | Correct interpretation |
|---|---:|---:|---|
| Ligand, direct GFN2 | 2 | 3 | Five invalid upstream structures; not six valid ligands rejected by GFN2 |
| Eu complex, direct GFN2 | 22 | 9 | 22 electronic/numerical failures plus 9 converged but chemically unacceptable structures |
| Eu complex, GFN-FF then GFN2 | 44 | 6 | 33 stop in GFN-FF; 11 reach and stop in GFN2; 6 complete both stages but fail final geometry acceptance |

The principal root causes are:

1. The current spin estimator reduces every even-electron system to `UHF=0`.
   This is not a physical electronic-state model for Eu(III), which is normally
   treated as an $4f^6$ center. With otherwise identical inputs, changing only
   `UHF=0` to `UHF=6` allowed 21/22 direct-GFN2 execution failures and 11/11
   post-GFN-FF GFN2 execution failures to complete.
2. Hotpot currently writes XYZ for every xTB method. xTB's GFN-FF implementation
   has a dedicated SDF/MOL branch that reads explicit bond endpoints, whereas
   its XYZ branch reconstructs a neighbour graph from coordinates. Re-running
   all 33 GFN-FF failures with the original SDF structures recovered 33/33
   native executions and 33/33 standard geometry gates.
3. Intermediate geometry findings must not be treated as unconditional
   pipeline stops. A finite but chemically poor UFF/GFN-FF structure may still
   be repaired by a later method. Intermediate checks should warn, rank and
   select recovery frames. Only backend-incompatible inputs should reject the
   current frame, and only the final acceptance gate should decide whether the
   scientific product passes.
4. A real MOL2 aromatic-charge drift exists in the current branch, but it did
   not cause the retained xTB benchmark failures because that benchmark did
   not re-infer the state through a MOL2 boundary. The drift must be fixed as a
   separate representation-invariance problem.

The proposed architecture therefore keeps one authoritative Hotpot molecule,
topology and electronic-state packet in memory, while using a method-specific
native input:

```text
GFN-FF       <- SDF/MOL adjacency + explicit total charge + optional charge/topology evidence
GFN1/2-xTB   <- XYZ geometry + explicit total charge + UHF
all outputs  -> coordinates only are committed to the authoritative Hotpot graph
```

## 2. Evidence identity

The complete original result tree is:

```text
/tmp/hotpot-xtb-benchmark-eu.gzWVgC
```

Its aggregate report is `summary.json`; every route retains its `report.json`,
stage report, native stdout/stderr and native workspace. The benchmark used:

| Item | Value |
|---|---|
| xTB executable | `$CONDA_PREFIX/bin/xtb` in the retained benchmark environment |
| Version | 6.7.1, revision `edcfbbe` |
| Executable SHA-256 | `271dd6cdd5da127a8e39919e7e32d924790fb346f0057b4183769ba8b3411887` |
| Ligand denominator | 187 |
| Eu-complex denominator | 182 |
| Quality gate | Hotpot `standard` |

Focused diagnostics produced the following temporary evidence:

| Diagnostic | Evidence directory | Result |
|---|---|---|
| Direct GFN2 failed cases, UHF=6 | `/tmp/hotpot-eu-uhf6-opt.NihRfw` | 21/22 execute; 18 pass geometry |
| Direct route quality cases, UHF=6 | `/tmp/eu-direct-gfn2-quality-uhf6-audit` | Three original quality failures fully recover |
| Post-GFN-FF GFN2 failures, UHF=6 | `/tmp/hotpot-eu-chain-uhf6-opt.23o5efe4` | 11/11 execute; 9 pass geometry |
| All 33 GFN-FF failures, native SDF input | `/tmp/hotpot-gfnff-sdf-33.o3wvk2tt` | 33/33 execute and 33/33 pass geometry |

These diagnostics change only one controlled variable at a time. They are
causal tests of execution stability, not a claim that every recovered state is
the physically correct electronic ground state.

For the 33-case SDF diagnostic, the native GFN-FF `blist` bond count also
matched the explicit SDF bond count in every case. This is direct evidence
that the recovery came from preserving adjacency rather than from an unrelated
runtime change.

The focused diagnostics are session-local evidence, not yet a committed
benchmark artifact. Their current identities are:

| Evidence | SHA-256 |
|---|---|
| Retained benchmark `summary.json` | `c2e3312baef2fd930dfc7d635e62e16be9a0b2673ac6b9bbf9dc1d5b5b801945` |
| Ordered 182-source-SDF checksum stream | `27fcc164bbd38d8651bbf6b8ea7d49224d3da73fc7cb13c3720e7fc9350d065d` |
| 33-case SDF diagnostic `summary.json` | `01ef400c8b43aed475ba13b0d303d1a0b34b8f33eeba15a2383013c9994f3204` |
| Ordered 33-case input/result/topology checksum stream | `e08e988639de0367c1aea14cd1167b88bd5828ef87ad673cb5529035572ba2b0` |
| Ordered nine-case UHF=6 diagnostic checksum stream | `7f557218286698878e8610e33c35dcab39b7950cd7a71c1198daeb56fb237d51` |

The report was assembled against Hotpot commit
`ecb78e9d347268fcb6559e0bc20faaee9a82789a`; only the new report/diagram and
its stage index were dirty at the final audit. The 33-case summary itself stores
native exit/output facts but not the geometry-gate reports. Thus “33/33 pass
standard geometry” is a session rerun result. Phase C acceptance must rerun the
same audit and persist commands, per-case gate reports, manifests and hashes
inside the benchmark result tree before treating it as durable regression
evidence.

The representative native GFN-FF diagnostic command contract was:

```bash
xtb input.sdf --gfnff --opt --json --chrg 3 --norestart --wrtopo json
```

The geometry audit reloads the authoritative source graph, applies only the
coordinates from `xtbopt.sdf`, and calls Hotpot's `standard` acceptance gate.
This distinction is essential: reloading `xtbopt.sdf` as a new chemical graph
would mix backend geometry evidence with file-parser topology inference.

## 3. Failure-stage audit

### 3.1 Ligands

The retained result contains five, not six, non-passing ligand cases:

| Case | Direct observation | Root cause |
|---:|---|---|
| 0068 | GFN2 converges, but N12-C13 reaches 3.827 A and a ring is pierced | UFF had already stretched the original 1.470 A bond to 2.609 A |
| 0084 | N1-C2 reaches 4.677 A; the chain route reaches 9.457 A | Initial build distance was 9.633 A; UFF never produced a valid connected geometry |
| 0089 | Ring-closing C75-C78 remains 3.871 A | Initial closure was 11.123 A and remained 3.057 A after UFF |
| 0124 | SCC finishes, but gradient reaches 1000.467 Eh/a0 | The upstream fused ring had collapsed; the shortest C-C distance was 0.0856 A |
| 0184 | xTB rejects H62-H63 at 0.000 A | The upstream force field placed two hydrogens at the same coordinate |

All five structures were already marked `failed_quality` before xTB. The 182
upstream-valid ligand structures passed direct GFN2 in 182/182 cases.

### 3.2 Eu complexes: direct GFN2

The 31 non-passing cases split into 22 execution failures and nine final
geometry failures.

#### Native execution failures

| Case | Failure location | Direct cause | UHF=6 diagnostic |
|---:|---|---|---|
| 0015 | Initial SCC | SCC does not converge in 250 iterations | Executes and passes |
| 0028 | Initial SCC | SCC non-convergence | Executes and passes |
| 0033 | Initial SCC | SCC non-convergence | Executes and passes |
| 0034 | Initial SCC | SCC non-convergence | Executes and passes |
| 0035 | Initial SCC | SCC non-convergence | Executes and passes |
| 0036 | Initial SCC | SCC non-convergence | Executes and passes |
| 0037 | Optimization cycle 7 | SCC non-convergence at a later geometry | Executes and passes |
| 0054 | Initial SCC | SCC non-convergence | Executes; final covalent-bond gate fails |
| 0060 | Optimization cycle 26 | SCC non-convergence | Executes and passes |
| 0091 | Initial SCC | SCC non-convergence | Executes and passes |
| 0101 | First gradient | Gradient becomes NaN, followed by NaN SCC | Still fails |
| 0103 | Initial SCC | SCC non-convergence | Executes and passes |
| 0125 | Optimization cycle 17 | SCC non-convergence | Executes and passes |
| 0133 | Initial SCC | SCC non-convergence | Executes and passes |
| 0136 | Initial SCC | SCC non-convergence | Executes; final covalent-bond gate fails |
| 0141 | Initial SCC | SCC non-convergence | Executes; final covalent-bond gate fails |
| 0151 | Initial SCC | SCC non-convergence | Executes and passes |
| 0162 | Initial SCC | SCC non-convergence | Executes and passes |
| 0178 | Initial SCC | SCC non-convergence | Executes and passes |
| 0180 | Initial SCC | SCC non-convergence | Executes and passes |
| 0181 | Optimization cycle 7 | SCC non-convergence | Executes and passes |
| 0184 | Optimization cycle 5 | SCC non-convergence | Executes and passes |

Case 0101 is not explained by spin alone. `UHF=0/2/4/6/8` all fail. Eu is
embedded in a crowded donor region, with Eu-N distances as short as 1.87 A;
moving Eu outward restores a finite gradient. This case requires a metal-local
geometry preflight and relocation path.

#### Numerically successful, geometrically unacceptable

| Case | Final observation | Root-cause class |
|---:|---|---|
| 0021 | Eu-O: 2.409 -> 5.748 A | Coordination dissociation; UHF=6 repairs Eu-O but another covalent finding remains |
| 0022 | O-H: 0.993 -> 1.708 A and Eu-N -> 5.441 A | Proton transfer plus coordination dissociation |
| 0026 | Two Eu-N bonds reach about 4.45 A | Unconstrained target coordination topology dissociates |
| 0029 | Eu-N reaches 4.408 A | Fully recovers with UHF=6 |
| 0075 | Eu-O reaches 4.323 A | Fully recovers with UHF=6 |
| 0098 | Eu-N reaches 4.529 A | Fully recovers with UHF=6 |
| 0100 | C-C rupture, two Eu-N dissociations and ring piercing | Abnormal initial metal placement and global topology rearrangement |
| 0108 | O-H transfer and Eu-N reaches 4.676 A | Protonation and donor-site assignment require review |
| 0134 | Eu-N reaches 5.397 A | The selected Eu-N contact is not retained for the anionic ligand state |

The same nine structures were rerun with `UHF=6`; this is a diagnostic of the
spin hypothesis, not an automatic replacement for the requested state:

| Case | UHF=6 native result | Standard geometry gate |
|---:|---|---|
| 0021 | Converged | Fail: covalent bond-length ratio |
| 0022 | Converged | Fail: two covalent bond-length-ratio findings |
| 0026 | Did not converge | No final geometry |
| 0029 | Converged | Pass |
| 0075 | Converged | Pass |
| 0098 | Converged | Pass |
| 0100 | Did not converge | No final geometry |
| 0108 | Converged | Fail: covalent bond-length ratio |
| 0134 | Converged | Fail: covalent bond-length ratio |

### 3.3 Eu complexes: GFN-FF then GFN2

#### GFN-FF execution failures

Thirty-three cases stop before GFN2. Thirty-one show the same native sequence:

```text
bonded atom energy = NaN
Solving linear equations failed
LAPACK factorisation failed
Geometry optimization did not converge
Could not read topology file
```

Cases 0091 and 0113 additionally terminate with `double free or corruption`
and `SIGABRT` after the numerical state has already become non-finite.

The decisive topology evidence is:

| Case | Hotpot explicit Eu degree -> XYZ-derived GFN-FF degree | Termination |
|---:|---:|---|
| 0022 | 5 -> 10 | NaN/LAPACK |
| 0027 | 3 -> 7 | NaN/LAPACK |
| 0028 | 3 -> 7 | NaN/LAPACK |
| 0029 | 3 -> 7 | NaN/LAPACK |
| 0030 | 3 -> 7 | NaN/LAPACK |
| 0031 | 3 -> 7 | NaN/LAPACK |
| 0032 | 3 -> 7 | NaN/LAPACK |
| 0033 | 3 -> 7 | NaN/LAPACK |
| 0034 | 3 -> 7 | NaN/LAPACK |
| 0035 | 3 -> 7 | NaN/LAPACK |
| 0036 | 3 -> 7 | NaN/LAPACK |
| 0037 | 3 -> 7 | NaN/LAPACK |
| 0042 | 3 -> 7 | NaN/LAPACK |
| 0078 | 3 -> 7 | NaN/LAPACK |
| 0091 | 3 -> 7 | NaN/SIGABRT |
| 0093 | 5 -> 7 | NaN/LAPACK |
| 0094 | 5 -> 7 | NaN/LAPACK |
| 0097 | 3 -> 7 | NaN/LAPACK |
| 0099 | 3 -> 7 | NaN/LAPACK |
| 0101 | 5 -> 11 | NaN/LAPACK |
| 0102 | 3 -> 7 | NaN/LAPACK |
| 0106 | 3 -> 7 | NaN/LAPACK |
| 0110 | 3 -> 7 | NaN/LAPACK |
| 0112 | 3 -> 7 | NaN/LAPACK |
| 0113 | 3 -> 7 | NaN/SIGABRT |
| 0114 | 3 -> 7 | NaN/LAPACK |
| 0117 | 5 -> 7 | NaN/LAPACK |
| 0124 | 4 -> 7 | NaN/LAPACK |
| 0129 | 3 -> 7 | NaN/LAPACK |
| 0130 | 3 -> 7 | NaN/LAPACK |
| 0131 | 3 -> 7 | NaN/LAPACK |
| 0180 | 3 -> 7 | NaN/LAPACK |
| 0181 | 3 -> 7 | NaN/LAPACK |

The current adapter always writes XYZ. xTB 6.7.1
`src/gfnff/gfnff_setup.f90` distinguishes input formats:

- SDF/MOL: iterate over `mol%bonds` and build the neighbour list from explicit
  endpoints;
- XYZ/coord: generate the GFN-FF topology from geometry and distance rules.

The controlled native-SDF rerun recovered all 33 cases and all 33 final
structures passed the same Hotpot standard geometry gate. This establishes
that the primary failure is the method-inappropriate XYZ adapter, not an
intrinsic inability of GFN-FF to optimize these 33 structures.

#### GFN-FF succeeds, GFN2 execution fails

| Case | Direct cause | Direct-GFN2 comparison | UHF=6 result |
|---:|---|---|---|
| 0015 | SCC non-convergence | Also fails directly | Executes and passes |
| 0054 | SCC non-convergence | Also fails directly | Executes; geometry fails |
| 0076 | SCC non-convergence | Direct GFN2 passes | Executes and passes |
| 0092 | SCC non-convergence | Direct GFN2 passes | Executes and passes |
| 0125 | SCC failure during optimization | Also fails directly | Executes and passes |
| 0133 | SCC/optimization recovery failure | Also fails directly | Executes and passes |
| 0136 | SCC non-convergence | Also fails directly | Executes; geometry fails |
| 0174 | SCC non-convergence | Direct GFN2 passes | Executes and passes |
| 0179 | SCC non-convergence | Direct GFN2 passes | Executes and passes |
| 0183 | SCC non-convergence | Direct GFN2 passes | Executes and passes |
| 0184 | SCC failure during optimization | Also fails directly | Executes and passes |

#### Both methods execute; final geometry fails

| Case | Final observation | First clear damage |
|---:|---|---|
| 0021 | Eu-O reaches 5.743 A | GFN2 |
| 0026 | Two Eu-N contacts reach about 4.449 A | GFN2 |
| 0073 | C-H reaches 3.518 A | GFN2 |
| 0075 | Eu-O reaches 4.332 A | GFN2 |
| 0098 | Eu-N reaches 4.528 A | GFN2 |
| 0100 | C-C is already 4.144 A and Eu-N is dissociated after GFN-FF | GFN-FF |

## 4. Intermediate geometry: warning, recovery and hard-stop policy

The concern that a poor UFF or GFN-FF geometry may still be repaired by xTB or
DFT is valid. A geometry quality finding must not automatically terminate the
complete pipeline. Conversely, a higher-level method is a local optimizer and
does not guarantee repair of a singular, disconnected or topologically
entangled input.

The current implementation does not yet meet that policy: an xTB stage returns
`StageStatus.QUALITY_FAILED`, and `pipeline/runner.py` immediately terminates
the ordered pipeline at that stage. The target behavior changes the controller
contract, not the scientific observation: an intermediate quality failure is
persisted as a warning/recovery fact, while the final requested product still
undergoes strict acceptance.

The design must separate three independent axes:

```text
execution_status: not_run | succeeded | failed_recoverable | failed_fatal
geometry_status:  pass | warning | uncertain | fail
next_action:      continue | recover | retry_state | skip_stage | stop_route
```

### 4.1 Three gate levels

| Gate | Purpose | Default action |
|---|---|---|
| Preflight executability | Determine whether the backend can safely receive this frame | Reject only the current unusable frame; select a recovery frame or rebuild |
| Intermediate observation | Record topology deviation, coordination change, piercing and convergence | Emit warning, retain input and output, rank recovery candidates and continue |
| Final acceptance | Decide whether the delivered scientific structure satisfies the requested contract | Pass, uncertain or failed_quality; always retain evidence |

### 4.2 Decision table

| Observation | Current frame | Current stage | Pipeline action |
|---|---|---|---|
| Finite coordinates with long/short bonds, coordination dissociation or ring piercing | Save and warn | May execute | Continue to a higher method or an explicit repair stage |
| Backend did not converge but coordinates, energy and gradient remain finite | Save `last_finite` | Stop or finish as reported by the backend | Continue from `best_recoverable` |
| SCC iteration limit | Preserve input and last finite frame | Stop this electronic-state attempt | Retry a scientifically justified state or skip the optional stage |
| Energy or gradient NaN/Inf, coordinates finite | Reject the non-finite step | Stop stage | Roll back; continue through another route |
| Coordinates NaN/Inf | Reject frame | Stop stage | Roll back; stop only this route if no recovery frame exists |
| Exact atom overlap or backend-rejected near singularity | Do not send that frame onward unchanged | Stop stage before expensive work | Rebuild, perturb or select an earlier frame |
| Unsupported element/method, changed atom mapping, invalid electron parity | Reject | Stop route | Require corrected configuration or another backend |
| Native SIGABRT/segfault | Preserve isolated workspace | Stop stage | Roll back and use another route; never crash the controlling process |

Thus ligand cases 0068/0084/0089 should be reported as warnings and may proceed
to GFN2/DFT. Cases 0124 and 0184 cannot be submitted unchanged because the
chosen frames are numerically singular; the pipeline should select an earlier
finite frame or rebuild, not terminate all downstream possibilities.

Each stage must retain at least:

```text
stage_input
stage_final, when finite
last_finite
best_recoverable
optional full trajectory
```

Frames produced by different methods must not be ranked by raw energy. The
deterministic selection order is atom mapping, backend executability,
catastrophic collisions, covalent topology findings, explicit ring piercing,
target coordination findings, then same-method convergence and energy as a
tie-break.

## 5. Independent inference specifications

Charge, oxidation state and electronic state are separate scientific
quantities and therefore separate implementation modules. Their normative
algorithms, data tables, confidence rules and references live only in these
documents:

| Module specification | Owns | Consumes | Must not decide |
|---|---|---|---|
| [Charge inference](charge_inference.md) | Net-charge evidence/candidates, intact-graph Lewis allocations and original connected-component charges | Immutable facts plus explicit net-charge/Lewis evidence | Coordination cleavage/electron counting, oxidation state, UHF or multiplicity |
| [Oxidation-state inference](oxidation_state_inference.md) | Convention-labelled coordination electron counting, per-atom oxidation-state candidates and globally consistent assignments | `MolecularFacts`, `NetChargeResult`, optional `LewisFormalChargeResult` and a coordination-counting-result tuple | Overwriting net/full-graph Lewis charge, partial charge, UHF, multiplicity or spin coupling |
| [Electronic-state inference](electronic_state_inference.md) | Electron count, UHF/multiplicity candidates and local-spin coupling | `MolecularFacts`, `NetChargeResult`, optional Lewis radical evidence and optional `OxidationStateInferenceResult` | Charge, oxidation state, topology or geometry quality |

The only permitted dependency and control-flow direction is:

```text
state_resolver(MolecularFacts)
├── charge.resolve_net_charge()
│   └── NetChargeResult
├── charge.infer_lewis_allocations(NetChargeResult)
│   └── optional LewisFormalChargeResult
├── for convention in requested_conventions:
│   └── oxidation_state.infer_coordination_electron_counting(
│           NetChargeResult, optional LewisFormalChargeResult, convention)
│       └── one CoordinationElectronCountingResult
├── aggregate coordination results
│   └── tuple[CoordinationElectronCountingResult, ...]
├── oxidation_state.infer_oxidation_states(
│       NetChargeResult,
│       optional LewisFormalChargeResult,
│       coordination-counting-result tuple)
│   └── optional OxidationStateInferenceResult
└── electronic_state.infer_electronic_states(
        NetChargeResult,
        optional LewisFormalChargeResult,
        optional OxidationStateInferenceResult)
    └── ElectronicStateInferenceResult
```

A calculator facade or pipeline resolver owns this sequence. None of the three
modules may call another module's inference implementation, mutate the source
graph, or silently replace an ambiguous result with a unique guess.

For interpreting the benchmark, xTB's `UHF` parameter is
$u=N_\alpha-N_\beta=2S=M-1$, not a request for the unrestricted
Hartree--Fock method. xTB 6.7.1 consumes `--uhf`, while a human-facing
multiplicity is converted by $u=M-1$. The retained Eu diagnostic shows that
`UHF=6` recovers 21/22 direct-GFN2 execution failures and all 11
post-GFN-FF GFN2 failures. This is execution-stability evidence only; the
electronic-state specification explains why it is not a universal Eu(III)
ground-state declaration.

The complete 1--118 neutral-atom UHF prior table is maintained only in the
[electronic-state specification](electronic_state_inference.md). The complete
1--118 selectable oxidation-state registry and the detailed
`infer_oxidation_states()` algorithm are maintained only in the
[oxidation-state specification](oxidation_state_inference.md).

## 6. Method-specific topology and state transport

| Information | GFN-FF native need | GFN1/2-xTB native need |
|---|---|---|
| Elements, atom order and 3D coordinates | Required | Required |
| Total molecular charge | Required; pass explicitly | Required; pass explicitly |
| UHF / unpaired-electron integer | Not consumed | Required execution state |
| Explicit bond endpoints | Required for reliable force-field topology in this workflow | Not consumed by the Hamiltonian |
| Numeric bond order and aromatic/dative semantics | Not consumed by the SDF neighbour builder | Not consumed |
| Intact-graph Lewis formal charges and original connected-component identities | Retained as sideband evidence; limited optional charge-file support | Not consumed directly |
| Geometric constraints | Optional xcontrol potential | Optional xcontrol potential |

### 6.1 Existing in-memory transport and the target contract

The controlled pipeline already has the correct core transport mechanism: the
current `MolecularRecord` carries the full Hotpot `Molecule` and one optional
resolved `ElectronicState`; `runner.py` passes each stage's returned payload
directly to the next stage instead of reloading the persisted SDF. The current
payload identity also hashes atoms, formal charges, implicit hydrogens, bond
endpoints, numeric orders, `BondKind` and electronic state. What is missing is
typed separation of the scientific-state results.

The loss occurs at a narrower boundary: `plugins/xtb/adapter.py` currently
writes `input.xyz` for every xTB method. The solution should therefore extend
the existing `MolecularRecord`/payload contract, not introduce a parallel
molecule container:

```text
MolecularRecord
├── molecule                    sole authoritative chemical graph
│   ├── stable atom identities/order and coordinates
│   ├── bond endpoints, numeric order and BondKind
│   ├── aromatic/formal-charge/hydrogen semantics
│   └── target coordination edges
├── net_charge                  NetChargeResult
├── lewis_state                 LewisFormalChargeResult or None
├── coordination_counting       tuple[CoordinationElectronCountingResult, ...]
├── oxidation_state             OxidationStateInferenceResult or None
├── electronic_state            ElectronicStateInferenceResult or None
└── boundary identity           derived immutable graph/state hash
```

The boundary identity is a canonical immutable fingerprint of facts already
owned by `Molecule`; it must not become a second mutable graph. Constraints and
stage provenance belong to the stage request/result, not to a duplicate graph.
Serialization remains a backend-adapter and evidence concern, not the
authoritative transport inside the Python pipeline. Independent shell
processes require an explicit sidecar or preserved state metadata because they
cannot share the in-memory record.

Convention-labelled ligand/core electron-counting assignments are oxidation-
state sideband evidence. They are not native xTB charge input and must not be
serialized as generic fragment charges.

### 6.2 GFN-FF input

GFN-FF requires a force-field neighbour topology. The planned order is:

1. Write a dedicated one-record SDF/MOL input containing the intended bond
   endpoints and coordinates.
2. Pass total charge explicitly with `--chrg`; do not re-infer it from the
   temporary SDF.
3. Keep full `BondKind`, aromatic, dative and formal-charge semantics in the
   in-memory packet, even though xTB consumes only the subset it needs.
4. Parse the format-dependent native result (`xtbopt.sdf` for SDF input),
   verify atom identities/order, and commit coordinates only onto the original
   Hotpot graph. The current hard-coded `xtbopt.xyz` result contract must change
   together with the input adapter.
5. In same-version diagnostics only, request xTB's JSON topology output and
   parse `gfnff_lists.json["blist"]`; compare that endpoint set with the
   requested adjacency. This is diagnostic evidence, never a topology input or
   a portable product contract. Production verification uses the input-SDF
   adjacency fingerprint, atom-order verification and coordinate-only commit.
6. Run in an isolated workspace with `--norestart`; `gfnff_topo` is a
   version-specific binary restart, not a portable interchange format.

xTB's SDF GFN-FF path reads bond endpoints; its setup code does not use the
third SDF bond-order field while constructing the initial neighbour list.
Therefore the SDF is sufficient for the adjacency problem demonstrated by the
33 cases, but it is not a lossless replacement for Hotpot aromatic/dative
semantics. Those remain in memory.

xTB 6.7.1 has no stable `$ffnb` input. A similarly named neighbour override is
present only on the unreleased upstream development branch and must not enter
the current production plan. If a later released xTB exposes it, support must
be version-gated and independently validated.

GFN-FF can also consume a `charges` file and, on some input paths, fragment
charges. This is optional electronic evidence rather than a substitute for
adjacency. Once metal-ligand bonds make the complex one connected component,
it cannot represent independent internal metal/ligand fragment charges without
additional backend support.

One native caveat must be tested explicitly: the SDF branch may attach a
zero-degree atom to a nearby earlier atom. Consequently, SDF protects the
declared bond endpoints for the demonstrated coordination structures, but is
not automatically a lossless interface for truly isolated ions.

### 6.3 GFN1/2-xTB input

GFN1/2-xTB computes from elements, coordinates, total charge and electronic
state. It does not require a user aromatic-bond assignment. Its adapter should
continue to use a strict XYZ plus:

```text
--chrg <total charge>
--uhf <Nalpha-Nbeta>
```

The absence of explicit bonds from the GFN Hamiltonian input is intentional,
not accidental loss. Nevertheless, target topology must remain available to:

- generate optional distance constraints for coordination edges;
- validate atom order and coordinate commit;
- detect covalent rupture, coordination dissociation and ring piercing;
- preserve the original chemical graph in the output.

No topology inferred from `xtbopt.xyz` should replace the authoritative graph.
This is also consistent with the official C API, whose molecule constructor
accepts atomic numbers, coordinates, total charge and UHF, but no bond or
aromaticity arrays.

### 6.4 Method transition

```text
authoritative MolecularRecord
       |
       +-- GFN-FF adapter -> input.sdf + --chrg + --norestart
       |                     -> native coordinates
       |                     -> intermediate warning/gate
       |
       +-- GFN-xTB adapter -> input.xyz + --chrg + --uhf
                             -> native coordinates
                             -> final or downstream gate

native output coordinates -> atom-order verification -> original graph update
```

If GFN-FF fails, the route rolls back to its input and may proceed directly to
GFN2. A GFN-FF failure must not force GFN2 to consume a partially corrupted
frame.

Coordinate-only commit does not mean that observed chemistry is discarded.
New contacts, ruptures, proton transfers and coordination changes are emitted
as a derived topology proposal/report. They may be accepted by an explicit
reaction or topology-reconciliation operation, but must never silently replace
the authoritative target graph during a geometry-optimization stage.

## 7. Aromaticity and charge drift

The colleague's reported MOL2/Kekule-dependent charge change was independently
reproduced without importing that patch:

```text
source SDF charge versus reloaded MOL2 charge
matches:       7 / 182
mismatches:  175 / 182
```

| MOL2 charge minus source charge | Cases |
|---:|---:|
| 0 | 7 |
| -2 | 98 |
| -4 | 29 |
| -6 | 44 |
| -8 | 4 |

In inspected case 0029, the SDF reload had no semantic aromatic bonds and
inferred charge `+3`; the MOL2 reload had 17 semantic aromatic bonds and
inferred `-1`. This establishes a real representation-invariance defect in
the current formal-charge path. It does not explain the retained xTB failures:
that benchmark resolved charge in memory, passed `--chrg` explicitly and did
not re-infer charge through a MOL2 boundary. Aromatic flags are also not a
direct GFN2 Hamiltonian input.

The corrective contract is specified in
[charge inference](charge_inference.md): preserve trusted in-memory charge
state across controlled boundaries, distinguish `preserve` from `infer`,
make inference invariant to equivalent aromatic/Kekule representations, and
bind serialized sidecars to atom mapping plus graph/state identity. Pipeline
boundary tests must keep these three concepts distinct:

```text
aromatic representation equivalence
formal-charge/protonation state
xTB total charge and UHF execution state
```

## 8. Planned implementation phases

No phase below is implemented by this report.

### Phase A: contracts and evidence states

Likely files:

```text
hotpot/cheminfo/calculator/contracts.py
hotpot/cheminfo/calculator/charge/contracts.py
hotpot/cheminfo/calculator/oxidation_state/contracts.py
hotpot/cheminfo/calculator/electronic_state/contracts.py
hotpot/cheminfo/calculator/state_resolver.py
hotpot/plugins/xtb/contracts.py
hotpot/pipeline/contracts.py
```

- add resolved/unresolved candidate states, confidence and provenance;
- split charge contracts into `NetChargeResult` and
  `LewisFormalChargeResult`; add convention-labelled
  `CoordinationElectronCountingResult` to oxidation-state contracts;
- separate execution, geometry and continuation status;
- retain `MolecularRecord` as the sole in-memory carrier and add only the
  immutable topology/state identity needed for boundary verification;
- preserve user overrides without silent fallback.

### Phase B: independent scientific-state inference

Likely files:

```text
hotpot/cheminfo/calculator/charge/
hotpot/cheminfo/calculator/oxidation_state/
hotpot/cheminfo/calculator/oxidation_state/ligand_electron_counting.py  # new
hotpot/cheminfo/calculator/electronic_state/spin.py
hotpot/cheminfo/calculator/electronic_state/atomic_spin_priors.py  # new
hotpot/cheminfo/calculator/state_resolver.py
```

- implement the independent contracts defined in
  [charge inference](charge_inference.md),
  [oxidation-state inference](oxidation_state_inference.md) and
  [electronic-state inference](electronic_state_inference.md);
- keep control flow in `state_resolver.py`; no inference module may call a
  sibling implementation;
- perform coordination cleavage/CBC bookkeeping only in
  `ligand_electron_counting.py`, always under an explicit convention;
- accept UHF or multiplicity as equivalent explicit inputs;
- replace the global lowest-parity default with metal/oxidation-aware logic;
- return candidate states instead of fabricating a unique state;
- add Eu(II)/Eu(III), Fe(II)/Fe(III), lanthanide and multimetal tests;
- never rank spin states with ordinary GFN2 energy alone.

### Phase C: method-specific native adapters

Likely files:

```text
hotpot/plugins/xtb/adapter.py
hotpot/plugins/xtb/runner.py
hotpot/plugins/xtb/workflow.py
```

- split `prepare_xtb_input()` into GFN-FF SDF and GFN-xTB XYZ adapters;
- pass resolved state explicitly;
- make optimized-result discovery and parsing input-format aware;
- commit coordinates only to the authoritative graph;
- retain native topology evidence;
- use isolated workspaces and `--norestart` for GFN-FF;
- do not depend on unreleased `$ffnb` behavior.

### Phase D: non-blocking intermediate gates and recovery

Likely files:

```text
hotpot/pipeline/runner.py
hotpot/pipeline/builtin_stages.py
hotpot/plugins/xtb/workflow.py
```

- introduce preflight, intermediate observation and final acceptance gates;
- preserve stage input, last finite and best recoverable frames;
- GFN-FF failure rolls back and skips to direct GFN2 when configured;
- geometry warnings continue downstream but remain visible;
- final failure still exports structure, trajectory and evidence.

### Phase E: aromatic charge invariance

Likely files:

```text
hotpot/cheminfo/calculator/charge/
hotpot/plugins/xtb/stream.py
hotpot/pipeline/artifacts.py
```

- implement the representation-invariance contract in
  [charge inference](charge_inference.md);
- avoid re-inference at a controlled serialization boundary;
- validate electronic state and semantic topology sideband;
- keep standalone MOL2 behavior explicit and testable.

## 9. Required validation

Each phase must be independently revertible and tested before the next phase.

| Validation | Acceptance criterion |
|---|---|
| Existing unit and CLI suites | No regression |
| UHF/multiplicity conversion | Exact equivalence and parity validation |
| Eu direct GFN2 diagnostic | The original 22 failures are reclassified with explicit electronic-state provenance; no silent retries |
| GFN-FF SDF adapter | Original 33 GFN-FF failures execute without Eu-neighbour inflation; target is 33/33 |
| GFN-FF native topology | `blist` endpoints match the requested SDF adjacency; isolated-ion behavior is explicitly covered |
| GFN-FF-to-GFN2 transition | Intermediate damage is reported and recovery-frame selection is deterministic |
| MOL2/SDF state round trip | Atom mapping, semantic topology and resolved electronic state remain invariant |
| Aromatic representation variants | Equivalent Kekule forms yield identical charge-state decisions |
| Complete 187/182 benchmark | Per-stage execution, warning, recovery and final-quality counts are reported separately |
| Scientific spot checks | Eu(III), Eu(II), Fe spin variants, ligand radicals and multimetal coupling |

The benchmark report must never collapse source-invalid, stage-execution,
intermediate-quality and final-quality outcomes into one failure count.

## 10. Sources and scientific limits

- xTB 6.7.1 command-line help and `xcontrol(7)`: `--uhf`/`$spin` is
  $N_\alpha-N_\beta$; geometric constraints are separate inputs and are not a
  replacement for force-field topology.
- xTB 6.7.1 source, `src/gfnff/gfnff_setup.f90`: SDF/MOL inputs consume
  explicit bond endpoints; XYZ inputs generate topology from geometry.
- xTB official command-line and spin-polarization documentation:
  <https://xtb-docs.readthedocs.io/en/latest/commandline.html> and
  <https://xtb-docs.readthedocs.io/en/latest/spgfn.html>.
- GFN2-xTB: Bannwarth et al., JCTC 2019,
  DOI `10.1021/acs.jctc.8b01176`.
- GFN-FF: Spicher and Grimme, Angew. Chem. Int. Ed. 2020,
  DOI `10.1002/anie.202004239`.
- Spin-polarized GFN: Neugebauer et al., JCC 2023,
  DOI `10.1002/jcc.27185`.

The SDF diagnostic proves that explicit adjacency repairs the observed GFN-FF
failure class; it does not prove that every GFN-FF minimum is the correct
physical complex. Likewise, UHF=6 removes most Eu SCC failures but must still
be backed by oxidation-state and electronic-state provenance. Scientific
sources for charge, oxidation state and spin priors are maintained in their
respective module specifications rather than duplicated here.
