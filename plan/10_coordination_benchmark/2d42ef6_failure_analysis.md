# Failure analysis for cases 54, 61, and 109 at `2d42ef6`

## Conclusions

Cases 54 and 109 have the same control-flow defect: a coordination bond is
accepted solely because it does not pierce a ring, although the metal already
occupies, or is subsequently pulled into, the ligand skeleton. Metal relocation
is only attempted when every pending bond pierces a ring and no coordination
bond has yet been restored, so this defect bypasses relocation completely.

Case 61 is chemically distinct. Five individually nonpiercing Eu--donor bonds
are restored and represented to UFF as ordinary single bonds. The fourth and
fifth restoration steps drive two Eu--N distances to about 1.50 Angstrom. This
requires a post-restoration metal-environment gate and, ultimately, a
coordination-aware restraint or parameterization; relaxing the quality
threshold would hide rather than solve the failure.

## Evidence

All atom indices below are zero-based Hotpot indices. The observations are
identical in the archived `2d42ef6` run and the fresh `1e14f28` run.

### Case 54: Eu collapses toward phosphorus after the second S bond

Input: `C[C@H](CC(C)(C)C)CP(=S)(S)C[C@H](C)CC(C)(C)C`

| Recorded state | P8--Eu54 | S9--Eu54 | S10--Eu54 |
|---|---:|---:|---:|
| Coordination ready | 1.9756 | 3.0277 | 3.9788 |
| First Eu--S bond relaxed | 3.1493 | 2.6889 | 5.2246 |
| Second Eu--S bond relaxed | 1.1509 | 2.3974 | 3.0971 |
| Selected final frame | 1.1443 | 2.4109 | 3.0918 |

The run reports convergence after six epochs, but fails `atom_too_close` for
P8--Eu54: 1.1443 Angstrom against a 1.6775 Angstrom threshold. This is not an
optimizer crash. It is a chemically invalid local minimum admitted by the
coordination-restoration acceptance rule.

### Case 109: the metal starts inside the ligand envelope

Input: `c1ccc2nc(-c3ccc4ccccc4n3)ccc2c1`

| Recorded state | C3--Eu32 | N4--Eu32 | N15--Eu32 |
|---|---:|---:|---:|
| Coordination ready | 0.6607 | 1.7772 | 4.5120 |
| First Eu--N bond relaxed | 1.2448 | 0.7413 | 3.7441 |
| Second Eu--N bond relaxed | 1.0819 | 0.5443 | 3.2965 |
| Selected final frame | 1.1130 | 0.3515 | 3.0768 |

The invalid metal placement exists before the first bond is restored. Both
bonds nevertheless pass the ring-piercing screen, so relocation is never
called. The 100-epoch run exhausts its budget with RMS/max gradients of
1,259/5,266 kJ/(mol Angstrom); only 11 distinct coordinate revisions are
stored, showing that the backend then remains effectively stalled. The final
Eu--N4 distance is 0.3515 Angstrom and its covalent-radius ratio is 0.1307.

### Case 61: high-denticity UFF collapse

Input: `CCCCn1cc[n+](CCNC(=O)c2ccc3ccc4c(c3n2)NC(C(=O)NCC[n+]2ccn(CCCC)c2C)C=C4)c1C`

CBond selects O12, N22, O26, N30, and N33. The five bonds are restored without
a piercing rejection or metal relocation. After the fifth short relaxation,
Eu90--N30 and Eu90--N33 are 1.3914 and 2.0819 Angstrom. At the selected final
frame they are 1.5073 and 1.5013 Angstrom; both are below the 1.7485 Angstrom
minimum and have radius ratios 0.5603 and 0.5581. The run exhausts 100 epochs
at 7,796.65 kJ/mol and does not converge.

Unlike cases 54/109, the initial Eu atom is not already collapsed onto the
failing donors. The failure develops as an overconstrained five-coordinate
topology is added to a generic UFF representation.

## Exact control-flow gap in `2d42ef6`

The relevant source is `hotpot/cheminfo/forcefields/repair.py` at commit
`2d42ef6`:

- lines 785--802 make relocation available only while the molecule has no
  active coordination bond;
- lines 812--820 additionally require every pending bond for a center to be in
  the ring-piercing rejection set;
- lines 883--923 accept a candidate whenever its piercing count is zero and
  immediately restore the bond;
- lines 1034--1073 call relocation only after no candidate bond was restored;
- lines 1131--1140 optimize the enlarged topology but perform no local
  metal--ligand distance or clearance gate before accepting that result.

Existing placement mathematics already provides most required facts in
`coordination.py` at `2d42ef6`: normalized atom clearance (lines 137--150),
normalized bond clearance (152--166), donor-distance ratio and ring-piercing
checks (211--273), and bounded metal candidate search (329--382). The missing
piece is workflow integration, not a new general geometry theory.

## Recommended repair sequence

1. Extract a single factual `CoordinationPlacementEvidence` result from the
   existing placement calculations: metal--donor distance ratios, minimum
   metal/segment-to-atom clearance, minimum segment-to-bond clearance, and
   definite ring-piercing relations. Keep the chemical pass/fail policy in
   `forcefields`, not `geometry`.
2. Before restoring the first bond for each metal center, evaluate the current
   placement. Relocate an unsafe unbound center even when its proposed bonds
   are nonpiercing. Scope active-bond checks per metal center; the current
   molecule-wide early return is also wrong for a future multi-metal complex.
3. Make each bond restoration transactional: snapshot coordinates/topology,
   screen the proposed segment, restore one bond, run the bounded short
   relaxation, then apply the local coordination gate. On failure, restore the
   snapshot and try the next bond ordering or a new metal placement.
4. Preserve the existing exclusion for a newly formed chelate ring containing
   both endpoints of the tested coordination bond. Such a ring is created by
   the proposed bond and is not evidence that the bond pierces an independent
   ligand ring.
5. For cases 54/109, first validate the existing relocation helper through the
   new proactive gate. A prior targeted probe placed case 109 donors at 2.690
   and 2.827 Angstrom and UFF subsequently relaxed both to about 2.343
   Angstrom, so this route has direct evidence.
6. For case 61, use the same transactional gate to prevent accepting the
   collapse, but do not claim this alone can produce a valid five-coordinate
   structure. Add a separate bounded strategy for alternative bond-addition
   orders and coordination templates. If generic UFF still contracts Eu--N,
   expose coordination-distance constraints through the native backend or use
   a coordination-specific force-field model. Never weaken the standard
   geometry thresholds to force a pass.

## Required regression tests before implementing the repair

- Case 54 and 109 trajectories must contain a placement evaluation before the
  first accepted coordination bond, and no accepted frame may contain the
  documented metal--ligand-skeleton clash.
- Every added bond must have explicit pre-screen, post-relaxation acceptance,
  and rollback evidence in the trajectory.
- Case 61 must either finish with all five intended bonds and pass the existing
  distance checks, or return the best finite inspectable frame with an explicit
  warning. It must not silently pass by threshold relaxation.
- The 175 currently passing cases must retain their pass status. The nine
  CBond-threshold failures remain outside the force-field repair denominator.
- Re-run all 187 cases with full trajectories and compare status, topology,
  geometry checks, wall time, and per-stage timing against this report.
