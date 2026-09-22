# Staged complex untangling refactor

## Verified current behavior

The current complex workflow has two sequential phases:

1. `_build_ligand_proxies()` hides every metal--ligand bond, builds and
   optimizes each ligand component, then restores every coordination bond at
   once.
2. `_OpenBabelOptimizer.optimize()` optimizes the fully connected complex and
   evaluates geometry after each epoch.

The second phase does **not** restart from identical coordinates: one Open
Babel molecule and optimizer state are initialized once, and every epoch
continues from the previous epoch.  Its structural limitation is different:
confirmed bond--ring piercing rejects an observed frame but does not trigger a
topology-changing repair.  Expensive full acceptance scans may therefore be
repeated while an ordinary force-field minimizer cannot escape the topology.

## Target workflow

```text
Stage 1: ligand construction
  OBBuilder once per independent candidate
    -> warm-up optimization
    -> inspect ligand-skeleton bond--ring relations
    -> while PIERCES and attempts < ligand_untangling_attempts (default 20):
         open one eligible ring edge
         perturb coordinates
         short optimization with the ring open
         restore the same edge
         inspect again
    -> retain the closed-topology frame with the lowest confirmed piercing count
    -> medium optimization and final candidate inspection

Stage 2.1: coordination-bond restoration
  hide all original metal--ligand bonds
    -> test each pending metal--donor segment against ligand-skeleton rings
    -> restore only non-piercing bonds
    -> short optimization
    -> retry pending bonds
    -> on a stalled round, perturb then optimize
    -> after the bounded budget, restore every still-pending original bond and warn

Stage 2.2: fully connected complex optimization
  apply the Stage-1 open-ring/perturb/restore loop to the complete complex
  (default 30 attempts)
    -> run the stateful final Open Babel relaxation
    -> retain the final frame, or the full trace when `save_movie=True`
```

## Semantic boundaries

- Geometry reports spatial facts only.  Forcefields owns the chemical policy.
- Every piercing decision used for repair is based on
  `ring_scope="ligand_skeleton"` and Relevant Cycles up to size 16.
- A coordination bond is tested as a finite segment before it is restored.
  Consequently, a newly closed chelate cycle is not part of the tested ring
  family.  A narrow forcefield-only guard may additionally ignore a finding
  only when the reported bond is the proposed coordination bond and the ring
  contains both of its endpoints.
- `UNDETERMINED` is recorded but is not actively opened.  Only confirmed
  `PIERCES` relations trigger ring opening.
- Topology changes always require a fresh Open Babel force-field setup.
- Independent candidate count, independent build-attempt budget, ligand
  untangling budget, coordination restoration budget, and full-complex
  untangling budget are separate controls.
- Exhausting an untangling budget is a warning, not a topology-loss error.
  The ligand stage uses the closed-topology frame with the lowest confirmed
  piercing count; coordination restoration always restores the complete
  original metal--ligand topology before Stage 2.2.

## Commit checkpoints

1. Add a bond-identity-preserving selective restore primitive and tests.
2. Add immutable workflow reports and shared ring-piercing repair helpers.
3. Replace ligand rebuild-on-piercing with bounded perturbative untangling.
4. Add incremental coordination-bond restoration with chelate-safe semantics.
5. Add full-complex untangling, trace preservation, and public parameter flow.
6. Run focused, integration, slow-case, lint, and compatibility tests.
