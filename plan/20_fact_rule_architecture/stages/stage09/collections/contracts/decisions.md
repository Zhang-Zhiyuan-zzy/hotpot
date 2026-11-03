# Stage9C collection completion

2026-11-04; predecessor Stage9B accepted and pushed at `e6908ab`. The existing
main Stage9 and rows9.2-026/9.6-006 own this bounded migration.

## S9C-01 — One collection, existing public behavior

Move the sole MolBundle to `structure/bundle.py`. Preserve its constructor
`list_mol: tp.Iterable[Molecule]`, ordered `list()` materialization and shared
Molecule references, plus `to_pyg_dataset(self, dataset_root: os.PathLike,
prefix: str = '')`. The method delegates locally to `_export_pyg_dataset`, iterating the existing
public member list without another copy. The public function materializes its
input iterable once before invoking that same exporter.
`operations/collections.py` owns the existing public convenience function,
retaining its eager annotations/signature and implicit None return. Bind both
existing exports in cheminfo; keep structure's existing five-name export set.
Delete the old bundle module without a shim or historical-pickle rewrite.
Current pickle and deepcopy preserve the collection and shared graph identity.

## S9C-02 — Export sequencing and external serialization

The operation converts the supplied root to Path and creates a missing leaf
directory with `parents=False`, then obtains the Torch save callable from
`adapters/pyg.py` once before iteration. The adapter imports real Torch lazily.
Keep molecule order, identifier-based filenames, prefix forwarding and implicit
None. Retain the existing `f"{mol.identifier}.pt" or f"{i}.pt"` expression; changing
that established behavior is outside this relocation. Duplicate identifiers
retain overwrite behavior. A later conversion/save failure propagates and leaves
previously saved graphs; no recovery, extra schema or speculative guard is added.
The existing Molecule.to_pyg_data/plugin conversion and feature meanings remain
unchanged. No eager Torch import is added to collection/operation import.

## S9C-03 — Bounded evidence and Stage9 exit

Use actual CP311 Torch/PyG in an isolated existing environment. Check membership
and public contracts, current pickle/deepcopy sharing, real ethanol/benzene graph
semantics, saved roundtrips, prefix/order/empty/duplicate/file-root behavior and
local failure side effects. Preserve observed prechange artifacts as references,
not as the sole chemical oracle or a byte-equality requirement. Check loaded
source/installed origins, zero old module and sole owner. Extend the existing
strict definition guard without weakening prior checks.

Production/support freeze and root source-ready review precede execution.
Affected source/installed collection and connected tests supplement unchanged
accepted9Bdefault/chemical evidence; no redundant broad rerun solely for relocation.
Complete an explicit Stage9 migration crosswalk before closing wholeStage9.
Native clean build/SDK/sdist, other versions and final science/performance/Plan12
obligations remain separate later gates. Root owns integration, acceptance and Git.
