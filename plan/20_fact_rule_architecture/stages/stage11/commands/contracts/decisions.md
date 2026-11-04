# Stage11B — One command owner

Root releases the bounded Decision12 cutover on2026-11-04 at cc9ae78.
The18-path authoring patch was reviewed against its actual before bytes; the
independent Stage12D resource delivery is committed and pushed. This is an
application/validation release, not Stage11 or whole-plan acceptance.

## C1 — Command ownership and supported conversion

`commands/convert.py` owns canonical-SMILES CLI conversion and the retained
batch helpers; `commands/optimize.py` and `commands/ml_train.py` own their
existing optional research handlers. Retire both main/works packages and the
uncalled works.gauss report helper under main-plan9.3. Preserve command names,
argument registrations/defaults and the xtb/run/ds owners. No old-path shim or
second scientific implementation remains.

Use current MolReader and Molecule.smiles to fix the observed obsolete class
read/canonical_smiles accesses. File and directory routes retain every record,
sorted input-file order, combined-file overwrite/append and numbered-directory
outputs. The existing CLI supports smi output only; unsupported output fails
before writing. Preserve explicit output/format requirements. README documents
this actual boundary; no general format-converter capability is claimed.

## C2 — Research-handler boundary

Retain existing algorithms/model registry. Shared Excel selection takes explicit
params/exclude_params or features/exclude_features field names, fixing the
actual sibling-namespace mismatch. Optimization imports its optional scientific
dependencies when used. Actual optional training/optimization needs its declared
dependencies and business evidence; missing packages or test stubs cannot
establish that capability. Do not repair unrelated optional algorithms here.

## C3 — Validation and acceptance

Freeze actual changed source/tests and retained native/resource inputs before
source/wheel execution. Verify cold help, real multirecord/file/directory
conversion, batch spawn/error/timeout cleanup, toolkit/native coexistence and
actual installed origins/package removal. Reuse unchanged scientific/model
evidence only within its real dependency scope. Preserve original failures,
skips and source-only checks. Root reviews observations and takes final bounded
acceptance separately. Decision13, D029 and final global gates remain open.
