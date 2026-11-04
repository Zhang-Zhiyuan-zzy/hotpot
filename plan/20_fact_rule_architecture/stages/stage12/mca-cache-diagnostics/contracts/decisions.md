# Stage12 — Derived MCA cache failure diagnostics

## C1 — Identified cache-resource failure, visible temporary materialization

Root accepts this bounded D013 treatment on2026-11-04. The optional persistent
derived cache may use its existing identical temporary materialization only
when an actual OSError identifies cache access/storage unavailability: EACCES,
EROFS, ENOSPC or EDQUOT. Emit a RuntimeWarning retaining the actual error's
errno/message/path and the temporary-copy consequence. This is a discovered
filesystem condition, not a result confidence label or a new speculative guard.
For any other OSError, re-raise the same exception; do not silently recover an
unexplained cause. Non-OSError transformation/content/integrity errors still
propagate. A failure during temporary materialization remains a failure.

Preserve the all-atom graph transformation, source/external hashes, model
encoding/weights, complete-cache reuse, lock release, atomic publication,
temporary lifetime/cleanup and existing source-only/no-transformation path.
Do not add retries, a new exception/result/status type, generic guard framework,
model fallback or extra numerical checks. The original historical OSError was
suppressed and cannot be reconstructed; its cause remains explicitly unknown.

Author only the existing all_atom_model adapter, its MCA companion guide and
focused synthetic cache tests in an isolated proposal. Test persistent success,
known actual filesystem-error injection, original unknown-error identity,
temporary lifetime/cleanup, temporary failure and content rejection. Existing
real-model corruption/selected-all-atom behavior remains in affected source and
installed validation after root application. Root retains application and all
business/acceptance/Git decisions.
