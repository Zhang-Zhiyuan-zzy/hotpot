# Stage12 — PyG export naming

## P1 — Explicit empty-identifier filenames

Root accepts this bounded correction on2026-11-04 for P20-D027. A nonempty
Molecule identifier retains `<identifier>.pt`; an empty identifier uses the
member’s zero-based export ordinal `<index>.pt`. This makes the existing
attempted ordinal policy effective; the current `f"{identifier}.pt" or ...`
always produces a nonempty string and gives Torch the rejected `.pt` name.

Keep member order and the existing later-write-wins behavior for any filename
collision, including named/ordinal collisions. Preserve graph features, prefix
handling, method/function materialization and partial-write/error propagation,
directory creation and the public `None` return. Do not add sanitization,
collision guards, broad catches, retries, new labels or alternate serializers.

Validate actual saved graph chemistry/attributes through both entrypoints and
actual source/installed packages. The old failure remains historical evidence;
its current expected behavior changes to a successful ordinal-named export.
Root owns application, acceptance, records and Git.
