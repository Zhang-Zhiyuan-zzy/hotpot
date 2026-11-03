# Stage10 native ownership

The Stage9 predecessor is accepted only within its recorded CP311 source and
installed scope. This entry releases the remaining physical native/build
migration; it does not accept a build or change scientific behavior.

- Geometry and graph keep their sole native implementations under the existing
  `rules/mathematics/geometry` and `rules/mathematics/graph` owners. Their existing
  private extension basenames and public C++ namespaces remain unchanged.
- `adapters/native/molecule_buffers.py` becomes the sole generic field-selective
  transport. Shared headers live at `adapters/native/include/hotpot/native`; the
  installed SDK still exposes `include/hotpot/native`.
- Migrate consumers, stubs, docs, extension/source lists, relative includes,
  CMake, workflows, package rules and source harnesses together. No shim, duplicate
  implementation, stale-binary build proof or algorithm change is authorized.
- CP311 leads. Clean-source/sdist/wheel, native imports/symbols, standalone SDK
  and supported-version evidence retain distinct scopes. Missing evidence stays
  open; Stage11 and Stage12 decisions are not absorbed by this migration.

The first static scan exposed H/aromatic preparation inside the formerly
unclassified buffer module. This is corrected before build: the named
`operations/native_packing.py` entry retains its original molecular lock and
aromatic → raw fields/partial-charge access → payload → missing-row H order,
then constructs one final buffer. The adapter owns the unchanged DTOs and pure
field/transport encoding. No scientific calculation or error precedence changes.
The stale-charge/atom-transport test explicitly checks that a charge failure
occurs before later H preparation. Initial failed static evidence remains.
