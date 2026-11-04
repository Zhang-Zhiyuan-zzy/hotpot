# Stage12A explicit facade and lazy toolkit imports

Root accepts this bounded CP311 delivery on 2026-11-04. Entry source is
`8c61532e429e368deec32793f7033b9c4a6a953b`; dirty production/test identities and
final allowlist rebind are recorded in [the checkpoint](checkpoint.json).

The root facade explicitly exports the30existing chemical names and `version`.
Importing it preserves object identity and `sys.path`, without importing RDKit,
Open Babel, Torch or ONNX Runtime. The sole private Open Babel runtime leaf loads
RDKit Chem before OB/pybel on actual adapter/wrapper use. The connected legacy
`works.convert` entry uses that same leaf; its command algorithms and paths stay
with their current owners. Package-root/temp conveniences and wildcard exports
are retired. The example derives its repository output path from its own file.

CCDC solvent values use packaged resources on first public attribute access.
The OrderedDict and set share one cached load. Missing SDK import failures remain
visible. Tests use an explicit fake SDK over74real packaged solvent files and an
explicit missing SDK; they qualify loading/resources, not CCDC chemistry.

| Actual selection | Passed | Failed/errors/skips | Observation |
| --- | ---: | --- | --- |
| Source fresh processes | 8 | 0/0/0 | Explicit facade/version/identity, both actual toolkit orders and native coexistence, CN cold batch conversion, lazy/missing CCDC contracts. |
| Source affected contracts | 178 | 0/0/0 | Conversion, transport, perception, crystal, charge, native/interface and batch-worker behavior. |
| Outside-checkout installed selection | 186 | 0/0/0 | Same ordered source8+178nodeids;13origin receipts, no checkout module/path/shared-library leaks. |
| Source architecture/version recheck | 199 | 0/0/0 | Three obsolete root exceptions retired; no new debt or layer/legacy findings. |

Fresh ethanol observations preserve atoms6/6/8, two single bonds, zero per-Atom
formal charge and zero molecular charge in both toolkits and Hotpot roundtrips.
Actual CN SDF-to-SMILES conversion preserves atoms6/7, one single bond and charge0.
Both native modules load from the selected source or installed product. These
hand-counted graphs and charges are the business checks; historical hash equality
is not used as a chemical truth criterion.

Actual setuptools wheel/sdist generation and pip target installation passed.
The wheel has891products and the sdist926files. Relative to accepted Stage11A,
only ten declared existing products and one new runtime leaf differ. Six accepted
Stage10 binaries and seven licenses were reused exactly, without compilation.

Commands were the root `fresh_source.py`, then `EXECUTE.sh source 01`,
`metadata 01`, `package`, `prepare-installed`, `installed 01`, followed by the
root `metadata_recheck.py`. Their exact argv/cwd/environment/interpreter, source
bindings, timings, resources and actual results are retained. The source fresh
cases were reused rather than rerun in the affected source selection.

Original failures remain available: a root AST checker needed import-alias
normalization; a launcher refused missing release before candidate start; and the
initial metadata run passed198/failed1because three completed root exceptions
remained. Only that allowlist changed before the199-pass recheck; all1511other
parent bindings were unchanged. A final key-order restoration keeps the Git diff
to the26deleted allowlist lines; parsed JSON equals the validated data exactly,
and both byte identities are explicitly retained without repeating business tests.

[Evidence graph](evidence_graph.json) provides selective entry to the181payload
archive manifest and [root verification](root-verification.json). The independent
local review found no concrete evidence defect. Root-review additions remain
outside the original immutable manifest and are explicitly indexed.

This completes Stage12A only. P20-D026 H representation, remaining Stage11/12,
other planned versions/release, Plan12 and global scientific/performance/final
acceptance remain open. No real CCDC SDK, new native compilation, broad default
suite, performance parity or all-version support is claimed here.
