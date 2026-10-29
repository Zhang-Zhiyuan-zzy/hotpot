# Stage1 assembly asset review

Status: read-only baseline audit and disposition recommendations, 2026-10-28.
No production asset, loader, packaging rule or default behavior changed.

The three asset locations contain four JSON files. They are **not interchangeable
copies of one schema**. Keep the active assembly definitions as the sole default;
retire the two obsolete fragment-definition files at their approved cleanup gate;
treat the independent framework collection as a Decision13 data-governance item.
The current wheels also omit the active default file, an existing packaging gap
that the assembly/package migration must close.

## Scope and source identity

Read the main plan's complete Section9 disposition, Stage1 requirements,
Section12 baseline/acceptance requirements, and the Stage8E/10/11 asset gates.
The explicit Stage1 comparison requirement is Section9.2, rows for
`FragTemplete.json`, `MolFragTemplate` and `MolFrameTemplate` (lines1269–1270).
The main-plan SHA256 at this audit is
`ee194ad91c18eb1106b68ff2b05bd4e6c8c97bd96a9ab72e0cad709b2a7683b8`.

All four current files match their raw Git blobs at B20 commit
`e6082f5fd67cb1054726ddcf43db7f963f7091ed`; the production subtree remains
`4a2086e717327c99056ef8cda200b0786e688219`. Paths below are repository-relative.

| Asset | Bytes | SHA256 |
|---|---:|---|
| `hotpot/cheminfo/mol_assemble/FragTemplete.json` | 2716 | `0ac6a05cd76c06d3a918bccf7c59acd053315c1b8629754153ad4b7185f08d52` |
| `hotpot/cheminfo/ChemData/MolFragTemplate/FragDef.json` | 31 | `9960ab8217ecba57417ba4bc20be51a3ac19ce3ff21bd5a023b569237779ef13` |
| `hotpot/cheminfo/ChemData/MolFragTemplate/phosphonic_acid.json` | 151 | `c022ec550e8a62be7138210978bb2ed49376deb3263f9f26f24588b17cb0c8a4` |
| `hotpot/cheminfo/ChemData/MolFrameTemplate/PhenBipyDerived.json` | 3691 | `92f5f61b6da110c9d8f8c7db1fa528b8890000c14481354ee111b480b7b16b70` |

## Content, schema and substitutability

| Asset | Actual structure/content | Same schema or replaceable content? | Recommended disposition |
|---|---|---|---|
| Active `FragTemplete.json` | JSON list of26 method-tagged objects:8 `EdgeShoulder`,15 `AtomLink`,1 `AtomReplace`,1 `BondAdding`,1 `AlkylGraft`. Fragment methods use lowercase `smiles` and nested integer `points`; replacement uses `elements`; alkyl generation uses `link_length`1–8. `name` is descriptive. The `BondAdding` row has null smiles/points, which its loader branch does not consume. | This is the only code-selected default schema. It describes assembly operations, including generated alkyl collections, rather than only molecule structures. | Preserve these definitions/ordering as the sole default. Move/rename to `operations/assembly/data/fragment_template.json` with its loader and explicit package-data inclusion. Do not import legacy records automatically or add a second default. |
| `MolFragTemplate/FragDef.json` | Object `{"Phosphonic_acid": {}}`; one empty definition. | No structure, action or attachment information exists to merge. Not accepted by the active list/method loader. | Recommend deletion at the approved asset-cleanup gate after its zero-consumer/build check; retain this digest in the audit. No replacement record is needed. |
| `MolFragTemplate/phosphonic_acid.json` | One object with `name`, uppercase `SMILES="P(=O)([OH])[OH]"`, flat `action_points=[0]`, `struct_query="H_carbon"`, `action="hydrogen_substitute"`. | Not the active schema. Its P-centered phosphonic-acid fragment and attachment index overlap the active `-PO3H2` row (`P(=O)(O)O`, `points=[[0]]`, `AtomLink`), but the query/action contract is not equivalent. Mere field renaming would lose the legacy carbon-site restriction. | Recommend retiring this unused definition format, retaining the existing active `-PO3H2` row unchanged. Do not count this as verified behavioral equivalence or silently recreate the unused action. Any future carbon-only operation needs its own concrete requirement. |
| `MolFrameTemplate/PhenBipyDerived.json` | Object keyed by strings`"1"`–`"6"`, containing respectively1/2/3/6/10/14 SMILES:36 entries,36 distinct literal strings. No `method`, `points`, schema version, provenance or explicit definition of the grouping keys. | A framework/input-molecule collection, not fragment operation definitions. No literal SMILES overlaps the active fragment rows. It cannot replace the active schema, and the active fragments do not preserve this collection. Literal uniqueness does not establish canonical molecular uniqueness or chemical validity. | Preserve unchanged pending Decision13 governance; if admitted, give the collection one governed dataset owner/manifest, separate from the sole assembly default. If not admitted, deletion needs the asset decision. An approved deferral retains its original path as an explicit exception, without creating a second copy/default. |

The legacy `struct_query` points to
`ChemData/StructQueryTemplate/H_carbon.json`, whose text describes carbon with at
least one H. It is not valid JSON: standard JSON decoding reports
`Expecting ':' delimiter: line 4 column 35 (char 71)`. Its SHA256 is
`68d545fd898d5e456e19292977f3423443209386880dbda064120abc871b59c9`.
No implementation of `hydrogen_substitute` or consumer of this query name was
found beyond the legacy asset itself. In contrast, active `AtomLink` searches
atomic numbers6/7 with hydrogen (`assembler.py:82`), and calls
`atom_link_atom_action` (`action_func.py:39`). Thus fragment overlap does not
establish the same admissible target sites, hydrogen handling or products.

## Code and packaging references

- `AssembleFactory.load_default_assembler` directly opens its sibling
  `FragTemplete.json` (`assembler.py:420`); `init_with_default_assembler`
  extends caller-supplied assemblers with that default (`:425`).
  `load_assembler_file` reads JSON and `load_assembler_contents` dispatches
  each list entry by `method` (`:450`, `:454`); there is no legacy-schema
  converter or automatic directory scan.
- `examples/mol_assemble/assemble.py:47` uses the default constructor.
  `examples/mol_assemble/xjz/fenzizuzhuang.py:11` uses an explicit caller file.
  The assembly README's template example (`:255`) uses the active list schema.
  These paths establish actual/documented consumers of the active schema.
- Searches of production code, tests, examples, workflows and build definitions
  found no loader or direct consumer of `MolFragTemplate`, `MolFrameTemplate`,
  `FragDef.json` or `PhenBipyDerived.json`. Production `ChemData` references
  address element/bond/solvent/other named files, not a generic scan of these
  directories. This is repository consumer evidence, not proof that external
  users never read packaged resources.
- `MANIFEST.in:10` unconditionally grafts `hotpot/cheminfo/ChemData`.
  `pyproject.toml:131` enables `include-package-data`; its explicit package-data
  table has no entry for the active assembly JSON. `MANIFEST.in` likewise has
  no assembly-template include. `setup.py` provides no alternative data hook.
- Read-only ZIP inspection of two already-built B20 wheels found **all three
  legacy ChemData JSON files present, byte-identical to source, and the active
  `FragTemplete.json` absent**. The default loader's sibling-file lookup
  therefore cannot find its default in these wheel contents. The existing
  installed smoke script contains no assembly default-resource check; its
  prior passing results do not establish this resource's presence.

| Inspected existing wheel | SHA256 |
|---|---|
| `/tmp/hotpot-plan20-20261028/build/wheels/hotpot_zzy-0.5.4.5-cp311-cp311-linux_x86_64.whl` | `a7a4c1580881be42a3651a92660efdc934164a4d47a386ffe584b4ef0d4c9a97` |
| `/tmp/hotpot-plan20-20261028/matrix/cp310/source/dist/hotpot_zzy-0.5.4.5-cp310-cp310-linux_x86_64.whl` | `258a6d7a98bcb89ef06d6157e3a89cee2041f14c9da1a9612af6e8e968e372dd` |

## Required later closure

Stage8E owns assembly semantics/default loading and documented examples;
Stage10 must verify the relocated resource in clean-source sdist/wheels and
from an installed package outside the checkout. Preserve the active definition
contents and method/point meaning when fixing its name/path/package inclusion.
Record the source-versus-installed baseline defect explicitly; do not add this
previously missing resource workload to a fabricated B20 timing comparison.

Stage11/Decision13 owns the proposed legacy deletion and framework dataset
disposition. Section9.2's merge/delete choice must reflect the framework file's
actual collection role: this audit recommends preserving its information for
that decision, not treating it as a redundant assembler or silently approving a
new dataset schema/deferral. No production deletion or movement occurs here.

This review used full asset-text/JSON inspection, source/reference searches,
Git-blob equality checks and ZIP-member/hash inspection only. No assembly
execution, SMILES sanitization, test suite, build or benchmark was run; no new
scientific-validity or product-equivalence result is claimed.
