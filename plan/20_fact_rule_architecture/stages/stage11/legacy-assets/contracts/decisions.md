# Stage11C — Chemistry reference assets and property queries

Root accepts L1–L4 on 2026-11-04 at `04e9d43c94c33e37bf3305ac03dc67d5bf7002b3`,
before authoring. These decisions implement the remaining Decision13 route;
they do not close its execution, delivery or whole-plan gates.

## L1 — One shipped resource owner within dataset

Move the existing 162 MOL2 files, the distinct 36-literal framework collection
and two property workbooks without changing their bytes to
`hotpot/dataset/resources/molecules/{coformers,solvent,frameworks}` and
`resources/properties`. A tracked `manifests/packaged-reference-assets.json`
inventory records original/final paths, roles, sizes and digests. It is resource
attribution, not a new public dataset/Bundle schema or chemical-validity verdict.
The framework file becomes `molecules/frameworks/phen_bipy_derived.json`; it is
not an assembly default and is not merged into fragment operation definitions.

These are already distributed small reference inputs. Keep them distributed and
read-only under dataset ownership, explicitly separate from the ignored local
`dataset/data` formal stores, licensed/private supplier payloads, generations,
certificates and audits. Update the dataset layout/docs for this bounded asset
relocation; do not change ds APIs, storage paths, schema, gates or ingestion.
No external data acquisition, scientific enrichment or model changes occur.
Package the resources in the wheel and sdist and verify actual installed readers.

## L2 — Sole query operation and existing public spelling

Move the four existing functions `get_medium_dataframe`,
`get_solvent_dataframe`, `query_medium` and `query_solvent` to
`hotpot.cheminfo.operations.solvent_properties`. Preserve parameter names,
positional/default behavior, DataFrame returns, identifier selection order and
duplicates. Use `importlib.resources.files/as_file` for the sole dataset resource
owner, and move the guide beside the operation. Migrate every actual repository
consumer. Retire the entire MolProps implementation without a compatibility shim
or wildcard export. No new top-level chemical facade export is added.

Root accepts the exact public import-path cutover. Correct the solvent
`sub_sheet` Literal annotation from `clean/fill` to `clean/total`; runtime already
passes those names to the actual workbook. Medium retains `clean/fill`. This is
the concrete domain of the same four functions, not a new query backend.

## L3 — Actual nondefault domains, no fabricated data

The actual medium sheets are clean92×29 and fill96×380; solvent sheets are
clean227×39 and total233×86. Both clean sheets provide CASs/Cid; both nondefault
sheets provide CASs only. Test clean and actual nondefault CAS selection with
order/duplicates; preserve missing-ID ValueError, nonexistent solvent fill
ValueError and unavailable nondefault Cid KeyError. Do not invent Cid values,
map total to fill, rename scientific columns, add a network fallback or turn
missing properties into model-valid observations. Document table/method limits.
These observations supply D029's query-domain disposition only after source and
installed execution pass; no all-property completeness claim follows.

## L4 — Retire unused formats and migrate the live reader

The Stage1 asset review and current product/test/example/build search establish
no executable consumer for the two MolFragTemplate and two StructQueryTemplate
JSON files. Delete them at this gate, retaining original identities/disposition
in the source inventory and Git history. Do not reinterpret their malformed or
obsolete query/action syntax or introduce a second assembly schema.

Migrate the live CCDC solvent resource reader to the sole dataset directory;
preserve its lazy SDK import, cached values, resource materialization and exposed
missing-SDK error. Fake-SDK tests qualify resource communication, not real CCDC
chemistry. Actual public MOL2 reads on known water/methanol/benzene inputs check
chemistry transport separately. Keep every moved asset and mapped consumer in
one coherent delivery; no old/new path fallback remains.

Root reviews the isolated patch before application, freezes source/native/assets,
and releases affected CP311 source and outside-checkout installed-wheel execution.
Actual wheel/sdist membership must preserve moved bytes and omit retired owners.
Keep original failures and unexecuted optional tiers visible. Other versions,
clean-native/SDK release, optional research computation, consolidated costs,
Plan12/final report/B_exp and the separate Stage13 experiment retain their gates.
