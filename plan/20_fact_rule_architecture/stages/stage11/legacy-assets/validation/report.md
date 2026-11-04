# Stage11C reference assets and property queries

Root accepts L1–L4 within the affected CP311 source and actual installed-package
boundary. The sole dataset resource owner now contains 162 MOL2 files, the
separate 36-literal framework collection and two unchanged property workbooks.
Four existing query functions and their companion guide live at
`hotpot.cheminfo.operations.solvent_properties`; repository consumers and the
lazy cached CCDC reader use that owner. Seven old files retire, including four
unused fragment/query definitions. There is no old-path shim or second assembly
schema. Formal dataset storage/APIs, chemical algorithms and native bytes did
not change.

The actual nondefault contract is now explicit: medium uses clean/fill, solvent
clean/total. Both nondefault sheets support CASs but lack Cid; order/duplicates
are preserved, unavailable Cid still raises KeyError, nonexistent solvent fill
and missing identifiers still raise ValueError. No missing property is invented
or treated as model-complete. This closes P20-D029's contract/evidence condition.
P20-D028 now retains only optional research execution; data ownership is delivered.

Actual main-source selection passed70/70; outside-checkout installed selection
passed69/69, zero failures or skips. All69 applicable ordered outcomes agree.
The one source-layout-only node passed at source; its four actual wheel/site
resources were independently verified. Both guide blocks executed verbatim;
public MolReader water, methanol and benzene observations preserve expected
composition, bonds, neutral charge, finite coordinates and benzene aromaticity.
Fake-CCDC tests qualify lazy resource/cache/error communication only.

Root independently ran `python -B /tmp/hotpot_plan20/long/stage11-legacy-assets-20261104/root-review01/verify.py`.
It verifies1540current source/test/resource bindings,6unchanged native identities,
172retired paths,183final patch bindings,738required source/wheel/sdist/site
members,892actual wheel product members and925frozen sdist members. Each boundary
has6174origin observations and0violations; five native modules actually imported.
The existing38affected authoring cases are included, not additional to70/69.

Original source01 (69passed/1failed) remains: two generated retired MolProps
pycs made a namespace importable. Root removed exactly those caches/empty
directories; tracked source/test inputs did not change. Source02 passes. The
metadata harness initially rejected equivalent reordered Python specifiers;
parsed-specifier comparison passes on identical artifacts. Both failures and
corrections remain; no test or chemistry threshold was weakened.

Raw drivers, exact argv/cwd/environment/interpreter/timing receipts, JUnit,
observations and actual distributables are under the [evidence graph](evidence_graph.json).
The wheel is7106093bytes, SHA25675adb579f64dad79f902528ecaab9ce9db90ff1d74b1ac86ce1bd1b4777a9c58;
the sdist is2839101bytes, SHA256f9ec37af09f0e8bae5b64225d54c4ab0c13c8ea4b16286130fe82e8ce21e17ba.
The archive has159verified payloads/18031072bytes; its manifest is bound by the
[checkpoint](checkpoint.json).

Six accepted CP311 native binaries were reused, without compilation. This is
not clean-native, standalone SDK, other-version, public-release, optional
training/optimization/real-CCDC execution, final science/cost/Plan12 acceptance,
B_exp or Stage13 evidence. Those obligations retain their owning gates.
