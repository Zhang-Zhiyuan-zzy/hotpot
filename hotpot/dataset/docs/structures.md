# Structure and asset identity

Hotpot may reuse identical structure-file bytes across oxidation, spin, and
other datasets, but byte reuse never merges their Cases or scientific targets.

The historical local inventory contains 2,280 unique coordinate objects.
Oxidation references 2,185; spin references all 2,280; and the collections
share 2,185 coordinate-identical objects. These counts describe asset reuse,
not a claim that the corresponding records have identical chemical identity or
labels.

## Separate identity namespaces

Hotpot keeps these identities distinct:

- Case identity: one independently assessable dataset record;
- sample identity: a policy-controlled scientific equivalence relation;
- chemical entity identity: the represented chemical subject;
- Structure identity: one scientific structure instance or variant;
- Asset identity: exact stored bytes, addressed by SHA-256.

Equal coordinates or files can prove asset reuse. They do not by themselves
prove equal protonation, bond order, stereochemistry, periodic context,
disorder model, measurement condition, or scientific sample. Sample matching
uses the persisted dataset identity schema and can return `same`, `different`,
or `indeterminate`.

`core_info` is a completeness contract, not an automatic identity key. Only
fields and Structure roles explicitly named by the persisted identity schema
participate in sample matching. Target values, evidence/quality levels,
benchmark decisions, and assessment results never participate.

The historical overlap audit compared ordered element symbols and Cartesian
coordinates rounded to eight decimal places while ignoring task-bearing XYZ
comments. That comparator remains useful migration evidence, but it is not a
universal chemical-identity rule.

Even a `same` sample match does not by itself make two observations comparable.
Scientific conflict assessment also requires the same property proposition,
subject/site, compatible conditions and phase, participant roles and
stoichiometry, method branch, units, and uncertainty semantics.

## Bundle and target representation

In a Bundle, every Case-to-Structure relation has an explicit Structure ID and
role. A Structure can declare one or more root-relative representations such as
SMI, MOL2, XYZ, or CIF. Multi-structure records retain all roles; neither query
nor conversion silently substitutes a different variant.

`bundle.json.required_structure_roles` lists scientific roles that every Case
must provide. The reserved value `"*"` instead requires at least one active
Structure with a present, hash-valid, parse-valid representation, regardless of
its role. It may be combined with named roles; each named role remains an
independent requirement. An empty list means that the dataset does not require
a Structure.

After ingestion, positive asset bytes live in target-owned content-addressed
storage:

```text
<database>.assets/objects/sha256/<prefix>/<sha256>
```

Canonical Case and Structure rows reference those objects. Equal bytes are
stored once, while all Case, Structure, entity, and role relations remain
separate. Missing, invalid, or remote-only declarations remain explicit facts
and never fabricate an object.

Historical relation files under local source/audit directories are migration
inputs. Current stores express the same valid links in canonical Case facts and
record every corrected legacy identity relationship in the certificate-bound
identity migration map. They are not interpreted through a separate benchmark
namespace.
