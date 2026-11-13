# Plan21: MetalCluster Goals and Framework Draft

Status: 2026-11-06 discussion draft covering goals and framework only.
This document records current owner direction and framework proposals for discussion.
Related: [Plan20](../20_fact_rule_architecture/fact_rule_architecture_implementation_plan.md),
[Plan20 goal](../20_fact_rule_architecture/goal.md), and the
[chemical type public interface contract](../20_fact_rule_architecture/chemical_type_interface_contract.md).

## 1. Goal and scope

`MetalCluster` represents a metal structural unit that can be recognized, connected,
assembled, or used in calculations as a whole. It should provide a shared mathematical
and logical model, extensible domain registration, external ligand interfaces, and
association with `Molecule`. Both mononuclear and polynuclear units are in scope;
no fixed metal count defines the framework.

Assembly and recognition are equally central workflows between registered definitions
and concrete molecules:

| Direction | Inputs | Result and operation meaning |
|---|---|---|
| Assembly | Named registered definition, concrete core/ligand components, and effective parameters | Creates a metal-complex `Molecule` and its member/role mappings and cluster associations by realizing the definition with those components |
| Recognition | General `Molecule` and a specified definition or selected catalog | Finds matching instances/mappings, if any, under graph, configuration, or combined conditions, without changing source facts |

Assembly components and effective parameters may come from caller inputs, registered
templates/defaults, or automatic resolution by the selected method.

Both directions share definitions, members/roles, boundary interfaces, and method
applicability domains. Assembly creates a concrete realization and its associations;
recognition finds mappings. A definition may admit multiple realizations, and a molecule
may contain multiple matching instances or role mappings. The workflows are therefore
complementary rather than strict inverse functions.

Definitions supply properties and rules within an explicit scope. Atoms, bonds,
coordinates, and supplied declarations retain their existing owners. Assigning a cluster
type interprets these facts without rewriting them.

| Application example | Scope to be explained by its definition |
|---|---|
| UO2, VO2, VO | Metal/oxygen unit, membership boundary, and external attachment |
| Cu/Ru paddlewheels, including (Cu-Cu)OOCR and (Ru-Ru)OOCR | Two metal roles, bridging units, peripheral ligands, and boundary interfaces |
| Zr6(u3-O)6(OH)6 | Metal/oxygen/hydroxyl unit, bridging relationships, and peripheral connections |
| Fe3O | Unit containing three metals and oxygen, and its connections |

These abbreviations preserve the intended application range. They do not determine
complete bond types, oxidation states, charge, spin, protonation, or ligand membership.
Examples without a detailed definition remain in scope; each specialization can state
its own applicability conditions.

## 2. Shared mathematical and logical model

The proposed model is an **attributed relation graph with roles and boundary interfaces,
plus optional geometric constraints**. Attributes and relations express required element,
role, connectivity, or other domain conditions. Geometric constraints describe spatial
conditions in a chosen configuration. Roles and interfaces may refer to atoms, atom
groups, or explicitly defined regions.

An instance can be expressed conceptually as `(D, M, phi)`: a registered definition and
version `D`, the authoritative `Molecule` `M`, and a mapping `phi` from structural roles
to actual members. The mapping identifies internal members and any external context
used to interpret them. A definition may describe a parameterized family, with instances
at several locations or in several molecules. Contextual ligand atoms need not all
become internal members.

The relation pattern expresses constraints. Member graphs and point clouds project
existing facts. The framework should reuse existing mathematics, graph, and geometry
foundations without introducing another authoritative molecular graph, a new hypergraph
system, or duplicating template/result/handle object families. Configuration patterns
can constrain roles and spatial relationships even when connectivity evidence is incomplete.

Membership boundaries come before the interpretation of external connections:

| Concept | Meaning in the framework |
|---|---|
| Allowed attachment pattern or assembly slot | Definition of permitted donor/ligand roles and attachment relationships; it may be unoccupied |
| Existing connection across the instance boundary | Actual bonds or supplied connections crossing the selected boundary, interpreted from `Molecule` facts |
| Port in an external superatom projection | Interface mapping the contracted unit back to internal members or boundary regions and to external connections, under the projection's conventions |

Assembly slot counts therefore do not determine external port counts. An interface
need not correspond to one atom; whether several connections share a port depends on
the projection's meaning. A complete ligand with several bridges can be the scope of
placement or transformation while only a defined subset belongs to the cluster.
Operating on the whole ligand does not make the whole ligand an internal member.

External ligand connection is expressed by mapping ligand attachment sites to cluster
interfaces, including connections spanning several sites or bridge units. The ligand
remains an intact molecular structure where its chemistry requires it. Recognizing
existing connections and proposing new ones are distinct capabilities; matching alone
does not create bonds. Constructed connections and coordinates use the existing
`Molecule` fact owners and do not implicitly redefine the registered membership boundary.

A superatom is a role in a task projection. `MetalCluster` and `Atom` retain distinct
domain identities; contraction creates no artificial atomic_number and inserts no
projection node into the real atom/bond collections. Actual member mappings remain
interpretable.

Overlapping or nested annotations may coexist in one `Molecule`. A task that contracts
instances into superatoms must select compatible instances, avoiding repeated contraction
of shared atoms or silent loss of real connections.

## 3. Typed annotation, discovery, and Molecule association

An instance acts as a typed substructure annotation, associating a registered type,
roles, and scoped properties/rules with actual members. The proposed ownership boundary is:

| Content | Scope and meaning |
|---|---|
| Actual elements, bonds, current coordinates, and supplied declarations | Existing `Molecule`, `Atom`, and `Bond` fact owners |
| Definition identity, roles, and member mapping | Interpretation of this instance, without copying member facts |
| Type-related properties and rules | Declared definition, instance, or method scope; no unconditional global atom attributes |
| Method-specific parameters | Selected method and calculation configuration, with units, applicability, source, and version |

For example, a Ru-Ru target distance may be a reference or target parameter used for a
definition under a particular method. The current supplied or measured distance still
comes from the existing coordinates and may differ. Parameters differing from ordinary
UFF should take effect as an overlay in a selected calculation configuration. Assigning
a cluster type alone must not silently alter ordinary UFF behavior or authoritative
bonds, bond orders, charges, or coordinates.

A cluster-aware calculation may resolve registered parameters automatically under its
named method and selection policy; it need not require manual configuration for each
sample. The selected method owns the effective parameter choice for each term.

The framework intends to retain the current `Molecule` fact layout and expose instances
through association and discovery. It does not require cluster types or force-field
values to be added to Atom/Bond fact columns. Association metadata ownership and storage
remain open.

Discovery has three distinct meanings, interpreted under the selected definition and inputs:

| Discovery basis | Meaning of a match |
|---|---|
| graph | Actual members and connections satisfy the selected graph and attribute conditions |
| configuration | Roles and spatial relationships satisfy geometric conditions; connectivity evidence may be incomplete |
| graph and configuration combined | Both the selected graph conditions and geometric conditions hold |

Geometric similarity alone establishes neither connectivity nor chemical identity.
Matching does not write back to `Molecule`, fill missing facts, or promote a pattern
into another authoritative representation. Explicit association and inferred matches
should remain distinguishable. Overlapping matches may be retained without silently
claiming exclusive members or assigning parameter precedence by discovery order.
Parameter selection belongs to the calculation configuration and its declared policy.

Whether a mapping remains applicable after source changes depends on its actual inputs;
a stale mapping cannot stand in for a current match. Existing identity and lifecycle
principles apply to member changes, copying, components, and merging. Batch use should
reuse graph, geometry, and transaction foundations, avoiding whole-molecule copies for
each annotation or exhaustive rematching on every read.

## 4. Open domain registration and user template

Registration covers `MetalCluster` definitions and families. Users should be able to
register definitions in their own namespaces, persist them, and enable them after import,
without waiting for a Hotpot release or Owner review of every definition. Built-in,
project, and user definitions should have explicit namespaces, identifiers, versions,
and sources, with clear activation scope and name resolution.

A family can parameterize metal combinations, bridging-unit counts, donor roles, or
allowed configurations. Parameter ranges and equivalences need domain meaning: grouping
must preserve chemical distinctions, while a new sample or conformer need not create
a new reusable definition.

The following **conceptual user registration template** describes content for discussion.
It is not a serialization schema or an API declaration:

| Content | What the definition should explain |
|---|---|
| Identity and source | Namespace, definition/family identifier, version, source, and applicability notes |
| Chemical domain | Supported metal/donor roles, necessary state conditions, and chemical interpretations left unresolved |
| Roles and relation pattern | Structural roles, attributes, and required relations, identifying conditions requiring actual connectivity |
| Assembly support | For assembly, component roles/inputs, available templates/defaults, or a named construction method; recognition conditions alone do not specify coordinate generation |
| Membership boundary | Roles mapped to internal members versus external ligands or surrounding context |
| Attachment and boundary interfaces | Allowed attachment patterns, assembly slots, and interpretation of existing connections and projection ports |
| Optional geometric constraints | Configuration conditions and context, distinguished from targets used by a method |
| Family parameters and equivalences | Variable counts/combinations, permitted ranges, and equivalent role mappings |
| Method parameter associations | Optional named methods and separate parameter configurations, with scope, units, and versions |
| Supporting material | Related database observations, literature, or calculations, with supported scope and known limits |

Content depends on the definition's needs; field names, required formats, JSON schema,
and public signatures remain open. New structures can be registered through user
definitions. New algorithms or chemical rules beyond the model's expressive scope belong
to named method extensions, rather than arbitrary script execution or field injection.

## 5. Database collection and knowledge layers

The long-term goal is systematic collection of all known clusters represented in CSD
and other selected structure databases, with reusable registrations and supporting
observations. The framework should support broad database scanning while retaining
these distinctions:

| Layer | Content |
|---|---|
| Database structural observation/instance | Structure and conditions in one source record, member associations, and original source references |
| Reusable definition/family | Shared roles, boundaries, relations, and parameter ranges that multiple instances may satisfy |
| Method-specific parameters | Parameters and sources for a particular calculation or geometry method within its applicability domain |
| Evidence material | What supports an observation, definition, or method, and what remains unsupported |

A crystal record does not automatically create a new definition. Repeated observations,
conformers, or crystallographic setting changes may share a family where appropriate,
while chemically meaningful differences remain visible. Observed database geometry does
not automatically become a universal target distance or a reliable force-field parameter.

Scanning supplies candidates and supporting material; scanning alone cannot guarantee
chemical interpretation of oxidation state, protonation, membership boundary, ligand
assignment, or family classification. This revision accesses no CSD data and creates no
complete cluster catalog. Database access or catalog completion is not a prerequisite
for independent user registration.

Source and method applicability belong with their definitions, parameter configurations,
and material records, without generic confidence labels or repeated known method inputs
in every ordinary return value.

## 6. Paddlewheel family scope

A paddlewheel specialization uses two metal roles and **N bridging units with two attachment
sites each**. A fixed M1/M2, B0-B3 model cannot define the general `MetalCluster` framework.
Permitted N values belong to the specialization and its input constraints, without an
unsupported universal lower bound or a fixed count of four.

N counts bridging units in the defined structure or assembly target, not necessarily
separate ligand molecules: one complete ligand with several bridges may cover multiple
units. Available assembly slots and currently occupied units remain distinct. The two
sites can be donor sites, atom groups, or defined regions, without requiring two O atoms
or one atom at each end. Bridges are not restricted to carboxylates, and units need not
use the same ligand. Domains can include mixed ligands and different metals at the two roles.

A definition may choose a metal core, fragments containing bridging donors, or a larger
unit as its membership boundary. Each choice must explain actual boundary connections
and the whole-unit projection interfaces. Bridge roles, complete ligand operation scope,
and cluster membership remain separate concepts; one sample's boundary does not become
a boundary for every family.

Equal N does not establish N-fold rotational equivalence, arbitrary slot permutations,
or exchange of the two metal roles, particularly with mixed bridges or different metals.
Allowed equivalences belong to the definition and method context; geometry generation
and symmetry algorithms remain outside this draft.

## 7. Public boundaries and open design choices

Retain the previously specified type and future association:

```text
MetalCluster
Molecule.metal_clusters -> list[MetalCluster]
```

The public domain name remains `MetalCluster`, without a parallel public `Cluster`.
This revision adds goals and framework discussion, not constructors, additional members,
registration/matching signatures, or export paths. List container semantics, ordering,
unknown/stale states, and association/discovery behavior on reads remain open.

Substantive choices for further discussion are the membership boundaries and interfaces
of initial specializations, family parameterization and the granularity of chemical
variants, and explicit applicability/selection of method parameters for overlapping
instances. These choices do not require a complete database catalog or distribution of
every user definition in Hotpot.

This directional draft does not prescribe implementation, algorithms, stage steps, or
acceptance criteria, and freezes no data layout, persistence format, native organization,
or new public API. Plan20 retains type/association registration; subsequent runtime work
belongs to Plan21. This revision introduces no additional Plan20 implementation prerequisite.

**Pre-implementation consideration:** Assess whether selected UFF4MOF parameters and
functional terms should be integrated into Hotpot's current UFF force field.
