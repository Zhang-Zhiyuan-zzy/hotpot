# Plan21: MetalCluster Goals and Framework

Status: 2026-11-06 discussion draft covering goals and framework only.

Plan21 has two equally central workflows: assemble a metal-complex `Molecule` from a named
registered definition, concrete core/ligand components, and effective parameters; or
recognize instances of a specified definition or selected catalog in a general `Molecule`
under graph, configuration, or combined conditions. Assembly may reuse registered
templates/defaults and automatic method parameter selection; recognition preserves source
facts. The directions share definitions and mappings; multiple realizations and matches
are possible, so they are complementary rather than strict inverse functions.

The proposed common model is an attributed relation graph with roles and boundary
interfaces, plus optional geometric constraints. An instance maps a definition to actual
`Molecule` members; atoms, bonds, and coordinates retain their existing fact owners,
and the current molecular fact layout is intended to remain in place.

The framework distinguishes membership boundaries, allowed attachment patterns/assembly
slots, existing boundary connections, and ports in an external superatom projection.
Discovery may use graph, configuration, or combined conditions. Parameter overlays apply
only to selected calculation configurations. Users can register and import definitions
independently. The long-term catalog goal covers known clusters in CSD and other selected
databases, distinguishing reusable definitions from individual observations. This revision
accesses no database and creates no complete catalog. Paddlewheels use N
bridging units with two attachment sites each, allowing non-carboxylate and mixed ligands
and one complete ligand spanning multiple units; N belongs to specialization/input constraints.

- [Goals and framework draft](metal_cluster_implementation_plan.md)
- [Plan20 architecture and type association](../20_fact_rule_architecture/fact_rule_architecture_implementation_plan.md)
- [Plan20 goal](../20_fact_rule_architecture/goal.md)
- [Chemical type public interface contract](../20_fact_rule_architecture/chemical_type_interface_contract.md)

The future entry remains `Molecule.metal_clusters -> list[MetalCluster]`. Documented
registration does not establish runtime implementation or a callable interface. This
revision does not prescribe implementation, algorithms, stage steps, or acceptance
criteria, and introduces no additional Plan20 implementation prerequisite.
