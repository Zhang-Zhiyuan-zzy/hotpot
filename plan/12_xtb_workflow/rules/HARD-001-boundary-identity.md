# HARD-001: computational-boundary identity

Type: hard, non-toggleable.

## Theory and basis

Net charge and electronic state refer to a defined computational boundary.
Changing atom membership, isotope facts, protonation, tautomer, connectivity
semantics or component membership changes the molecular-facts identity.
Different charge, oxidation or spin candidates may share those facts but have
different scientific-state identities. Backend/ECP identity is bound later by
a projection and is not folded into either scientific identity.

## Implementation

- Build an immutable `MolecularFactsIdentity` from atom IDs/nuclear identities,
  hydrogen inventory, semantic graph, component boundary and relevant source
  facts.
- Keep coordinates outside that chemical identity. Bind geometry-dependent
  evidence to the separate `MolecularFacts.geometry_digest`, so a coordinate
  update invalidates that evidence without invalidating topology-only state.
- Bind every scientific candidate to the molecular-facts digest and derive a
  separate `ScientificStateIdentity` from that digest plus candidate values and
  conventions.
- Bind backend projections to both the scientific-state identity and a separate
  `BackendIdentity`.
- Reject composition of results with different identities. Do not repair the
  mismatch by remapping unless an explicit mapping operation produces a new
  attributable result.

## Acceptance

- Atom reordering with a verified map preserves chemistry and remaps results.
- Coordinate-only changes preserve `MolecularFactsIdentity` while changing the
  geometry digest.
- Adding/removing H, ions or atoms invalidates prior state results.
- Changing a declared isotope mass number changes molecular facts identity.
- Two legal redox/spin candidates share molecular facts but have distinct
  scientific-state identities.
- A backend change invalidates only the projection, not the scientific result.
- Mismatched identities cannot reach a backend process call.
