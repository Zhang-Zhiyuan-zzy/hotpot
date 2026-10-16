# HARD-011: method-specific topology and state transport

Type: hard, non-toggleable backend contract.

## Theory and basis

GFN-FF needs a neighbour topology; GFN1/2-xTB consumes nuclei, coordinates,
charge and UHF. The authoritative Hotpot graph must survive both routes.

## Implementation

- Write GFN-FF input as one-record SDF/MOL with explicit bond endpoints and
  pass $Q$ separately.
- Write GFN1/2-xTB input as strict XYZ and pass $Q,u$ separately.
- Keep semantic bond kind, aromaticity, formal charges and target coordination
  topology in the in-memory/sidecar record.
- Verify atom count/order/map and commit only finite coordinates to the
  authoritative graph.
- Never reload native output as a replacement chemical graph.

## Acceptance

- Historical GFN-FF adjacency failures are covered by an SDF regression.
- Requested and native endpoint evidence is recorded where supported.
- Coordinate commit preserves graph, aromaticity, formal charge and state
  provenance.
- Isolated ions and unsupported native topology cases are explicit fixtures.
