# Stage 14: geometry reconstruction

Status: deferred post-processing work.

## Scope

This stage will reconstruct a chemically interpretable molecular graph from
coordinates produced by topology-free electronic-structure backends such as
GFN1/2-xTB and ORCA.

The future work must distinguish:

- preserved atom identity and composition;
- covalent bond perception;
- metal--ligand contacts;
- bond-order and aromaticity hypotheses;
- alternative or uncertain reconstructions; and
- reconstruction confidence and provenance.

Geometry reconstruction must return a proposed graph or candidate set. It must
not silently overwrite the input molecular graph.

## Boundary

This is a downstream interpretation problem. It is not part of the current
work on preparing a metal--ligand pair for xTB/ORCA submission or determining
whether the resulting electronic energy is scientifically meaningful.
