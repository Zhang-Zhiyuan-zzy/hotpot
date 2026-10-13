# Metal oxidation-state data

This collection targets metal-centre oxidation state only. Spin, multiplicity, and molecular
formal charge remain separate properties even when they refer to the same structure.

| Source subset | Labels | Structures | Evidence character |
| --- | ---: | ---: | --- |
| SSE17 | 17 | 27 XYZ | Explicit literature assignment with composition and charge context |
| SCO-95 Fe(II) | 95 | 95 MOL2 | Chemist-curated experimental crystal structures |
| TM oxidation | 2,063 | 2,063 XYZ | Algorithmic cell2mol assignments from CSD-derived graphs |

The TM source is suitable as an algorithmic candidate set, not as a gold reference. The
SSE17 and SCO-95 sources have stronger attribution but still require the quality and evidence
levels recorded by Hotpot; source names such as “gold” are not automatic quality upgrades.

Structure relations use shared IDs in `data/structures/`. Reuse of coordinates never copies
oxidation labels into the spin dataset. Source files are local under
`data/sources/oxidation/`; attribution records are tracked in `docs/attributions/`.
