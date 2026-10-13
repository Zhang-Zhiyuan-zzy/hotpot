# Metal-complex spin-state data

This collection targets spin multiplicity, categorical spin state, transition temperature,
or spin-state energy differences where the source supports them. Oxidation state and formal
charge remain independent properties.

| Source subset | Label scope | Structures | Evidence character |
| --- | ---: | ---: | --- |
| SSE17 | 17 complexes | 27 XYZ | Literature spin-crossover thermochemistry or spectroscopy |
| SCO-95 | 95 low/high-temperature pairs | 190 MOL2 | Author-distributed experimental structures |
| TM-GSspin | 2,063 complexes | 2,063 XYZ | Multi-functional DFT ground-state screening |

The TM-GSspin labels are computed candidates. SCO-95 temperature pairs must remain linked;
their two structures are not independent random-split samples. A categorical phrase such as
`high_spin` must not be converted to numerical `S` or multiplicity without explicit evidence.

Source files are local under `data/sources/spin/`. Shared coordinate objects reside in
`data/structures/`; attribution records are tracked in `docs/attributions/`.
