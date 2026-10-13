# Shared structure objects

Oxidation and spin records reuse a content-addressed coordinate store without merging their
datasets or targets.

The current local inventory contains 2,280 unique coordinate objects. Oxidation references
2,185 of them; spin references all 2,280. The two collections share 2,185 coordinate-identical
objects, while 95 high-temperature SCO-95 structures occur only in spin data.

Coordinate identity is based on ordered element symbols and three-dimensional coordinates
rounded to eight decimal places. It intentionally ignores task-bearing XYZ comments. This
supports byte reuse, but it does not establish chemical identity when protonation, bond order,
periodicity, or disorder differ.

The local relation files and objects live in `data/structures/`:

- `structures.csv` maps structure IDs to physical objects and hashes;
- `record_structure_links.csv` maps dataset records and structure roles to shared objects;
- `oxidation_spin_overlap.csv` records cross-dataset coordinate matches;
- `objects/` contains the content-addressed coordinate payload.

These payloads are excluded from Git. New normalized stores should express the same links in
the current dataset schema rather than interpreting “benchmark” as a storage namespace.
