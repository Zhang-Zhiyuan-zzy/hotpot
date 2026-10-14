# Quality assessment: CSD electronic-state text

Assessment date: 2026-10-12.

## Decision

The source can be admitted to Hotpot DS as candidate evidence, but the complete collection is
not an oxidation- or spin-state truth benchmark. Database integrity and extraction
reproducibility are strong; scientific support is usually limited to an explicit CSD text
statement rather than an independent primary-literature adjudication.

- The full source is a collection-level dataset.
- Active-core OX-T1 mappings may become oxidation candidates with explicit-reported evidence.
- Spin text remains an unresolved claim until a reviewer verifies subject coreference, metal
  site, phase, temperature, pressure, illumination, and metastability conditions.
- Incomplete cores and ambiguous or conflicting mappings remain unresolved evidence.
- No automatic subset qualifies as a curated, validated, or reference benchmark.

Quality and evidence levels are derived independently by the persisted Gates;
they are not assigned by the source. A structurally consistent CSD entry does
not make its electronic-state phrase a validated label.

## Known risks

- Preparation prose may describe a reagent, comparison compound, or earlier literature entry.
- Fixed categories may represent a population, isomer, partial transition, or photoinduced
  metastable state.
- Records without a matching phrase are missing labels, not negative examples.
- Refcode families can leak near-identical structures and source text across random splits.
- Chemical names or CIF annotations can leak the target directly into model inputs.
- A one-site mapping proves relational uniqueness, not chemical correctness.

The source audit identifies explicit conflicts, including ACAHUJ and ZEKDOK, which must remain
contested until primary sources and editorial history are reviewed. A chemically implausible
text-derived value such as the SIQKEJ Hg(+9) extraction must be retained as raw evidence but
rejected as an active observation.

## Requirements for higher quality

Promotion requires documented domain review of every promoted record, primary-source support,
condition completeness, conflict adjudication, family-aware splitting, and an uncertainty
policy. Validated levels additionally require independent review and leakage auditing;
reference status requires external audit and multi-source corroboration.

The detailed machine audit is stored locally under
`data/sources/csd-electronic-state/records/` and is excluded from Git.
