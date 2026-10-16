# OPTIONAL-003: neutral L-type donor prior

Type: optional empirical rule. Planned screening default: **enabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `COORDINATION_COUNTING` |
| Requires rules | none |
| Consumes | fixed protonation/coordination-boundary facts, compatible charge/Lewis evidence, requested convention |
| Produces | convention-specific `CoordinationFragmentAssignment` candidates |
| Required providers/data | versioned L-donor motif/protonation registry |
| Review state | `DRAFT` |
| Owner approval | pending individual review |
| Primary basis | Green CBC, DOI `10.1016/0022-328X(95)00508-N` |
| Applicability artifact | L-donor motif/protonation registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

Under Green CBC/donor classifications, common L-type donors such as water,
ammonia, ordinary amines, pyridines, ethers, phosphines and CO are neutral
ligands in their ordinary fixed-protonation forms.

## Implementation

- Match curated semantic motifs with explicit protonation and valence facts.
- Operate on the immutable coordination-separated ligand-component view and do
  not mutate or delete source bonds.
- Action emits a separate typed result for each explicitly requested
  convention: CBC may propose class `L`, donor-return may propose a neutral
  returned fragment, and ionic counting may propose a zero bookkeeping charge
  only when its own template supports that result. A numeric zero is never
  reused across convention types.
- No result overwrites intact-graph formal charges or system $Q$.
- Ionic bookkeeping candidates are inputs to the joint composer; this rule does
  not independently select system $Q$.
- Dative direction, hapticity and unusual redox/protonation states are explicit
  applicability checks.
- Motifs, sources and counterexamples are versioned.

## Acceptance

- Ordinary H2O, NH3, amine, pyridine, ether, phosphine and CO examples match.
- Protonated, deprotonated, radical and non-innocent variants abstain or retain
  alternatives.
- CBC/donor-return output cannot populate an ionic-charge field, even when all
  displayed numerical charges would be zero.
- Equivalent graph representations yield the same motif result.
