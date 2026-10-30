# Aromaticity and Kekule assignment

Stage5 retains the existing scientific model in private, read-only rules.
`aromaticity.py` evaluates an immutable `_BondingSnapshot`; `kekule.py` returns
one `_AromaticAssignment`. `operations/chemistry.py` selects the ligand
skeleton, captures source identities, invokes the rule and commits the whole
Atom/Bond assignment through `structure/_commit.py`. The old writable
`cheminfo/kekulize` implementation and its five structural protocols are
retired; there is no compatibility implementation or runtime selector.

`Molecule.determine_rings_aromatic()` retains the greedy Kekule workflow.
`Molecule.calc_atom_valence(assign_aromatic=...)` retains the three-state
aromatic-perception gate before hydrogen preparation. `Ring.is_aromatic` is
a source-checked read. Ring's aromatic setter, `determine_aromatic()` and
`kekulize()` are removed. `JointRing.check_kekulize()` uses the same local
numeric check through its source-checked operation.

## Scientific behavior retained

- An already entirely aromatic ring immediately returns True.
- Metal-containing rings return False; the molecule operation selects its
  metal-free ligand skeleton without hiding or modifying source edges.
- The separate coordinate and no-coordinate electron rules remain unchanged.
- The strict planarity ratio remains `maximum_deviation / length_scale < 0.03`.
- The existing `Ring.has_3d` tuple-versus-Atom comparison is characterized as
  always true, even for coincident coordinates. Bound snapshots preserve that
  branch selection. Repairing this predicate would change the scientific route
  and is outside this extraction.
- Rings are processed in the existing Relevant Cycle sequence. A later ring
  still overwrites shared Atom flags in the task-local proposal.
- Greedy Kekule assignment sets each accepted ring's edge orders to one and
  selects eligible double bonds in the existing edge order. It is not a global
  matching algorithm. Normal kind/order updates follow the former Bond setter;
  zero and dative categories remain distinct.
- The joint-ring numeric test and the one-shared-edge relation are retained.

The immutable proposal contains all Atom flags, selected Bond flags, orders,
kinds, coverage and member/source bindings. Flags outside every selected cycle
are cleared during a complete explicit assignment. This is a representation
normalization effect of the complete operation; it does not add a new
aromaticity model. Explicit/source hydrogen counts remain protected while
inferred counts are prepared against the accepted representation.

## Ring families and model encoding

`mol.rings`, `mol.ligand_rings` and `mol.rings_for_scope(...)` use Relevant
Cycles. `cycle_basis`/`basis_rings` use the deterministic native Paton basis;
`all_cycle_basis`/`all_basis_rings` enumerate complete simple-cycle bases.
The default Relevant Cycle limit is 10,000; an exceeded limit raises
`RelevantCycleLimitExceeded` without truncated output. An explicit
`max_cycles=None` requests an unbounded family.

The NumPy and PyG trained ring tensors continue to use the native Paton basis
on the ligand skeleton. Their feature order, ring count/order and shapes do
not switch to Relevant Cycles. Numeric atom features use one field-selective
projection; a strict consumer prepares the required H/aromatic groups before
encoding known values. No changed missing-value encoding is inferred for an
unchanged model.

SMARTS `R/r` semantics remain separately specified: `R<n>` counts selected
Relevant Cycles containing an atom; `r<n>` means membership in at least one
selected cycle of size n. Cubane therefore has six symmetric four-membered
public rings, while the trained Paton tensor retains five rings and its
characterized composite cycle.

## Known limitations and technical changes

The three existing strict xfails remain executable scientific records:

1. Reversing overlapping Relevant Cycles can change final Atom aromatic flags.
2. Clearing imported aromatic flags from azulene misses its global ten-electron
   aromatic system under the retained local five-/seven-ring rules.
3. Aromatic joint-ring grouping rejects Relevant Cycles sharing multiple edges.

Existing aromatic flags can still bypass scientific perception. Greedy
assignment still does not establish a globally valence-valid solution. These
limitations are not hidden by a toolkit fallback or a cycle-basis fallback.

Stage5 deliberately replaces two technical defects: scientific rules no
longer mutate carriers one ring at a time, and the molecule operation no
longer has a hide/process/recover exception window. A source change, failed
member mapping or failed commit leaves the previous complete facts intact.
Ring/JointRing membership is immutable and source-bound; changed relevant
facts reject current calculations with the existing RuntimeError, while held
member identities remain inspectable. Ring-derived caches are not retained
on Molecule.

The exact current technical/scientific runs and their coverage limits are
maintained under `plan/20_fact_rule_architecture/stages/stage05/`; historical
counts from the old package are not current validation evidence. Global
conjugation, fused-system perception, and constrained matching remain separate
unapproved scientific redesigns.
