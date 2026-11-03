# Stage8C search decisions

## C1 — Canonical BondKind contract prerequisite

Root selects an early extraction of the existing seven-value `BondKind` strEnum
from core.py to `structure/bond_kind.py`. Lower chemical rules must not import
a concrete carrier owner to name a bond category. This is the already planned
structure-owned type, with one definition and no new members, class, scientific
rule or representation. Core and ordinary top-level exports refer to that one
object; existing coercion and kind/order mapping implementations stay unchanged.
Four SMARTS/IR imports now name the canonical contract, retiring their exact
legacy-core debt. This does not move the concrete Bond or other carriers early.

New pickles identify the actual physical class owner. The captured current
core-path enum pickles must still load through core's existing export during
this prerequisite; Stage9 retains the separately owned old-core deletion and
historical-asset decision. Add no compatibility module, alias class or duplicate
implementation. `hotpot.BondKind` remains the ordinary public entry point and
is identical to the object used by Molecule/Bond construction and copy.

Validation uses the actual pre-edit CP311 enum/pickle capture at c13970d, a
contract-leaf import without the execution tree, cold imports/pickle identity,
real bond construction/copy, existing SMARTS/native matching and structural
bonding/transport tests. An exact class-AST comparison checks this mechanical
type move; chemical meaning is judged by the actual behavior tests. Native
source/binary/schema/feature encoding does not change or rebuild for C1.

The full parser/recursive-query/chemical-predicate/RI/packing/public-operation
separation remains under read-only inventory. The current molecule-keyed
recursive SMARTS cache still requires its owning Stage8C treatment. No full
Stage8C, installed successor, native-package or global acceptance follows
from this prerequisite. Root will record the connected split before authoring.
