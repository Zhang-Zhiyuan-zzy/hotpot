# Stage4 technical validation checkpoint

The frozen implementation passes its technical, supported-Python and package
checks. Stage4 remains in progress for its measured performance regressions;
this report supplies no final scientific/global efficiency approval.
[Execution evidence](evidence.json) binds exact argv, cwd, environment,
interpreter, production/native/tests/runner hashes, artifacts and elapsed time.

## Actual results

| Check | Result |
|---|---|
| Complete default suite |4571 passed,123 skipped,3 xfailed,49 subtests;10 warnings |
| Python3.10–3.14 source, each |2273 passed,42 skipped |
| Python3.10–3.14 debug allocator, each |162 passed |
| Python3.10–3.14 installed runtime, each |2143 passed,42 skipped |
| Five installed core smokes and wheel audits |passed |
| Standalone graph/geometry C++ API builds |7 passed; seven executable identities retained |
| Actual source distribution content |542 current source/stub/license/documentation members match; no prebuilt `.so` |

The affected selection covers all Stage3/4 existing/new contracts, actual
rewired consumers and the complete DS contract suite after the default gate
exposed its source-content defect. Architecture and four packaging source scans
stay source-only; they are not relabeled installed tests. Source/installed42
skips are41 requiring`--run-external` and1 requiring`--run-scientific-boundary`;
those tiers were not enabled by these commands.
The default's123 skips include optional/model/scientific/high-compute/external
tiers, not successful execution of those tiers. JUnit passed-node totals also
include49 subtests; the primary default count above uses pytest's actual summary.

## Package and native identities

Each current wheel assembles and audits current Python/package bytes. All
loaded Hotpot/MCA module origins resolve to the intended checkout or isolated
installation. Two frozen benchmark/operator scripts and build metadata are
copied as hashed test support, not runtime implementations. Superseded eleven
Python modules are absent from wheels and sdist.

Existing native source/recipe/root-binary identities match Stage3. Reused wheel
members bind independently to actual prior installed wheel binaries; four OB
root/installed builds have different bytes and remain distinct identities.
This verifies current Python-package integration with unchanged native
components. Stage10's clean all-native rebuild, ABI and SDK gates remain open.
The seven standalone C++ tests retain their unchanged current source/test
closure after later Python-only fixes, so they were not redundantly rerun.

## Preserved failures and fixes

The original allocator test failed on3.13/3.14 because dict buckets use MEM,
where3.11 uses OBJ. The corrected helper wraps both real allocator domains,
retains a rejecting negative control and verifies original exception/owner/
fact restoration. Other Python allocations still occur; this is targeted
bucket-rejection evidence, not arbitrary-OOM immunity.

All preflight/source/installed failures remain in their original attempt
folders. Installed retries first supplied the missing Plan17 benchmark runner,
then the DS operator script and its `pyproject.toml` test input. The final
metadata-only support retry changed no production, tests or wheel bytes and
reran the entire installed selection plus smoke on all five versions.

The original DS source and test also missed same-size byte changes when real
filesystem timestamps collided:98 of100 direct rewrites retained the same
stat signature. A deterministic unchanged-stat regression failed before repair.
The minimal production fix checks current source content once at cache release,
while preserving shared snapshot/parse work, error types and cleanup. Full DS,
default and five-version source/installed checks cover it. The original common
technical benchmark node maps to its`[False]` case; the new collision`[True]`
case is additional work. That mapping is recorded before final common-suite
measurement and changes no frozen membership or weights.
