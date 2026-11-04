# Current standalone SDK qualification

Root accepts this bounded current-source Linux x86_64 manylinux2.28 SDK
qualification. D007, the five-version wheel cycle and global gates remain open.
The exact pinned image/compiler/tools and offline six-archive closure build a
Python-independent SDK with SDK=ON/Python=OFF and at most eight compile workers.
All46selected current inputs remain unchanged at root review after Stage11C.
The changed H molecule_builder.cpp is included in the newly compiled core.

Seven unique existing C11/C++17 header and installed SDK consumer cases pass,
zero failures/errors/skips. Actual ABI0/ABI1 find_package/link/run consumers and
supplemental assertion-enabled consumers pass. All31direct commands pass,
including C smoke, header-only C++ facade, symbols and runtime linkage. Exactly15
C functions are exported, plus the ELF version symbol; no private RDKit/Python
function, RPATH/RUNPATH or private header is exposed. DT_NEEDED contains only the
declared system C/C++ libraries and loader, resolved in the selected image.

Nine predeclared H projection fixtures match expectations: five finite-coordinate
results on the original graph atom count and four named local UNSUPPORTED
results for inconsistent/current-mask/unregistered inputs. Coordinates are not
direct inspection of hidden H or proof of a unique chemical structure. The
versioned assumptions and diagnostics remain in the raw report and fixtures.

Original failures are retained: missing Ninja prevented the first build; the
second completed162compile/link actions but its isolated capsule omitted the
LICENSE required by the existing install rule. Binding the unchanged repository
LICENSE allowed an install-only retry. The original build-driver exit remains1;
no clean full-driver rerun is claimed. Original release consumers are preserved;
the assertion-enabled variants supplement their Release/NDEBUG limitations.
An optional per-runtime-library hashing attempt was interrupted before execution
when the same selected image was no longer locally present. Completed image-
bound linkage checks remain evidence; those individual hashes were not obtained.

Root ran `python -B /tmp/hotpot_plan20/long/stage12-current-sdk-20261104/root-review/verify_and_archive.py`.
It independently checks46current source bindings,25compiled-artifact bindings,
19installed SDK entries, raw JUnit,31commands, symbols, H outcomes and actual
library identity. The installed/build libraries match at6164136bytes,
SHA2565e05b3e81fe9f366606cc0bd356e58b6053109d7eccf89d52b5876c87cf34d10.
The archive includes the usable SDK with symlinks, original attempts and finite
compiled/input/tool/runtime evidence:154payload entries/10068609file bytes.

[Evidence graph](evidence_graph.json) selects exact commands, outputs and the
archived SDK; [checkpoint](checkpoint.json) binds the source, artifact and archive.
This run does not qualify Python wheels, other Python versions/platforms,
public upload, models/engines/GPU, performance, final science/Plan12 acceptance,
B_exp or Stage13. Reuse this unchanged SDK closure during final delivery;
source/configuration changes require an affected rebind, not a gratuitous rebuild.
