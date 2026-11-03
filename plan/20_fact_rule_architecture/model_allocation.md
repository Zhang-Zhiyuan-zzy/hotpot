# Model allocation

Owner-approved execution reference, updated 2026-11-04. The active goal and
applicable agent instructions retain authority. Choose effort for each bounded
task; do not inherit max into every child.

| Work | Model / effort | Boundary |
| --- | --- | --- |
| Main-agent ownership and final decisions | GPT-6.1 Sol / max | Own route, integration, scientific judgment, acceptance and Git delivery. |
| Main implementation/design document or global interface/architecture change | GPT-6.1 Sol / max | Root performs it when already Sol/max; otherwise use the bounded Sol/max role. |
| Local business implementation under a settled design | GPT-6.1 Sol / medium initially; high/xhigh when needed | Business code always at least medium; escalate for a concrete obstacle. |
| Difficult bounded problem unresolved at lower effort | GPT-6.1 Sol / progressively through max | Keep the problem and actual attempts attributable; owner joins discussion if max cannot resolve it. Applicable agent instructions govern any separately authorized Astra escalation. |
| Narrow lookup/extraction | GPT-6.1 Sol / low | No unreviewed design or scientific decision. |
| Explicit mechanical task without business-code changes or reasoning | GPT-6.1 Terra or GPT-6 Luna / no reasoning when supported, otherwise lowest supported effort | Examples: exact file moves, applying an already reviewed documentation patch, executing a fixed command, packaging a fixed artifact list. Return actual outputs; Sol owns interpretation and acceptance. |

Use only a model actually exposed by the execution interface. The current
collaboration tool exposes `gpt-6-luna`; it exposes no Terra ID. This records the
owner's permitted choices, not a claim of account availability, equal quality,
price or measured token savings. Do not change main-session settings to apply a
mechanical-task exception. Unresolved choices, diagnosis, chemical meaning and
evidence judgment remain substantive work.
