# Automatic scientific-state core validation

Date: 2026-11-05. Core handoff status: ready for root integration/activation review; no production activation, native/model execution, install, commit or push was performed by this partition. Working checkout base at final core handoff: `8715c9b3fdfb960a4068e671c04956617c9d43ee`. Root/other workers own periodic data, CBond/OB/stale-frame projection, backend active spin, stream codec, payload identity, final profile activation and actual/delivery work. This record qualifies the core source evidence only.

## Commands and outcomes

Interpreter: `/tmp/hotpot-plan20-env/bin/python`, CPython3.11, resolving to `/home/zhangzhiyuan/usr/conda3/envs/hp-usage/bin/python`; imports current checkout. Exact start/end, argv and return code are retained in the adjacent command records.

1. Initial focused source/rule run: `-m pytest -q tests/test_cheminfo/calculator/test_automatic_scientific_state.py tests/test_cheminfo/calculator/test_state_profile_manifest.py tests/test_cheminfo/calculator/test_neutral_l_donor_rule.py tests/test_cheminfo/calculator/test_anionic_x_ligand_rule.py tests/test_cheminfo/calculator/test_scientific_state_facts.py` —64passed in1.40s, one existing missing-element-H warning. Two initial fixture failures were corrected: the partial-charge fixture accidentally satisfied the stronger neutral-L template, and Open Babel did not supply the radical field assumed by a radical test; the latter now uses an actual RDKit radical source. Expectations in the frozen holdout cohort were not changed.
2. Latest pre-stress check including monoatomic rule: `-m pytest -q tests/test_cheminfo/calculator/test_automatic_scientific_state.py tests/test_plugin/test_xtb/test_automatic_state_resolution.py tests/test_cheminfo/calculator/test_monoatomic_ion_rule.py` —42passed in1.10s.
3. Broad core command: `-m pytest -q tests/test_cheminfo/calculator tests/test_plugin/test_xtb/test_automatic_state_resolution.py` —5 failed, 675 passed, 1 warning in 6.94s. [Exact command](core_broad_command.json), [original full log](core_broad_pytest.log). The five failures are two old no-init L/X constructor fingerprints, two old profile resource digest assertions and the old10-rule count. The new additive constructor digest is `1ea139550de51c42d87ded0d20499456400367fbccf6390c51f4372113123971` versus old `33e02ef8c75f6322a5531f52672728217cfbec6bf87ca0f3925165566ac5556c`. Root owns qualified updates at activation; this failing run is retained rather than relabeled a full pass.
4. Final connected command: `-m pytest -q tests/test_cheminfo/calculator/test_automatic_scientific_state.py tests/test_plugin/test_xtb/test_automatic_state_resolution.py tests/test_plugin/test_xtb/test_automatic_stream.py tests/test_pipeline/test_builtin_stages.py tests/test_cli/test_xtb_cli.py` —143 passed in 6.91s. [Exact command](core_final_focused_command.json), [full log](core_final_focused_pytest.log). This follows the source-absence/CLI corrections and root's bound correction. Stubbed automatic-state tests launch no xTB executable; existing CLI cases use their declared fake executable, not scientific energy evidence.

The first connected CLI replay run retained six failures (86passed): source absence `False/None/None` became `False/0/None` through current-factual installation, changing an over-specific source digest. Both raw and source-evidence digests now normalize absent declaration/atom slots' unused values/sources toNone. A regression changes absent backing storage while preserving semantic absence and requires both identities to stay equal. Actual supplied presence/value/source still changes identity. No installer/schema weakening or relaxed scientific identity comparison was used. Root's independent original failure/diagnosis remains under `/tmp/hotpot_plan20/short/automatic-stream-20261105`.

CLI replay now resolves restored source authority first, then uses a recorded total charge to reconstruct prior explicit CLI input only when exact scientific identity matches. Six GFN-FF/GFN2-to-GFN2 replay cases cover automatic Q, explicit Q2 and explicit Q3, with exactly one public workflow invocation. A foreign scientific identity starts no workflow. A GFN2 UHF2 record passed to GFN-FF produces a charge-only request/projection while retaining previous metadata. The pipeline fixture now obtains its reusable bundle from the actual shared source-bound resolver, preserving its single public workflow call and identity/object assertions. Changed source Q/spin/formal-charge provenance/radicals and changed profile artifact have separate invalidation tests; a bundle without source binding is resolved again.

Root accepted the independent last-rank frontier fix after red1failed/3passed and green69passed in `/tmp/hotpot_plan20/short/automatic-payload-identity-20261105/spin-bound-summary.json`. This partition did not edit `defaults.py` after ownership transfer.

## Frozen holdouts and bounded stress

Protocol SHA256 `e4de3e64d6eafcce3c2a20f37cad305bb3f0f71f9f98fe5ad74158018ceeff18`. All12 predeclared positives selected the expected Q and physical2S and satisfied represented-electron parity/range under isolated test-only active profile. Frozen protocol bytes/status/results were left unchanged; this separate record contains execution evidence. Production was still `planned` at core handoff. The planned-profile negative test now uses an explicit isolated planned fixture so future owner activation cannot delete that boundary.

| Frozen case | Expected / actual Q | Expected / actual physical2S | GFN2 activeUHF |
|---|---:|---:|---:|
| neutral_water | 3 / 3 | 6 / 6 | 0 |
| neutral_ether | 3 / 3 | 6 / 6 | 0 |
| remote_cation | 4 / 4 | 6 / 6 | 0 |
| remote_dication | 5 / 5 | 6 / 6 | 0 |
| anionic_alkoxide | 2 / 2 | 6 / 6 | 0 |
| dianionic_dicarboxylate | 1 / 1 | 6 / 6 | 0 |
| tetraanionic_edta | -1 / -1 | 6 / 6 | 0 |
| trivalent_lanthanum | 3 / 3 | 0 / 0 | 0 |
| trivalent_gadolinium | 3 / 3 | 7 / 7 | 0 |
| generic_zinc | 2 / 2 | 0 / 0 | 0 |
| generic_copper | 2 / 2 | 1 / 1 | 1 |
| radical_ligand | 3 / 3 | 5 / 5 | 1 |

The supplied-spectator fixture `[Na+].CC[O-].O.[Cl-]` with Eu coordination retains four distinct component counts `+1,-1,0,-1` exactly once, totalQ2, physical2S6; source-covered ions do not retain OPTIONAL-002 template alternatives. Explicit EuII Q2 remains physical2S7. Explicit legal molecular spin wins; invalid parity and contradictory complete `[Eu+3]` source/Q2 fail through existing hard findings. Source absence, partial charged completion, radicals, unknown H, unsupported Am and ordinary strict default are covered.

Two-/three-Eu stress fixtures obey64 values/factor,4096 expanded paths,1024 frontier bounds. Both retained a64-value truncated molecular factor, zero expanded certified paths and one frontier entry; both return `resource_limit_no_certified_decision` / `no_automatic_state`, never a canonical guess. A materialization-budget1 generic Cu fixture retains explicit omitted-domain evidence and is ineligible under the changed budget. Three real decision snapshots (charged ligand, radical ligand, generic Zn) cross JSON and restore exact candidates, constraints, traces, resolution, decision and canonical serialized bytes. Existing broader tests cover composition/search, explicit-state priority, local-spin coupling and serializer contracts.

## Source cost observations and limits

The retained [measurement](core_measurement.json) and [harness](measure_core.py) identify source file SHA256 values, profile digests, interpreter and exact command. Three warmup passes precede ten measured passes over the12 fixed positives,120 samples total. The backend context/projection is stubbed at xTB6.7.1 capability level; no executable probe, FF/native calculation, install or energy evaluation is timed. The final pytest ran concurrently for part of this shared-host observation.

| Phase | Median wall time per positive (ms) |
|---|---:|
| Facts capture | 2.657 |
| Source binding | 0.073 |
| Manifest | 0.175 |
| Common resolve | 13.165 |
| Pure backend projection | 0.074 |
| Total | 16.992 |

Observed total range6.657–60.103ms. The two-Eu single stress observation took359.07ms, three-Eu2950.68ms, both abstaining. These are explicit current coverage/cost limits. There is no before/after or dedicated-resource comparator, throughput claim or speedup claim. Root's actual187 and installed/delivery resource observations remain separate.

## Remaining owning gates

Root must disposition/activate profile1.1.0, update only qualified current-profile/constructor frozen fixtures, then run integrated source and representative installed-CP311/default delivery checks and the actual original187 route. Existing old broad failures are not waived by the143-case connected pass. No other interpreter, wheel/native artifact or scientific energy result is established here. Full arbitrary saved-bundle/session replay remains outside scope.

Carried binding includes the profile artifact digest but no independent aggregate of every periodic/prior data file. Root accepted the2026-11-05 maintenance contract that any periodic/prior/default data change used by this profile MUST accompany a profile artifact/version revision; data are fixed within an invocation/install. The changed-profile regression proves carried state rebinds under that revision. No hot-edit API or broad aggregate hashing is introduced. Data-only manual edits without the required profile revision violate this contract. Multi-center truncation and the intra-component radical highest-spin assumption remain documented scientific/coverage limits.

## Artifact identities

- Broad original log SHA256 `95c349d8986f1c54939862420ef201f9c5afd02653cfe50ab78732a10e9ecd60`.
- Final connected log SHA256 `b52f30d5baea2e9498b5cab2dfae05a7796138ec73c81c79c57500bd2c86ff4c`.
- Core measurement SHA256 `3013b0d0a7dc2abff194ae98a5fc228ca5f27694bc7414d05392d0463aecc4d4`.
- Frozen production planned profile at measured checkpoint SHA256 `afc2615f74ccf70c088b38e62785480f448a2f978d31e1c676d68bc3df6544be`; root's eventual activation necessarily has a new digest.


## Root post-handoff integration, 2026-11-05

Root reviewed the12/12 frozen holdouts, explicit/source conflicts, bounded stress,
serializer and independently reproduced frontier-bound correction, and activated
profile1.1.0 in [the separate decision](activation_decision.json). Inactive gates
remain explicit planned-profile fixtures; current-resource and only the two
accepted source-evidence constructor fingerprints were qualified.

The first frozen182-file affected run used hp-usage CP311 with the existing
Gemmi0.7.5 dependency site:4244 tests,4196passed,7failed,41skipped in136.23s.
Every production Python/resource/native binding stayed unchanged. Five failures
were the historical pre-activation benchmark's intentional planned-only
precondition; two were old N–Cu DATIVE cutoff fixtures after the accepted mapped
metal–nonmetal projection. Benchmark-local explicit planned fixtures preserve
that historical test, and N–C unmapped DATIVE fixtures preserve the cutoff and
all119 original conversion assertions. Both complete conversion files and the
entire scientific-state benchmark test directory then passed84/84 in1.09s,
against exactly the original production bytes. Original failures are retained.
This is composed affected evidence, not a relabeled all-green initial run.

Exact argv/environment/source hashes, JUnit and original logs are under
`/tmp/hotpot_plan20/short/automatic-state-integration-20261105/attempt-01/`
and `fixture-correction-01/`. Default-suite, installed-package and original187
actual native gates remain pending; this source checkpoint closes none of them.
