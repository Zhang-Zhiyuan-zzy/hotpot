# GitHub Actions

Two entry workflows share one internal native-wheel workflow. Installed-package
checks run outside the checkout; test support exposes only `tests` and `plan`.
The ordinary result of a chemical calculation and its scientific meaning are
unchanged by CI scheduling.

| Definition | Trigger and purpose |
| --- | --- |
| [Inference compatibility](workflows/inference_compatibility.yml) | Pushes to `main` and numeric version branches, pull requests, manual dispatch and a weekly schedule. Selects lightweight documentation checks or installed CP311 business validation. Manual `full_matrix=true` and scheduled runs add CP310/312/313/314 compatibility checks. It is also callable by the release workflow. |
| [Build and optionally publish Python distributions](workflows/publish_pypi.yml) | `v*` tags or manual dispatch. Validates release metadata, calls the complete five-version validation and reuses its wheel/sdist artifacts. PyPI publishing requires manual `publish=true` and the configured `pypi` environment/OIDC publisher. |
| [Native wheels](workflows/native_wheels.yml) | Internal `workflow_call` only, with no independent push, PR, schedule or manual trigger. Owns the pinned manylinux build, both native/toolkit load-order smokes, wheel audit and wheel artifacts for both entry workflows. |

## Scope and frequency

- Ordinary source/configuration pushes and PRs build one CP311 wheel, run the
  installed core/docs/MCA/CBond/FF smokes and the complete maintained CP311
  default/model-artifact selection with coverage and Codecov upload.
- Feature branches run CI on PR or manual dispatch, avoiding duplicate push/PR
  runs for the same feature work. Numeric version branches such as `0.5.5.0`
  retain direct push validation.
- Changes confined to documentation outside runtime/test/tool/input owners use
  whitespace and workflow/scope contract checks. Runtime-owned Markdown,
  executable plan helpers, unknown paths, unavailable old history and initial
  branch pushes retain core checks. PR changes are compared with the merge base;
  code deleted or renamed into documentation still triggers business checks.
- Manual full validation, release validation and weekly validation retain the
  CPython 3.10–3.14 native builds and the four non-CP311 inference matrices.
  Lower ordinary frequency does not waive applicable release/final-plan gates.
- Weekly full validation is scheduled for Monday 02:30 UTC. GitHub schedules run
  from the default branch. The current default branch is `main`; the schedule
  in this version branch takes effect only after that definition reaches `main`.
  Manual dispatch can select this branch because the entry file already exists
  on `main`.
- Superseded ordinary push/PR runs on the same ref are canceled. Full and daily
  validation have separate concurrency groups. Manual, scheduled and release
  validations have unique run groups, so they cannot replace each other while
  pending. Publishing is never automatically canceled by the ordinary-CI policy.

The fixed scientific/external/high-compute test switches remain explicit. A
business failure fails its job. A skipped unrelated tier is not a passed tier.
Release wheel and source-distribution audits live in `tests/test_packaging/`,
not as duplicate inline YAML programs. Tag pushes always build without publishing;
GitHub does not apply path filters to tag pushes.

## Historical registrations

The following YAML files are absent from both this branch and the inspected
`main` tree. Existing runs remain historical evidence. GitHub can retain their
registry entries even though the current branch has no corresponding definition.

| Registration | ID | Disposition |
| --- | --- | --- |
| Diagnose installed inference smoke | 397804761 | Superseded one-time fixed-wheel MCA/CBond diagnostic; disable server registration. |
| Export validated PyPI artifacts | 397728709 | Superseded one-time artifact transfer; disable server registration. |
| Tests and coverage | 64041508 | Old Python 3.9 Conda/unittest/Codecov route; disable server registration. |
| Dependency Graph | Platform-generated | Keep dependency-graph updates; no repository YAML. |

At cleanup entry on 2026-11-06, all three legacy registrations still reported
`active`. This session has working SSH Git push access but no GitHub API
credential, so their server-side disablement is **pending**. It requires a
GitHub login with repository Actions write permission or the repository Actions
web UI. No token, historical upload credential or external export path is added
to this branch. Removing a registry entry is separate from changing repository
files; historical run records are preserved.
