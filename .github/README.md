# GitHub Actions

The `0.5.5.0` branch contains two workflow definitions. Actions runs are bound to
commits and retain the branch name used when they started; renaming a branch does
not change historical run names or conclusions.

| Workflow | File | Trigger | Work performed |
| --- | --- | --- | --- |
| Inference compatibility | [inference_compatibility.yml](workflows/inference_compatibility.yml) | Push and pull request with matching source, test, packaging or workflow paths; manual dispatch. | Build Linux x86-64 native wheels for CPython3.10–3.14; audit wheel contents; check installed CP311 core/docs, MCA, CBond and FF; run maintained inference/graph-search tests on3.10/3.12/3.13/3.14; run CP311 default tests with coverage and upload Codecov reports. |
| Build and optionally publish Python distributions | [publish_pypi.yml](workflows/publish_pypi.yml) | `v*` tags; manual dispatch. | Validate tag/version agreement; build and smoke-test CPython3.10–3.14 manylinux x86-64 wheels and an sdist; upload distribution artifacts. Publish to PyPI only on manual dispatch with `publish=true`, through the configured `pypi` environment and OIDC Trusted Publisher. Tag runs build without publishing. |

Tests run against installed wheels outside the checkout. Their isolated support
directory exposes repository `tests` and `plan` helpers without adding the
checkout root to `PYTHONPATH`. Scientific tests, model tests and optional/external
execution retain their explicit switches. A test failure fails the job; collection
errors and missing targets do not establish business results.

## Additional entries in the GitHub workflow registry

GitHub's Actions catalogue also retains the following entries. They are not
workflow definitions in the current `0.5.5.0` or `main` trees. Their presence is
not evidence that they run when this version branch is pushed.

| Registry name | Historical function | Scope |
| --- | --- | --- |
| Diagnose installed inference smoke | Download a fixed earlier CP311 wheel and diagnose installed MCA/CBond behavior. | Historical `fix/github-workflows` push diagnostic; successful diagnostic completion is not an inference correctness verdict. |
| Export validated PyPI artifacts | Copy distribution artifacts from a fixed earlier run to a transfer endpoint. | Historical one-time export branch; it does not itself publish to PyPI. |
| Tests and coverage | Create a Python3.9 Conda environment, run unittest coverage and upload through Codecov. | Older branches, including `old_version_discard`; superseded by current installed-package CI. |
| Dependency Graph | GitHub-managed dependency graph update. | Platform-generated workflow, not a YAML file maintained here. |

The current workflows use read-only repository access. The coverage upload and
manual PyPI publishing jobs additionally request `id-token: write` for OIDC.
No external export workflow or historical upload credential is incorporated into
this branch.
