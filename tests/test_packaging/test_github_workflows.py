"""Workflow entry points share artifacts and preserve validation boundaries."""

from pathlib import Path
import re
from types import SimpleNamespace


ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / '.github' / 'workflows'


def _workflow(name):
    return (WORKFLOWS / name).read_text(encoding='utf-8')


def _jobs(workflow):
    sections = re.split(r'(?m)^  ([\w-]+):\n', workflow.split('\njobs:\n', 1)[1])
    return dict(zip(sections[1::2], sections[2::2]))


def _expression(job, key):
    match = re.search(rf'(?m)^\s+{key}:\s*\$\{{\{{(.+?)\}}\}}\s*$', job)
    return match.group(1).strip()


def _evaluate(expression, **context):
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    return eval(expression, {'__builtins__': {}}, context)


def test_native_builder_is_internal_and_builds_the_exact_wheels_consumers_test():
    workflow = _workflow('native_wheels.yml')
    events = workflow.split('\non:\n', 1)[1].split('\npermissions:', 1)[0]
    assert 'workflow_call:' in events
    assert not re.search(r'^  (push|pull_request|workflow_dispatch|schedule):', events, re.M)
    for tag in ('cp310', 'cp311', 'cp312', 'cp313', 'cp314'):
        assert tag in events
    native = _jobs(workflow)['native-wheel']
    assert 'fromJSON(inputs.python-tags)' in native
    assert 'pypa/cibuildwheel@v4.3.0' in native
    assert 'HOTPOT_RDWRAPPERS_BUILD_PROFILE=manylinux_2_28_x86_64' in native
    assert 'quay.io/pypa/manylinux_2_28_x86_64@sha256:39df0042d5cc900b085aa25a0659368b42a0006c54c474299b785b44c1b4ff82' in native
    assert 'docker; disable_host_mount: true' in native
    assert '--exclude libopenbabel.so.7' in native
    assert '--exclude libopenbabel.so.8' in native
    assert 'native_wheel_smoke.py hotpot-first' in native
    assert '&& python {project}/tests/test_packaging/native_wheel_smoke.py rdkit-first' in native
    assert 'audit_rdwrapper_wheel.py wheelhouse/*.whl' in native
    assert 'name: wheel-${{ matrix.python-tag }}' in native
    assert 'if-no-files-found: error' in native
    entrypoints = [_workflow(name) for name in ('inference_compatibility.yml', 'publish_pypi.yml')]
    assert all('pypa/cibuildwheel' not in text for text in entrypoints)
    assert all('python -m build --wheel' not in text for text in entrypoints)


def test_inference_routes_core_docs_and_full_matrix_through_shared_artifacts():
    workflow = _workflow('inference_compatibility.yml')
    events = workflow.split('\non:\n', 1)[1].split('\npermissions:', 1)[0]
    assert 'pull_request:' in events and 'push:' in events
    assert 'branches:' in events and '- main' in events and "'[0-9]*'" in events
    assert '"**"' not in events
    assert 'paths:' not in events and 'paths-ignore:' not in events
    assert 'workflow_call:' in events and 'workflow_dispatch:' in events
    assert events.count('full_matrix:') == 2
    assert events.count('default: true') == 2
    assert events.count('default: false') == 1
    assert 'export_sdist:' in events
    assert 'cron: "30 2 * * 1"' in events
    jobs = _jobs(workflow)
    scope = jobs['scope']
    assert 'fetch-depth: 0' in scope
    assert 'python .github/scripts/ci_scope.py' in scope
    assert 'steps.scope.outputs.core' in scope
    assert 'steps.scope.outputs.python_tags' in scope
    expression = _expression(scope, 'HOTPOT_CI_FULL_MATRIX')
    for event, requested, expected in (
        ('push', False, False), ('pull_request', False, False),
        ('schedule', False, True), ('workflow_dispatch', True, True),
        ('workflow_dispatch', False, False),
    ):
        assert _evaluate(expression, github=SimpleNamespace(event_name=event), inputs=SimpleNamespace(full_matrix=requested)) is expected
    assert 'uses: ./.github/workflows/native_wheels.yml' in jobs['native-wheel']
    assert 'python-tags: ${{ needs.scope.outputs.python_tags }}' in jobs['native-wheel']
    for core, tags, native, cpu, docs in (
        ('true', '["cp311"]', True, False, False),
        ('true', '["cp310","cp311","cp312","cp313","cp314"]', True, True, False),
        ('false', '["cp311"]', False, False, True),
    ):
        needs = SimpleNamespace(scope=SimpleNamespace(outputs=SimpleNamespace(core=core, python_tags=tags)))
        for job, expected in (('native-wheel', native), ('package-smoke', native), ('cpu-inference', cpu), ('doc-check', docs)):
            assert _evaluate(_expression(jobs[job], 'if'), needs=needs) is expected
    assert 'needs: [scope, package-smoke]' in jobs['cpu-inference']
    assert 'needs: [scope, native-wheel]' in jobs['package-smoke']
    assert 'name: wheel-cp311' in jobs['package-smoke']
    assert 'name: wheel-${{ matrix.python-tag }}' in jobs['cpu-inference']
    for version in ('3.10', '3.12', '3.13', '3.14'):
        assert f'python-version: "{version}"' in jobs['cpu-inference']
    assert 'audit_sdist.py dist/*.tar.gz' in jobs['package-smoke']
    assert 'inputs.export_sdist' in jobs['package-smoke']
    assert 'name: sdist' in jobs['package-smoke']
    assert 'fetch-depth: 0' in jobs['doc-check']
    assert 'python -m pip install pytest' in jobs['doc-check']
    assert '--confcutdir=tests/test_packaging' in jobs['doc-check']
    assert 'tests/test_packaging/test_github_workflows.py' in jobs['doc-check']
    assert 'tests/test_packaging/test_ci_scope.py' in jobs['doc-check']
    runner = (ROOT / 'tests/run_inference_compatibility.sh').read_text()
    assert 'HOTPOT_TEST_INSTALLED_PACKAGE=1' in runner
    assert 'PYTHONSAFEPATH=1' in runner
    assert 'export PYTHONPATH="$test_support"' in runner


def test_coverage_keeps_default_installed_and_model_validation_with_oidc():
    workflow = _workflow('inference_compatibility.yml')
    jobs = _jobs(workflow)
    coverage = jobs['coverage']
    assert not (WORKFLOWS / 'test_push.yml').exists()
    assert 'needs: package-smoke' in coverage
    assert 'id-token: write' in coverage
    assert 'name: wheel-cp311' in coverage
    assert './tests/run_coverage.sh' in coverage
    assert 'HOTPOT_TEST_INSTALLED_PACKAGE: "1"' in coverage
    assert 'HOTPOT_RUN_MODEL_ARTIFACT_TESTS: "1"' in coverage
    assert 'codecov/codecov-action@v5' in coverage
    assert 'use_oidc: true' in coverage
    runner = (ROOT / 'tests/run_coverage.sh').read_text()
    assert '"$repo_root/tests"' in runner
    assert '--run-scientific-boundary' not in runner


def test_release_reuses_full_business_validation_and_requires_explicit_publication():
    workflow = _workflow('publish_pypi.yml')
    events = workflow.split('\non:\n', 1)[1].split('\npermissions:', 1)[0]
    assert 'tags:' in events and '"v*"' in events
    assert 'paths:' not in events and 'branches:' not in events
    assert re.search(r'publish:.*?default: false', events, re.S)
    jobs = _jobs(workflow)
    assert set(jobs) == {'release-metadata', 'validation', 'publish'}
    assert 'Match a release tag to the project version' in jobs['release-metadata']
    assert 'os.environ["RELEASE_TAG"] == f"v{version}"' in jobs['release-metadata']
    validation = jobs['validation']
    assert 'needs: release-metadata' in validation
    assert 'uses: ./.github/workflows/inference_compatibility.yml' in validation
    assert 'full_matrix: true' in validation
    assert 'export_sdist: true' in validation
    assert 'contents: read' in validation and 'id-token: write' in validation
    publish = jobs['publish']
    assert 'needs: validation' in publish
    expression = _expression(publish, 'if')
    for event, requested, expected in (
        ('push', True, False), ('push', False, False),
        ('workflow_dispatch', False, False), ('workflow_dispatch', True, True),
    ):
        assert _evaluate(expression, github=SimpleNamespace(event_name=event), inputs=SimpleNamespace(publish=requested)) is expected
    assert 'pattern: wheel-*' in publish
    assert 'name: sdist' in publish
    assert 'pattern: "*"' not in publish
    assert 'name: pypi' in publish and 'id-token: write' in publish
    assert 'pypa/gh-action-pypi-publish@v1.14.2' in publish


def test_inference_cancellation_never_cancels_tag_or_manual_release_validation():
    workflow = _workflow('inference_compatibility.yml')
    concurrency = workflow.split('\nconcurrency:\n', 1)[1].split('\njobs:', 1)[0]
    assert '${{ github.workflow }}' in concurrency and '${{ github.ref }}' in concurrency
    expression = _expression(concurrency, 'cancel-in-progress')
    for event, ref_type, expected in (
        ('push', 'branch', True), ('pull_request', 'branch', True),
        ('push', 'tag', False), ('workflow_dispatch', 'branch', False),
        ('workflow_dispatch', 'tag', False), ('schedule', 'branch', False),
    ):
        assert _evaluate(expression, github=SimpleNamespace(event_name=event, ref_type=ref_type)) is expected
    group = re.search(r'(?m)^  group: (.+)$', concurrency).group(1)

    def render_group(event, requested, workflow_name='Inference compatibility', run_id=1, ref_type='branch'):
        context = {
            'github': SimpleNamespace(event_name=event, workflow=workflow_name, ref='refs/heads/main', run_id=run_id, ref_type=ref_type),
            'inputs': SimpleNamespace(full_matrix=requested),
        }
        return re.sub(r'\$\{\{(.+?)\}\}', lambda match: str(_evaluate(match.group(1), **context)), group)

    assert render_group('push', False) != render_group('schedule', False)
    assert render_group('push', False) != render_group('workflow_dispatch', True)
    assert render_group('push', False) != render_group('workflow_dispatch', True, 'Build and optionally publish Python distributions')
    for event in ('push', 'pull_request'):
        assert render_group(event, False, run_id=1) == render_group(event, False, run_id=2)
    for event in ('schedule', 'workflow_dispatch'):
        assert render_group(event, True, run_id=1) != render_group(event, True, run_id=2)
    assert render_group('push', True, run_id=1, ref_type='tag') != render_group('push', True, run_id=2, ref_type='tag')
    assert 'concurrency:' not in _workflow('publish_pypi.yml')
