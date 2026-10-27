"""Security and determinism tests for the mandatory checker sandbox."""

from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from hotpot.ds.gates._checker import (
    CHECKER_PROTOCOL_DIGEST,
    _CHECKER_LIMITS,
    _inspect_source,
    CheckerExecutionError,
    CheckerIsolationUnavailableError,
    OSIsolatedCheckerRunner,
)

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def runner() -> OSIsolatedCheckerRunner:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this Linux host")
    return OSIsolatedCheckerRunner()


def _checker(source: str, *, imports: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "kind": "custom",
        "source": source,
        "entrypoint": "checker",
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "imports": list(imports),
        "checker_protocol_digest": CHECKER_PROTOCOL_DIGEST,
    }


def test_valid_deterministic_checker(runner: OSIsolatedCheckerRunner) -> None:
    source = """\
def checker(case, bundle):
    return case["value"] == bundle["expected"], "value compared"
"""

    assert runner.run(
        _checker(source), {"value": 3}, {"expected": 3}
    ) == (True, "value compared")


def test_future_annotations_and_static_import_are_transformed_safely(
    runner: OSIsolatedCheckerRunner,
) -> None:
    source = """\
from __future__ import annotations
import math as numeric
def checker(case: MissingType, bundle: MissingType) -> tuple[bool, str]:
    return numeric.isfinite(case["value"]), "future annotations accepted"
"""

    assert runner.run(
        _checker(source, imports=("__future__", "math")), {"value": 1.0}, {}
    ) == (True, "future annotations accepted")


def test_set_iteration_is_deterministic_across_workers(
    runner: OSIsolatedCheckerRunner,
) -> None:
    source = """\
def checker(case, bundle):
    values = {"cobalt", "dysprosium", "europium", "americium", "terbium"}
    return True, ",".join(values)
"""

    results = {runner.run(_checker(source), {}, {}) for _ in range(12)}

    assert len(results) == 1


def test_import_function_globals_escape_fails(
    runner: OSIsolatedCheckerRunner,
) -> None:
    source = """\
def checker(case, bundle):
    restricted_import = __builtins__["__import__"]
    real_import = restricted_import.__globals__["_REAL_IMPORT"]
    real_import("os")
    return True, "escaped"
"""

    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source), {}, {})


def test_dynamic_dunder_operator_escape_is_not_admitted(
    runner: OSIsolatedCheckerRunner,
) -> None:
    source = """\
import operator
def checker(case, bundle):
    d = "_" * 2
    namespace = operator.attrgetter(d + "globals" + d)(checker)
    builtins = operator.itemgetter(d + "builtins" + d)(namespace)
    restricted_import = operator.itemgetter(d + "import" + d)(builtins)
    real_import = operator.attrgetter(d + "globals" + d)(restricted_import)["_REAL_IMPORT"]
    return True, str(len(real_import("os").urandom(1)))
"""

    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source, imports=("operator",)), {}, {})


@pytest.mark.parametrize(
    ("module", "body"),
    [
        ("re", "re.enum.sys.modules['os']"),
        ("json", "json.codecs.sys.modules['os']"),
    ],
)
def test_transitive_module_surface_escape_is_not_admitted(
    runner: OSIsolatedCheckerRunner,
    module: str,
    body: str,
) -> None:
    source = f"""\
import {module}
def checker(case, bundle):
    os = {body}
    return True, str(len(os.urandom(1)))
"""

    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source, imports=(module,)), {}, {})


@pytest.mark.parametrize(
    ("source", "imports"),
    [
        (
            """\
import operator
def checker(case, bundle):
    dunder = "_" * 2
    namespace = operator.attrgetter(dunder + "globals" + dunder)(checker)
    builtins = operator.itemgetter(dunder + "builtins" + dunder)(namespace)
    restricted = operator.itemgetter(dunder + "import" + dunder)(builtins)
    worker_globals = operator.attrgetter(dunder + "globals" + dunder)(restricted)
    real_import = operator.itemgetter("_REAL_IMPORT")(worker_globals)
    real_import("os")
    return True, "escaped dynamically"
""",
            ("operator",),
        ),
        (
            """\
import string
def checker(case, bundle):
    dunder = "_" * 2
    field = ("0." + dunder + "globals" + dunder +
             "[" + dunder + "builtins" + dunder + "]" +
             "[" + dunder + "import" + dunder + "]." +
             dunder + "globals" + dunder + "[_REAL_IMPORT]")
    real_import = string.Formatter().get_field(field, (checker,), {})[0]
    real_import("os")
    return True, "escaped dynamically"
""",
            ("string",),
        ),
    ],
)
def test_dynamic_dunder_construction_escape_fails(
    runner: OSIsolatedCheckerRunner,
    source: str,
    imports: tuple[str, ...],
) -> None:
    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source, imports=imports), {}, {})


@pytest.mark.parametrize(
    ("module", "sys_expression"),
    [("re", "re.enum.sys"), ("json", "json.codecs.sys")],
)
def test_allowlisted_module_cannot_expose_worker_sys_modules(
    runner: OSIsolatedCheckerRunner,
    module: str,
    sys_expression: str,
) -> None:
    source = f'''\
import {module}
def checker(case, bundle):
    os = {sys_expression}.modules["os"]
    return True, str(os.getpid())
'''

    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source, imports=(module,)), {}, {})


def test_generator_frame_cannot_recover_real_builtins(
    runner: OSIsolatedCheckerRunner,
) -> None:
    source = """\
def checker(case, bundle):
    holders = []
    def expose():
        yield holders[0].gi_frame.f_back.f_back.f_builtins
    generator = expose()
    holders.append(generator)
    builtins_map = next(generator)
    dunder = "_" * 2
    real_import = builtins_map[dunder + "import" + dunder]
    os = real_import("os")
    return True, str(os.getpid())
"""

    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source), {}, {})


@pytest.mark.parametrize(
    "source",
    [
        """\
def checker(case, bundle):
    yield True, "generator result"
""",
        """\
def checker(case, bundle):
    values = (value for value in case["values"])
    return True, str(next(values))
""",
        """\
async def suspended():
    return True
def checker(case, bundle):
    coroutine = suspended()
    return True, str(coroutine.cr_frame.f_builtins)
""",
    ],
)
def test_suspended_execution_constructs_are_rejected_by_source_admission(
    source: str,
) -> None:
    with pytest.raises(ValueError, match="generators|asynchronous"):
        _inspect_source(source, "checker")


@pytest.mark.parametrize(
    ("source", "imports"),
    [
        (
            """\
import typing
def checker(case, bundle):
    return typing.sys is not None, "private module reached"
""",
            ("typing",),
        ),
        (
            """\
def checker(case, bundle):
    return __builtins__["__import__"] is not None, "private builtins reached"
""",
            (),
        ),
    ],
)
def test_private_runtime_access_fails(
    runner: OSIsolatedCheckerRunner,
    source: str,
    imports: tuple[str, ...],
) -> None:
    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source, imports=imports), {}, {})


def test_wall_or_cpu_limit_terminates_checker(
    runner: OSIsolatedCheckerRunner,
) -> None:
    source = """\
def checker(case, bundle):
    while True:
        pass
"""

    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source), {}, {})


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_worker_transport_bounds_oversized_output(
    runner: OSIsolatedCheckerRunner,
    stream: str,
) -> None:
    emitted = _CHECKER_LIMITS["response_bytes"] * 4
    probe_source = f"import sys;sys.{stream}.write('x'*{emitted})"

    completed = runner._execute(("-c", probe_source), b"")

    assert completed.returncode == 0
    assert len(getattr(completed, stream)) == _CHECKER_LIMITS["response_bytes"] + 1


def test_worker_process_count_is_hard_limited(
    runner: OSIsolatedCheckerRunner,
) -> None:
    probe_source = """\
import json, os, time
children = []
failure = None
for index in range(64):
    try:
        child = os.fork()
    except OSError as error:
        failure = [index, error.errno]
        break
    if child == 0:
        time.sleep(0.05)
        os._exit(0)
    children.append(child)
for child in children:
    os.waitpid(child, 0)
print(json.dumps({"children": len(children), "failure": failure}))
"""

    completed = runner._execute(("-c", probe_source), b"")

    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["children"] < _CHECKER_LIMITS["processes"]
    assert result["failure"][1] == errno.EAGAIN


def test_runtime_mount_overlap_is_rejected_before_execution() -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this Linux host")

    with pytest.raises(CheckerIsolationUnavailableError, match="overlaps"):
        OSIsolatedCheckerRunner((Path("/lib/hotpot-dataset"),))


def test_sandbox_cannot_see_workspace_or_use_network(
    runner: OSIsolatedCheckerRunner,
) -> None:
    workspace_file = str(Path(__file__).resolve())
    probe_source = (
        "import json,os,socket;"
        "sock=socket.socket();sock.settimeout(0.2);"
        "result={'workspace_visible':os.path.exists("
        + repr(workspace_file)
        + "),'network_result':sock.connect_ex(('1.1.1.1',9))};"
        "sock.close();print(json.dumps(result,sort_keys=True))"
    )

    completed = runner._execute(("-c", probe_source), b"")

    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["workspace_visible"] is False
    assert result["network_result"] in {
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
    }


@pytest.mark.parametrize(
    "body",
    [
        "return True",
        "return True, ''",
        "return 1, 'not an exact bool'",
        "return [True, 'not a tuple']",
    ],
)
def test_malformed_checker_result_fails(
    runner: OSIsolatedCheckerRunner,
    body: str,
) -> None:
    source = f"def checker(case, bundle):\n    {body}\n"

    with pytest.raises(CheckerExecutionError):
        runner.run(_checker(source), {}, {})
