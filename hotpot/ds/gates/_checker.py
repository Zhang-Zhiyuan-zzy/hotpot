"""Admission and execution boundary for custom Gate checkers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import errno
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

try:
    import resource as _resource
except ModuleNotFoundError:  # pragma: no cover - non-Linux import boundary
    _resource = None


Checker = Callable[[Mapping[str, Any], Mapping[str, Any]], Tuple[bool, str]]

def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


# Modules that are deterministic and do not themselves grant filesystem,
# process, network, clock, entropy, or project-package access.  The isolated
# runner additionally supplies a restricted built-in namespace.
_ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "decimal",
        "itertools",
        "math",
    }
)

# Suspended execution objects expose interpreter frames through otherwise
# non-dunder attributes (for example ``generator.gi_frame.f_builtins``).  A
# checker has no need for generators or coroutines, so reject both their
# construction and every direct frame/tracing attribute at admission time.
_FORBIDDEN_AST_NODE_NAMES = frozenset(
    {
        "AsyncFor",
        "AsyncFunctionDef",
        "AsyncWith",
        "Await",
        "GeneratorExp",
        "Yield",
        "YieldFrom",
    }
)
_FORBIDDEN_RUNTIME_ATTRIBUTES = frozenset(
    {
        "ag_await",
        "ag_code",
        "ag_frame",
        "ag_running",
        "cr_await",
        "cr_code",
        "cr_frame",
        "cr_origin",
        "cr_running",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_lasti",
        "f_lineno",
        "f_locals",
        "f_trace",
        "gi_code",
        "gi_frame",
        "gi_running",
        "gi_suspended",
        "gi_yieldfrom",
        "tb_frame",
        "tb_lasti",
        "tb_lineno",
        "tb_next",
    }
)

_CHECKER_LIMITS = {
    "address_space_bytes": 512 * 1024 * 1024,
    "cpu_seconds": 2,
    "file_size_bytes": 1024 * 1024,
    "open_files": 32,
    "processes": 16,
    "request_bytes": 8 * 1024 * 1024,
    "response_bytes": 64 * 1024,
    "wall_seconds": 5,
}

_CHECKER_WORKER_SOURCE = f'''\
import __future__ as future_features
import ast
import hashlib
import json
import sys
from types import MappingProxyType

_ALLOWED_IMPORTS = frozenset({tuple(sorted(_ALLOWED_IMPORTS))!r})
_FORBIDDEN_AST_NODE_NAMES = frozenset({tuple(sorted(_FORBIDDEN_AST_NODE_NAMES))!r})
_FORBIDDEN_RUNTIME_ATTRIBUTES = frozenset({tuple(sorted(_FORBIDDEN_RUNTIME_ATTRIBUTES))!r})
def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({{key: _freeze(item) for key, item in value.items()}})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value

def _main():
    request = json.load(sys.stdin)
    if set(request) != {{'protocol_digest', 'checker', 'case', 'bundle'}}:
        raise ValueError('invalid checker request fields')
    if request['protocol_digest'] != sys.argv[1]:
        raise ValueError('checker protocol digest mismatch')
    checker = request['checker']
    if set(checker) != {{'kind', 'source', 'entrypoint', 'source_sha256', 'imports', 'checker_protocol_digest'}}:
        raise ValueError('invalid checker definition fields')
    if checker['kind'] != 'custom' or checker['checker_protocol_digest'] != sys.argv[1]:
        raise ValueError('invalid checker protocol binding')
    source = checker['source']
    if hashlib.sha256(source.encode('utf-8')).hexdigest() != checker['source_sha256']:
        raise ValueError('checker source digest mismatch')
    if set(checker['imports']) - _ALLOWED_IMPORTS:
        raise ValueError('checker import inventory mismatch')
    tree = ast.parse(source)
    import_bindings = {{}}
    retained = []
    observed_imports = set()
    compiler_flags = 0
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if '.' in alias.name or alias.name not in _ALLOWED_IMPORTS:
                    raise ValueError('checker import is outside the fixed allowlist')
                if (alias.asname or alias.name).startswith('__'):
                    raise ValueError('dunder import aliases are forbidden')
                observed_imports.add(alias.name)
                import_bindings[alias.asname or alias.name] = __import__(alias.name)
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module == '__future__' and node.level == 0:
                observed_imports.add('__future__')
                if ({{alias.name for alias in node.names}} != {{'annotations'}}
                        or any(alias.asname is not None for alias in node.names)):
                    raise ValueError('only future annotations is supported')
                compiler_flags |= future_features.annotations.compiler_flag
                continue
            raise ValueError('checker from-imports are forbidden')
        retained.append(node)
    if observed_imports != set(checker['imports']):
        raise ValueError('checker import inventory mismatch')
    for node in ast.walk(tree):
        if type(node).__name__ in _FORBIDDEN_AST_NODE_NAMES:
            raise ValueError('generators and asynchronous execution are forbidden')
        if isinstance(node, ast.comprehension) and node.is_async:
            raise ValueError('asynchronous comprehensions are forbidden')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('dunder runtime names are forbidden')
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('__'):
                raise ValueError('dunder runtime attributes are forbidden')
            if node.attr in _FORBIDDEN_RUNTIME_ATTRIBUTES:
                raise ValueError('interpreter frame attributes are forbidden')
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and '__' in node.value):
            raise ValueError('dunder runtime keys are forbidden')
    safe_builtins = {{
        'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
        'enumerate': enumerate, 'filter': filter, 'float': float, 'int': int,
        'isinstance': isinstance, 'len': len, 'list': list, 'map': map,
        'max': max, 'min': min, 'next': next, 'range': range,
        'reversed': reversed, 'round': round, 'set': set, 'sorted': sorted,
        'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
        'Exception': Exception, 'ValueError': ValueError,
    }}
    namespace = {{'__builtins__': safe_builtins, '__name__': '__hotpot_checker__',
                 **import_bindings}}
    tree.body = retained
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<persisted-checker>', 'exec', flags=compiler_flags,
                 dont_inherit=True), namespace, namespace)
    entrypoint = namespace.get(checker['entrypoint'])
    if not callable(entrypoint):
        raise ValueError('checker entrypoint is unavailable')
    result = entrypoint(_freeze(request['case']), _freeze(request['bundle']))
    if (type(result) is not tuple or len(result) != 2 or
            type(result[0]) is not bool or not isinstance(result[1], str) or
            not result[1].strip()):
        raise ValueError('invalid checker result')
    return {{'ok': True, 'passed': result[0], 'reason': result[1]}}

try:
    response = _main()
except BaseException:
    response = {{'ok': False, 'error': 'checker_error'}}
sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False,
                            separators=(',', ':'), sort_keys=True))
'''

_CHECKER_PROTOCOL = {
    "arguments": [
        {"name": "case", "type": "canonical_deep_read_only_json_mapping"},
        {"name": "bundle", "type": "bounded_deep_read_only_json_mapping"},
    ],
    "entrypoint": "top_level_python_function",
    "imports": sorted(_ALLOWED_IMPORTS),
    "isolation": {
        "filesystem": "empty_root_with_read_only_python_runtime",
        "network": "dedicated_network_namespace",
        "process": "dedicated_pid_user_ipc_uts_namespaces",
        "environment": {
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "TZ": "UTC",
        },
        "limits": dict(_CHECKER_LIMITS),
    },
    "request_fields": ["protocol_digest", "checker", "case", "bundle"],
    "response_variants": {
        "success_fields": ["ok", "passed", "reason"],
        "error_fields": ["ok", "error"],
    },
    "result": ["exact_bool", "non_empty_reason_string"],
    "worker_source_sha256": hashlib.sha256(
        _CHECKER_WORKER_SOURCE.encode("utf-8")
    ).hexdigest(),
}

CHECKER_PROTOCOL_DIGEST = hashlib.sha256(
    _canonical_bytes(_CHECKER_PROTOCOL)
).hexdigest()


class CheckerIsolationUnavailableError(RuntimeError):
    """Raised when custom code cannot run behind the required OS boundary."""


class CheckerExecutionError(RuntimeError):
    """Raised when an isolated checker fails or violates its result protocol."""


def _apply_worker_limits() -> None:
    _resource.setrlimit(
        _resource.RLIMIT_AS,
        (
            _CHECKER_LIMITS["address_space_bytes"],
            _CHECKER_LIMITS["address_space_bytes"],
        ),
    )
    _resource.setrlimit(
        _resource.RLIMIT_CPU,
        (_CHECKER_LIMITS["cpu_seconds"], _CHECKER_LIMITS["cpu_seconds"]),
    )
    _resource.setrlimit(
        _resource.RLIMIT_FSIZE,
        (_CHECKER_LIMITS["file_size_bytes"], _CHECKER_LIMITS["file_size_bytes"]),
    )
    _resource.setrlimit(
        _resource.RLIMIT_NOFILE,
        (_CHECKER_LIMITS["open_files"], _CHECKER_LIMITS["open_files"]),
    )
    _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
    _resource.setrlimit(
        _resource.RLIMIT_NPROC,
        (_CHECKER_LIMITS["processes"], _CHECKER_LIMITS["processes"]),
    )


class OSIsolatedCheckerRunner:
    """Execute persisted checker source in the mandatory Linux/bubblewrap cage."""

    os_isolation_verified = True

    def __init__(self, forbidden_roots: Tuple[Path, ...] = ()) -> None:
        if sys.platform != "linux" or _resource is None:
            raise CheckerIsolationUnavailableError(
                "custom checker isolation requires Linux namespaces"
            )
        executable = shutil.which("bwrap")
        if executable is None:
            raise CheckerIsolationUnavailableError(
                "custom checker isolation requires bubblewrap"
            )
        python = Path(sys.executable).resolve()
        prefix = Path(sys.prefix).resolve()
        if prefix == Path("/"):
            raise CheckerIsolationUnavailableError(
                "the Python runtime prefix cannot be the filesystem root"
            )
        try:
            python.relative_to(prefix)
        except ValueError as exc:
            raise CheckerIsolationUnavailableError(
                "the Python runtime is not contained in its isolated prefix"
            ) from exc
        mounted_roots = (
            prefix,
            *(root.resolve() for root in (Path("/lib"), Path("/lib64")) if root.exists()),
        )
        for raw_root in forbidden_roots:
            root = Path(raw_root).resolve()
            if any(
                root == mounted
                or root in mounted.parents
                or mounted in root.parents
                for mounted in mounted_roots
            ):
                raise CheckerIsolationUnavailableError(
                    "an isolated runtime mount overlaps Bundle/store data"
                )
        self._forbidden_roots = tuple(Path(root).resolve() for root in forbidden_roots)
        self._command_prefix = self._build_command(Path(executable), python, prefix)
        self._verify_isolation()

    @staticmethod
    def _build_command(bwrap: Path, python: Path, prefix: Path) -> Tuple[str, ...]:
        command = [
            str(bwrap),
            "--unshare-user",
            "--uid",
            "65534",
            "--gid",
            "65534",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--new-session",
            "--die-with-parent",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            str(prefix),
            str(prefix),
        ]
        for library_root in (Path("/lib"), Path("/lib64")):
            if library_root.exists():
                command.extend(
                    ("--ro-bind", str(library_root), str(library_root))
                )
        command.extend(
            (
                "--tmpfs",
                "/tmp",
                "--chdir",
                "/tmp",
                str(python),
                "-S",
                "-s",
                "-B",
            )
        )
        return tuple(command)

    def _execute(
        self,
        arguments: Tuple[str, ...],
        payload: bytes,
        *,
        bootstrap: bool = False,
    ) -> subprocess.CompletedProcess:
        try:
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                completed = subprocess.run(
                    self._command_prefix + arguments,
                    input=payload,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=_CHECKER_LIMITS["wall_seconds"],
                    check=False,
                    close_fds=True,
                    env={
                        "LANG": "C",
                        "LC_ALL": "C",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONHASHSEED": "0",
                        "PYTHONNOUSERSITE": "1",
                        "TZ": "UTC",
                    },
                    preexec_fn=_apply_worker_limits,
                )
                stdout.seek(0)
                stderr.seek(0)
                return subprocess.CompletedProcess(
                    completed.args,
                    completed.returncode,
                    stdout.read(_CHECKER_LIMITS["response_bytes"] + 1),
                    stderr.read(_CHECKER_LIMITS["response_bytes"] + 1),
                )
        except subprocess.TimeoutExpired as exc:
            error = (
                CheckerIsolationUnavailableError
                if bootstrap
                else CheckerExecutionError
            )
            raise error("the isolated checker worker exceeded its wall-time limit") from exc
        except OSError as exc:
            raise CheckerIsolationUnavailableError(
                "the custom checker isolation worker could not start"
            ) from exc

    def _verify_isolation(self) -> None:
        hidden_paths = [str(Path.cwd().resolve()), *map(str, self._forbidden_roots)]
        probe_source = (
            "import json,os,socket,sys;"
            "s=socket.socket();s.settimeout(0.2);"
            "r={'cwd':os.getcwd(),'hidden':all(not os.path.exists(p) for p in json.loads(sys.argv[1])),"
            "'proc':os.path.exists('/proc'),'dev':os.path.exists('/dev'),"
            "'network':s.connect_ex(('1.1.1.1',9))};s.close();"
            "sys.stdout.write(json.dumps(r,sort_keys=True,separators=(',',':')))"
        )
        probe = self._execute(
            ("-c", probe_source, json.dumps(hidden_paths)), b"", bootstrap=True
        )
        expected_probe = {
            "cwd": "/tmp",
            "hidden": True,
            "proc": False,
            "dev": False,
        }
        try:
            probe_value = json.loads(probe.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            probe_value = {}
        if (
            probe.returncode != 0
            or any(probe_value.get(key) != value for key, value in expected_probe.items())
            or probe_value.get("network")
            not in {errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN}
        ):
            raise CheckerIsolationUnavailableError(
                "the host cannot establish the required custom checker isolation"
            )

    def run(
        self,
        checker: Mapping[str, Any],
        case: Mapping[str, Any],
        bundle: Mapping[str, Any],
    ) -> Tuple[bool, str]:
        try:
            definition = _CheckerDefinition.from_info(checker)
        except (TypeError, ValueError) as exc:
            raise CheckerExecutionError("checker definition violates the protocol") from exc
        request = _canonical_bytes(
            {
                "protocol_digest": CHECKER_PROTOCOL_DIGEST,
                "checker": dict(definition.info()),
                "case": case,
                "bundle": bundle,
            }
        )
        if len(request) > _CHECKER_LIMITS["request_bytes"]:
            raise CheckerExecutionError("checker request exceeds the protocol limit")
        completed = self._execute(
            ("-c", _CHECKER_WORKER_SOURCE, CHECKER_PROTOCOL_DIGEST), request
        )
        if completed.returncode != 0:
            raise CheckerExecutionError("isolated checker worker failed")
        if (
            len(completed.stdout) > _CHECKER_LIMITS["response_bytes"]
            or len(completed.stderr) > _CHECKER_LIMITS["response_bytes"]
            or completed.stderr
        ):
            raise CheckerExecutionError("checker output violates the protocol limit")
        try:
            response = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckerExecutionError("checker response is not JSON") from exc
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise CheckerExecutionError("isolated checker reported checker_error")
        if set(response) != {"ok", "passed", "reason"}:
            raise CheckerExecutionError("checker response fields are invalid")
        return _validate_result((response["passed"], response["reason"]))


@dataclass(frozen=True)
class _CheckerDefinition:
    kind: str
    enabled: Optional[bool] = None
    source: Optional[str] = None
    entrypoint: Optional[str] = None
    source_sha256: Optional[str] = None
    imports: Tuple[str, ...] = ()
    checker_protocol_digest: str = CHECKER_PROTOCOL_DIGEST

    @classmethod
    def builtin(cls, enabled: bool) -> "_CheckerDefinition":
        return cls(kind="builtin", enabled=enabled)

    @classmethod
    def custom(cls, checker: Checker) -> "_CheckerDefinition":
        source, entrypoint, imports = _inspect_callable(checker)
        return cls(
            kind="custom",
            source=source,
            entrypoint=entrypoint,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            imports=imports,
        )

    @classmethod
    def from_info(cls, value: Mapping[str, Any]) -> "_CheckerDefinition":
        kind = value.get("kind")
        if kind == "builtin":
            if set(value) != {"kind", "enabled"} or type(value["enabled"]) is not bool:
                raise ValueError("invalid built-in checker configuration")
            return cls.builtin(value["enabled"])
        if kind != "custom":
            raise ValueError("invalid checker kind")
        expected = {
            "kind",
            "source",
            "entrypoint",
            "source_sha256",
            "imports",
            "checker_protocol_digest",
        }
        if set(value) != expected:
            raise ValueError("invalid custom checker configuration fields")
        source = value["source"]
        entrypoint = value["entrypoint"]
        source_sha256 = value["source_sha256"]
        protocol_digest = value["checker_protocol_digest"]
        imports = value["imports"]
        if not isinstance(source, str) or not isinstance(entrypoint, str):
            raise ValueError("custom checker source and entrypoint must be strings")
        if protocol_digest != CHECKER_PROTOCOL_DIGEST:
            raise ValueError("custom checker protocol digest does not match")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != source_sha256:
            raise ValueError("custom checker source digest does not match")
        parsed_imports = _inspect_source(source, entrypoint)
        if not isinstance(imports, list) or tuple(imports) != parsed_imports:
            raise ValueError("custom checker import inventory does not match")
        return cls(
            kind="custom",
            source=source,
            entrypoint=entrypoint,
            source_sha256=source_sha256,
            imports=parsed_imports,
        )

    def info(self) -> Mapping[str, Any]:
        if self.kind == "builtin":
            return MappingProxyType({"kind": "builtin", "enabled": self.enabled})
        return MappingProxyType(
            {
                "kind": "custom",
                "source": self.source,
                "entrypoint": self.entrypoint,
                "source_sha256": self.source_sha256,
                "imports": list(self.imports),
                "checker_protocol_digest": self.checker_protocol_digest,
            }
        )


def _inspect_callable(checker: Checker) -> Tuple[str, str, Tuple[str, ...]]:
    if not inspect.isfunction(checker):
        raise TypeError("a custom checker must be a Python function")
    if checker.__name__ == "<lambda>" or "<locals>" in checker.__qualname__:
        raise ValueError("a custom checker must be a top-level named function")
    if checker.__closure__ or checker.__code__.co_freevars:
        raise ValueError("a custom checker must not close over external state")
    _validate_signature(inspect.signature(checker))

    source_path = inspect.getsourcefile(checker)
    if source_path is None or source_path.startswith("<"):
        raise ValueError("custom checker source is unavailable")
    try:
        source = Path(source_path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("custom checker source is not an accessible UTF-8 file") from exc
    imports = _inspect_source(source, checker.__name__)
    return source, checker.__name__, imports


def _validate_signature(signature: inspect.Signature) -> None:
    parameters = tuple(signature.parameters.values())
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if (
        len(parameters) != 2
        or tuple(parameter.name for parameter in parameters) != ("case", "bundle")
        or any(parameter.kind not in positional for parameter in parameters)
        or any(parameter.default is not inspect.Parameter.empty for parameter in parameters)
    ):
        raise TypeError(
            "a custom checker must have the signature checker(case, bundle)"
        )


def _inspect_source(source: str, entrypoint: str) -> Tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("custom checker source is not valid Python") from exc
    entrypoints = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint
    ]
    if len(entrypoints) != 1:
        raise ValueError("custom checker entrypoint is not a unique top-level function")
    node = entrypoints[0]
    arguments = node.args
    if (
        arguments.posonlyargs
        or tuple(argument.arg for argument in arguments.args) != ("case", "bundle")
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kwarg is not None
        or arguments.defaults
        or arguments.kw_defaults
    ):
        raise TypeError(
            "a custom checker must have the signature checker(case, bundle)"
        )

    imports = set()
    top_level_imports = {
        id(candidate)
        for candidate in tree.body
        if isinstance(candidate, (ast.Import, ast.ImportFrom))
    }
    for candidate in ast.walk(tree):
        if type(candidate).__name__ in _FORBIDDEN_AST_NODE_NAMES:
            raise ValueError(
                "generators and asynchronous execution are not allowed "
                "in custom checkers"
            )
        if isinstance(candidate, ast.comprehension) and candidate.is_async:
            raise ValueError(
                "asynchronous comprehensions are not allowed in custom checkers"
            )
        if isinstance(candidate, ast.Name) and candidate.id.startswith("__"):
            raise ValueError("dunder runtime names are not allowed in custom checkers")
        if isinstance(candidate, ast.Attribute):
            if candidate.attr.startswith("__"):
                raise ValueError(
                    "dunder runtime attributes are not allowed in custom checkers"
                )
            if candidate.attr in _FORBIDDEN_RUNTIME_ATTRIBUTES:
                raise ValueError(
                    "interpreter frame attributes are not allowed in custom checkers"
                )
        if (
            isinstance(candidate, ast.Constant)
            and isinstance(candidate.value, str)
            and "__" in candidate.value
        ):
            raise ValueError("dunder runtime keys are not allowed in custom checkers")
        if isinstance(candidate, ast.Import):
            if id(candidate) not in top_level_imports:
                raise ValueError("checker imports must be top-level")
            if any("." in alias.name for alias in candidate.names):
                raise ValueError("dotted imports are not allowed in custom checkers")
            if any(
                (alias.asname or alias.name).startswith("__")
                for alias in candidate.names
            ):
                raise ValueError("dunder import aliases are not allowed")
            imports.update(alias.name for alias in candidate.names)
        elif isinstance(candidate, ast.ImportFrom):
            if id(candidate) not in top_level_imports:
                raise ValueError("checker imports must be top-level")
            if (
                candidate.level
                or candidate.module != "__future__"
                or {alias.name for alias in candidate.names} != {"annotations"}
                or any(alias.asname is not None for alias in candidate.names)
            ):
                raise ValueError(
                    "only 'from __future__ import annotations' is allowed"
                )
            imports.add("__future__")
    forbidden = imports - _ALLOWED_IMPORTS
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise ValueError(f"custom checker imports are not allowlisted: {names}")
    return tuple(sorted(imports))


def _validate_result(result: Any) -> Tuple[bool, str]:
    if (
        type(result) is not tuple
        or len(result) != 2
        or type(result[0]) is not bool
        or not isinstance(result[1], str)
        or not result[1].strip()
    ):
        raise ValueError("a checker result must be tuple[exact bool, non-empty str]")
    return result[0], result[1]


def evaluate_configured_check(
    checker: _CheckerDefinition,
    case: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    builtin_result: Optional[Tuple[bool, str]] = None,
    isolated_runner: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Evaluate one admitted checker without an in-process custom-code path.

    A synchronization engine computes built-in results itself.  A custom
    definition is passed only to a runner that explicitly advertises the
    complete OS isolation contract; the original Python callable is never
    retained by the Gate.
    """

    if checker.kind == "builtin":
        if checker.enabled is False:
            raise ValueError("a disabled checker cannot be evaluated")
        return _validate_result(builtin_result)

    if (
        isolated_runner is None
        or getattr(isolated_runner, "os_isolation_verified", False) is not True
    ):
        raise CheckerIsolationUnavailableError(
            "custom checker execution requires a verified OS-isolated runner"
        )
    result = isolated_runner.run(
        checker.info(),
        json.loads(json.dumps(case, ensure_ascii=False, allow_nan=False)),
        json.loads(json.dumps(bundle, ensure_ascii=False, allow_nan=False)),
    )
    return _validate_result(result)
