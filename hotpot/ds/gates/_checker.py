"""Admission and execution boundary for custom Gate checkers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple


Checker = Callable[[Mapping[str, Any], Mapping[str, Any]], Tuple[bool, str]]

_CHECKER_PROTOCOL = {
    "arguments": [
        {"name": "case", "type": "canonical_read_only_json_mapping"},
        {"name": "bundle", "type": "bounded_read_only_json_mapping"},
    ],
    "entrypoint": "top_level_python_function",
    "result": ["exact_bool", "non_empty_reason_string"],
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


CHECKER_PROTOCOL_DIGEST = hashlib.sha256(
    _canonical_bytes(_CHECKER_PROTOCOL)
).hexdigest()

# Modules that are deterministic and do not themselves grant filesystem,
# process, network, clock, entropy, or project-package access.  The isolated
# runner additionally supplies a restricted built-in namespace.
_ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "collections",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "re",
        "statistics",
        "string",
        "typing",
    }
)


class CheckerIsolationUnavailableError(RuntimeError):
    """Raised when custom code cannot run behind the required OS boundary."""


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
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in candidate.names)
        elif isinstance(candidate, ast.ImportFrom):
            if candidate.level or candidate.module is None:
                raise ValueError("relative imports are not allowed in custom checkers")
            imports.add(candidate.module.split(".", 1)[0])
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
