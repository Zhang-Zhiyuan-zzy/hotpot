from typing import Any, Iterator


class MappingCursor(Iterator[tuple[int, ...]]):
    def __iter__(self) -> MappingCursor: ...
    def __next__(self) -> tuple[int, ...]: ...


class NativeTarget:
    def __init__(self, target_spec: dict[str, Any]) -> None: ...


class NativeProblem:
    def __init__(
        self,
        query_spec: dict[str, Any],
        target: NativeTarget,
    ) -> None: ...
    @property
    def query_keys(self) -> tuple[int, ...]: ...
    def has_match(self) -> bool: ...
    def cursor(self) -> MappingCursor: ...


class _NativeAutomorphism:
    def __init__(
        self,
        target: NativeTarget,
        node_labels: tuple[int, ...],
        edge_labels: tuple[int, ...],
    ) -> None: ...
    def equivalent(self, first: int, second: int) -> bool: ...
