from typing import Any, Iterator, Sequence


class MappingCursor(Iterator[tuple[int, ...]]):
    def __iter__(self) -> MappingCursor: ...
    def __next__(self) -> tuple[int, ...]: ...


class SyntheticProblem:
    def __init__(
        self,
        query_keys: Sequence[int],
        query_edges: Sequence[tuple[int, int]],
        target_keys: Sequence[int],
        target_edges: Sequence[tuple[int, int]],
        node_compatible: Sequence[Sequence[bool]],
        edge_compatible: Sequence[Sequence[bool]],
    ) -> None: ...
    @property
    def query_keys(self) -> tuple[int, ...]: ...
    def cursor(self) -> MappingCursor: ...


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
    def cursor(self) -> MappingCursor: ...
