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
    def cursor(self) -> MappingCursor: ...
