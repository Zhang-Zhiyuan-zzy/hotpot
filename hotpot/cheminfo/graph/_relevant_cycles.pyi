from typing import List, Optional, Sequence, Tuple


class RelevantCycleLimitExceeded(RuntimeError): ...


class RelevantCyclesNotImplemented(NotImplementedError): ...


def relevant_cycles(
    edges: Sequence[Tuple[int, int]],
    max_size: Optional[int] = ...,
    max_cycles: Optional[int] = ...,
) -> List[List[int]]: ...


def _interface_probe(
    edges: Sequence[Tuple[int, int]],
    max_size: Optional[int] = ...,
    max_cycles: Optional[int] = ...,
) -> Tuple[List[List[int]], Optional[int], Optional[int]]: ...
