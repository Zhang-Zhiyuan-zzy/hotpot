from typing import List, Optional, Sequence, Tuple


class RelevantCycleLimitExceeded(RuntimeError): ...


def relevant_cycles(
    edges: Sequence[Tuple[int, int]],
    max_size: Optional[int] = ...,
    max_cycles: Optional[int] = ...,
) -> List[List[int]]: ...
