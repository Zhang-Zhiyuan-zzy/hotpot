"""Run the fixed native-FAST-then-complex automatic benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from .workflow_runner import main_for_backend


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    return main_for_backend("hotpot_auto", argv)


if __name__ == "__main__":
    main()
