"""Run the fixed native Open Babel ligand and complex benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from .workflow_runner import main_for_backend


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    return main_for_backend("openbabel", argv)


if __name__ == "__main__":
    main()
