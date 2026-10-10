"""Isolated temporary workspace lifecycle for native processes."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


__all__ = ["isolated_workspace"]


@contextmanager
def isolated_workspace(
    parent: Optional[Path] = None,
    *,
    prefix: str = "hotpot-",
    retain: bool = False,
) -> Iterator[Path]:
    """Yield a unique workspace and remove it unless retention is requested."""

    parent_path = None if parent is None else os.fspath(parent)
    workspace = Path(tempfile.mkdtemp(prefix=prefix, dir=parent_path)).resolve()
    try:
        yield workspace
    finally:
        if not retain:
            shutil.rmtree(workspace)
