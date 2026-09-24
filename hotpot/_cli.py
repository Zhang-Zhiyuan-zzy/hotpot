"""Shared command-line presentation helpers."""

from __future__ import annotations

import argparse
from typing import Callable, Optional, Sequence


__all__ = (
    "MarkdownDocumentationAction",
)


class MarkdownDocumentationAction(argparse.Action):
    """Render packaged Markdown without loading command business dependencies."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        nargs: Optional[int] = None,
        *,
        document_loader: Callable[[], str],
        **kwargs: object,
    ) -> None:
        if nargs != 0:
            raise ValueError("MarkdownDocumentationAction requires nargs=0")
        self._document_loader = document_loader
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: Optional[str] = None,
    ) -> None:
        from rich.console import Console
        from rich.markdown import Markdown

        Console().print(Markdown(self._document_loader()))
        parser.exit()
