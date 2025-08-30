# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : logging_config
 Created   : 2025/8/30 9:36
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import sys
import logging
from rich.console import Console
from rich.logging import RichHandler

def setup_logging(debug=True, to_stdout=True):
    console = Console(file=sys.stdout) if to_stdout else Console()  # Console() defaults to stderr
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",                # let Rich handle the rest
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True, console=console)],
        force=True,                          # override any prior logging config
    )

__all__ = ["setup_logging"]
