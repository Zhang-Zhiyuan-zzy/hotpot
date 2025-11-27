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
import time
import logging
from collections import defaultdict
from functools import partial

from rich.console import Console
from rich.logging import RichHandler

__all__ = [
    "setup_logging",
    "RateLimitLogger",
    "LoggerDict"
]

def setup_logging(debug=True, to_stdout=True):
    console = Console(file=sys.stdout) if to_stdout else Console()  # Console() defaults to stderr
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",                # let Rich handle the rest
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True, console=console)],
        force=True,                          # override any prior logging config
    )


class RateLimitLogger:
    def __init__(
            self,
            interval_seconds=0,
            interval_count=0,
            just_once=False,
    ):
        self.interval_seconds = interval_seconds
        self.interval_count = interval_count
        self.just_once = just_once

        self._last_time = 0
        self._counter = 0
        self._once = False

    def allow(self) -> bool:
        """ Determine whether to allow logging """
        if self.just_once:
            if self._once:
                return False
            self._once = True
            return True

        should_trigger = False

        # 1. 检查时间
        if self.interval_seconds > 0:
            now = time.time()
            if now - self._last_time > self.interval_seconds:
                self._last_time = now
                should_trigger = True

        # 2. 检查次数
        if self.interval_count > 0:
            self._counter += 1
            if self._counter % self.interval_count == 1:
                should_trigger = True

        return should_trigger

    def info(self, msg: str, *args, **kwargs):
        if self.allow():
            logging.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        if self.allow():
            logging.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        if self.allow():
            logging.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        if self.allow():
            logging.debug(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        if self.allow():
            logging.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        if self.allow():
            logging.exception(msg, *args, **kwargs)


class LoggerDict(defaultdict):
    def __init__(
            self,
            interval_seconds=0,
            interval_count=0,
            just_once=False,
    ):
        super().__init__(
            partial(
                RateLimitLogger,
                interval_seconds=interval_seconds, interval_count=interval_count, just_once=just_once
            ))
