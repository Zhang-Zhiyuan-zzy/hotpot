# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : types
 Created   : 2025/7/14 15:12
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 The definition of logical tuple for being distinguished in programs,
 and applying different operations on them.
===========================================================
"""
from abc import ABC

# Definition of the base class
class LogicalTuple(tuple, ABC): ...


# define logical tuple
class AndTuple(LogicalTuple): ...
class OrTuple(LogicalTuple): ...
class NotTuple(LogicalTuple): ...


__all__ = [name for name, value in globals().items() if 'Tuple' in name and issubclass(value, LogicalTuple)]
