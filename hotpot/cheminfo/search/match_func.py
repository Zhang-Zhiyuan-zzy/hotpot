# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : match_func
 Created   : 2025/7/14 10:42
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from enum import Enum, auto
from typing import Iterable, Any, Optional, Union, Callable

from ._logic_tuple import AndTuple, OrTuple, NotTuple, LogicalTuple



class MatcherType(Enum):
    IN_CONTAINER = auto()
    NUM_MIX_MAX = auto()
    NOT_CONTAIN = auto()


class LogicType(Enum):
    AND = auto()
    OR = auto()



def in_matcher(targets: Iterable[Any]):
    def judge(other):
        return other in targets
    return judge


def min_max_matcher(min_value: Optional[int, float] = None, max_value: Optional[int, float] = None):
    if min_value is None and max_value is None:
        raise ValueError("The `min_value` and `max_value` should be given at least one`")
    elif min_value is None and isinstance(max_value, int):
        return lambda other: other <= max_value
    elif min_value is None and isinstance(max_value, float):
        return lambda other: other < max_value
    elif isinstance(min_value, int) and max_value is None:
        return lambda other: min <= other
    elif isinstance(min_value, int) and isinstance(max_value, int):
        return lambda other: min <= other <= max_value
    elif isinstance(min_value, int) and isinstance(max_value, float):
        return lambda other: min <= other < max_value
    elif isinstance(min_value, float) and max_value is None:
        return lambda other: min_value < other
    elif isinstance(min_value, float) and isinstance(max_value, int):
        return lambda other: min_value < other <= max_value
    elif isinstance(min_value, float) and isinstance(max_value, float):
        return lambda other: min_value < other < max_value
    else:
        raise TypeError(f"`min_value` and `max_value` should be `int` or `float`")


def logic_merge(obj, logic_group: list[Union[list, tuple, bool, Callable, LogicalTuple]]) -> bool:
    if isinstance(logic_group, bool):
        return logic_group
    elif isinstance(logic_group, Callable):
        return logic_group(obj)
    elif isinstance(logic_group, (tuple, AndTuple)):
        return all(logic_merge(obj, g) for g in logic_group)
    elif isinstance(logic_group, (list, OrTuple)):
        return any(logic_merge(obj, g) for g in logic_group)
    elif isinstance(logic_group, NotTuple):
        assert isinstance(logic_group, (bool, Callable))
        if isinstance(logic_group, bool):
            return not logic_group
        else:
            raise not logic_group(obj)
    else:
        raise TypeError(f"`logic_group` should be `bool`, `Callable`, `tuple`, `list`")




