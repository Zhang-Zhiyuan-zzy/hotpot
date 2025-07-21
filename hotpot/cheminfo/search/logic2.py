# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : logic2
 Created   : 2025/7/18 14:30
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import sympy as sp


class LogicSet:
    def __init__(self, _set, field):
        if isinstance(_set, sp.Set):
            self.set = _set
        else:
            self.set = sp.FiniteSet(_set)

        assert not self.set - field
        self.field = field

    def __bool__(self):
        return bool(self.set)

    def __and__(self, other):
        assert self.field == other.field
        return self.__class__(self.set & other.set, self.field)

    def __or__(self, other):
        assert self.field == other.field
        return self.__class__(self.set | other.set, self.field)

    def __sub__(self, other):
        assert self.field == other.field
        return self.__class__(self.set - other.set, self.field)

    def __invert__(self):
        return self.__class__(self.set - self.field, self.field)


class ProductDict:
    def __init__(self, *args, **kwargs):
        _dict = dict(*args, **kwargs)
        self._keys = list(_dict.keys())

        # Check ...
        assert all(isinstance(v, LogicSet) for v in self)

    def __and__(self, other):
        attrs = set(self.keys()) | set(other.keys())
        and_attr = {}
        for attr in attrs:
            self_set = self.get(attr, None)
            other_set = other.get(attr, None)
            if self_set is None:
                assert other_set is not None
                and_attr[attr] = other_set

            elif other_set is None:
                assert self_set is not None
                and_attr[attr] = self_set

            else:
                and_set = self_set & other_set
                if and_set:
                    and_attr[attr] = and_set
                else:
                    return self.__class__()

    def __invert__(self):
        ...
