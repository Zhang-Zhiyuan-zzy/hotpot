# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : bools
 Created   : 2025/7/14 20:37
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from abc import ABC, abstractmethod
from numbers import Number
from typing import Union, Iterable, Hashable, Any, Optional, override
from copy import copy
from functools import reduce
from enum import Enum, auto

import sympy as sp
from hotpot.cheminfo.search.fields import *

sand = lambda _iterable: reduce(lambda x, y: x & y, _iterable)
sand.__doc__ = """ A sum operation for `&` operation, make sure all elements in the _iterable supports the `&` """
sor = lambda _iterable: reduce(lambda x, y: x | y, _iterable)
sor.__doc__ = """ A sum operation for `|` operation, make sure all elements in the _iterable supports the `|` """


def set_replace_add(base: Union[set, 'LogicSet'], other: Iterable):
    """
    Add elements in other into base.
    If an element existed both in other and base, the one in the base will
    be replaced by the one in the other, which is different from the set
    built-in method set.add(other)
    """
    other = set(other)
    return (base - other) | other


class MutexValue(ABC):
    def __init__(self, value):
        self._value_check(value)
        self.value = value
    def __eq__(self, other):
        return self.__class__ is other.__class__
    def __hash__(self):
        return hash(self.__class__)
    def __repr__(self):
        return f'{self.__class__.__name__}({self.value})'
    def __gt__(self, other):
        return hash(self) > hash(other)
    @abstractmethod
    def _value_check(self, value):
        raise NotImplementedError


class MutexBool(MutexValue):
    def _value_check(self, value):
        if not isinstance(value, bool):
            raise TypeError(f'The value should be bool, not {type(value)}')
    def __bool__(self):
        return bool(self.value)


class LogicSet:
    def __init__(
            self,
            _values: Optional[Any] = None,
            *,
            not_flag=False,
            field: Optional[Field] = None
    ):
        self.sp_set, self.set = self._create_values_set(_values)
        self.not_flag = not_flag

    def __new__(cls, _values: Optional[Any] = None):
        if isinstance(_values, cls):
            return cls._create_from_sets(copy(_values.sp_set), copy(_values.set))
        return super().__new__(cls)

    def __repr__(self):
        return f'{self.__class__.__name__}(Num={self.sp_set}, Other={self.set})'

    def __invert__(self):
        self.not_flag = not self.not_flag
        return self

    @classmethod
    def _create_from_sets(cls, sp_set: sp.Set, _set: set):
        obj = cls()
        obj.sp_set, obj.set = sp_set, _set
        return obj

    @staticmethod
    def _create_values_set(values) -> (sp.Set, set):
        if values is None:
            return sp.EmptySet, set()

        elif (    # The target structure
                isinstance(values, tuple) and
                len(values) == 2 and
                isinstance(values[0], sp.Set) and
                isinstance(values[1], set)
        ):
            return values

        elif isinstance(values, sp.Set):
            return values, set()
        elif isinstance(values, Number):
            return sp.FiniteSet(values), set()
        elif isinstance(values, (str, MutexValue)) or not isinstance(values, Iterable):
            return sp.EmptySet, {values}
        else:  # For Iterable objs
            num, non_num = [], []
            for v in values:
                if isinstance(v, Number):
                    num.append(v)
                else:
                    if isinstance(v, MutexValue) and v in non_num:
                        non_num.remove(v)
                    non_num.append(v)
            return sp.FiniteSet(num), set(non_num)

    def add(self, e):
        if isinstance(e, sp.Set):
            self.sp_set |= e
        elif isinstance(e, Number):
            self.sp_set |= {e}
        elif isinstance(e, MutexValue):
            shared_set = {e} & self.set
            if not shared_set:
                self.set.add(e)
            else:
                existed = list(shared_set)[0]
                if existed.value != e.value:
                    raise AttributeError('Same MutexValue with different values cannot exist in a same set')
        else:
            self.set.add(e)

    @property
    def mutex_values(self) -> set[MutexValue]:
        return {v for v in self.set if isinstance(v, MutexValue)}

    @staticmethod
    def shared_mutex_values(
            mutex_value_set1: set[MutexValue],
            mutex_value_set2: set[MutexValue]
    ) -> (set[MutexValue], set[MutexValue], set[MutexValue]):
        """
        Retrieve the sets containing the shared mutex values both in two LogicSet
        The shared values are split into there set, according to their values
        The first: MutexValues with same value in both LogicSet
        The second: value-difference MutexValues in LogicSet1
        The third: value-difference MutexValues in LogicSet2
        """
        shared_mut_v = mutex_value_set1 & mutex_value_set2

        smv1 = shared_mut_v & mutex_value_set1
        smv2 = shared_mut_v & mutex_value_set2

        same_value_smv = {
            smv
            for smv, v1, v2 in zip(sorted(shared_mut_v), sorted(smv1), sorted(smv2))
            if smv.value == v1.value == v2.value
        }

        return same_value_smv, smv1 - same_value_smv, smv2 - same_value_smv

    def is_empty(self) -> bool:
        return bool(self)

    def issubset(self, other: "LogicSet") -> bool:
        return bool(self - other)

    def issuperset(self, other: "LogicSet") -> bool:
        return bool(other - self)

    @staticmethod
    def _extract_other_sets(other: Union[sp.Set, set, 'LogicSet']) -> (set, set, sp.Set):
        if isinstance(other, sp.Set):
            other_set = set()
            other_mut_set = set()
            other_sp_set = other

        elif isinstance(other, set):
            other_set = other
            other_mut_set = {v for v in other if isinstance(v, MutexValue)}
            other_sp_set = sp.EmptySet

        elif isinstance(other, LogicSet):
            other_set = other.set
            other_mut_set = other.mutex_values
            other_sp_set = other.sp_set

        else:
            raise TypeError('The other is not a LogicSet!')

        return other_set, other_mut_set, other_sp_set

    def __contains__(self, item):
        if isinstance(item, MutexValue):
            intersect = {item} & self.set
            assert len(intersect) <= 1
            judge = len(intersect) == 1 and list(intersect)[0].value == item.value
        else:
            judge = (item in self.set) or (item in self.sp_set)

        if self.not_flag:
            return not judge
        return judge

    def __bool__(self):
        return bool(self.sp_set) or bool(self.set)

    def __and__(self, other: Union[sp.Set, set, 'LogicSet']) -> 'LogicSet':
        other_set, other_mut_set, other_sp_set = self._extract_other_sets(other)
        smv_value_same, smv_self_uni, smv_other_uni = self.shared_mutex_values(self.mutex_values, other_mut_set)

        return self._create_from_sets(
            self.sp_set & other_sp_set,
            (self.set & other_set) - (smv_self_uni | smv_other_uni)
        )

    def __or__(self, other: Union[sp.Set, set, 'LogicSet']) -> 'LogicSet':
        other_set, other_mut_set, other_sp_set = self._extract_other_sets(other)
        smv_value_same, smv_self_uni, smv_other_uni = self.shared_mutex_values(self.mutex_values, other_mut_set)

        if smv_self_uni or smv_other_uni:
            raise ValueError(f'Cannot merge two LogicSets containing MutexValues with different `values`')

        return self._create_from_sets(
            self.sp_set | other_sp_set,
            self.set | other_set
        )


    def __sub__(self, other: Union[sp.Set, set, 'LogicSet']) -> 'LogicSet':
        other_set, other_mut_set, other_sp_set = self._extract_other_sets(other)
        smv_value_same, smv_self_uni, smv_other_uni = self.shared_mutex_values(self.mutex_values, other_mut_set)

        return self._create_from_sets(
            self.sp_set - other.sp_set,
            (self.set - other_set) | smv_self_uni
        )

    def __rsub__(self, other):
        self_set, self_mut_set, self_sp_set = self._extract_other_sets(self)
        other_set, other_mut_set, other_sp_set = self._extract_other_sets(other)
        smv_value_same, smv_self_uni, smv_other_uni = self.shared_mutex_values(self.mutex_values, other_mut_set)

        return self._create_from_sets(
            other_sp_set - self_sp_set,
            (other_set - self_set) | smv_other_uni
        )


class SetUnion:
    def __init__(self, *sets: 'LogicSet'):
        assert all(isinstance(s, LogicSet) for s in sets)
        self.sets = sets


class AndDict(dict):
    def __init__(self, *args, **kw):
        _tmp_dict = dict(*args, **kw)
        items = [(k, LogicSet(v)) for k, v in _tmp_dict.items()]
        super().__init__(items)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        value = self.default_factory
        self[key] = value
        return value

    @property
    def default_factory(self) -> LogicSet:
        return LogicSet()

    @staticmethod
    def create_logic_dict(item: Union[dict, LogicSet]) -> "AndDict":
        if isinstance(item, LogicSet):
            raise item
        elif isinstance(item, dict):
            return AndDict(item)
        else:
            raise TypeError('The item is not a LogicSet!')

    @classmethod
    def _directly_create(cls, dict_items: Union[dict[Hashable, LogicSet], tuple[Hashable, LogicSet]]):
        obj = cls()
        obj.update(dict_items)
        return obj

    def difference(
            self,
            other: Union['AndDict', dict, 'LogicTuple'['AndDict']]
    ) -> Union['AndDict', 'LogicTuple']:
        other = self.create_logic_dict(other)
        clone_dict = copy(self)
        for k, logic_set in other.items():
            if k in clone_dict:
                clone_dict[k] = clone_dict[k] - logic_set

        for key in clone_dict.keys():
            if clone_dict[key].is_empty():
                del clone_dict[key]

        return clone_dict

    def intersect(self, other: Union['AndDict', dict]) -> 'AndDict':
        other = self.create_logic_dict(other)
        shared_items = {}
        for k, other_set in other.items():
            if self_set := self.get(k, None):
                if shared_set := (self_set & other_set):
                    shared_items[k] = shared_set
                else:
                    return AndDict()

        return AndDict._directly_create(shared_items)

    def union(self, other: Union['AndDict', dict]):
        other = self.create_logic_dict(other)
        for k, other_set in other.items():
            if k not in self:
                self[k] = other_set
            else:
                self[k] = other_set | self[k]

    def values(self) -> (sp.Set, set):
        return super().values()

    def items(self) -> (Hashable, (sp.Set, set)):
        return super().items()

    def isin(self, key, value):
        if key in self:
            s1, s2 = self[key]
            return value in s2 or value in s1
        return False

    @staticmethod
    def _subsets(d1: Union['AndDict', dict], d2: Union['AndDict', dict]):
        """ Whether the d2 is the subset of d1"""
        # The key of dict2 must in dict1
        if any(k not in d1 for k in d2):
            return False

        if not isinstance(d1, AndDict):
            d1 = AndDict(d1)
        if not isinstance(d2, AndDict):
            d2 = AndDict(d2)

        # The LogicSet of dict2[k] must be subset of the LogicSet of dict1[k]
        return all(s2.issubset(d1[k2]) for k2, s2 in d2.items())

    def subsets(self, other: Union['AndDict', dict]):
        return self._subsets(self, other)

    def supersets(self, other: Union['AndDict', dict]):
        return self._subsets(other, self)

    def __or__(self, other: Union['AndDict', dict]) -> 'AndDict':
        clone_dict = copy(self)
        clone_dict.union(other)
        return clone_dict

    def __and__(self, other: Union['AndDict', dict]) -> 'AndDict':
        return self.intersect(other)

    def __sub__(self, other: Union['AndDict', dict]) -> 'AndDict':
        return self.difference(other)

    def __invert__(self):
        for v in self:
            self[v] = ~self[v]
        return self


class LogicTuple(tuple, ABC):
    @staticmethod
    def reduce():
        raise NotImplementedError


class AndTuple(LogicTuple):
    def __contains__(self, item: Union[dict, AndDict]):
        item = AndDict(item)
        return all(item in d for d in self)

class OrTuple(LogicTuple):
    @override
    def __init__(self, __iterable: Iterable):
        self._check_values(__iterable)
        super(OrTuple, self).__init__(__iterable)

    def __contains__(self, item: Union[dict, AndDict]):
        item = AndDict(item)
        return any(item in d for d in self)

    @staticmethod
    def _check_values(values: Iterable[AndDict]):
        assert all(isinstance(v, AndDict) for v in values)

    def reduce(self) -> 'OrTuple':
        """
        Reduce the AndDict items to be irreducible:
        (d0, d1, d2, ...) -> (d0, d1-d0, d2-d1-d0, ...)
        Where, d0, d1, ... is AndDict items in the OrTuple.
        if any d0
        """
        reduced = [self[0]]


    def __or__(self, other: Union[tuple[AndDict], 'OrTuple'[AndDict], AndDict, dict]):
        if isinstance(other, dict):
            other = OrTuple((AndDict(other)))
        elif isinstance(other, tuple):
            if not all(isinstance(v, AndDict) for v in other):
                raise TypeError('all values in tuple must be AndDict')
            other = OrTuple(other)
        else:
            raise TypeError('The other item in the or`|` operation just allow to be dict and tuple')

        return self + other

if __name__ == '__main__':
    logic_dict = AndDict()
    logic_dict['atomic_number'].add(1)
    logic_dict['charge'].add(sp.Interval(-2, 3))

    print(1 in logic_dict['atomic_number'])
    inv_logic_dict = ~logic_dict
    print(1 in logic_dict['atomic_number'])
