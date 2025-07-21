# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : _logic
 Created   : 2025/7/17 20:55
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import logging
from enum import Enum, auto
from abc import ABC, abstractmethod,ABCMeta
from typing import Iterable, Callable, Union
from functools import wraps

import sympy as sp


# Enum types
class Chiral(Enum):
    UnSpecified = 0
    CIS = 1
    TRANS = -1

################################################
#################################################

############### Definition of Sets ###################
def value_check(ope_method: Callable):
    if hasattr(ope_method, '__value_check_has_wrapped__'):
        return ope_method

    @wraps(ope_method)
    def wrapper(self, other):
        logging.info(f"{self.__class__.__name__} instance could invoke the value_check wrapper")
        if isinstance(other, sp.Set):
            return ope_method(self, other)

        if set(other) - self.field:
            raise ValueError(
                'All elements of the given other set should be in the defined field:\n'
                f'In the {ope_method.__qualname__} method, the field:\n'
                f'{self.field}'
            )
        return ope_method(self, other)

    wrapper.__value_check_has_wrapped__ = True
    return wrapper


class BinaryLogicMeta(ABCMeta):
    _binary_logic_operator = {
        '__and__', '__or__', '__sub__'
    }

    def __new__(mcs, name, bases, attrs):
        logging.info(f'Manufacture: {name}')
        cls = super(BinaryLogicMeta, mcs).__new__(mcs, name, bases, attrs)
        for op_name in mcs._binary_logic_operator:
            setattr(cls, op_name, value_check(getattr(cls, op_name)))
        return cls


class FieldSetABC(ABC):
    def __init__(self, __iterable: Iterable):
        super().__init__(__iterable)
        self.check_all_values_in_field()

    def __invert__(self):
        return self.__class__(self.field - self)

    @abstractmethod
    def __sub__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __and__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __or__(self, other):
        raise NotImplementedError

    @property
    @abstractmethod
    def set_base_type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def field(self):
        raise NotImplementedError('The field set is not implemented')

    @abstractmethod
    def is_empty(self):
        raise NotImplementedError

    def check_all_values_in_field(self):
        if not hasattr(self, 'field'):
            raise NotImplementedError(
                "The attribute 'field' has not been defined, the subclass of FieldSet should specify it")
        if not isinstance(self.field, self.set_base_type):
            raise ValueError(f'The "field" should be a {self.set_base_type} instance')
        if self - self.field:
            raise AttributeError('All elements in the Field set must be in the defining field')


class SpFieldSet(sp.Set, FieldSetABC, ABC, metaclass=BinaryLogicMeta):
    set_base_type = sp.Set
    def is_empty(self):
        return isinstance(self, sp.EmptySet),


class RealsSet(SpFieldSet):
    field = sp.Reals


class FieldSet(set, FieldSetABC, ABC, metaclass=BinaryLogicMeta):
    set_base_type = set
    def is_empty(self):
        return len(self) == 0


class AtomicNumSet(FieldSet):
    field = set(range(120))

class BoolSet(FieldSet):
    field = {False, True}

class ChiralSet(FieldSet):
    field = set(Chiral)

########################################################################
########################################################################


################# LogicDict ##############################
class LogicDict(dict):
    def __init__(self, *args, default_factory=None, **kw):
        super().__init__(*args, **kw)
        # Check whether all value in the dict is a FieldSet
        assert all(isinstance(s, FieldSetABC) for s in self.values())
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        value = self.default_factory
        self[key] = value
        return value

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

        return self.__class__(and_attr)

    # def __invert__(self):
    #     raise self.__class__({k: ~v for k, v in self.items()})

    def __sub__(self, other: "LogicDict"):
        ...


    def __or__(self, other: Union['LogicDict', 'LogicTuple']) -> Union["LogicTuple", "LogicDict"]:
        if isinstance(other, (LogicDict, AndTuple)):
            return OrTuple((self, other))
        elif isinstance(other, OrTuple):
            return OrTuple((self,) + other)
        else:
            raise TypeError(f"Unsupported type {type(other)}")


class LogicTuple(tuple, ABC):
    def __new__(cls, *args, **kwargs):
        super().__new__(cls, *args, **kwargs)

    def __init__(self, _iterable: Iterable, *, auto_reduce: bool = True):
        super().__init__(_iterable)
        self.check_value_types()
        self.auto_reduce = auto_reduce

    def check_value_types(self):
        for v in self:
            if not isinstance(v, (LogicDict, LogicTuple)):
                raise TypeError(
                    f'All elements in the LogicTuple should be LogicDict, '
                    f'LogicTuple instances, got {v.__class__.__name__}'
                )

    @abstractmethod
    def __or__(self, other):
        raise NotImplementedError
    @abstractmethod
    def __and__(self, other):
        raise NotImplementedError
    @abstractmethod
    def __sub__(self, other):
        raise NotImplementedError
    @abstractmethod
    def reduce(self):
        raise NotImplementedError


class OrTuple(LogicTuple):
    def __or__(self, other):
        if isinstance(other, (LogicDict, AndTuple)):
            return self + OrTuple(other,)
        elif isinstance(other, OrTuple):
            return self + other


class AndTuple(LogicTuple):
    ...


if __name__ == '__main__':
    an1 = AtomicNumSet([2, 3, 7])
    an2 = AtomicNumSet([2, 3, 10])
