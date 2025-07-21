# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : fields
 Created   : 2025/7/17 19:57
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from abc import ABC
from enum import Enum, auto, EnumMeta

fields = []
def register_field(field_type):
    if not issubclass(field_type, Field):
        raise TypeError(f'register type should be subclass of Field, not {field_type}')
    fields.append(field_type)
    return field_type


class Field(Enum): ...

@register_field
class Chiral(Field):
    CIS = auto
    TRANS = auto()

class BoolFieldMeta(EnumMeta):
    def __contains__(cls, item):
        return isinstance(item, bool) or super().__contains__(item)

@register_field
class BoolField(Field, metaclass=BoolFieldMeta):
    TRUE = True
    FALSE = False


def get_field(value):
    for field in fields:
        try:
            if value in field:
                return field
        except TypeError:
            continue

    raise RuntimeError(f'field {value} not found in:\n{fields}')


if __name__ == '__main__':
    print(list(BoolField))
    print(True in BoolField)
    print(get_field(True))
