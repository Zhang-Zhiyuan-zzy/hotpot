"""
python v3.9.0
@Project: hotpot
@File   : _base
@Auther : Zhiyuan Zhang
@Data   : 2023/10/14
@Time   : 15:47
"""
import json

from abc import ABC
from typing import *

from openbabel import openbabel as ob


# New OB data Wrapper
class Wrapper(ABC):
    """
    A python wrapper for invoke chemical information saved on OpenBabel objects.
    This class provide an ensemble methods to covert variable data type to str,
    and save the str data into the OBCommentData item of OpenBabel.
    """
    _registered_ob_comment_data = {}
    _conv = ob.OBConversion()

    def __init__(self, ob_obj):
        self._obj = ob_obj

    def _get_ob_comment_data(self, data_name: str) -> Optional[str]:
        """ Retrieve OBCommentData according to specific data_name """
        comment = self._obj.GetData(data_name)
        if comment:
            comment = ob.toCommentData(comment)
            return comment.GetData()
        return None

    def _set_ob_comment_data(self, attr_name: str, value: str):
        """ Set the OBCommentData for ob_obj """
        comment_data = ob.OBCommentData()

        comment_data.SetAttribute(attr_name)
        comment_data.SetData(value)

        if self._obj.HasData(attr_name):
            self._obj.DeleteData(attr_name)

        self._obj.CloneData(comment_data)

    def _get_ob_bool_data(self, attr_name: str) -> bool:
        value = self._get_ob_comment_data(attr_name)
        if value == "True":
            return True
        elif value == "False":
            return False

    def _get_ob_float_data(self, attr_name: str) -> float:
        """ Retrieve custom attribute with float value """
        value = self._get_ob_comment_data(attr_name)
        if value:
            return float(value)

    def _get_ob_int_data(self, attr_name: str) -> int:
        """ Retrieve custom attribute with int value """
        value = self._get_ob_comment_data(attr_name)
        if value:
            return int(value)

    def _get_ob_list_data(self, attr_name: str) -> list:
        value = self._get_ob_comment_data(attr_name)
        if value:
            return json.loads(value)

    def _set_ob_bool_data(self, attr_name: str, value: bool):
        """ set custom attribute with bool value """
        if not isinstance(value, bool):
            raise TypeError(f'the given value must be a float, got {type(value)} instead')

        if value:
            self._set_ob_comment_data(attr_name, 'True')
        else:
            self._set_ob_comment_data(attr_name, 'False')

    def _set_ob_float_data(self, attr_name: str, value: float):
        """ set custom attribute with float value """
        if not isinstance(value, float):
            raise TypeError(f'the given value must be a float, got {type(value)} instead')

        self._set_ob_comment_data(attr_name, str(value))

    def _set_ob_int_data(self, attr_name: str, value: int):
        """ set custom attribute with int value """
        if not isinstance(value, int):
            raise TypeError(f'the given value must be an int, got {type(value)} instead')

        self._set_ob_comment_data(attr_name, str(value))

    def _set_ob_list_data(self, attr_name, value: list):
        if not isinstance(value, list):
            raise TypeError(f"the given value must be list, got {type(value)} instead")

        str_value = json.dumps(value)
        self._set_ob_comment_data(attr_name, str_value)


class ObjCollection(ABC):
    """ Representing a collection of Chemical objects, like Molecule, Atom, Ring, Crystal and so on """
    def __init__(self, *objs):
        self._objs = objs

    def __repr__(self):
        return "[" + ", ".join(str(obj) for obj in self._objs) + "]"

    def __contains__(self, item):
        return item in self._objs

    def __len__(self):
        return len(self._objs)

    def __iter__(self):
        return iter(self._objs)

    def __getitem__(self, item: int):
        return self._objs[item]
