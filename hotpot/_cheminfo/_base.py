"""
python v3.9.0
@Project: hotpot
@File   : _base
@Auther : Zhiyuan Zhang
@Data   : 2023/10/14
@Time   : 15:47
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable
from functools import wraps

from openbabel import openbabel as ob


# New OB data Wrapper
class Wrapper(ABC):
    """
    A python wrapper for invoke chemical information saved on OpenBabel objects.
    This class provide an ensemble methods to covert variable data type to str,
    and save the str data into the OBCommentData item of OpenBabel.
    """
    _registered_ob_comment_data = {}
    _obj = None  # the abstract definition for wrapped Openbabel object

    @abstractmethod
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

    def _int_comment_attr(self, attr_setter: Callable):
        """convert the input int data to str and register the given attr_setter.__name__
        to invoke the str->int decoder"""
        @wraps(attr_setter)
        def decoder(value: str) -> int:
            return int(value)

        @wraps(attr_setter)
        def encoder(value: int):
            self._set_ob_comment_data(attr_setter.__name__, str(value))

        class_dict = Wrapper._registered_ob_comment_data.setdefault(self.__class__.__name__, {})
        class_dict[attr_setter.__name__] = decoder

        return encoder
