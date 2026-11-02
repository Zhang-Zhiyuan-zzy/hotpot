"""Source and format mechanics shared by the reader and writer operations."""

import io
import os
from os import PathLike
from pathlib import Path
from typing import Literal


class IoBase:
    """
    Provides functionality for handling input data sources of various types, determining their
    formats, and processing them in a unified manner.

    The IoBase class allows users to initialize with different types of data sources, validate their
    types, and determine the format based on the input. It includes checks for compatibility, format
    deduction, and additional options for further customization.

    Attributes:
    src: The input source that is processed. Can be of various types, including file paths,
         strings, bytes, and stream objects.
    src_type: Represents the type of the input source, determined during initialization.
    fmt: The format of the input source, auto-determined or explicitly passed by the user.
    ob_opt: A dictionary of custom options passed during initialization for operational behavior.
    kwargs: Any additional arguments passed for extended functionality.

    Methods:
    __init__: Initializes the IoBase instance by processing the input source, determining its
              type, and setting the format.
    _src_checks: A static method to determine the type of the input source.
    _determine_fmt: Determines the format of the input source based on its type.

    Raises:
    TypeError: Raised by _src_checks method if the type of src is unsupported.
    ValueError: Raised by _determine_fmt if the format of the source cannot be determined.
    """
    def __init__(self, src, fmt=None, **kwargs):
        self.src = src
        self.src_type = self._src_checks(src)
        self.fmt = self._determine_fmt(src, fmt)
        self.ob_opt = kwargs.pop('ob_opt', {})
        self.kwargs = kwargs

    @staticmethod
    def _src_checks(src) -> Literal['path', 'str', 'bytes', 'StringIO', 'BytesIO', 'FileIO', 'None']:
        if src is None:
            return 'None'

        if isinstance(src, str):
            return 'path' if os.path.isfile(src) else 'str'
        elif isinstance(src, PathLike):
            return 'path'
        elif isinstance(src, bytes):
            return 'bytes'
        elif isinstance(src, io.StringIO):
            return 'StringIO'
        elif isinstance(src, io.BytesIO):
            return 'BytesIO'
        elif isinstance(src, io.FileIO):
            return 'FileIO'
        else:
            raise TypeError(f'get unsupported input type {type(src)}')

    def _determine_fmt(self, src, fmt):
        if not fmt:
            if isinstance(src, (str, PathLike)):
                p_src = Path(src)
                fmt = p_src.suffix[1:]

                if not fmt:
                    fmt = "smi"

        if not fmt:
            raise ValueError('the fmt has not been known!')

        return fmt
