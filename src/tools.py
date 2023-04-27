"""
python v3.7.9
@Project: hotpot
@File   : utils.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/29
@Time   : 0:30

Notes:
    Private common functions
"""
from os import PathLike
from pathlib import Path
from typing import *


class PathNotExistError(Exception):
    """ Raise when can't find a file or dir """


def check_path(
        path: Optional[Union[str, PathLike]],
        none_allowed: Optional[bool] = True,
        check_exist: Optional[bool] = False,
        mkdir: Optional[bool] = False,
        file_or_dir: Optional[Literal['file', 'dir']] = None
) -> Union[NoReturn, Path]:
    """
    Check whether the given path is valid and process str path to a Path object.

    Args:
        path (Optional[Union[str, PathLike]]): The path to be checked.
        none_allowed (Optional[bool], default=True): Whether to allow the path to be None.
        check_exist (Optional[bool], default=False): Whether to check if the path exists.
        mkdir (Optional[bool], default=False): Whether to create the directory if it doesn't exist.
        file_or_dir (Optional[Literal['file', 'dir']], default=None): Whether to check if the path is a file or directory.

    Returns:
        Union[None, Path]: The processed path as a Path object or None if allowed and given as None.

    Raises:
        ValueError: If the path is None and not allowed, or if `mkdir=True` and `file_or_dir='file'`.
        TypeError: If the given path is neither a str nor a PathLike object.
        PathNotExistError: If the path doesn't exist and check_exist is True or mkdir is False.
        IsADirectoryError: If the path is a directory and file_or_dir is 'file'.
        NotADirectoryError: If the path is a file and file_or_dir is 'dir'
    """
    # If get the None, keep it or raise!
    if not path:
        if none_allowed:
            return path
        else:
            raise ValueError("the path shouldn't to be None")

    # Check the given path type and transform the str path to Path
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, PathLike):
        raise TypeError(f'the given path should be str or PathLick, instead of{type(path)}')

    is_exist = path.exists() if (check_exist or mkdir or file_or_dir) else None

    if is_exist is False:
        if file_or_dir == 'file' and mkdir:
            raise ValueError("the mkdir=True and file_or_dir='file' can't to be set simultaneously!")

        if mkdir:
            path.mkdir()
        else:
            raise PathNotExistError(f'the path {str(path)} not exist!')

    if file_or_dir == 'file' and not path.is_file():
        raise IsADirectoryError(f'{str(path)} is a directory!')
    if file_or_dir == 'dir' and not path.is_dir():
        raise NotADirectoryError(f'{str(path)} is a file!')

    return path
