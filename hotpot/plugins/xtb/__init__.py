from .core import *


def __create_cache_file():
    """
    Creates a cache file in the module's directory if it does not already exist.

    The function checks for the existence of a cache file named '.cache.json'
    in the module's root directory. If the file does not exist, it will create
    the file and initialize it with an empty JSON object.

    Returns
    -------
    None
    """
    import os.path as osp
    import json
    module_root = osp.dirname(__file__)
    cache_dir = osp.join(module_root, '.cache.json')

    if not osp.exists(cache_dir):
        with open(cache_dir, 'w') as writer:
            json.dump({}, writer)

__create_cache_file()
