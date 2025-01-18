from .core import *


def __create_cache_file():
    import os.path as osp
    import json
    module_root = osp.dirname(__file__)
    cache_dir = osp.join(module_root, '.cache.json')

    if not osp.exists(cache_dir):
        with open(cache_dir, 'w') as writer:
            json.dump({}, writer)

__create_cache_file()
