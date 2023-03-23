"""
python v3.7.9
@Project: hotpot
@File   : datasets.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/22
@Time   : 4:58
"""
from functools import wraps


class DataSets:
    _datasets = {}

    @classmethod
    def register(cls, dataset_name: str):
        """ register a Dataset with a name as the handle for get it easily """
        def decorator(decorated_class: type):
            """ Decorate a Dataset class """
            cls._datasets[dataset_name] = cls

            @wraps(decorated_class)
            def wrapper(*args, **kwargs):
                return decorated_class(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def get(cls, dataset_name: str):
        return cls._datasets.get(dataset_name)


# Just an example give your self dataset names
@DataSets.register('graph')
class Dataset1:
    ...


@DataSets.register('huge_organ')
class Dataset2:
    ...
