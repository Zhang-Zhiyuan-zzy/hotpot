"""
python v3.9.0
@Project: hotpot
@File   : elements
@Auther : Zhiyuan Zhang
@Data   : 2024/1/6
@Time   : 15:38
"""
import os
import pandas as pd

__all__ = ['ele']


class ElementFeature:
    def __init__(self, *, data=None):
        if data is None:
            self._data = pd.read_pickle(os.path.join(os.path.dirname(__file__), '_elements.pd.xz'))
        elif isinstance(data, pd.DataFrame):
            self._data = data
        elif isinstance(data, pd.Series):
            self._data = pd.DataFrame([data])
        else:
            raise AttributeError

    def __repr__(self):
        return repr(self._data)

    def __getitem__(self, item):
        return self.__class__(data=self._data.loc[item, :])

    def dropna(self):
        return self.__class__(data=self._data.dropna(axis=1))

    @property
    def dataframe(self):
        return pd.DataFrame(self._data)


ele = ElementFeature()


if __name__ == "__main__":
    Am = ele['Am']
