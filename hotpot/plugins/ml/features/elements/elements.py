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

__all__ = ['element_features']


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


def determine_elements_block(atom_numbers: pd.Series):
    num_period = [2, 8, 8, 18, 18, 32, 32]
    names = ['groups', 'is_main', 'trans_metal', 'lanthanide', 'actinide']
    values = []
    for atom_num in atom_numbers:
        atom_num = int(atom_num)
        for i, p in enumerate(num_period, 1):
            atom_num = atom_num - p
            if atom_num <= 0:
                atom_num = atom_num + p
                if i <= 3:
                    group = atom_num if atom_num <= 2 else atom_num + 10
                    is_main = 1
                    trans_metal = 0
                    lanthanide = 0
                    actinide = 0
                elif 3 < i <= 5:
                    group = atom_num
                    is_main = 1 if atom_num <= 2 or atom_num > 12 else 0
                    trans_metal = 0 if is_main else 1
                    lanthanide = 0
                    actinide = 0
                elif 5 < i <= 7:
                    group = atom_num if atom_num <= 2 else 3 if atom_num <= 17 else atom_num - 14
                    is_main = 1 if atom_num <= 2 or atom_num > 26 else 0
                    trans_metal = 1 if 17 < atom_num <= 26 else 0
                    lanthanide = 0 if is_main or trans_metal or i == 7 else 1
                    actinide = 0 if is_main or trans_metal or i == 6 else 1
                else:
                    raise ValueError("The elements upper than 7 is not discovered!")

                values.append([group, is_main, trans_metal, lanthanide, actinide])
                break

    return pd.DataFrame(values, columns=names, index=atom_numbers.index)


element_features = ElementFeature()


if __name__ == "__main__":
    # Am = ele['Am']
    data = pd.read_pickle(os.path.join(os.path.dirname(__file__), '_elements.pd.xz'))
    # df_ = determine_elements_block(ChemData['atomic_number'])
    # ChemData = pd.concat([df_, ChemData], axis=1)
    # ChemData.to_pickle(os.path.join(os.path.dirname(__file__), '_elements.pd.xz'))
