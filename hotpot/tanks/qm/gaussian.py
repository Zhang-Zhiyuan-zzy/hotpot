"""
python v3.9.0
@Project: hotpot
@File   : gaussian
@Auther : Zhiyuan Zhang
@Data   : 2023/7/19
@Time   : 2:28

TODO: New implementation for running Gaussian
TODO: How to set options to be both convenient for input and easy to debug
"""
import os
import json
from typing import *


class _OptionValues:
    def __init__(self, values: list):
        self.values = values

    def __repr__(self):
        return f"{self.values}"

    def __dir__(self) -> Iterable[str]:
        return self.values

    def __getattr__(self, item):
        if item not in self.values:
            raise ValueError(f'The option {self.option} does not have the value {item}')


class _Option:
    """ The representation of certain Gaussian option """
    def __init__(
            self,
            raw_options: dict,
            title: Literal['link0', 'route'],
            keyword: str,
            option: str = None,
            value: Any = None
    ):
        self._options = raw_options

        self.title = title
        self.keyword = keyword
        self.option = option
        self.value = value

    def __dir__(self) -> Iterable[str]:
        if isinstance(self.value, _OptionValues):
            return self.value.values
        else:
            return []

    def __getattr__(self, item):
        """
        The values of some options are fixed, and they can be obtained through attribute selection to
        avoid conflicts when writing options.
        For example, the value of route->opt->Algorithm is choose from: GEDIIS, RFO and EF, one may want
        to apply the RFO as the optimization algorithm. To do so, the one just to choose the option by:

            gauss = Gaussian(...)
            gauss.options.route.opt.Algorithm.RFO  # select RFO as the optimizing algorithm

        the Gaussian instance will record the options immediately.

        Sometimes, you could need to handle the GaussError by change the RFO to other method, say GEDIIS.
        You can do this change just by:

            gauss.options.route.opt.Algorithm.GEDIIS

        the Gaussian instance will change the optimizing algorithm to GEDIIS
        """
        if not isinstance(self.value, _OptionValues) or item not in self.value.values:
            raise ValueError(f'The option {self.option} does not have the value {item}')

        key = self.title + f'.{self.keyword}' + (f".{self.option}" if self.option else "")
        self._options[key] = self

    def __repr__(self):
        return f"{self.keyword}" \
               + (f"={self.option}" if self.option else "") \
               + (f"({self.value})" if self.value else "")

    def __hash__(self):
        key = f"{self.title}" + f".{self.keyword}" \
               + (f".{self.option}" if self.option else "") \

        return hash(key)

    def __eq__(self, other):
        return self.title == other.title and self.keyword == other.keyword and self.option == other.option

    def __call__(self, value=None):
        if self.value is None and value is not None:
            raise ValueError(f"the option {self.keyword}.{self.option} doesn't have any value")

        elif self.value == 'float' or isinstance(self.value, float):
            if isinstance(value, float):
                self.value = value
            else:
                raise TypeError('the input value should be a float')

        elif self.value == "int" or isinstance(self.value, int):
            if isinstance(value, int):
                self.value = value
            else:
                raise TypeError('the input value should be an int')

        elif self.value == 'str' or isinstance(self.value, str):
            if isinstance(value, str):
                self.value = value
            else:
                raise TypeError('the input value should be a string')

        else:
            raise NotImplemented

        key = self.title + f'.{self.keyword}' + (f".{self.option}" if self.option else "")
        self._options[key] = self


class _Options:
    """
    A handle to the link0 and route options.
    This class should be initialized by `Gaussian.options` attributes, initializing directly is not recommended.
    """
    from hotpot import data_root
    _path_option_json = os.path.join(data_root, 'goptions.json')

    _tree: dict = json.load(open(_path_option_json))

    def __init__(self, raw_options: dict, path: str = ""):

        self._paths = path
        self._options = raw_options

    def __repr__(self):
        return self._paths if self._paths else 'RootOptions'

    def __dir__(self) -> Iterable[str]:
        if not self._paths:
            return list(self._tree.keys())
        else:
            paths = self._paths.split('.')

            tree = self._tree
            for option in paths:
                tree = tree.get(option)

            if isinstance(tree, list):
                return tree
            elif isinstance(tree, dict):
                return list(tree.keys())
            else:
                return []

    def __getattr__(self, item: str):
        tree = self._tree
        if self._paths:
            for option in self._paths.split('.'):
                tree = tree.get(option)

        if isinstance(tree, dict):
            option = tree.get(item)
        else:
            assert isinstance(tree, list)
            if item in tree:
                option = item
            else:
                raise AttributeError(f'{option} not have option: {item}')

        if isinstance(option, (dict, list)):
            if self._paths:
                return _Options(self._options, f"{self._paths}.{item}")
            else:
                return _Options(self._options, f"{item}")
        else:
            paths = self._paths.split('.')
            if len(paths) == 1:
                return _Option(self, paths[0], option)
            elif len(paths) == 2:
                return _Option(self, paths[0], paths[1], option)
            elif len(paths) == 3:
                return _Option(self, paths[0], paths[1], paths[3])
            else:
                raise AttributeError

    def __call__(self):
        """
        Sometimes, the keywords could be without the options, so call the _Options instance to specify it

        """
        paths = self._paths.split('.')
        if len(paths) == 2:
            option = _Option(self._options, paths[0], paths[1])
            option()


def _cook_raw_options_to_struct_input_dict(options: _Option, raw_options: dict) -> dict:
    """"""
    for op in options:
        title = raw_options.setdefault(op.title, {})
        ops = title.setdefault(op.keyword, [])
        ops.append(op)

    _input = {}
    for title, item in raw_options.items():

        keywords = _input.setdefault(title, {})

        for kwd, ops in item.items():
            ops = [op for op in ops if op.option is not None]

            if not ops:
                keywords[kwd] = None

            elif len(ops) == 1:
                if ops[0].value is None:
                    keywords[kwd] = ops[0].option
                else:
                    keywords[kwd] = {ops[0].option: ops[0].value}

            else:
                options = keywords.setdefault(kwd, {})
                for op in ops:
                    if op.value is None:
                        options[op.option] = None
                    else:
                        options[op.option] = op.value

    return _input