# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : data
 Created   : 2025/6/19 16:12
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 Definition of Custom Data class inheriting from PyG Data
===========================================================
"""
from typing import Any

from torch import Tensor
from torch_geometric.index import Index
from torch_geometric.data import Data


class ExtractionData(Data):
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key.startswith(('sol', 'med')):
            if 'batch' in key and isinstance(value, Tensor):
                if isinstance(value, Index):
                    return value.get_dim_size()

                if value.size(0) == 0:
                    return 1
                else:
                    return int(value.max()) + 1

            elif 'index' in key or key == 'face':
                stores = args[0]
                prefix = key.split('_')[0]

                graph_key = prefix + '_x'
                graph_x = getattr(stores, graph_key)

                return graph_x.size(0)

            else:
                return 0

        else:
            return super(ExtractionData, self).__inc__(key, value, *args, **kwargs)
