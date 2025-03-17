from typing import Union
import torch
import numpy as np


from hotpot.cheminfo.elements import elements


def oh2label(inp_vec: Union[torch.Tensor, np.ndarray]):
    if isinstance(inp_vec, torch.Tensor):
        return torch.argmax(inp_vec, dim=1)
    elif isinstance(inp_vec, np.ndarray):
        return np.argmax(inp_vec, axis=1)
    else:
        raise TypeError('the input vectors must be of type torch.Tensor or np.ndarray')


_np_metal = np.array(list(elements.metal|elements.metalloid_2nd))
_torch_metal = torch.from_numpy(_np_metal)
def where_metal(type_labels: Union[torch.Tensor, np.ndarray]):
    if isinstance(type_labels, torch.Tensor):
        return torch.isin(type_labels, _torch_metal)
    elif isinstance(type_labels, np.ndarray):
        return np.isin(type_labels, _torch_metal)
