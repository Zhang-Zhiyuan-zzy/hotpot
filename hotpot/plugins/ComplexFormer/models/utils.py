from functools import wraps
import inspect

from typing import Union, Literal, Callable, Optional
import torch
import torch.nn.functional as F
import numpy as np


from hotpot.cheminfo.elements import elements

electron_config_tensor = torch.cat((torch.arange(0, 120).unsqueeze(-1), torch.tensor(elements.electron_configs)), dim=1)

################################ Onehot Encode ###############################################
def norm_binary_to_zero_one(inp_vec: Union[torch.Tensor, np.ndarray]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(inp_vec, np.ndarray):
        inp_vec = torch.from_numpy(inp_vec)
        return np.round(F.sigmoid(inp_vec).numpy())
    elif isinstance(inp_vec, torch.Tensor):
        return torch.round(F.sigmoid(inp_vec))
    else:
        raise TypeError(f"Input type {type(inp_vec)} not supported")

def oh2label(inp_vec: Union[torch.Tensor, np.ndarray]):
    if len(inp_vec.shape) == 1:
        return inp_vec
    elif len(inp_vec.shape) == 2:
        if inp_vec.shape[-1] == 1:
            return inp_vec.flatten()
        else:
            return inp_vec.argmax(dim=-1) if isinstance(inp_vec, torch.Tensor) else inp_vec.argmax(axis=-1)
    else:
        raise AttributeError('The input tensor or vector must have 1 or 2 dimensions.')


def inverse_onehot(is_onehot, *onehot_vecs: Union[torch.Tensor, np.ndarray]):
    if is_onehot:
        inv_vecs = []
        for vec in onehot_vecs:
            if isinstance(vec, torch.Tensor):
                inv_vecs.append(torch.argmax(vec, dim=1).reshape(-1, 1))
            elif isinstance(vec, np.ndarray):
                inv_vecs.append(np.argmax(vec, axis=1).reshape(-1, 1))
            else:
                raise TypeError('the input vectors must be of type torch.Tensor or np.ndarray')
        return tuple(inv_vecs)
    else:
        return onehot_vecs


_np_metal = np.array(list(elements.metal|elements.metalloid_2nd))
_torch_metal = torch.from_numpy(_np_metal)
def where_metal(type_labels: Union[torch.Tensor, np.ndarray]):
    if isinstance(type_labels, torch.Tensor):
        return torch.isin(type_labels, _torch_metal.to(type_labels.device))
    elif isinstance(type_labels, np.ndarray):
        return np.isin(type_labels, _torch_metal)


def weight_labels(
        labels: torch.Tensor,
        num_types: int = 119,
        weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'cross-entropy',
) -> object:
    # labels = torch.argmax(labels, dim=-1)  # Is one hot vector
    labels = labels.long()
    values, counts = torch.unique(labels, return_counts=True)

    if weight_method == 'cross-entropy':
        weight = counts / counts.sum()
        weight = -weight*torch.log(weight)
    elif weight_method == 'inverse-count':
        weight = counts.sum() / counts
        weight = weight / weight.max()
    elif weight_method == 'sqrt-invert_count':
        weight = counts.sum() / counts.sqrt()
        weight = weight / weight.max()
    else:
        raise ValueError('weight_method must be either "inverse-count" or "cross-entropy"')

    onehot_weight = torch.zeros(num_types).to(labels.device).to(weight.dtype)
    onehot_weight[values] = weight

    return onehot_weight

def weight_binary(labels: torch.Tensor, eps: float = 1e-7):
    if not torch.is_floating_point(labels):
        labels = labels.float()

    dtype = labels.dtype

    values, counts = torch.unique(labels, return_counts=True)
    assert len(values) <= 2 and all(v in [0, 1] for v in values)
    weight_true = torch.sum(labels, dtype=labels.dtype) / len(labels)
    weight_false = 1 - weight_true

    weight_true = (weight_true / max(weight_true, weight_false, eps)).to(dtype)
    weight_false = (weight_false / max(weight_true, weight_false, eps)).to(dtype)

    weights = torch.empty_like(labels, dtype=dtype)
    weights[labels == 1] = weight_true
    weights[labels == 0] = weight_false

    return weights

######################################################################################################

####################################### Input Preprocessor ###########################################
def get_x_input_attrs(
        inputs: Union[dict, list, tuple],
        input_x_index: Union[list, torch.Tensor],
        dtype: Optional[torch.dtype] = None,
):
    """ Extract node features from Data.x and convert the Tensor dtype """
    if isinstance(inputs, dict):
        inputs['x'] = inputs['x'][:, input_x_index].to(dtype)
        return inputs
    else:
        x = inputs[0][:, input_x_index].to(dtype)
        return (x,) + inputs[1:]

def get_labeled_x_input_attrs(inputs, input_x_index: Union[list, torch.Tensor]=None):
    return (inputs[0][:, 0],) + inputs[1:]
#############################################################################################



######################## Masker Function ######################################
def _to_mask(
        inp_vec: torch.Tensor,
        masked_idx: torch.Tensor,
        mask: Union[int, torch.Tensor],
        inp_atom_labels: torch.Tensor,
        vocab_sheet: torch.Tensor,
):
    """
    Given an input vector and the index which to be masked, return the masked vector and masked labels.
    Args:
        inp_vec (torch.Tensor): the input vector (i.e., representing vector)
        masked_idx (torch.Tensor): the index to be masked.
        mask (torch.Tensor): The masked label or vector.
        inp_atom_labels (torch.Tensor): the input labels (i.e., targets or labels)
        label_mask (bool）: whether the input vectors are (n-dim) continuous or (1-dim) discrete.
        to_mask_label (int): used when label_mask is True, specify which label index to be the masking label.
    """
    atom_labels = inp_atom_labels[masked_idx]

    # Prepare masked input
    masked_vec = inp_vec.clone()
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    mask2mask_idx = masked_idx & (torch.rand(inp_vec.shape[0]) < 0.90).to(inp_vec.device)
    masked_vec[mask2mask_idx] = torch.as_tensor(mask, dtype=masked_vec.dtype).to(masked_vec.device)

    # Set 10% to a random token
    mask2rand_idx = mask2mask_idx & (torch.rand(inp_vec.shape[0]) < 1 / 9).to(inp_vec.device)
    # masked_vec[mask2rand_idx] = inp_vec[torch.randint(0, len(inp_vec), (torch.sum(mask2rand_idx),))]
    masked_vec[mask2rand_idx] = vocab_sheet[torch.randint(0, vocab_sheet.shape[0], (torch.sum(mask2rand_idx), ))].to(masked_vec.device)

    return masked_vec, atom_labels, masked_idx

metal_index = torch.tensor(list(elements.metal | elements.metalloid_2nd))
def _select_atom_type_index(inp_vec: torch.Tensor, inp_labels: torch.Tensor):
    # 10% atoms to mask
    masked_idx = torch.from_numpy(np.random.uniform(size=inp_vec.shape[0]) < 0.10).to(inp_vec.device)
    # Randomly select 40% metals to mask
    masked_metal_idx = _select_metal_type_index(inp_vec, inp_labels, 0.4)
    return masked_idx | masked_metal_idx

def _select_metal_type_index(inp_vec: torch.Tensor, inp_labels: torch.Tensor, ratio: float = 1.0):
    masked_idx = torch.isin(inp_labels, metal_index.to(inp_vec.device))
    return masked_idx & (torch.rand(masked_idx.shape, device=inp_vec.device) < ratio)


def get_masked_input_and_labels(
        inp_vec: torch.Tensor,
        masked_idx: torch.Tensor,
        mask_vec: torch.Tensor,
        inp_atom_labels: torch.Tensor,
        label_mask: bool = False,
        to_mask_label: int = 0,
):

    return _to_mask(inp_vec, masked_idx, mask_vec, inp_atom_labels, label_mask, to_mask_label)



# Decorator for masker function applying for pretrain workflow
def _masker_func(mask_idx_getter: Callable):
    def decorator(signature_func: Callable):
        define_sign = {'inputs': tuple, 'node_mask': Union[torch.Tensor, int]}
        signature = inspect.signature(signature_func)
        # Check the signature
        for i, ((sn, sp), (dn, dp)) in enumerate(zip(signature.parameters.items(), define_sign.items())):
            if sn != dn:
                raise ValueError(f'The {i}th signature parameter name should be {dn}, but got {sn}')
            # elif not issubclass(sp.annotation, dp):
            #     raise ValueError(f'The {i}th signature parameter type should be {dp}, but got {sp.annotation}')

        @wraps(signature_func)
        def wrapper(inputs: tuple, node_mask: Union[int, torch.Tensor]):
            x = inputs[0]
            if x.ndim == 2:
                inp_labels = x[:, 0].long()
                masked_idx = mask_idx_getter(x, x[:, 0].long())
                vocab_sheet = electron_config_tensor[1:104].int()
            else:
                inp_labels = x
                masked_idx = mask_idx_getter(x, x.long())
                vocab_sheet = torch.arange(1, 104).int()

            masked_x, atom_label, masked_node_idx = (
                _to_mask(x, masked_idx, node_mask, inp_labels, vocab_sheet))

            return (masked_x,) + inputs[1:], masked_node_idx

        # Return of decorator
        return wrapper

    # Return of _masker_func
    return decorator

###################### Masker Signature Functions ##############################################
@_masker_func(_select_atom_type_index)
def mask_atom_type(inputs: tuple, node_mask: Union[int, torch.Tensor]):
    """
    A masker function to mask the atom types of input nodes features.
    Args:
        inputs (tuple): the input node features (i.e., representing node features),
        node_mask (torch.Tensor): the masking vector or label

    Returns:
        (tuple): the masked node features (i.e., representing node features) and other input information.
        (torch.Tensor): the masked index in the label vector.
    """

@_masker_func(_select_metal_type_index)
def mask_metal_type(inputs: tuple, node_mask: Union[int, torch.Tensor]):
    """
    A masker function to mask the metal types of input nodes features.
    Args:
        inputs (tuple): the input node features (i.e., representing node features),
        node_mask (torch.Tensor): the masking vector or label

    Returns:
        (tuple): the masked node features (i.e., representing node features) and other input information.
        (torch.Tensor): the masked index in the label vector.
    """


def x_masker_func(inputs: tuple, masked_vec: torch.Tensor):
    x = inputs[0]
    if x.dim == 2:
        masked_x, atom_label, masked_node_idx = get_masked_input_and_labels(inputs[0], masked_vec, x[:, 0].long())
    else:
        masked_x, atom_label, masked_node_idx = get_masked_input_and_labels(inputs[0], masked_vec, x.long(), label_mask=True)

    return (masked_x,) + inputs[1:], masked_node_idx

def metal_masker_func(inputs: tuple, masked_vec: torch.Tensor):
    x = inputs[0]


######################### Interact with Numpy ################################################
def torch_numpy_exchanger(nf: Callable, **kw):
    # torch-numpy exchanger
    def wrapper(*inputs: Union[torch.Tensor, np.ndarray]):
        return nf(*inputs, **kw)

    return wrapper


########################## Operate XYZ ################################################
PerturbMode = Literal['norm', 'uniform']
def perturb_xyz(xyz: torch.Tensor, sigma: float = 1.0, mode: PerturbMode = 'uniform'):
    if mode == 'norm':
        pert = sigma*torch.randn_like(xyz)
    elif mode == 'uniform':
        pert = sigma*torch.rand_like(xyz)
    else:
        raise ValueError(f'Unknown perturbation mode {mode}')
    return xyz + pert, pert

