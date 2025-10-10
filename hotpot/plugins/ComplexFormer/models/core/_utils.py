# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : utils
 Created   : 2025/6/10 15:31
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import torch


__all__ = [
    'complete_graph_generator',
    'split_padding',
    'split_padding_deploy',
    'seq_absmax_pooling'
]


def complete_graph_generator(ptr):
    return torch.cat(
        [torch.combinations(torch.arange(ptr[i], ptr[i+1]), with_replacement=True) for i in range(len(ptr)-1)]
    ).T.to(ptr.device)


def split_padding_deploy(x: torch.Tensor, nums: torch.Tensor):
    """
    A fully symbolic function to convert a packed sequence tensor (PyG Batch style)
    into a padded batch tensor.

    This version is compatible with ONNX export by replacing `repeat_interleave`
    with an equivalent implementation using more basic, ONNX-supported operators.

    Args:
        x (torch.Tensor): A packed tensor of features, with shape
                          [total_elements, feature_dim]. This is the result
                          of batching graphs in PyG.
        nums (torch.Tensor): A 1D tensor containing the number of nodes (length)
                             for each graph in the batch, with shape [batch_size].
        max_len (int): The maximum length of the rings sequence.

    Returns:
        padded_X (torch.Tensor): The resulting padded batch tensor, with shape
                                 [batch_size, max_len, feature_dim].
        padding_mask (torch.Tensor): A boolean padding mask, with shape
                                     [batch_size, max_len], where `True`
                                     indicates a padded element.
    """
    device = x.device
    # total_elements = x.shape[0]
    batch_size = nums.shape[0]
    # max_len = torch.max(nums)
    max_len = 32

    # # 1. Symbolically get Max Length (L). This logic remains the same.
    # # max_len = torch.max(nums)
    #
    # # 2. Create `batch_idx` WITHOUT `repeat_interleave`.
    # # This is the core fix. It creates an index like [0, 0, ..., 1, 1, ..., 2, ...]
    # # by comparing an arange tensor with the cumulative sum of lengths.
    # arange_total = torch.arange(total_elements, device=device)
    # # Get the index where each new sequence begins.
    # boundaries = torch.cumsum(nums, 0)[:-1]
    # # The batch index for each element is the number of boundaries it's greater than or equal to.
    # batch_idx = torch.sum(arange_total.unsqueeze(1) >= boundaries.unsqueeze(0), dim=1)
    #
    # # 3. Create `seq_idx` also WITHOUT `repeat_interleave`.
    # # We can reuse the `batch_idx` we just created to achieve this.
    # ends = torch.cumsum(nums, dim=0)
    # starts = torch.cat((torch.tensor([0], device=device), ends[:-1]))
    # # Instead of `starts.repeat_interleave(nums)`, we can simply use advanced indexing
    # # with the `batch_idx` we already computed to get the correct start offset for each element.
    # seq_idx = arange_total - starts[batch_idx]

    # 6. Symbolically create the padding mask. This logic also remains the same.
    # indices = torch.arange(max_len, device=device).expand(batch_size, -1)
    indices = torch.arange(max_len, device=device)
    padding_mask = indices >= nums.unsqueeze(1)

    # 4. Create the destination tensor with symbolic dimensions.
    padded_X = torch.zeros(
        (batch_size, max_len, x.shape[-1]),
        device=device,
        dtype=x.dtype
    )

    # 5. Scatter the data from packed to padded tensor using the computed indices.
    # padded_X[batch_idx, seq_idx] = x
    padded_X[~padding_mask] = x

    return padded_X, padding_mask


def split_padding(x: torch.Tensor, nums: torch.Tensor):
    """
    A fully symbolic function to convert a packed sequence tensor (PyG Batch style)
    into a padded batch tensor.

    This function is designed for `torch.export` and avoids all graph-breaking
    operations like `.tolist()` or `.cpu()`.

    Args:
        x (torch.Tensor): A packed tensor of features, with shape
                          [total_elements, feature_dim]. This is the result
                          of batching graphs in PyG.
        nums (torch.Tensor): A 1D tensor containing the number of nodes (length)
                             for each graph in the batch, with shape [batch_size].

    Returns:
        padded_X (torch.Tensor): The resulting padded batch tensor, with shape
                                 [batch_size, max_len, feature_dim].
        padding_mask (torch.Tensor): A boolean padding mask, with shape
                                     [batch_size, max_len], where `True`
                                     indicates a padded element.
    """
    device = x.device

    # 1. Symbolically get Batch Size (B) and Max Length (L).
    # These are now symbolic `SymInts`, not concrete Python integers.
    batch_size = nums.shape[0]
    max_len = torch.max(nums)

    # 2. Symbolically construct the indices required for scattering.
    # `batch_idx` indicates which batch item each element in `x` belongs to.
    # e.g., [0, 0, 0, 1, 1, 2, 2, 2, 2, ...]
    batch_idx = torch.arange(batch_size, device=device).repeat_interleave(nums)

    # `seq_idx` indicates the position of each element within its own sequence.
    # This is calculated using the cumulative sum trick.
    # e.g., [0, 1, 2, 0, 1, 0, 1, 2, 3, ...]
    ends = torch.cumsum(nums, dim=0)
    starts = torch.cat((torch.tensor([0], device=device), ends[:-1]))
    seq_idx = torch.arange(x.shape[0], device=device) - starts.repeat_interleave(nums)

    # 3. Create the destination tensor with symbolic dimensions.
    padded_X = torch.zeros(
        batch_size, max_len, x.shape[-1],
        device=device,
        dtype=x.dtype
    )

    # 4. Use advanced indexing to "scatter" the data from the packed tensor `x`
    # into the padded tensor `padded_X`. This is the out-of-place, symbolic
    # equivalent of a for-loop with in-place assignment.
    padded_X[batch_idx, seq_idx] = x

    # 5. Symbolically create the padding mask.
    # `indices` is a tensor like [[0, 1, ..., L-1], [0, 1, ..., L-1], ...]
    indices = torch.arange(max_len, device=device).expand(batch_size, -1)

    # The mask is `True` wherever the index is greater than or equal to the
    # actual sequence length specified in `nums`.
    padding_mask = indices >= nums.unsqueeze(1)

    return padded_X, padding_mask

def __split_padding(x: torch.Tensor, nums: torch.Tensor):
    # B = nums.shape[0]
    # L = torch.max(nums)
    # D = x.shape[-1]

    device = x.device
    split_x = list(torch.split(x, nums.tolist()))
    padded_X = torch.nn.utils.rnn.pad_sequence(split_x, batch_first=True)

    # padded_X = torch.zeros((B, L, D)).to(device).to(x.dtype)

    indices = torch.arange(padded_X.shape[1]).to(device)

    nums = nums.unsqueeze(1)
    padding_mask = indices < nums

    # for i, (msk, split_x) in enumerate(zip(padding_mask, split_x)):
    #     padded_X[i][msk] = split_x
    return padded_X, torch.logical_not(padding_mask)


def _split_padding(x: torch.Tensor, nums: torch.Tensor):
    """
    Split X in PyG-style batch and padding.
    :param x: PyG-style batch node vectors
    :param nums: node numbers in each sample
    :return:
    """
    B = nums.shape[0]
    # L = max(nums).item()
    L = torch.max(nums)
    D = x.shape[-1]

    padded_X = torch.zeros((B, L, D)).to(x.device).to(x.dtype)
    padding_mask = torch.ones((B, L), dtype=torch.bool, device=x.device)

    start = torch.zeros(1, dtype=torch.int, device=x.device)
    for i, size in enumerate(nums.long()):
        size = torch.tensor(size, dtype=torch.int, device=x.device)  # For onnx
        i = torch.tensor(i, dtype=torch.int, device=x.device)  # For onnx
        padding_mask[i, :size] = 0
        padded_X[i, :size] = x[start:start + size]
        start += size

    return padded_X, padding_mask

def _seq_absmax_pooling(seq):
    """ Pooling each seq[L, E] to a vec[1, E] """
    pooled_vec = torch.zeros((seq.shape[0], seq.shape[-1])).to(seq.device)
    for i, t in enumerate(seq):
        pooled_vec[i, :] = t.gather(-2, torch.argmax(torch.abs(t), dim=-2).unsqueeze(-2))
    return pooled_vec


def seq_absmax_pooling(seq):
    """ Pooling each seq[B, L, E] to a vec[B, 1, E] """
    X_abs = torch.abs(seq)
    abs_idx = torch.argmax(X_abs, dim=-2).unsqueeze(-2)
    pool_vec = torch.gather(seq, -2, abs_idx)
    return pool_vec.squeeze(-2)