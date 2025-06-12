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
    'seq_absmax_pooling'
]


def complete_graph_generator(ptr):
    return torch.cat(
        [torch.combinations(torch.arange(ptr[i], ptr[i+1]), with_replacement=True) for i in range(len(ptr)-1)]
    ).T.to(ptr.device)

def split_padding(x: torch.Tensor, nums: torch.Tensor):
    """
    Split X in PyG-style batch and padding.
    :param x: PyG-style batch node vectors
    :param nums: node numbers in each sample
    :return:
    """
    B = nums.shape[0]
    L = max(nums).item()
    D = x.shape[-1]

    padded_X = torch.zeros((B, L, D)).to(x.device)
    padding_mask = torch.ones((B, L), dtype=torch.bool, device=x.device)

    start = 0
    for i, size in enumerate(nums.long()):
        size: int
        padded_X[i, :size] = x[start:start + size]
        padding_mask[i, :size] = 0
        start += size

    return padded_X, padding_mask

def seq_absmax_pooling(seq):
    """ Pooling each seq[L, E] to a vec[1, E] """
    pooled_vec = torch.zeros((len(seq), seq.shape[-1])).to(seq.device)
    for i, t in enumerate(seq):
        pooled_vec[i, :] = t.gather(-2, torch.argmax(torch.abs(t), dim=-2).unsqueeze(-2))
    return pooled_vec