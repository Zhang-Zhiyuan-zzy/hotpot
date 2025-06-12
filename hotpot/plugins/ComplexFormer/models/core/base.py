# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : base
 Created   : 2025/6/10 15:31
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from abc import abstractmethod
from typing import Optional
import torch.nn as nn

class CoreBase(nn.Module):
    """"""
    extractor_class = None
    extractor_keys = {
        'atom': 'extract_atom_vec',
        'bond': 'extract_bond_vec',
        'pair': 'extract_pair_vec',
        'ring': 'extract_ring_vec',
        'mol': 'extract_mol_vec',
        'cbond': 'extract_cbond_pair',
        'metal': 'extract_metal_vec'
    }
    def __init__(self, vec_dim: int, x_label_nums: Optional[int] = None):
        super(CoreBase, self).__init__()
        self.vec_size = vec_dim
        self.x_label_nums = x_label_nums

        self.feature_extractor = {
            key:getattr(self.extractor_class, method_name)
            for key, method_name in self.extractor_keys.items()
            if hasattr(self.extractor_class, method_name)
        }

    @property
    @abstractmethod
    def x_mask_vec(self):
        raise NotImplementedError('the property `x_mask_vec` is not implemented.')
