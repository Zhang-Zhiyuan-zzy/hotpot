import torch
from torch import nn

import torch_geometric.nn as pyg_nn


__all__ = [
    'CompleteGraph'
]

class CompleteGraph(nn.Module):
    def __init__(
            self,
            vec_dim: int,
            n_layers: int,
            nheads: int
    ):
        super(CompleteGraph, self).__init__()
        self.graph = pyg_nn.GAT(
            vec_dim, vec_dim, n_layers,
            vec_dim, 0.1, norm=pyg_nn.LayerNorm(vec_dim),
            edge_dim=vec_dim, v2=True, heads=nheads,
        )

        self.seq_mol_pooling = pyg_nn.SAGPooling(vec_dim, 1)

    @staticmethod
    def _nums2ptr(nums, offset=0):
        return torch.cat([torch.tensor([0]).to(nums.device), torch.cumsum(nums, 0)]) + offset

    @staticmethod
    def _ptr2nums(ptr):
        return ptr[1:] - ptr[:-1]

    @staticmethod
    def _ptr2complete_edge_index(*ptrs):
        if len(ptrs) == 1:
            ptr = ptrs[0]
            return torch.cat(
                [torch.combinations(torch.arange(ptr[i], ptr[i+1]), 2, True) for i in range(len(ptr) - 1)]
            )
        else:
            # Check whether the length of each ptr is same
            for i in range(len(ptrs) - 1):
                if len(ptrs[i]) != len(ptrs[i+1]):
                    raise AssertionError(f'length of ptr between ptrs[{i}] and ptrs[{i + 1}] is not same')

            # combining them
            return torch.cat([
                torch.combinations(torch.cat([torch.arange(ptr[i], ptr[i + 1]) for ptr in ptrs]), 2, True)
                for i in range(len(ptrs[0]) - 1)
            ])

    @staticmethod
    def _nums2batch(nums):
        return torch.repeat_interleave(torch.arange(len(nums)), nums)

    def _get_complete_graph(self, ptr=None, nums=None):
        """"""
        if ptr is not None:
            nums = self._ptr2nums(ptr)
        elif nums is not None:
            ptr = self._nums2ptr(nums)
        else:
            raise ValueError('The ptr and nums should be provided at least one.')

        edge_index = self._ptr2complete_edge_index(ptr)
        batch = self._nums2batch(nums)
        return edge_index, batch

    def ring_forward(self, x, rings_node_index, rings_node_nums):
        xr = x[rings_node_index]
        ring_edge_index, ring_batch = self._get_complete_graph(nums=rings_node_nums)

        xr = self.graph(xr, ring_edge_index)
        return pyg_nn.global_max_pool(xr, ring_batch)

    def mol_forward(self, x, xr, mol_ring_nums, mol_ptr, mol_batch) -> (torch.Tensor, list[torch.Tensor], list[torch.Tensor]):
        mol_ring_ptr = self._nums2ptr(mol_ring_nums, len(x))
        mol_edge_index = self._ptr2complete_edge_index(mol_ptr, mol_ring_ptr)
        ring_batch = self._nums2batch(mol_ring_nums)

        x = torch.cat([x, xr])
        mol_batch = torch.cat([mol_batch, ring_batch])

        x = self.graph(x, mol_edge_index)
        vec_mol = self.seq_mol_pooling(x, mol_edge_index)

        split_x = torch.split(x, mol_ptr.detach().tolist() + mol_ring_ptr.detach().tolist())

        return vec_mol, split_x[:len(mol_ptr)-1], split_x[len(mol_ptr)-1:]

    def forward(self, *args, **kwargs):
        if len(args) == 3:
            return self.ring_forward(*args, **kwargs)
        elif len(args) == 5:
            return self.mol_forward(*args, **kwargs)
        else:
            raise NotImplementedError("the input of CompleteGraph should be 3 or 5")
