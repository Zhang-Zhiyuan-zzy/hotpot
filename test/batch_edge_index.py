import os
import torch
from torch_geometric.data import Data, Batch

def remove_cbond_edges(batch):
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
    is_cbond = batch.is_cbond

    cbond_indices = torch.nonzero(is_cbond == 1).squeeze()

    if len(cbond_indices) > 0:
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[cbond_indices] = False

        edge_index = edge_index[:, mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

    batch.edge_index = edge_index
    batch.edge_attr = edge_attr if edge_attr is not None else torch.empty((0,))

    return batch
