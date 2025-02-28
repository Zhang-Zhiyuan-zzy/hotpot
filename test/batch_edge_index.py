import os
import torch
from torch_geometric.data import Data, Batch

def remove_cbond_edges(batch):
    # 获取批量中的所有信息
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
    is_cbond = batch.is_cbond

    cbond_indices = torch.nonzero(is_cbond == 1).squeeze()

    # 如果cbond_indices非空，删除对应的边
    if len(cbond_indices) > 0:
        # 创建一个mask来标记不需要删除的边
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[cbond_indices] = False  # 将要删除的边标记为False

        edge_index = edge_index[:, mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

    batch.edge_index = edge_index
    batch.edge_attr = edge_attr if edge_attr is not None else torch.empty((0,))

    return batch
