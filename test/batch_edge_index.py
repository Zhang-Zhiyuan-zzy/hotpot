import torch

def remove_cbond_edges(batch: Batch):
    """ Remove the cbond edges for predict """
    is_cbond = batch['is_cbond']
    cbond_index = batch['cbond_index']
    edge_index = batch['edge_index']
    edge_attr = batch['edge_attr']

    device = edge_index.device

    is_cbond = is_cbond.to(device)
    cbond_index = cbond_index.to(device)

    cbond_pairs_to_remove = cbond_index[:, is_cbond == 1]

    mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)

    for pair in cbond_pairs_to_remove.t():
        pair = pair.to(device)
        pair = pair.view(-1, 1).to(device)
        mask &= ~(torch.all(edge_index == pair, dim=0))

    edge_index = edge_index[:, mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    batch['edge_index'] = edge_index
    if edge_attr is not None:
        batch['edge_attr'] = edge_attr

    return batch
