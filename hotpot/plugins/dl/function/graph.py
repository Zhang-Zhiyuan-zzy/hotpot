"""
python v3.9.0
@Project: hotpot
@File   : func
@Auther : Zhiyuan Zhang
@Data   : 2024/8/23
@Time   : 8:53
"""
from typing import *
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.utils import from_smiles

import hotpot as hp
from hotpot.dataset import load_dataset
from hotpot.cheminfo.graph import calc_spectrum

from . import base


def _spectral_similarity(
        matrices1: Union[torch.Tensor, Sequence[torch.Tensor]],
        matrices2: Union[torch.Tensor, Sequence[torch.Tensor]]
):
    """
    Determine the similarity of two graphs by spectral methods,
    where, each graph is represented by a group of ordered squared matrices.

    Args:
        matrices1 (Union[torch.Tensor, Sequence[torch]]): the matrices to represented first graph, each matrix
            should be a real symmetric tensor
        matrices2 (Union[torch.Tensor, Sequence[torch]]): the matrices to represented second graph, each matrix
            should be a real symmetric tensor
    """
    if isinstance(matrices1, torch.Tensor):
        matrices1 = [matrices1]
    if isinstance(matrices2, torch.Tensor):
        matrices2 = [matrices2]

    assert len(matrices1) == len(matrices2)

    spec1, spec2 = [], []
    for mat1, mat2 in zip(matrices1, matrices2):
        assert mat1.shape == mat2.shape
        spec1.append(graph_spectrum(mat1))
        spec2.append(graph_spectrum(mat2))

    spec1 = torch.concat(spec1)
    spec2 = torch.concat(spec2)

    return torch.cosine_similarity(torch.unsqueeze(spec1, 0), torch.unsqueeze(spec2, 0))


def spectral_similarity(
        spec1 : Union[torch.Tensor, Sequence[torch.Tensor]],
        spec2 : Union[torch.Tensor, Sequence[torch.Tensor]],
        metric: Union[Literal['min', 'mean'], Callable] = None
):
    """ applied to """
    spec1 = torch.tensor(spec1)
    spec2 = torch.tensor(spec2)

    assert len(spec1.shape) == len(spec2.shape) <= 2
    if len(spec1.shape) == 1:
        spec1 = spec1.unsqueeze(0)
        spec2 = spec2.unsqueeze(0)

    assert spec1.shape[0] == spec2.shape[0]

    if spec1.shape[1] > spec2.shape[1]:
        torch.pad(spec2, (0, spec1.shape[1] - spec2.shape[1]))
    elif spec1.shape[1] < spec2.shape[1]:
        torch.pad(spec1, (0, spec2.shape[1] - spec1.shape[1]))

    spectrum = torch.cosine_similarity(spec1, spec2, dim=-1)

    if isinstance(metric, Callable):
        return metric(spectrum)
    elif metric == 'min':
        return spectrum.min()
    elif metric == 'mean':
        return spectrum.mean()
    else:
        return spectrum


def smiles_to_spectrum(smiles: str, length: int = 4) -> torch.Tensor:
    mol = next(hp.Molecule.read(smiles, 'smi'))
    return torch.tensor(mol.graph_spectrum(length).spectrum)


def edge_index2adj(note_num: int, edge_index: torch.Tensor) -> torch.Tensor:
    """"""
    assert len(edge_index.shape) == 2
    assert edge_index.T.shape[1] == 2
    edge_index = edge_index.T

    adj = torch.zeros((note_num, note_num))
    adj[edge_index[:, 0], edge_index[:, 1]] = 1
    adj[edge_index[:, 1], edge_index[:, 0]] = 1

    return adj


def graph_spectrum(mat: torch.Tensor, descending: bool = False):
    """ Compute the spectral for a given group of ordered squared matrices """
    assert base.is_symmetric(mat)
    return torch.sort(torch.linalg.eigvals(mat).real, descending=descending)[0]


def adj2laplacian(adj: torch.Tensor, norm=True) -> torch.Tensor:
    """"""
    deg = torch.diag(torch.sum(adj, dim=1))
    eye = torch.eye(deg.shape[0])

    if norm:
        return eye - torch.linalg.inv(deg ** 0.5) @ adj @ torch.linalg.inv(deg ** 0.5)
    else:
        raise deg - adj


def graph_data2spectrum(
        data: Data,
        confs_length: int = 4,
        atomic_numbers_col=0,
) -> torch.Tensor:
    adj = edge_index2adj(data.x.shape[0], data.edge_index)
    atomic_numbers = data.x[:, atomic_numbers_col]

    spectrum = calc_spectrum(adj, atomic_numbers, confs_length)
    return torch.tensor(spectrum)


def smiles2spectrum_data(
        smiles: str,
        length: int = 4,
        atomic_numbers_col: int = 0,
) -> Data:
    data = from_smiles(smiles, with_hydrogen=True)
    data['spectrum'] = graph_data2spectrum(data, length, atomic_numbers_col)
    data['smiles'] = smiles

    return data


def graph_spectrum_similarity(
        spec1: torch.Tensor,
        spec2: torch.Tensor,
        metric: Union[None, Literal['mean', 'mix'], Callable] = 'min'
) -> torch.Tensor:
    """ Compute graph similarity between two graph spectrum """
    # Check arguments
    assert len(spec1.shape) == len(spec2.shape) in [2, 3]
    if len(spec1.shape) == 2:
        spec1 = spec1.unsqueeze(0)
        spec2 = spec2.unsqueeze(0)
    assert spec1.shape[0] == spec2.shape[0]

    if spec1.shape[-1] > spec2.shape[-1]:
        pad = 4*(0,) + (0, spec1.shape[-1] - spec2.shape[-1])
        spec2 = F.pad(spec2, pad)
    elif spec1.shape[-1] < spec2.shape[-1]:
        pad = 4*(0,) + (0, spec2.shape[-1] - spec1.shape[-1])
        spec1 = F.pad(spec1, pad)

    similarity_vector = torch.cosine_similarity(spec1, spec2, dim=-1)

    if isinstance(metric, Callable):
        return metric(similarity_vector)
    elif metric == 'mean':
        return similarity_vector.mean(-2)
    elif metric == 'min':
        return similarity_vector.min(-2)
    else:
        raise ValueError(f'the metric {metric} is not implemented')


def batch_spectrum(batch_matrix: torch.Tensor, batch_atom_num: torch.Tensor):
    """"""
    batch_atom_num = batch_atom_num.flatten().int()
    assert batch_matrix.shape[0] == len(batch_atom_num)

    # Mask those elements whose row or col indices exceed
    for i, atom_num in enumerate(batch_atom_num):
        batch_matrix[i, atom_num:, :] = batch_matrix[i, :, atom_num:] = 0.

    eigenvalues = torch.linalg.eigvals(batch_matrix).real

    return tensor_sort(eigenvalues, dim=-1, key=torch.abs)


def tensor_sort(inputs: torch.Tensor, key: Callable = None, dim: int = None, descending: bool = True):
    """"""
    temp = key(inputs)
    sort_indices = torch.argsort(temp, dim=dim, descending=descending)

    return torch.gather(inputs, -1, sort_indices)


if __name__ == '__main__':
    smiles_loader = load_dataset('SMILES')
    # for smi in smiles_loader:
    #     mol = next(hp.Molecule.read(smi, 'smi'))
    #
    #     ChemData = data_with_spectrum(smi)
    #
    #     print(ChemData['spectrum'].shape)
