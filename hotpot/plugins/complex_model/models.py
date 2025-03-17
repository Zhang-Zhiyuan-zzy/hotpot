import datetime
import os.path as osp
from typing import Literal, Union, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch.nn import TransformerEncoderLayer

from hotpot.cheminfo.elements import elements
from . import attn
from . import utils


def complete_graph_generator(ptr):
    return torch.cat([torch.combinations(torch.arange(ptr[i], ptr[i+1]), with_replacement=True) for i in range(len(ptr)-1)]).T.to(ptr.device)


def model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming 32-bit floating point numbers (4 bytes per parameter)
    total_size_bytes = total_params * 4
    total_size_gb = total_size_bytes / (1024 ** 3)
    return total_params, total_size_gb


def print_cuda_status(device_ = torch.device('cuda:0')):
    total_memory =torch.cuda.get_device_properties(device_).total_memory
    used_memory =torch.cuda.memory_allocated(device_)

    print(f"{device_} Memory used: {used_memory/1024**2}MB/{total_memory/1024**2}MB")


def atom_label_weight_(
        atom_labels,
        num_types: int = 119,
        weight_method: Literal['inverse-count', 'cross-entropy'] = 'cross-entropy',
):
    atom_labels = torch.argmax(atom_labels, dim=-1)  # Is one hot vector
    values, counts = torch.unique(atom_labels, return_counts=True)

    if weight_method == 'cross-entropy':
        weight = counts / counts.sum()
        weight = -weight*torch.log(weight)
    elif weight_method == 'inverse-count':
        weight = counts.sum() / counts
        weight = weight / weight.max()
    else:
        raise ValueError('weight_method must be either "inverse-count" or "cross-entropy"')

    onehot_weight = torch.zeros(num_types).to(atom_labels.device)
    onehot_weight[values] = weight

    return onehot_weight


def atom_label_weight(atom_labels, scale: float = 10, num_types: int = 119):
    atom_labels = atom_labels.to(torch.int)
    values, counts = torch.unique(atom_labels, return_counts=True)
    weight = len(atom_labels) / counts
    weight = weight / torch.max(weight) * scale

    onehot_weight = torch.zeros(num_types).to(atom_labels.device)
    onehot_weight[values] = weight

    return onehot_weight


def _to_mask(
        inp_vec: torch.Tensor,
        masked_idx: torch.Tensor,
        mask_vec: torch.Tensor,
        inp_atom_labels: torch.Tensor,
        label_mask: bool = False,
        to_mask_label: int = 0
):
    atom_labels = inp_atom_labels[masked_idx]

    # Prepare masked input
    masked_vec = inp_vec.clone()
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    mask2mask_idx = masked_idx & (torch.rand(inp_vec.shape[0]) < 0.90).to(inp_vec.device)
    masked_vec[mask2mask_idx] = to_mask_label if label_mask else mask_vec # mask token is the last in the dict

    # Set 10% to a random token
    mask2rand_idx = mask2mask_idx & (torch.rand(inp_vec.shape[0]) < 1 / 9).to(inp_vec.device)
    masked_vec[mask2rand_idx] = inp_vec[torch.randint(0, len(inp_vec), (torch.sum(mask2rand_idx),))]

    return masked_vec, atom_labels, masked_idx


metal_index = torch.tensor(list(elements.metal | elements.metalloid_2nd))
def masked_metal(
        inp_vec: torch.Tensor,
        mask_vec: torch.Tensor,
        inp_atom_labels: torch.Tensor,
):
    masked_idx = torch.isin(inp_vec[0], metal_index)
    return _to_mask(inp_vec, masked_idx, mask_vec, inp_atom_labels)


def get_masked_input_and_labels(
        inp_vec: torch.Tensor,
        mask_vec: torch.Tensor,
        inp_atom_labels: torch.Tensor,
        label_mask: bool = False
):
    # 15% BERT masking
    masked_idx = torch.from_numpy(np.random.uniform(size=inp_vec.shape[0]) < 0.10).to(inp_vec.device)

    # Randomly select 40% metals to mask
    masked_metal_idx = torch.isin(inp_atom_labels, metal_index.to(inp_vec.device))
    masked_metal_idx = masked_metal_idx & (torch.rand(masked_metal_idx.shape, device=inp_vec.device) < 0.6)

    masked_idx = masked_idx | masked_metal_idx

    return _to_mask(inp_vec, masked_idx, mask_vec, inp_atom_labels, label_mask)


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


def padded_to_nested(padded_tensor, padding_value, layout=torch.jagged):
    indices = padded_tensor[..., 0] != padding_value
    return torch.nested.nested_tensor([t[i] for i, t in zip(indices, padded_tensor)], layout=layout)

def _get_mol_num_from_ptr(ptr):
    return ptr[1:] - ptr[:-1]

class LossMethods:
    """ A collection of loss functions """
    @staticmethod
    def calc_atom_type_loss(pred, target, weight=None, acc=torch.tensor(1.)):
        """ Cross Entropy Loss """
        # return F.cross_entropy(pred, target.float(), weight=weight.to(pred.device)) - acc*torch.log(acc)
        return F.cross_entropy(pred, target.float(), weight=weight.to(pred.device))


class Metrics:
    """ A collection of metrics functions """
    @staticmethod
    def calc_oh_accuracy(pred, target, is_onehot: bool = True):
        if is_onehot:
            pred_label, target_label = utils.oh2label(pred), utils.oh2label(target)
        else:
            pred_label, target_label = pred, target

        if isinstance(pred, torch.Tensor):
            return (pred_label == target_label).float().mean()
        elif isinstance(pred, np.ndarray):
            return (pred_label == target_label).mean()
        else:
            raise TypeError('pred_oh must be of type torch.Tensor or np.ndarray')

    @staticmethod
    def metal_oh_accuracy(pred, target, is_onehot: bool = True):
        if is_onehot:
            pred_label, target_label = utils.oh2label(pred), utils.oh2label(target)
        else:
            pred_label, target_label = pred, target

        metal_idx = utils.where_metal(target_label)
        pred_label = pred_label[metal_idx]
        target_label = target_label[metal_idx]

        if isinstance(pred, torch.Tensor):
            return (pred_label == target_label).float().mean()
        elif isinstance(pred, np.ndarray):
            return (pred_label == target_label).mean()
        else:
            raise TypeError('pred_oh must be of type torch.Tensor or np.ndarray')


    @staticmethod
    def binary_accuracy(pred: np.ndarray, target: np.ndarray):
        return (target == np.round(pred)).mean()

    @staticmethod
    def r2_score(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Computes the R^2 (coefficient of determination) score between y_true and y_pred.

        R^2 = 1 - (SS_res / SS_tot),
        where SS_res = Σ(y_true - y_pred)²
              SS_tot = Σ(y_true - mean(y_true))²
        """
        # Ensure y_true and y_pred are float tensors
        # target = target.float()
        # pred = pred.float()

        # Mean of true values
        mean_y_true = target.mean()

        # Sum of squares of residuals
        ss_res = ((target - pred) ** 2).sum()

        # Total sum of squares (relative to the mean)
        ss_tot = ((target - mean_y_true) ** 2).sum()

        # Handle the case where ss_tot can be zero (e.g., constant targets)
        if ss_tot <= 1e-8:
            if isinstance(pred, torch.Tensor):
                return torch.tensor(1.0 if torch.allclose(target, pred) else 0.0)
            else:
                return np.array(1.0 if np.allclose(target, pred) else 0.0)

        return 1 - ss_res / ss_tot

    @staticmethod
    def rmse(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(target, torch.Tensor):
            return torch.sqrt(F.mse_loss(pred, target))
        else:
            return np.sqrt(np.mean((pred - target) ** 2))

    @staticmethod
    def mse(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(target, torch.Tensor):
            return F.mse_loss(pred, target)
        else:
            return np.mean((pred - target) ** 2)

    @staticmethod
    def mae(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(target, torch.Tensor):
            return torch.mean(torch.abs(target - pred))
        else:
            return np.mean(np.abs(target - pred))


class FeatureExtractors:
    """ A collection of feature extractor functions """
    @staticmethod
    def extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        if seq.is_nested:
            return FeatureExtractors._extract_atom_vec_from_nested(seq, batch.ptr)  # inputs.ptr
        else:
            return FeatureExtractors._extract_atom_vec_from_padded(seq, X_mask)

    @staticmethod
    def _extract_atom_vec_from_nested(seq, ptr):
        node_num = _get_mol_num_from_ptr(ptr)
        return torch.cat([t[:num] for num, t in zip(node_num, seq)])

    @staticmethod
    def _extract_atom_vec_from_padded(seq, X_mask):
        Znode = []
        node_seq = seq[:, 1:X_mask.shape[-1]+1]
        for s, m in zip(node_seq, X_mask.sum(dim=-1)):
            Znode.append(s[:m])

        return torch.cat(Znode, dim=0)

    @staticmethod
    def extract_cbond_pair(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = FeatureExtractors.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)
        cbond_index = batch.cbond_index

        upper_Znode = Znode[cbond_index[0]]
        lower_Znode = Znode[cbond_index[1]]

        cbond_feature = torch.cat([upper_Znode, lower_Znode], dim=1)

        assert cbond_feature.shape == (upper_Znode.shape[0], upper_Znode.shape[1] * 2)

        return cbond_feature

    @staticmethod
    def extract_pair_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        Znode = FeatureExtractors.extract_atom_vec(seq, X_mask, R_mask, batch, batch_getter)

        pair_index = batch_getter(batch)

        upper_Znode = Znode[pair_index[0]]
        lower_Znode = Znode[pair_index[1]]

        return (upper_Znode + lower_Znode) / 2

    @staticmethod
    def extract_ring_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        Zring = []

        ring_seq = seq[:, -R_mask.shape[-1]-1:-1]
        assert ring_seq.shape[:2] == R_mask.shape

        for s, m in zip(ring_seq, R_mask.sum(dim=-1)):
            Zring.append(s[:m])

        return torch.cat(Zring, dim=0)

    @staticmethod
    def extract_mol_vec(seq, X_mask, R_mask, batch, batch_getter=None):
        return seq[:, 1]


class CloudGraph(nn.Module):
    """ Graph network to perform complete graph operation """
    def __init__(self, vec_dim: int):
        super(CloudGraph, self).__init__()
        self.xyz_proj = nn.Linear(3, vec_dim, bias=False)
        self.xyz_proj_norm = nn.BatchNorm1d(vec_dim)

        self.lin1 = nn.Linear(vec_dim, vec_dim)
        self.norm1 = nn.LayerNorm(vec_dim)

        self.lin2 = nn.Linear(vec_dim, vec_dim)
        self.norm2 = nn.LayerNorm(vec_dim)

    def forward(self, x, xyz, ptr):
        # TODO: add vector cross-multiply module
        # TODO: Add CNN module

        cloud_edge_index = complete_graph_generator(ptr)
        rela_xyz = xyz[cloud_edge_index[0]] - xyz[cloud_edge_index[1]]
        rela_x = x[cloud_edge_index[0]] - x[cloud_edge_index[1]]

        dist_xyz = torch.norm(rela_xyz, p=2, dim=-1)
        weight = torch.exp(-dist_xyz).unsqueeze(-1)

        x = x + self.norm1(pygnn.global_add_pool(
            F.relu(self.lin1(weight * rela_x)),
            cloud_edge_index[0]
        ))

        x = x + self.xyz_proj_norm(
            pygnn.global_add_pool(
                F.relu(self.xyz_proj(rela_xyz)),
                cloud_edge_index[0]
            )
        )

        return x


class AssembleModule(nn.Module):
    def __init__(self):
        super(AssembleModule, self).__init__()

    def forward(self, X, Xr, CLS, RING, END):
        return AssembleNestedSeqFn.apply(X, Xr, CLS, RING, END)


class AssembleNestedSeqFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Xr, CLS, RING, END):
        """
        ctx: a context object used to store information for backward
        X, Xr: your input tensors
        X_mask, Xr_mask: masks
        CLS, RING, END: special tokens
        is_nested: boolean flag
        """
        # Save anything needed for backward
        ctx.save_for_backward(X, Xr, CLS, RING, END)
        seq = torch.nested.as_nested_tensor(
            [torch.cat([CLS, x, RING, xr, END]) for x, xr in zip(X, Xr)],
            layout=torch.jagged)

        return seq

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        grad_seq, grad_seq_mask: Gradients wrt the outputs from forward.
        We must return as many gradients as forward had inputs.
        (X, Xr, X_mask, Xr_mask, CLS, RING, END, is_nested)
        """
        print('go int assemble')
        grad_seq, = grad_outputs
        X, Xr, CLS, RING, END = ctx.saved_tensors

        grad_X = []
        grad_Xr = []
        for grad_s, x, xr in zip(grad_seq, X, Xr):
            grad_X.append(grad_s[1:1+x.shape[0]])
            grad_Xr.append(grad_s[x.shape[0]+2:x.shape[0]+2+xr.shape[0]])

        grad_X = torch.nested.as_nested_tensor(grad_X, layout=torch.jagged)
        grad_Xr = torch.nested.as_nested_tensor(grad_Xr, layout=torch.jagged)
        grad_CLS = None
        grad_RING = None
        grad_END = None

        return None, None, grad_CLS, grad_RING, grad_END



class CoreBase(nn.Module):
    def __init__(
            self,
            vec_dim: int,
            # Rings Transformer arguments
            ring_layers: int = 1,
            ring_nheads: int = 2,
            ring_encoder_kw: dict = None,
            ring_encoder_block_kw: dict = None,

            # Molecular Transformer arguments
            mol_layers: int = 1,
            mol_nheads: int = 4,
            mol_encoder_kw: dict = None,
            mol_encoder_block_kw: dict = None,
            *,
            mode: Literal['fast', 'default'] = 'default',
            **kwargs,
    ):
        super(CoreBase, self).__init__()
        self.ring_encoder_kw = ring_encoder_kw if ring_encoder_kw else {}
        self.ring_encoder_block_kw = ring_encoder_block_kw if ring_encoder_block_kw else {}
        if mode == 'fast':
            self.ring_encoder = attn.Encoder(
                n_layers=ring_layers,
                d_model=vec_dim,
                nheads=ring_nheads,
                **self.ring_encoder_kw,
            )
        else:
            self.ring_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=vec_dim,
                    nhead=ring_nheads,
                    dim_feedforward=1024,
                    batch_first=True,
                ), num_layers=ring_layers,
            )

        self.mol_encoder_kw = mol_encoder_kw if mol_encoder_kw else {}
        self.mol_encoder_block_kw = mol_encoder_block_kw if mol_encoder_block_kw else {}
        if mode == 'fast':
            self.mol_encoder = attn.Encoder(
                n_layers=mol_layers,
                d_model=vec_dim,
                nheads=mol_nheads,
                **self.mol_encoder_kw,
            )
        else:
            self.mol_encoder = nn.TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=vec_dim,
                    nhead=mol_nheads,
                    dim_feedforward=1024,
                    batch_first=True,
                ), num_layers=mol_layers,
            )

        self.CLS = nn.Parameter(torch.randn(1, vec_dim))
        self.RING = nn.Parameter(torch.randn(1, vec_dim))
        self.END = nn.Parameter(torch.randn(1, vec_dim))

        self.CLS_proj = nn.Linear(vec_dim, vec_dim)
        self.RING_proj = nn.Linear(vec_dim, vec_dim)
        self.END_proj = nn.Linear(vec_dim, vec_dim)
        # self.cre_emb = nn.Embedding(3, vec_dim)
        self.assemble = AssembleModule()
        self.is_nested = kwargs.get('is_nested', False)

    def _padding_encode(self, x, ptr, rings_node_index, rings_node_nums, mol_rings_nums):
        X, X_mask = self._nodes_padding(x, ptr)
        not_padded_X = torch.logical_not(X_mask)
        Xr = self._rings_attention(x, rings_node_index, rings_node_nums)
        Xr, Xr_mask = self._split_padding(Xr, mol_rings_nums)

        not_padded_Xr = torch.logical_not(Xr_mask)

        seq, seq_padding_mask = self._assemble_sequence(X, Xr, X_mask, Xr_mask)
        seq = self.mol_encoder(seq, src_key_padding_mask=seq_padding_mask)

        return seq, not_padded_X, not_padded_Xr

    def _nesting_encode(self, x, ptr, rings_node_index, rings_node_nums, mol_rings_nums):
        X, _ = self._node_nesting(x, ptr)
        Xr = self._rings_attention(x, rings_node_index, rings_node_nums)
        Xr, _ = self._split_nested(Xr, mol_rings_nums, layout=torch.jagged)

        seq, _ = self._assemble_sequence(X, Xr, is_nested=self.is_nested)
        seq = self.mol_encoder(seq)

        return seq, None, None

    @staticmethod
    def _split_nested(x: torch.Tensor, nums: torch.Tensor, layout=None):
        return torch.nested.nested_tensor(list(torch.split(x, nums.tolist())), layout=layout), None

    @staticmethod
    def _split_padding(x: torch.Tensor, nums: torch.Tensor):
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

    def _node_nesting(self, x, ptr):
        mol_node_nums = ptr[1:] - ptr[:-1]
        return self._split_nested(x, mol_node_nums, layout=torch.jagged)

    def _nodes_padding(self, x, ptr):
        mol_node_nums = ptr[1:] - ptr[:-1]
        return self._split_padding(x, mol_node_nums)

    def _rings_attention(self, x, rings_node_index, rings_node_nums):
        x = x[rings_node_index]  # Get rings node's x

        if self.is_nested:
            X, _ = self._split_nested(x, rings_node_nums, layout=torch.jagged)
            X = self.ring_encoder(X)
        else:
            X, padding_mask = self._split_padding(x, rings_node_nums)
            X = self.ring_encoder(X, src_key_padding_mask=padding_mask)

        # Max pooling, extracting the value with max absolute.
        rings_vec = torch.zeros((len(X), x.shape[-1])).to(x.device)
        for i, t in enumerate(X):
            rings_vec[i, :] = t.gather(-2, torch.argmax(torch.abs(t), dim=-2).unsqueeze(-2))

        return rings_vec

    def _assemble_sequence(self, X, Xr, X_mask=None, Xr_mask=None, is_nested: bool = False):
        if is_nested:
            # CLS = torch.randn(1, X.shape[-1], device=X.device)
            # RING = torch.randn(1, X.shape[-1], device=X.device)
            # END = torch.randn(1, X.shape[-1], device=X.device)
            # CLS = self.CLS_proj(CLS)
            # RING = self.RING_proj(RING)
            # END = self.END_proj(END)
            #
            # seq = torch.nested.nested_tensor(
            #     [torch.cat([CLS, x, RING, xr, END]) for x, xr in zip(X, Xr)],
            #     layout=torch.jagged,
            #     # requires_grad=True
            # )

            # seq = AssembleNestedSeqFn.apply(X, Xr, self.CLS, self.RING, self.END)
            seq = self.assemble(X, Xr, self.CLS, self.RING, self.END)

            # seq = torch.cat((CLS, X.to_padded_tensor(float("-inf")), RING, Xr.to_padded_tensor(float("-inf")), END), dim=-2)
            # return padded_to_nested(seq, float("-inf"), torch.jagged), None
            return seq, None

        else:
            CLS = torch.tile(self.CLS, (X.shape[0], 1, 1))
            RING = torch.tile(self.RING, (X.shape[0], 1, 1))
            END = torch.tile(self.END, (X.shape[0], 1, 1))

            seq = torch.cat((CLS, X, RING, Xr, END), dim=-2)
            seq_padding_mask = torch.cat([
                torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
                X_mask,
                torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
                Xr_mask,
                torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
            ], dim=1)

            return seq, seq_padding_mask


class Core(CoreBase):
    def __init__(
            self,
            x_dim: int,
            vec_dim: int = 512,
            x_label_nums: Optional[int] = None,
            graph_model: nn.Module = None,

            # Rings Transformer arguments
            ring_layers: int = 1,
            ring_nheads: int = 2,
            ring_encoder_kw: dict = None,
            ring_encoder_block_kw: dict = None,

            # Molecular Transformer arguments
            mol_layers: int = 1,
            mol_nheads: int = 4,
            mol_encoder_kw: dict = None,
            mol_encoder_block_kw: dict = None,
            **kwargs,
    ):
        super(Core, self).__init__(
            vec_dim=vec_dim,
            ring_layers=ring_layers,
            ring_nheads=ring_nheads,
            ring_encoder_kw=ring_encoder_kw,
            ring_encoder_block_kw=ring_encoder_block_kw,
            mol_layers=mol_layers,
            mol_nheads=mol_nheads,
            mol_encoder_kw=mol_encoder_kw,
            mol_encoder_block_kw=mol_encoder_block_kw,
            **kwargs)

        self.vec_size = vec_dim
        self.x_label_nums = x_label_nums

        if isinstance(x_label_nums, int):
            self.x_emb = nn.Embedding(x_label_nums+1, vec_dim)
            self.x_mask_vec = self.x_emb.weight[0]
        else:
            self.x_proj = nn.Linear(x_dim, vec_dim)
            self.x_mask_vec = nn.Parameter(torch.randn(x_dim))

        self.cloud_graph = CloudGraph(vec_dim)
        self.lin = nn.Linear(vec_dim, vec_dim)
        self.norm = nn.BatchNorm1d(vec_dim)

        if graph_model:
            self.graph = graph_model
        else:
            self.graph = pygnn.GAT(
                vec_dim, vec_dim, 6,
                vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim),
                edge_dim=vec_dim, v2=True
            )

    def forward(
            self,
            x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr,
            *,
            xyz: Optional[Union[torch.Tensor, torch.nested.nested_tensor]] = None,
    ):
        if isinstance(self.x_label_nums, int):
            x = self.x_emb(x.long())
        else:
            x = self.x_proj(x)

        xg = self.graph(x, edge_index)

        # coordinates side
        if xyz is not None:
            x_cloud = self.cloud_graph(x, xyz, ptr)
            x = self.norm(self.lin(x + xg + x_cloud))
        else:
            x = self.norm(self.lin(x + xg))

        if not self.is_nested:
            return self._padding_encode(x, ptr, rings_node_index, rings_node_nums, mol_rings_nums)
        else:
            return self._nesting_encode(x, ptr, rings_node_index, rings_node_nums, mol_rings_nums)

    @staticmethod
    def block_diag_split(tensor: torch.Tensor, block_sizes: list[int]) -> list[torch.Tensor]:
        """
        将 (n, n, m) 形状的张量(沿前两个维度)拆分为若干对角方块，
        每个对角方块的大小由 block_sizes 指定。
        """
        blocks = []
        start = 0
        for size in block_sizes:
            end = start + size
            # 取第 0、1 维度为矩形切片，保留第三维度 (m)
            block = tensor[start:end, start:end, :]
            blocks.append(block)
            start = end
        return blocks


class CoreModule(nn.Module):
    def __init__(
            self,
            x_dim: int,
            edge_dim: int,
            vec_dim: int = 512,
            graph_model: nn.Module = None,
            rings_kwargs: dict = None,
            transformer_kwargs: dict = None,

            # Rings Transformer arguments
            ring_layers: int = 1,
            ring_nheads: int = 2,
            ring_encoder_kw: dict = None,
            ring_encoder_block_kw: dict = None,

            # Molecular Transformer arguments
            mol_layers: int = 1,
            mol_nheads: int = 4,
            mol_encoder_kw: dict = None,
            mol_encoder_block_kw: dict = None,
            **kwargs,
    ):
        super(CoreModule, self).__init__()
        self.vec_size = vec_dim
        self.rings_kwargs = rings_kwargs if rings_kwargs else {}
        self.transformer_kwargs = transformer_kwargs if transformer_kwargs else {}

        self.x_mask_vec = nn.Parameter(torch.randn(x_dim))
        self.x_project = nn.Linear(x_dim, vec_dim)
        self.e_project = nn.Linear(edge_dim, vec_dim)

        self.NODE_MASK = nn.Parameter(torch.randn(vec_dim, 1))
        self.RING_MASK = nn.Parameter(torch.randn(vec_dim, 1))

        self.CLS = nn.Parameter(torch.randn(1, vec_dim))
        self.RING = nn.Parameter(torch.randn(1, vec_dim))
        self.END = nn.Parameter(torch.randn(1, vec_dim))

        if graph_model:
            self.graph = graph_model
        else:
            self.graph = pygnn.GAT(
                vec_dim, vec_dim, 6,
                vec_dim, 0.1, norm=pygnn.LayerNorm(vec_dim),
                edge_dim=vec_dim, v2=True
            )

        self.ring_encoder_kw = ring_encoder_kw if ring_encoder_kw else {}
        self.ring_encoder_block_kw = ring_encoder_block_kw if ring_encoder_block_kw else {}
        self.ring_encoder = attn.Encoder(
            n_layers=ring_layers,
            d_model=vec_dim,
            nheads=ring_nheads,
            **self.ring_encoder_kw,
        )

        self.mol_encoder_kw = mol_encoder_kw if mol_encoder_kw else {}
        self.mol_encoder_block_kw = mol_encoder_block_kw if mol_encoder_block_kw else {}
        self.mol_encoder = attn.Encoder(
            n_layers=mol_layers,
            d_model=vec_dim,
            nheads=mol_nheads,
            **self.mol_encoder_kw,
        )

        # TODO: convert to False later
        self.is_nested = kwargs.get('is_nested', True)

    def _padding_encode(self, x, ptr, rings_node_index, rings_node_nums, mol_rings_nums):
        X, X_mask = self._nodes_padding(x, ptr)
        not_padded_X = torch.logical_not(X_mask)
        Xr = self._rings_attention(x, rings_node_index, rings_node_nums)
        Xr, Xr_mask = self._split_padding(Xr, mol_rings_nums)

        not_padded_Xr = torch.logical_not(Xr_mask)

        seq, seq_padding_mask = self._assemble_sequence(X, Xr, X_mask, Xr_mask)
        seq = self.mol_encoder(seq, src_key_padding_mask=seq_padding_mask)

        return seq, not_padded_X, not_padded_Xr

    def _nesting_encode(self, x, ptr, rings_node_index, rings_node_nums, mol_rings_nums):
        X, _ = self._node_nesting(x, ptr)
        Xr = self._rings_attention(x, rings_node_index, rings_node_nums)
        Xr, _ = self._split_nested(Xr, mol_rings_nums)

        seq, _ = self._assemble_sequence(X, Xr, is_nested=True)
        seq = self.mol_encoder(seq)

        return seq, None, None

    def forward(
            self,
            x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr,
            *,
            xyz=None,
    ):
        x = self.x_project(x.bfloat16())
        e = self.e_project(edge_attr.bfloat16())

        x = self.graph(x, edge_index, edge_attr=e)

        if not self.is_nested:
            return self._padding_encode(x, ptr, rings_node_index, rings_node_nums, mol_rings_nums)
        else:
            return self._nesting_encode(x, ptr, rings_node_index, rings_node_nums, mol_rings_nums)

        # X, _ = self._node_nesting(x, ptr)
        # Xr = self._rings_attention(x, rings_node_index, rings_node_nums)
        # Xr, _ = self._split_nested(Xr, mol_rings_nums)
        #
        # seq, seq_padding_mask = self._assemble_sequence(X, Xr)
        # seq = self.mol_encoder(seq, src_key_padding_mask=seq_padding_mask)
        #
        # return seq, not_padded_X, not_padded_Xr

    @staticmethod
    def _split_nested(x: torch.Tensor, nums: torch.Tensor, layout=None):
        return torch.nested.nested_tensor(list(torch.split(x, nums.tolist())), layout=layout), None

    def _split_padding(self, x: torch.Tensor, nums: torch.Tensor):
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
            padded_X[i, :size] = x[start:start + size]
            padding_mask[i, :size] = 0
            start += size

        return padded_X, padding_mask

    def _node_nesting(self, x, ptr):
        mol_node_nums = ptr[1:] - ptr[:-1]
        return self._split_nested(x, mol_node_nums, layout=torch.jagged)

    def _nodes_padding(self, x, ptr):
        mol_node_nums = ptr[1:] - ptr[:-1]
        return self._split_padding(x, mol_node_nums)

    def _rings_attention(self, x, rings_node_index, rings_node_nums):
        x = x[rings_node_index]

        nested_X, padding_mask = self._split_nested(x, rings_node_nums, layout=torch.jagged)
        nested_X = self.ring_encoder(nested_X)

        # Max pooling, extracting the value with max absolute.
        rings_vec = torch.zeros((len(nested_X), x.shape[-1])).to(x.device)
        for i, t in enumerate(nested_X):
            rings_vec[i, :] = t.gather(-2, torch.argmax(torch.abs(t), dim=-2).unsqueeze(-2))

        return rings_vec

        # padded_X, padding_mask = self._split_padding(x, rings_node_nums)

        # weight_padding = torch.logical_not(padding_mask).float().unsqueeze(-1)
        # padded_X = weight_padding * self.ring_encoder(padded_X, src_key_padding_mask=padding_mask)

        # X = self.mha(padded_X, padded_X, padded_X)

        # padded_X = padded_X.to_padded_tensor(0.0)
        # pooling_indices = torch.argmax(abs(padded_X), dim=-2).unsqueeze(dim=-2)
        # return padded_X.gather(-2, pooling_indices).squeeze(dim=-2)

    def _assemble_sequence(self, X, Xr, X_mask=None, Xr_mask=None, is_nested: bool = False):
        CLS = torch.tile(self.CLS, (X.shape[0], 1, 1))
        RING = torch.tile(self.RING, (X.shape[0], 1, 1))
        END = torch.tile(self.END, (X.shape[0], 1, 1))

        if is_nested:
            seq = torch.cat((CLS, X.to_padded_tensor(float("-inf")), RING, Xr.to_padded_tensor(float("-inf")), END), dim=-2)
            return padded_to_nested(seq, float("-inf"), torch.jagged), None

        else:
            seq = torch.cat((CLS, X, RING, Xr, END), dim=-2)
            seq_padding_mask = torch.cat([
                torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
                X_mask,
                torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
                Xr_mask,
                torch.zeros((X.shape[0], 1), dtype=torch.bool, device=seq.device),
            ], dim=1)

            return seq, seq_padding_mask


############################# Predictors #################################
TargetTypeName = Literal['num', 'xyz', 'onehot', 'binary']
class Predictor(nn.Module):
    def __init__(
            self,
            in_size: int,
            target_pattern: TargetTypeName,
            num_layers: int = 2,
            dropout: float = 0.1,
            act: Type[nn.Module] = nn.ReLU,
            out_act: Type[nn.Module] = nn.ReLU,
            **kwargs
    ):
        super(Predictor, self).__init__()
        self.hidden_layers = pygnn.MLP(num_layers * [in_size], dropout=dropout)

        self.target_pattern = target_pattern
        if target_pattern == 'num':
            self.out_layer = nn.Linear(in_size, 1)
            self.out_act = out_act()
        elif target_pattern == 'xyz':
            self.out_layer = nn.Linear(in_size, 3)
            self.out_act = out_act()
        elif target_pattern == 'onehot':
            self.out_layer = nn.Linear(in_size, kwargs.get("onehot_type", 119))
            self.out_act = nn.Softmax(dim=-1)
        elif target_pattern == 'binary':
            self.out_layer = nn.Linear(in_size, 1)
            self.out_act = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{target_pattern} is not implemented")

    def forward(self, z):
        # z = self.atom_type_predictor(z) + z
        # z = self.out_layer(z)
        # return F.softmax(z, dim=-1)
        z = self.hidden_layers(z) + z
        return F.softmax(self.out_layer(z), dim=-1)

############################### ComplexFormer ##################################


class ComplexFormer(nn.Module):
    def __init__(
            self,
            x_dim: int,
            edge_dim: int,
            vec_dim: int = 512,
            x_label_nums: Optional[int] = None,
            graph_model: nn.Module = None,

            # Rings Transformer arguments
            ring_layers: int = 1,
            ring_nheads: int = 2,
            ring_encoder_kw: dict = None,
            ring_encoder_block_kw: dict = None,

            # Molecular Transformer arguments
            mol_layers: int = 4,
            mol_nheads: int = 8,
            mol_encoder_kw: dict = None,
            mol_encoder_block_kw: dict = None,

            # Predictors arguments
            atom_types: int = 119,
            layer_atom_types: int = 2,
            mol_attrs: Union[list, str] = None,

            *,
            # Load from core
            core_module: Union[Type[CoreBase], nn.Module] = None,
            target_type: TargetTypeName = 'num',
            **kwargs
    ):
        super(ComplexFormer, self).__init__()

        if isinstance(core_module, nn.Module):
            self.core = core_module
        elif issubclass(core_module, CoreBase):
            self.core = core_module(
                x_dim=x_dim,
                edge_dim=edge_dim,
                vec_dim=vec_dim,
                x_label_nums=x_label_nums,
                graph_model=graph_model,
                ring_layers=ring_layers,
                ring_nheads=ring_nheads,
                ring_encoder_kw=ring_encoder_kw,
                ring_encoder_block_kw=ring_encoder_block_kw,
                mol_layers=mol_layers,
                mol_nheads=mol_nheads,
                mol_encoder_kw=mol_encoder_kw,
                mol_encoder_block_kw=mol_encoder_block_kw,
                **kwargs
            )
        else:
            self.core = CoreModule(
                x_dim=x_dim,
                edge_dim=edge_dim,
                vec_dim=vec_dim,
                x_label_nums=x_label_nums,
                graph_model=graph_model,
                ring_layers=ring_layers,
                ring_nheads=ring_nheads,
                ring_encoder_kw=ring_encoder_kw,
                ring_encoder_block_kw=ring_encoder_block_kw,
                mol_layers=mol_layers,
                mol_nheads=mol_nheads,
                mol_encoder_kw=mol_encoder_kw,
                mol_encoder_block_kw=mol_encoder_block_kw
            )
        self.is_labeled_x = isinstance(getattr(self.core, 'x_label_nums', None), int)

        ###########  Predictors  ##############
        self.predictor = Predictor(
            in_size=vec_dim,
            target_pattern=target_type,
        )

    def forward(
            self,
            x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr,
            *,
            xyz=None,
    ):
        return self.core(
            x, edge_index, edge_attr, rings_node_index, rings_node_nums, mol_rings_nums, batch, ptr, xyz=xyz)

    def save_core(self, save_dir):
        torch.save(self.core, osp.join(save_dir, 'core.pt'))
        torch.save(self.core.state_dict(), osp.join(save_dir, 'core.state_dict.pt'))

    def save_model(self, save_dir):
        self.save_core(save_dir)

        torch.save(self, osp.join(save_dir, 'model.pt'))
        torch.save(self.state_dict(), osp.join(save_dir, 'state_dict.pt'))

    def predict_atom_type(self, z: torch.Tensor) -> torch.Tensor:
        z = self.atom_type_predictor(z) + z
        z = self.atom_type_linear(z)
        return F.softmax(z, dim=-1)

    def predict_atom_charge(self, z: torch.Tensor) -> torch.Tensor:
        z = self.batch_norm(self.atom_partial_charge_predictor(z) + z)
        z = self.atom_partial_charge_linear(z)
        return z

    def predict_atom_aromatic(self, z: torch.Tensor) -> torch.Tensor:
        z = self.batch_norm(self.atom_aromatic_predictor(z) + z)
        z = self.atom_aromatic_linear(z)
        return F.sigmoid(z)

    def predict_pair_step(self, z: torch.Tensor) -> torch.Tensor:
        z = self.pair_step_predictor(z) + z
        z = self.pair_step_linear(z)
        return z

    def predict_rings_aromatic(self, z: torch.Tensor) -> torch.Tensor:
        z = self.ring_aromatic_predictor(z) + z
        z = self.ring_aromatic_linear(z)
        return F.sigmoid(z)

    def predict_mol_attrs(self, zs: torch.Tensor, z_names) -> list[torch.Tensor]:
        return [self.mol_attr_predictors[n](z) for z, n in zip(zs, z_names)]

    def save_checkpoint(self, save_dir, which: Literal['both', 'model', 'state_dict'] = 'both'):
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%y%m%d%H%M%S")
        if which == 'both':
            torch.save(self.state_dict(), osp.join(save_dir, f'state_dict_{formatted_datetime}.pt'))
            torch.save(self, osp.join(save_dir, f'model_{formatted_datetime}.pt'))
        elif which == 'model':
            torch.save(self, osp.join(save_dir, f'model_{formatted_datetime}.pt'))
        elif which == 'state_dict':
            torch.save(self.state_dict(), osp.join(save_dir, f'state_dict_{formatted_datetime}.pt'))
