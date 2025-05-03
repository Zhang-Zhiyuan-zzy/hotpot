import sys
import os.path as osp
import socket

import torch

machine_name = socket.gethostname()
torch.set_default_dtype(torch.bfloat16)
if torch.cuda.is_available():
    if machine_name == '4090':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# Initialize paths.
if machine_name == '4090':
    project_root = '/home/zzy/docker_envs/pretrain/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'DESKTOP-G9D9UUB':
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'docker':
    project_root = '/app/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == '3090':
    project_root = '/home/zz1/docker/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))

# Running in Super
elif str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('In Super')
    project_root = '/HOME/scz0s3z/run/proj/'
    sys.path.append(osp.join(project_root, 'hotpot-zzy'))
else:
    raise ValueError(__file__)

# from hotpot.plugins.complex_model import (
#     models as M,
#     pretrain,
#     dataset as D,
# )
from hotpot.plugins.cmodel import (
    models as M,
    pretrain,
    dataset as D,
)

models_dir = osp.join(project_root, 'models')
# dataset save paths
_tmqm_data_dir = osp.join(project_root, 'datasets', 'tmqm_data0207')

if str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('in /dev/shm')
    tmqm_getter = D.DatasetGetter('/dev/shm', 'tmqm')
else:
    tmqm_getter = D.DatasetGetter(project_root, "tmqm")

dataset, dataset_test = tmqm_getter.get_datasets()
INPUT_X_INDEX = tmqm_getter.get_index('x', ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z'))
XYZ_INDEX = tmqm_getter.get_index('x', ('x', 'y', 'z'))
TYPE_INDEX = tmqm_getter.get_index('x', 'atomic_number')
ATOM_CHRG_INDEX = tmqm_getter.get_index('x', 'partial_charge')
ATOM_AROMATIC_INDEX = tmqm_getter.get_index('x', 'is_aromatic')
RING_AROMATIC_INDEX = tmqm_getter.get_index('ring_attr', 'is_aromatic')
PAIR_STEP_INDEX = tmqm_getter.get_index('pair_attr', 'length_shortest_path')
PAIR_WBO_INDEX = tmqm_getter.get_index('pair_attr', 'wiberg_bond_order')
Y_ATTR_NAMES = tmqm_getter.get_y_attrs()


EPOCHS = 100
OPTIMIZER = torch.optim.Adam
X_DIM = len(INPUT_X_INDEX)
EDGE_DIM = dataset[0].edge_attr.shape[-1]
VEC_DIM = 64
MASK_VEC = (-1 * torch.ones(X_DIM)).to(device)
RING_LAYERS = 1
RING_HEADS = 2
MOL_LAYERS = 1
MOL_HEADS = 2

ATOM_TYPES = 119  # Arguments for atom type loss


hypers = pretrain.Hypers()
hypers.batch_size = 1024
hypers.lr = 2e-4
hypers.weight_decay = 4e-5

core = M.Core(
    x_dim=X_DIM,
    edge_dim=EDGE_DIM,
    vec_dim=VEC_DIM,
    x_label_nums=ATOM_TYPES,
    ring_layers=RING_LAYERS,
    ring_nheads=RING_HEADS,
    mol_layers=MOL_LAYERS,
    mol_nheads=MOL_HEADS,
)

def atom_types():
    pretrain.run(
        work_name="AtomType",
        work_dir=models_dir,
        core=core,
        train_dataset=dataset,
        test_dataset=dataset_test,
        hypers=hypers,
        epochs=EPOCHS,
        device=device,
        eval_steps=1,
        # checkpoint_path=-1,
        load_core_only=True,
        # save_model=False,
        # x_masker=pretrain.x_masker_func,
        load_all_data=True,
        show_batch_pbar=True,
        constant_lr=True,
        other_metric='metal_accuracy',
        early_stopping=True,
        loss_weight_calculator=True,
        loss_weight_method='sqrt-invert_count',
        debug=True,
    )

def atom_charges():
    pretrain.run(
        work_name="AtomCharge",
        work_dir=models_dir,
        core=core,
        train_dataset=dataset,
        test_dataset=dataset_test,
        hypers=hypers,  # pretrain.Hyper instance
        epochs=EPOCHS,  # int
        device=device,  # cuda or cpu
        eval_steps=1,  # the interval (epoch) steps to eval
        checkpoint_path=-1,  # (path or index)
        load_core_only=True,  # if just load core params
        with_xyz=True,  # xyz as input
        # save_model=False,
        # x_masker=pretrain.x_masker_func,  # predict masked nodes
        load_all_data=True,  # if load all data before training
        constant_lr=True,  #
        # other_metric='metal_accuracy',
        early_stopping=True,
        # debug=True,
    )

def cbond():
    ...


def main():
    ...


if __name__ == '__main__':
    print('run!')
    atom_types()
    # atom_charges()

