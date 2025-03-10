from typing import Literal, Optional, Iterable, Union
import os.path as osp
import socket
from sklearn.metrics import root_mean_squared_error as rmse
from torch import EnumType

from hotpot.plugins.complex_model import (
    models as M,
    pretrain
)
from datasets import DatasetGetter


import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.bfloat16)
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")


# Initialize paths.
machine_name = socket.gethostname()
if machine_name == '4090':
    project_root = '/home/zzy/docker_envs/pretrain/proj'
elif machine_name == 'DESKTOP-G9D9UUB':
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
elif machine_name == 'docker':
    project_root = '/app/proj'
else:
    raise ValueError

models_dir = osp.join(project_root, 'models')

# dataset save paths
_tmqm_data_dir = osp.join(project_root, 'datasets', 'tmqm_data0207')


tmqm_getter = DatasetGetter(project_root, "tmqm")

tmqm_train, tmqm_test = tmqm_getter.get_datasets()
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
lr_schedular = torch.optim.lr_scheduler.ExponentialLR
schedular_kw = {'gamma': 0.95}
X_DIM = len(INPUT_X_INDEX)
EDGE_DIM = tmqm_train[0].edge_attr.shape[-1]
VEC_DIM = 64
MASK_VEC = (-1 * torch.ones(X_DIM)).to(device)
RING_LAYERS = 1
RING_HEADS = 2
MOL_LAYERS = 1
MOL_HEADS = 2

ATOM_TYPES = 119  # Arguments for atom type loss

hypers = pretrain.Hypers()
hypers.batch_size = 1024
hypers.lr = 1e-3
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

model = M.ComplexFormer(core, )

general_init_kw = dict(
    work_dir=models_dir,
    model=model,
    optimizer=OPTIMIZER,
    lr_scheduler=lr_schedular,
    scheduler_kw=schedular_kw,
    hypers=hypers,
    epochs=EPOCHS,
    device=device,
    eval_steps=1,
)

def atom_types():
    with pretrain.PretrainComplex(
        work_name="atom types",
        train_dataset=tmqm_train,
        test_dataset=tmqm_test,
        # not_save=True,
        # eval_first=True,
        # early_stopping=True,
        # primary_metric="Accuracy",
        **general_init_kw
    ) as pt:
        print(pt.work_dir)
        pt.load_model_params()
        pt.run(
            feature_extractor=M.FeatureExtractors.extract_atom_vec,
            predictor=model.predict_atom_type,
            input_x_index=INPUT_X_INDEX,
            xyz_index=XYZ_INDEX,
            target_getter=lambda batch: batch.x[:, TYPE_INDEX],
            x_masker=pretrain.x_masker_func,
            loss_fn=M.LossMethods.calc_atom_type_loss,
            to_onehot=True,
            onehot_types=ATOM_TYPES,
            loss_weight_calculator=lambda t, n: M.atom_label_weight_(t, n, 'inverse-count'),
            metrics={'Accuracy': lambda p, t: M.Metrics.calc_oh_accuracy(p, t, is_onehot=True)}
        )

def atom_charges():
    with pretrain.PretrainComplex(
        work_name="atom charges",
        eval_first=True,
        # not_save=True,
        early_stopping=True,
        primary_metric="R^2 score",
        train_dataset=tmqm_train,
        test_dataset=tmqm_test,
        **general_init_kw
    ) as pt:
        print(pt.work_dir)
        pt.load_model_params()
        pt.run(
            feature_extractor=M.FeatureExtractors.extract_atom_vec,
            predictor=model.predict_atom_charge,
            input_x_index=INPUT_X_INDEX,
            xyz_index=XYZ_INDEX,
            target_getter=lambda batch: batch.x[:, ATOM_CHRG_INDEX],
            # x_masker=pretrain.x_masker_func,
            loss_fn=F.mse_loss,
            # to_onehot=False,
            # onehot_types=ATOM_TYPES,
            # loss_weight_calculator=lambda t, n: M.atom_label_weight_(t, n, 'inverse-count'),
            metrics={
                'R^2 score': M.Metrics.r2_score,
                'RMSE': M.Metrics.rmse
            }
        )


if __name__ == '__main__':
    # atom_types()
    # atom_charges()
    pretrain.run(
        work_name="AtomType",
        work_dir=models_dir,
        core_model=core,
        train_dataset=tmqm_train,
        test_dataset=tmqm_test,
        hypers=hypers,
        epochs=EPOCHS,
        device=device,
        eval_steps=1,
        checkpoint_path=-1,
        load_core_only=True,
        save_model=False,
        x_masker=pretrain.x_masker_func,
        load_all_data=True,
        show_batch_pbar=True,
    )
