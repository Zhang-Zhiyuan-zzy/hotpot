import sys
import os.path as osp
import socket

import torch

machine_name = socket.gethostname()
torch.set_default_dtype(torch.bfloat16)
if torch.cuda.is_available():
    if machine_name == '4090':
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# Initialize paths.
if machine_name == '4090':
    project_root = '/home/zzy/docker_envs/pretrain/proj'
elif machine_name == 'DESKTOP-G9D9UUB':
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
elif machine_name == 'docker':
    project_root = '/app/proj'
elif machine_name == '3090':
    project_root = '/home/zz1/docker/proj'
else:
    raise ValueError

# Import hotpot module
sys.path.append(osp.join(project_root, 'hotpot'))
from hotpot.plugins.complex_model import (
    models as M,
    pretrain,
    dataset as D,
)


models_dir = osp.join(project_root, 'models')
# dataset save paths
_tmqm_data_dir = osp.join(project_root, 'datasets', 'tmqm_data0207')

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
hypers.batch_size = 512
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

def atom_types():
    with pretrain.PretrainComplex(
        work_name="atom types",
        not_save=True,
        work_dir=models_dir,
        model=model,
        dataset_=dataset,
        dataset_test_=dataset_test,
        optimizer=OPTIMIZER,
        hypers=hypers,
        epochs=EPOCHS,
        device=device,
        eval_steps=1,
        eval_first=True,
    ) as pt:
        print(pt.work_dir)
        # pt.load_model_params()
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
            metrics={'accuracy': lambda p, t: M.Metrics.calc_oh_accuracy(p, t, is_onehot=True)}
        )


def main():
    ...


if __name__ == '__main__':
    print('run!')
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
        save_model=False,
        x_masker=pretrain.x_masker_func,
        load_all_data=True,
        show_batch_pbar=True,
        constant_lr=True,
        other_metric='metal_accuracy',
        # debug=True,
    )
