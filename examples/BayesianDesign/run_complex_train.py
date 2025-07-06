import sys
import os.path as osp
import socket
import json
from collections import defaultdict

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
elif str.split(__file__, '/')[1:4] == ['data', 'user', 'hd54396']:
    print('In zksl Super')
    project_root = '/data/user/hd54396/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
else:
    raise ValueError(__file__)

from hotpot.plugins.ComplexFormer import (
    models as M,
    tools,
    run
)
from hotpot.plugins.ComplexFormer.data.data_module import get_first_data

models_dir = osp.join(project_root, 'models')
# dataset save paths

if str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('in /dev/shm')
    dir_datasets = osp.join('/dev', 'shm', 'datasets')
else:
    dir_datasets = osp.join(project_root, 'datasets')


EPOCHS = 50
OPTIMIZER = torch.optim.Adam
X_ATTR_NAMES = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
X_DIM = len(X_ATTR_NAMES)
VEC_DIM = 128
MASK_VEC = (-1 * torch.ones(X_DIM)).to(device)
RING_LAYERS = 1
RING_HEADS = 2
MOL_LAYERS = 1
MOL_HEADS = 2

ATOM_TYPES = 119  # Arguments for atom type loss


hypers = tools.Hypers()
hypers.batch_size = 128
hypers.lr = 2e-4
hypers.weight_decay = 4e-5

core = M.Core(
    x_dim=X_DIM,
    vec_dim=VEC_DIM,
    x_label_nums=ATOM_TYPES,
    ring_layers=RING_LAYERS,
    ring_nheads=RING_HEADS,
    mol_layers=MOL_LAYERS,
    mol_nheads=MOL_HEADS,
    med_props_nums=22,
    sol_props_nums=34,
    with_sol_encoder=True,
    with_med_encoder=True,
)

def which_datasets_train(
        *datasets,
        work_name: str = None,
        debug: bool = False,
        **kwargs
):
    if not datasets:
        raise ValueError('No datasets')
    datasets = list(datasets)

    json_file = osp.join(osp.dirname(__file__), 'def_tasks.json')
    task_definition = json.load(open(json_file, 'r'))

    if len(datasets) == 1:
        feature_extractors = task_definition[datasets[0]]['feature_extractors']
        predictors = task_definition[datasets[0]]['predictors']
        target_getters = task_definition[datasets[0]]['target_getters']
        loss_fn = task_definition[datasets[0]]['loss_fn']
        primary_metric = task_definition[datasets[0]]['primary_metric']

        options = task_definition[datasets[0]].get('options', {})

    else:
        feature_extractors = [task_definition[ds]['feature_extractors'] for ds in datasets]
        predictors = [task_definition[ds]['predictors'] for ds in datasets]
        target_getters = [task_definition[ds]['target_getters'] for ds in datasets]
        loss_fn = [task_definition[ds]['loss_fn'] for ds in datasets]
        primary_metric = [task_definition[ds]['primary_metric'] for ds in datasets]

        _options = [task_definition[ds].get('options', {}) for ds in datasets]
        all_opt_keys = set(k for opt in _options for k in opt.keys())

        options = defaultdict(list)
        for key in all_opt_keys:
            for opt in _options:
                options[key].append(opt.get(key, None))

    options.update(kwargs)

    run.run(work_name=work_name, work_dir=models_dir, core=core, dir_datasets=dir_datasets, hypers=hypers,
            dataset_names=datasets, target_getter=target_getters, epochs=EPOCHS, feature_extractor=feature_extractors,
            predictor=predictors, loss_fn=loss_fn, primary_metric=primary_metric, xyz_perturb_sigma=0.5,
            load_all_data=True, debug=debug, device=device, eval_steps=1, **options)


def combined_training():
    work_name = 'MultiTask'
    feature_extractors = [
        {
            'AT': 'atom',  # AtomType,
            'MT': 'metal',  # MetalType
            'RA': 'ring',  # RingsAromatic
            'CB': 'cbond',  # CBond
            'BO': 'bond',  # BondOrder
            'PSP': 'pair',  # PairShortPath
            'WBO': 'pair',  # WibergBondOrder
            'xyzC': 'atom',  # xyzInCrystal

        },
        {
            'AT': 'atom',  # AtomType
            'AC': 'atom',  # AtomCharge
            'MT': 'metal',  # MetalType
            'ME': 'mol',  # MolEnergy
            'MDs': 'mol',  # MolDisp
            'MDi': 'mol',  # MolDipole
            'MMQ': 'mol',  # MolMetalQ
            'MH': 'mol',  # MolHl
            'MHM': 'mol',  # MolHOMO
            'MLM': 'mol',  # MolLOMO
            'MP': 'mol',  # MolPolar,
            'PSP': 'pair',  # PairShortPath
            'WBO': 'pair',  # WibergBondOrder,
            'xyzX': 'atom',  # xyzInXtb
        },
    ]
    predictors = [
        {
            'AT': 'onehot',
            'MT': 'onehot',
            'RA': 'binary',
            'CB': 'binary',
            'BO': 'onehot',
            'PSP': 'num',
            'WBO': 'num',
            'xyzC': 'xyz'
        },
        {
            'AT': 'onehot',
            'AC': 'num',
            'MT': 'onehot',
            'ME': 'num',
            'MDs': 'num',
            'MDi': 'num',
            'MMQ': 'num',
            'MH': 'num',
            'MHM': 'num',
            'MLM': 'num',
            'MP': 'num',
            'PSP': 'num',
            'WBO': 'num',
            'xyzX': 'xyz',
        },
    ]

    first_data = get_first_data(dir_datasets)
    target_getters = [
        {   # Tasks for mono datasets
            'AT': tools.TargetGetter(first_data['mono'], 'x', 'atomic_number'),
            'RA': tools.TargetGetter(first_data['mono'], 'rings_attr', 'is_aromatic'),
            'MT': lambda batch: batch.x[:, 0][M.where_metal(batch.x[:, 0])],
            'CB': lambda batch: batch.is_cbond,
            'BO': tools.TargetGetter(first_data['mono'], 'edge_attr', 'bond_order'),
            'PSP': tools.TargetGetter(first_data['mono'], 'pair_attr', 'length_shortest_path'),
            'WBO': tools.TargetGetter(first_data['mono'], 'pair_attr', 'wiberg_bond_order'),
            'xyzC': tools.TargetGetter(first_data['mono'], 'x', ('x', 'y', 'z')),
        },
        {  # Tasks for tqdm datasets
            'AT': tools.TargetGetter(first_data['tmqm'], 'x', 'atomic_number'),
            'AC': tools.TargetGetter(first_data['tmqm'], 'x', 'partial_charge'),
            'MT': lambda batch: batch.x[:, 0][M.where_metal(batch.x[:, 0])],
            'ME': tools.TargetGetter(first_data['tmqm'], 'y', 'energy'),
            'MDs': tools.TargetGetter(first_data['tmqm'], 'y', 'dispersion'),
            'MDi': tools.TargetGetter(first_data['tmqm'], 'y', 'dipole'),
            'MMQ': tools.TargetGetter(first_data['tmqm'], 'y', 'metal_q'),
            'MH': tools.TargetGetter(first_data['tmqm'], 'y', 'Hl'),
            'MHM': tools.TargetGetter(first_data['tmqm'], 'y', 'HOMO'),
            'MLM': tools.TargetGetter(first_data['tmqm'], 'y', 'LUMO'),
            'MP': tools.TargetGetter(first_data['tmqm'], 'y', 'polarizability'),
            'PSP': tools.TargetGetter(first_data['tmqm'], 'pair_attr', 'length_shortest_path'),
            'WBO': tools.TargetGetter(first_data['tmqm'], 'pair_attr', 'wiberg_bond_order'),
            'xyzX': tools.TargetGetter(first_data['tmqm'], 'x', ('x', 'y', 'z')),
        },
    ]
    loss_fn = [
        {
            'AT': 'cross_entropy',
            'MT': 'cross_entropy',
            'RA': 'binary_cross_entropy',
            'CB': 'binary_cross_entropy',
            'BO': 'cross_entropy',
            'PSP': 'mse',
            'WBO': 'mse',
            'xyzC': 'amd',
        },
        {
            'AT': 'cross_entropy',
            'AC': 'mse',
            'MT': 'cross_entropy',
            'ME': 'mse',
            'MDs': 'mse',
            'MDi': 'mse',
            'MMQ': 'mse',
            'MH': 'mse',
            'MHM': 'mse',
            'MLM': 'mse',
            'MP': 'mse',
            'PSP': 'mse',
            'WBO': 'mse',
            'xyzX': 'amd'
        },
    ]
    primary_metric = [
        {
            'AT': 'acc',
            'MT': 'macc',
            'RA': 'bacc',
            'CB': 'bacc',
            'BO': 'acc',
            'PSP': 'r2',
            'WBO': 'r2',
            'xyzC': 'amd',
        },
        {
            'AT': 'acc',
            'AC': 'r2',
            'MT': 'macc',
            'ME': 'r2',
            'MDs': 'r2',
            'MDi': 'r2',
            'MMQ': 'r2',
            'MH': 'r2',
            'MHM': 'r2',
            'MLM': 'r2',
            'MP': 'r2',
            'PSP': 'r2',
            'WBO': 'r2',
            'xyzX': 'amd',
        },
    ]

    run.run(work_name=work_name, work_dir=models_dir, core=core, dir_datasets=dir_datasets, hypers=hypers,
            target_getter=target_getters, epochs=EPOCHS, feature_extractor=feature_extractors, predictor=predictors,
            loss_fn=loss_fn, primary_metric=primary_metric, xyz_perturb_sigma=0.5, load_all_data=True, debug=True,
            device=device, eval_steps=1)


def multi_task():
    work_name = 'MultiTask'
    feature_extractors = {
            'AT': 'atom',  # AtomType,
            'MT': 'metal',  # MetalType
            'RA': 'ring',  # RingsAromatic
            'CB': 'cbond',  # CBond
            'BO': 'bond',  # BondOrder
            'PSP': 'pair',  # PairShortPath
            'WBO': 'pair',  # WibergBondOrder
            'xyzC': 'atom',  # xyzInCrystal

    }
    predictors =  {
            'AT': 'onehot',
            'MT': 'onehot',
            'RA': 'binary',
            'CB': 'binary',
            'BO': 'onehot',
            'PSP': 'num',
            'WBO': 'num',
            'xyzC': 'xyz'
    }
    first_data = get_first_data(dir_datasets)
    target_getters = {   # Tasks for mono datasets
            'AT': tools.TargetGetter(first_data['mono'], 'x', 'atomic_number'),
            'RA': tools.TargetGetter(first_data['mono'], 'rings_attr', 'is_aromatic'),
            'MT': lambda batch: batch.x[:, 0][M.where_metal(batch.x[:, 0])],
            'CB': lambda batch: batch.is_cbond,
            'BO': tools.TargetGetter(first_data['mono'], 'edge_attr', 'bond_order'),
            'PSP': tools.TargetGetter(first_data['mono'], 'pair_attr', 'length_shortest_path'),
            'WBO': tools.TargetGetter(first_data['mono'], 'pair_attr', 'wiberg_bond_order'),
            'xyzC': tools.TargetGetter(first_data['mono'], 'x', ('x', 'y', 'z')),
    }
    loss_fn = {
            'AT': 'cross_entropy',
            'MT': 'cross_entropy',
            'RA': 'binary_cross_entropy',
            'CB': 'binary_cross_entropy',
            'BO': 'cross_entropy',
            'PSP': 'mse',
            'WBO': 'mse',
            'xyzC': 'amd',
    }
    primary_metric = {
            'AT': 'acc',
            'MT': 'macc',
            'RA': 'bacc',
            'CB': 'bacc',
            'BO': 'acc',
            'PSP': 'r2',
            'WBO': 'r2',
            'xyzC': 'amd',
    }

    run.run(work_name=work_name, work_dir=models_dir, core=core, dir_datasets=dir_datasets, hypers=hypers,
            dataset_names=['mono'], target_getter=target_getters, epochs=EPOCHS, feature_extractor=feature_extractors,
            predictor=predictors, loss_fn=loss_fn, primary_metric=primary_metric, xyz_perturb_sigma=0.5, devices=1,
            load_all_data=True, device=device, eval_steps=1)



if __name__ == '__main__':
    # combined_training()
    # multi_task()
    which_datasets_train(
        'tmqm', 'mono', 'SclogK',
        # 'mono_ml_pair',
        work_name='MultiTask',
        debug=True,
        # devices=2,
        # with_sol=True,
        # with_med=True,
    )
