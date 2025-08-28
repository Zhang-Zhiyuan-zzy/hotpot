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
from hotpot.utils import fmt_print

models_dir = osp.join(project_root, 'models')

if str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('in /dev/shm')
    dir_datasets = osp.join('/dev', 'shm', 'datasets')
else:
    dir_datasets = osp.join(project_root, 'datasets')


# Hyperparameters definition
EPOCHS = 200
OPTIMIZER = torch.optim.Adam
X_ATTR_NAMES = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
X_DIM = len(X_ATTR_NAMES)
VEC_DIM = 128
MASK_VEC = (-1 * torch.ones(X_DIM)).to(device)
RING_LAYERS = 1
RING_HEADS = 2
MOL_LAYERS = 4
MOL_HEADS = 4

ATOM_TYPES = 119  # Arguments for atom type loss


hypers = tools.Hypers()
hypers.batch_size = 512
hypers.lr = 1e-4
hypers.weight_decay = 4e-6

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
        refine: bool = False,
        checkpoint_path: str = None,
        **kwargs
):
    if not datasets:
        raise ValueError('No datasets')
    datasets = list(datasets)

    json_file = osp.join(osp.dirname(__file__), 'def_tasks.json')

    if not refine:
        task_definition = json.load(open(json_file, 'r'))['PreTrain']
    else:
        task_definition = json.load(open(json_file, 'r'))['Refine']

    if len(datasets) == 1:
        feature_extractors = task_definition[datasets[0]]['feature_extractors']
        predictors = task_definition[datasets[0]]['predictors']
        target_getters = task_definition[datasets[0]]['target_getters']
        loss_fn = task_definition[datasets[0]]['loss_fn']
        primary_metrics = task_definition[datasets[0]]['primary_metric']
        other_metrics = task_definition[datasets[0]].get('other_metrics', None)

        options = task_definition[datasets[0]].get('options', {})

    else:
        feature_extractors = [task_definition[ds]['feature_extractors'] for ds in datasets]
        predictors = [task_definition[ds]['predictors'] for ds in datasets]
        target_getters = [task_definition[ds]['target_getters'] for ds in datasets]
        loss_fn = [task_definition[ds]['loss_fn'] for ds in datasets]
        primary_metrics = [task_definition[ds]['primary_metric'] for ds in datasets]
        other_metrics = [task_definition[ds].get('other_metrics', None) for ds in datasets]

        _options = [task_definition[ds].get('options', {}) for ds in datasets]
        all_opt_keys = set(k for opt in _options for k in opt.keys())

        options = defaultdict(list)
        for key in all_opt_keys:
            for opt in _options:
                options[key].append(opt.get(key, None))

    options.update(kwargs)

    run.run(
        work_name=work_name,
        work_dir=models_dir,
        core=core,
        dir_datasets=dir_datasets,
        hypers=hypers,
        dataset_names=datasets,
        target_getter=target_getters,
        epochs=EPOCHS,
        checkpoint_path=checkpoint_path,
        feature_extractor=feature_extractors,
        predictor=predictors,
        loss_fn=loss_fn,
        primary_metrics=primary_metrics,
        other_metrics=other_metrics,
        xyz_perturb_sigma=0.5,
        load_all_data=True,
        debug=debug,
        device=device,
        eval_steps=1,
        **options,
    )


if __name__ == '__main__':
    which_datasets_train(
        # 'tmqm', 'mono', 'SclogK',
        'mono_ml_pair',
        # work_name='MultiTask',
        work_name='CBond',
        # debug=True,
        devices=1,
        loss_fn_wrap_tasks=['CB'],
        # with_sol=True,
        # with_med=True,
        refine=True,
        # show_pbar=False,
        # checkpoint_path='/data/user/hd54396/proj/models/MDTask(3)/logs/lightning_logs/version_2/checkpoints/epoch=56-step=78375.ckpt',
        # checkpoint_path='/home/zz1/docker/proj/models/MDTask(3)/logs/lightning_logs/version_0/checkpoints/epoch=5-step=594.ckpt'
        # checkpoint_path='/data/user/hd54396/proj/models/MultiTask(1)/logs/lightning_logs/version_4/checkpoints/epoch=10-step=18238.ckpt',
        # checkpoint_path='/data/user/hd54396/proj/models/MultiTask(1)/logs/lightning_logs/version_7/checkpoints/epoch=74-step=124350.ckpt',
        early_stop_step=20,
        # test_only=True,
        # checkpoint_path=-1,
    )
