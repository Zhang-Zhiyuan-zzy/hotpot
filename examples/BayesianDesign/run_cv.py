"""
@File Name:        run_cv
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/19 20:05
@Project:          Hotpot
"""
import os
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
print(machine_name)
if machine_name == '4090':
    project_root = '/home/zzy/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'DESKTOP-G9D9UUB':  # 221 PC
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'LAPTOP-K2H04HI4':
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

import hotpot as hp
from hotpot.plugins.ComplexFormer.run import cv, datacls
from hotpot.cheminfo.AImodels.cbond.deploy.models import deploy
from hotpot.plugins.opti import ParamSpace, ParamSets

models_dir = osp.join(project_root, 'models')

if str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('in /dev/shm')
    dir_datasets = osp.join('/dev', 'shm', 'datasets')
else:
    dir_datasets = osp.join(project_root, 'datasets')


def which_cv_run(
        *datasets,
        work_name: str = None,
        log_dir: str = None,
        debug: bool = False,
        refine: bool = False,
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

    print(f'Defined Datasets: {list(task_definition.keys())}')
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

    cfg_args = datacls.ConfigArgs(
        target_getter=target_getters,
        feature_extractor=feature_extractors,
        predictor=predictors,
        loss_fn=loss_fn,
        primary_metrics=primary_metrics,
        other_metrics=other_metrics,
    )

    study = cv.run_cv(
        work_name=work_name,
        work_dir=models_dir,
        log_dir=log_dir,
        config_args=cfg_args,
        dir_datasets=dir_datasets,
        dataset_names=datasets,
        other_metrics=other_metrics,
        xyz_perturb_sigma=0.5,
        load_all_data=True,
        debug=debug,
        device=device,
        eval_steps=1,
        early_stop_step=20,
        **options,
    )
    return study


def gibbs_test():
    return which_cv_run(
        'GibbsBeta_CB',
        # 'SclogK_with_cb',
        work_name = 'GibbsBeta_CB',
        log_dir='/data/user/hd54396/proj/models/logK/logs/lightning_logs/10251139_0.904',
        target_metrics = 'logK',
        devices=[7],
        refine=True,
        # debug=True,
    )

if __name__ == '__main__':
    gibbs_test()
