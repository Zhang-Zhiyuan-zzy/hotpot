import itertools
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
from hotpot.plugins.ComplexFormer import (
    models as M,
    run
)
from hotpot.cheminfo.AImodels.cbond.deploy.models import deploy
from hotpot.plugins.opti import ParamSpace, ParamSets

models_dir = osp.join(project_root, 'models')

if str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('in /dev/shm')
    dir_datasets = osp.join('/dev', 'shm', 'datasets')
else:
    dir_datasets = osp.join(project_root, 'datasets')


# Hyperparameters definition
# EPOCHS = 200
# OPTIMIZER = torch.optim.Adam
X_ATTR_NAMES = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
# X_DIM = len(X_ATTR_NAMES)
# VEC_DIM = 256
# MASK_VEC = (-1 * torch.ones(X_DIM)).to(device)
# RING_LAYERS = 2
# RING_HEADS = 2
# MOL_LAYERS = 2
# MOL_HEADS = 4
# GRAPH_LAYERS = 6



def load_hyper(hyper_dict):
    _hypers = ParamSets()
    for k, v in hyper_dict.items():
        setattr(_hypers, k, v)
    return _hypers


ATOM_TYPES = 119  # Arguments for atom type loss


hypers = ParamSets()
hypers.EPOCHS = 200
hypers.batch_size = 512
hypers.lr = 1e-3
hypers.weight_decay = 4e-6
hypers.ATOM_TYPES = 119
hypers.OPTIMIZER = torch.optim.Adam
hypers.X_DIM = len(X_ATTR_NAMES)
hypers.VEC_DIM = 128
hypers.RING_LAYERS = 2
hypers.RING_HEADS = 2
hypers.MOL_LAYERS = 4
hypers.MOL_HEADS = 2
hypers.GRAPH_LAYERS = 6
hypers.DIM_FEEDFORWARD = 2048


# core = M.Core(
#     x_dim=hypers.X_DIM,
#     vec_dim=hypers.VEC_DIM,
#     x_label_nums=hypers.ATOM_TYPES,
#     ring_layers=hypers.RING_LAYERS,
#     ring_nheads=hypers.RING_HEADS,
#     ring_encoder_kw={'dim_feedforward': hypers.DIM_FEEDFORWARD},
#     mol_layers=hypers.MOL_LAYERS,
#     mol_nheads=hypers.MOL_HEADS,
#     mol_encoder_kw={'dim_feedforward': hypers.DIM_FEEDFORWARD},
#     graph_layer=hypers.GRAPH_LAYERS,
#     med_props_nums=22,
#     sol_props_nums=34,
#     with_sol_encoder=True,
#     with_med_encoder=True,
# )

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

    study = run.run(
        work_name=work_name,
        work_dir=models_dir,
        # core=core,
        dir_datasets=dir_datasets,
        # hypers=hypers,
        dataset_names=datasets,
        target_getter=target_getters,
        checkpoint_path=checkpoint_path,
        feature_extractor=feature_extractors,
        predictor=predictors,
        loss_fn=loss_fn,
        primary_metrics=primary_metrics,
        other_metrics=other_metrics,
        xyz_perturb_sigma=0.25,
        load_all_data=True,
        debug=debug,
        device=device,
        eval_steps=1,
        early_stop_step=20,
        **options,
    )
    return study


def deploy_model():
    import os
    # Set the environment variables for detailed logging
    os.environ['TORCH_LOGS'] = "dynamic"
    os.environ['TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL'] = "u27"
    os.environ['TORCHDYNAMO_EXTENDED_DEBUG_CPP'] = "1"

    onnx_version = 21
    export_path = osp.join(hp.package_root, 'cheminfo', 'AImodels', 'cbond', 'onnx')
    for n, s in itertools.product((2, 4, 8), (64,)):
        deploy(
            work_dir=models_dir,
            export_path=export_path,
            checkpoint_path='/mnt/d/zhang/OneDrive/Papers/BayesDesign/results/cbond/CB_0.958/checkpoints/epoch=23-step=39792.ckpt',
            dir_datasets=osp.join(dir_datasets, 'mono_ml_pair'),
            extract_predictors='CB',
            opset_version=onnx_version,
            max_rings_nums=n,
            max_rings_size=s,
        )


def pretrain_model():
    hypers = ParamSets()
    EPOCHS = 5
    BATCH_SIZE = 64
    hypers.lr = 1e-3
    hypers.weight_decay = 4e-5
    hypers.ATOM_TYPES = None
    hypers.OPTIMIZER = torch.optim.Adam
    # hypers.X_DIM = 7
    hypers.VEC_DIM = 256
    hypers.EMB_TYPE = 'atom'  # atom, proj, or orbital
    hypers.RING_LAYERS = 1
    hypers.RING_HEADS = 1
    hypers.MOL_LAYERS = 1
    hypers.MOL_HEADS = 1
    hypers.GRAPH_LAYERS = 6
    hypers.DIM_FEEDFORWARD = 1024
    hypers.HIDDEN_DIM = 256
    hypers.HIDDEN_LAYERS = 4
    work_name = 'PreT'
    debug = False

    which_datasets_train(
        'tmqm',
        'mono',
        'SclogK',
        # 'mono_ml_pair',
        # 'SclogK_with_cb',
        # work_name='MultiTask',
        work_name=work_name + ('_debug' if debug else ''),
        hypers=hypers,
        debug=debug,
        devices=[0],
        # overfit_test=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        # show_pbar=False,
        # checkpoint_path='/data/user/hd54396/proj/models/PreTrain/logs/lightning_logs/20251123-232018/checkpoints/epoch=4-step=51430.ckpt',
        # stages='test',
        # loss_fn_wrap_tasks=['CB'],
        # refine=True,
        # early_stop_step=30,
    )


def CBondModel():
    space = ParamSpace()
    space.add_categorical_params('EPOCHS', [200])
    space.add_categorical_params('BATCH_SIZE', [512])
    space.add_float_params('lr', 1e-5, 1e-2, log=True)
    space.add_float_params('weight_decay', 4e-7, 4e-4, log=True)
    space.add_categorical_params('ATOM_TYPES', [119])
    space.add_categorical_params('OPTIMIZER', [torch.optim.Adam])
    space.add_categorical_params('X_DIM', [len(X_ATTR_NAMES)])
    space.add_int_params('VEC_DIM', 60, 720, step=60)
    space.add_int_params('RING_LAYERS', 1, 4)
    space.add_int_params('RING_HEADS', 1, 4)
    space.add_int_params('MOL_LAYERS', 2, 6)
    space.add_int_params('MOL_HEADS', 3, 6)
    space.add_int_params('GRAPH_LAYERS', 2, 10)
    space.add_int_params('DIM_FEEDFORWARD', 512, 4096, log=True)
    space.add_int_params('HIDDEN_DIM', 256, 4096, step=256)
    space.add_int_params('NUM_LAYERS', 2, 6)

    which_datasets_train(
        # 'tmqm', 'mono', 'SclogK',
        'mono_ml_pair',
        # 'SclogK_with_cb',
        # work_name='MultiTask',
        work_name='CBond',
        hypers=space,
        # debug=True,
        devices=[0],
        loss_fn_wrap_tasks=['CB'],
        # stages='test',
        # with_sol=True,
        # with_med=True,
        refine=True,
        # show_pbar=False,
        # checkpoint_path='/data/user/hd54396/proj/models/MDTask(3)/logs/lightning_logs/version_2/checkpoints/epoch=56-step=78375.ckpt',
        # checkpoint_path='/home/zz1/docker/proj/models/MDTask(3)/logs/lightning_logs/version_0/checkpoints/epoch=5-step=594.ckpt'
        # checkpoint_path='/data/user/hd54396/proj/models/MultiTask(1)/logs/lightning_logs/version_4/checkpoints/epoch=10-step=18238.ckpt',
        # checkpoint_path='/data/user/hd54396/proj/models/MultiTask(1)/logs/lightning_logs/version_7/checkpoints/epoch=74-step=124350.ckpt',
        early_stop_step=30,
        # test_only=True,
        # checkpoint_path=-1,
    )

def logKmodel():
    # space = ParamSpace()
    # space.add_categorical_params('EPOCHS', [200])
    # space.add_categorical_params('BATCH_SIZE', [512])
    # space.add_float_params('lr', 1e-5, 1e-2, log=True)
    # space.add_float_params('weight_decay', 4e-7, 4e-4, log=True)
    # space.add_categorical_params('ATOM_TYPES', [119])
    # space.add_categorical_params('OPTIMIZER', [torch.optim.Adam])
    # space.add_categorical_params('X_DIM', [len(X_ATTR_NAMES)])
    # space.add_int_params('VEC_DIM', 60, 720, step=60)
    # space.add_int_params('RING_LAYERS', 1, 4)
    # space.add_int_params('RING_HEADS', 1, 4)
    # space.add_int_params('MOL_LAYERS', 1, 6)
    # space.add_int_params('MOL_HEADS', 1, 6)
    # space.add_int_params('GRAPH_LAYERS', 2, 12)
    # space.add_int_params('DIM_FEEDFORWARD', 64, 4096, log=True)
    hypers = ParamSets()
    EPOCHS = 30
    BATCH_SIZE = 64
    hypers.lr = 1e-3
    hypers.weight_decay = 4e-5
    hypers.ATOM_TYPES = None
    hypers.OPTIMIZER = torch.optim.Adam
    # hypers.X_DIM = 7
    hypers.VEC_DIM = 256
    hypers.EMB_TYPE = 'atom'  # atom, proj, or orbital
    hypers.RING_LAYERS = 1
    hypers.RING_HEADS = 1
    hypers.MOL_LAYERS = 2
    hypers.MOL_HEADS = 1
    hypers.GRAPH_LAYERS = 6
    hypers.DIM_FEEDFORWARD = 1024
    hypers.HIDDEN_DIM = 256
    hypers.HIDDEN_LAYERS = 4

    return which_datasets_train(
        # 'mono_ml_pair',
        # 'tmqm', 'mono',
        'SclogK_with_cb',
        # 'SclogK',
        work_name = 'logK-optuna',
        target_metrics = 'logK',
        refine=True,
        checkpoint_path='/data/user/hd54396/proj/models/PreTrain/logs/lightning_logs/20251123-232018/checkpoints/epoch=4-step=51430-v1.ckpt',
        devices=1,
        hypers=hypers,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        # data_split_ratios=(0., 0., 1.)
        # debug=True,
        # show_pbar=False,
    )


def gibbs_test():
    # model_dir = '/data/user/hd54396/proj/models/logK-optuna/logs/lightning_logs/20251030-143123_0.904'
    model_dir = '/data/user/hd54396/proj/models/logK/logs/lightning_logs/10251139_0.904'
    hp_path = osp.join(model_dir, 'hparams.json')
    hp_dict = json.load(open(hp_path))
    hypers = load_hyper(hp_dict)

    ckpt_dir = osp.join(model_dir, 'checkpoints')
    ckpt_file = os.listdir(ckpt_dir)[0]
    ckpt_path = osp.join(ckpt_dir, ckpt_file)

    return which_datasets_train(
        # 'GibbsBeta_CB',
        'SclogK_with_cb',
        work_name = 'GibbsBeta_CB',
        target_metrics = 'logK',
        refine=True,
        checkpoint_path=ckpt_path,
        devices=[0],
        hypers=hypers,
        # data_split_ratios=(0., 0., 1.),
        stages='test',
        # debug=True,
        # show_pbar=False,
        external_datasets='GibbsBeta_CB'
    )


if __name__ == '__main__':
    # CBondModel()
    # deploy_model()
    # s = logKmodel()
    # gibbs_test()
    pretrain_model()
