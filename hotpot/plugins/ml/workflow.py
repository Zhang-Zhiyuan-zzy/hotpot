"""
python v3.9.0
@Project: hotpot
@File   : workflow
@Auther : Zhiyuan Zhang
@Data   : 2024/1/13
@Time   : 9:30
Notes: The convenient workflows for training ordinary(non-deep) machine learning models
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Union, Sequence, Literal, Callable
from multiprocessing import Process, Queue
from copy import copy
from itertools import combinations

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.base import clone, BaseEstimator
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, minmax_scale
from sklearn.model_selection import LeaveOneOut, train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, BaseDecisionTree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBRegressor

import shap
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors

from hotpot.plots import SciPlotter, R2Regression, PearsonMatrix, HierarchicalTree, FeatureImportance, SHAPlot, Pearson


def leave_one_out_score(X: np.ndarray, y: np.ndarray, model) -> float:
    """"""
    split_method = LeaveOneOut()
    pred, true = [], []
    for train_index, val_index in split_method.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]

        _model = clone(model)
        _model.fit(X_train, y_train)
        p_val = _model.predict(X_val, return_std=True)
        pred.append(p_val)
        true.append(y_val)

    pred, true = np.concatenate(pred), np.concatenate(true)

    return r2_score(true, pred)


def expectation_improvement(y, mu, sigma):
    ymax = np.max(y)

    diff = mu - ymax
    u = diff / sigma

    ei = diff * norm.cdf(u) + sigma * norm.pdf(u)
    ei[sigma <= 0.] = 0.
    return ei


def get_rdkit_mol_descriptor(list_smi) -> pd.DataFrame:
    attr3d = [
        'InertialShapeFactor', "NPR1", 'NPR2', 'Asphericity', 'Eccentricity',
        'PMI1', 'PMI2', 'PMI3', 'RadiusOfGyration', 'SpherocityIndex'
    ]

    list_mol = []
    for smi in list_smi:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        list_mol.append(mol)

    ligand_2d = pd.DataFrame([CalcMolDescriptors(m) for m in list_mol])
    ligand_3d = pd.DataFrame([{n: getattr(Chem.Descriptors3D, n)(mol) for n in attr3d} for mol in list_mol])

    return pd.concat([ligand_2d, ligand_3d], axis=1)


class Params:
    def __init__(self, name: str, params: Sequence):
        self.name = name
        self.params = list(params)

        # transform discrete labels to integer index
        if any(isinstance(p, str) for p in self.params):
            self._param = [i for i in range(len(params))]
            self.is_labels = True
        else:
            self._param = self.params
            self.is_labels = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, N={len(self.params)})"

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        return self.params

    def __getitem__(self, item):
        return self.params[item]

    def invert(self, value):
        """ given the labels params, return the actual label """
        return self.params[self._param.index(value)]


class ParamsRange(Params):
    def __init__(self, name: str, start: Union[int, float], stop: Union[int, float], step: [int, float] = None):
        if (isinstance(start, int) and isinstance(stop, int)) or (isinstance(start, float) and isinstance(stop, float)):
            if start < stop:
                self.start = start
                self.stop = stop
            else:
                raise ValueError('The start must larger than stop!')
        else:
            raise TypeError('The start and stop should both int or both float!')

        if isinstance(self.start, int):
            if step is None:
                self.step = 1
            elif isinstance(step, int):
                self.step = step
            else:
                raise TypeError('The step should be int, same with start and stop')
        else:
            if step is None:
                self.step = (self.stop - self.start) / 10
            elif isinstance(step, float):
                self.step = step
            else:
                raise TypeError('The step should be float, same with start and stop')

        self.num_params = int((self.stop - self.start) // self.step)
        params = [self.start + i*self.step for i in range(self.num_params)]
        super().__init__(name, params)


class ParamsSpace:
    def __init__(self, *params: Params):
        self._params = params
        self.embed_params = self._embedding()

        self.tried_params = None
        self.tried_scores = None
        self.tried_indices = None

    def __getitem__(self, item) -> dict:
        embde_param = self.embed_params[item]
        return {p.name: p.invert(ep) for p, ep in zip(self._params, embde_param)}

    def _check(self):
        """ Check if the given params have duplicate names """

    def invert(self, param_values):
        return [p.invert(v) for p, v in zip(self._params, param_values)]

    @property
    def optimal_params(self) -> dict:
        return self.tried_params[np.argmax(self.tried_scores)]

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self._params]

    def sample(self, seed=None) -> (int, dict):
        np.random.seed(seed)
        idx = np.random.randint(self.size)
        return idx, self[idx]

    @property
    def size(self) -> int:
        return len(self.embed_params)

    def to_params_score_sheet(self) -> pd.DataFrame:
        df = pd.DataFrame(self.tried_params)
        df['Scores'] = self.tried_scores

        return df

    def _embedding(self):
        mesh_params = np.meshgrid(*[np.array(getattr(p, '_param')) for p in self._params])
        design_params = np.vstack([p.flatten() for p in mesh_params]).T

        return design_params


class WorkFlow:
    """ The convenient workflows for training ordinary(non-deep) machine learning models """
    def __init__(
            self, data: pd.DataFrame, scaler=StandardScaler, model=GradientBoostingRegressor,
    ):
        self.data = data
        self.Xscaler = scaler()
        self.yscaler = scaler()
        self._model = model

        self.features = None
        self.target = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def cross_validation(self, split_method=KFold(), model_kwargs: dict = None) -> np.ndarray:
        if model_kwargs is None:
            model_kwargs = {}
        if self.X_train is None or self.y_train is None:
            self.X_train = self.X
            self.y_train = self.y

        pred, true = [], []
        for train_index, val_index in split_method.split(self.X_train):
            X_train, y_train = self.X_train[train_index], self.y_train[train_index]
            X_val, y_val = self.X_train[val_index], self.y_train[val_index]

            model = self._model(**model_kwargs)
            model.fit(X_train, y_train)
            pred.append(model.predict(X_val))
            true.append(y_val)

        pred, true = np.concatenate(pred), np.concatenate(true)
        return np.vstack([true, pred])

    def gaussian_process_cross_validation(self, split_method=LeaveOneOut(), model_kwargs: dict = None):
        if self.X_train is None or self.y_train is None:
            self.X_train = self.X
            self.y_train = self.y

        if model_kwargs is None:
            model_kwargs = {
                "kernel": ConstantKernel() * RBF(np.ones(self.X_train.shape[1]), (1e-6, 1e6)) + WhiteKernel(),
                "n_restarts_optimizer": 50
            }

        pred, true, list_sigma = [], [], []
        for train_index, val_index in split_method.split(self.X_train):
            X_train, y_train = self.X_train[train_index], self.y_train[train_index]
            X_val, y_val = self.X_train[val_index], self.y_train[val_index]

            model = GaussianProcessRegressor(**model_kwargs)
            model.fit(X_train, y_train)
            mu, sigma = model.predict(X_val,return_std=True)
            pred.append(mu)
            true.append(y_val)
            list_sigma.append(sigma)

        pred, true = np.concatenate(pred), np.concatenate(true)
        return np.vstack([true, pred]), np.array(list_sigma)

    def feature_engineering(self):
        pass

    @staticmethod
    def metric(pt: np.ndarray, work_type: Literal['reg', 'cls'] = 'reg'):
        metric_methods = {
            'reg': {'R2': r2_score, 'MAE': mean_absolute_error, 'RMSE': lambda p, t: np.sqrt(mean_squared_error(p, t))},
        }
        methods = metric_methods[work_type]

        metrics_results = {n: m(*pt) for n, m in methods.items()}
        for name, result in metrics_results.items():
            print(f"{name}: {result}")

        return metrics_results

    def params_optim(
            self, X, y, params_space: ParamsSpace, split_method=LeaveOneOut(), seed=None, score=r2_score,
            n_init: int = 10, n_iter: int = 10, n_job=10
    ):
        """"""
        def _initialize(queue):
            idx_, kwargs_ = params_space.sample(seed)
            queue.put((idx_, kwargs_, score(*self.cross_validation(split_method, model_kwargs=kwargs_))))

        def get_processes_data():
            for proc, queue in processes.items():
                proc.join()
                idx_, kwargs_, scores = queue.get()
                indices.append(idx_)
                list_kwargs.append(kwargs_)
                list_score.append(scores)

            processes.clear()

        processes = {}
        indices, list_kwargs, list_score = [], [], []
        for i in tqdm(range(n_init), "Initialize"):
            q = Queue()
            p = Process(name=str(i), target=_initialize, args=(q,))
            p.start()
            processes[p] = q

            if len(processes) % n_job == 0:
                get_processes_data()

        get_processes_data()

        for i in range(n_iter):
            kernel = ConstantKernel(1.0, (1e-6, 1e6)) * RBF(np.ones(len(params_space.param_names)), (1e-6, 1e6)) + \
                     WhiteKernel(1.0, (1e-6, 1e6))
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=50, normalize_y=True)

            gp.fit(params_space.embed_params[indices], np.array(list_score))
            mu, sigma = gp.predict(params_space.embed_params, return_std=True)
            ei = expectation_improvement(np.array(list_score), mu, sigma)

            idx = np.argmax(ei)
            model_kwargs = params_space[idx]
            pt = self.cross_validation(split_method, model_kwargs=model_kwargs)
            list_score.append(score(*pt))

            indices.append(idx)
            list_kwargs.append(model_kwargs)

            print(f"iter {i}: Score={list_score[-1]}")

        # Store the tried params
        params_space.tried_params, params_space.tried_scores, params_space.tried_indices = \
            list_kwargs, list_score, indices

        return indices, list_kwargs, list_score

    def _params_optim(self, params_space: ParamsSpace, score_func: Callable):
        """"""
        pass

    def plot_r2(self, pt1, pt2=None, err1=None, err2=None, show=True) -> (plt.Figure, plt.Axes):
        r2_plotter = R2Regression(pt1, pt2, err1=err1, err2=err2)
        sci_plot = SciPlotter(r2_plotter)
        fig, ax = sci_plot()
        if show:
            fig.show()

        return fig, ax

    def plot_pearson_matrix_and_tree(
            self, X: np.ndarray, labels: list[str], show_values=False
    ) -> (plt.Figure, np.ndarray[plt.Axes]):
        pmat_plotter = PearsonMatrix(X, labels, show_values=show_values)
        tree_plotter = HierarchicalTree(X, labels)

        sci_plot = SciPlotter([[pmat_plotter], [tree_plotter]])
        fig, axs = sci_plot()

        return fig, axs

    def preprocessing(
            self, Xoffset: int = 0, Xindex: Union[Sequence, np.ndarray] = None, y_index: int = -1
    ) -> (np.ndarray, np.ndarray):
        if Xindex is not None:
            X, y = self.data.iloc[:, Xindex], self.data.iloc[:, y_index]
        else:
            X, y = self.data.drop(columns=[self.data.columns[y_index]]).iloc[:, Xoffset:], self.data.iloc[:, y_index]

        self.features, self.target = X.columns, y.name
        self.Xscaler.fit(X.values)
        self.yscaler.fit(y.values.reshape(-1, 1))

        self.X = self.Xscaler.transform(X.values)
        self.y = self.yscaler.transform(y.values.reshape(-1, 1)).flatten()

        return self.X, self.y

    def split_train_test(self, test_size=0.2, random_state=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, model_kwargs: dict = None):
        if model_kwargs is None:
            model_kwargs = {}
        if self.X_train is None or self.y_train is None:
            self.X_train = self.X
            self.y_train = self.y

        model = self._model(**model_kwargs)
        model.fit(self.X_train, self.y_train)
        return model


class SklearnWorkFlow:
    """
    The convenient workflows based on sklearn module.
    Examples:
        X, y, other_info = read_data_(...)
        work_dir = Path('...')

        workflow = wf.SklearnWorkFlow(
            X, y, work_dir, other_info=other_info,
            seed=_seed,
            test_indices=test_indices,
            show_mae=True, show_rmse=True,
            estimator=GradientBoostingRegressor(),
            max_feature=6,
            required_features=required_feature
        )
        workflow()  # Run the workflow, get the results from work_dir
    """
    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            work_dir: Union[str, os.PathLike],
            other_info: Union[pd.Series, pd.DataFrame] = None,
            split_stratify: int = None,
            stratify_style: Literal['linear', 'logarithm'] = 'linear',
            feature_measurer=GradientBoostingRegressor(),
            estimator=XGBRegressor(),
            **kwargs
    ):
        """
        Args:
            X(pandas.DataFrame): all feature dataset
            y(pandas.Series): all target dataset
            work_dir: working directory for saving data and pictures when workflow running
            other_info: additional info with same order with X and y
            feature_measurer: feature measurer in feature engineering
            estimator: machine learning estimator

        Keyword Args:
            required_features: list of features to be required for model building.
            fixed_features: skip the feature engineering, training and testing model by the given features
            test_indices: Specify which samples to be used for testing
            show_mae: Whether to show the MAE value in the prediction plot
            show_rmse: Whether to show the RMSE value in the prediction plot
        """
        if isinstance(other_info, (pd.Series, pd.DataFrame)):
            if (X.index == other_info.index).all():
                self.other_info = other_info
            else:
                raise ValueError('the given other_info must have the identical index with X')
        else:
            self.other_info = None

        self.X = X.values
        self.y = y.values.flatten()
        self.feature_names = X.columns.to_numpy()
        self.target_name = y.name
        self.samples_index = X.index.to_numpy()

        self.split_stratify = split_stratify
        self.stratify_style = stratify_style

        self.feature_measurer = feature_measurer
        self.estimator = estimator
        self.seed = kwargs.get("seed", np.random.randint(1000000))
        self.kwargs = kwargs

        # Create work dirs
        self.work_dir = Path(work_dir)
        self.picture_dir = self.work_dir.joinpath('pictures')
        self.sheet_dir = self.work_dir.joinpath('sheet')
        if self.work_dir.is_dir():
            if not self.picture_dir.exists():
                self.picture_dir.mkdir()
            if not self.sheet_dir.exists():
                self.sheet_dir.mkdir()
        else:
            raise NotADirectoryError(f"Output directory {self.work_dir} is not a directory!")

        self.xscalar = None
        self.yscaler = None
        self.scaled_X = None
        self.scaled_y = None
        self.cov4scaledX = None

        self.idx_train = None
        self.idx_test = None
        self.sample_idx_train = None
        self.sample_idx_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.feature_importance = None
        self.feature_drop_index = None

        self.pearson_mat = None
        self.feature_cluster = None
        self.cluster2feature = {}
        self.cluster2feature_format = 'dict {cluster_name: [array of feature names, array of feature components]}'

        self.necessary_feature_indices = None
        self.necessary_features = None
        self.necessary_X = None
        self.necessary_feature_importance = None
        self.cov4necsX = None

        self.p_cross = None
        self.p_train = None
        self.p_test = None
        self.pt_diff_train = None
        self.pt_diff_test = None

        self.explainer = None
        self.shap_value = None

        # Status control argument to indicate whether the automatic workflow is running.
        self.running = False

    def __call__(self, *args, **kwargs):
        """

        Args:
        Keyword Args:
            make_plot: Whether to make plots and save them in the output_dir/pictures dir
            save_sheet: Whether to save results to a sheet in the output_dir/sheet dir
        """
        self.running = True

        self.save_input_data()
        self.preprocessing()
        # self._split_train_test()

        if self.kwargs.get('fixed_features'):
            # Skip the feature engineering
            _nf_names = self.kwargs['fixed_features']  # necessary feature names in input pipe
            necessary_features_indices = np.isin(self.feature_names, _nf_names)
            self.necessary_features = self.feature_names[necessary_features_indices].tolist()
            self.necessary_X = self.scaled_X[:, necessary_features_indices]

            reg = GradientBoostingRegressor()
            reg.fit(self.necessary_X, self.scaled_y)
            self.necessary_feature_importance = reg.feature_importances_

        else:
            # Feature Engineering
            self.determine_feature_importance(*args, **kwargs)
            self._filter_feature_by_importance()
            self._cluster_by_hierarchical_tree()
            self.pca_dimension_reduction()
            # self._determine_necessary_features()
            self._determine_final_features()
            self.make_imp_plot_for_necessary_feat()

        self.cross_validation(*args, **kwargs)
        self.make_prediction_target_plot(*args, **kwargs)
        self.shap_analysis()
        self.train_final_model()
        self.pickle_workflow()

        self.running = False

    def save_input_data(self):
        with pd.ExcelWriter(self.sheet_dir.joinpath('input_data.xlsx')) as writer:
            pd.DataFrame(self.X, columns=self.feature_names, index=self.samples_index).to_excel(writer, sheet_name='X')
            pd.Series(self.y, index=self.samples_index, name=self.target_name).to_excel(writer, sheet_name='y')
            if isinstance(self.other_info, (pd.DataFrame, pd.Series)):
                self.other_info.to_excel(writer, sheet_name='other_info')

    @staticmethod
    def _get_seed(X, y, estimitor, max_=10000):
        list_scores = []
        for i in tqdm(range(max_)):
            list_scores.append(
                np.mean([
                    cross_val_score(estimitor, X, y, cv=KFold(shuffle=True, random_state=i))
                    for _ in range(10)
                ])
            )

        return list_scores, np.argmax(list_scores)

    @staticmethod
    def _get_best_split(X, y, estimitor, test_size=0.2, max_=10000):
        list_scores = []
        train_test_indices = []
        for seed in tqdm(range(max_)):
            i_train, i_test, X_train, X_test, y_train, y_test = (
                train_test_split(np.arange(len(X)), X, y, shuffle=True, test_size=test_size, random_state=seed))

            estimitor.fit(X_train, y_train)
            list_scores.append(estimitor.score(X_test, y_test))

            train_test_indices.append([i_train, i_test])

        return np.array(list_scores), train_test_indices

    def get_best_split(self, test_size=0.2, max_=10000):
        if self.kwargs.get('get_best_split'):
            scores, indices = self._get_best_split(self.necessary_X, self.scaled_y, self.estimator, test_size, max_)

            print(f'Best score: {max(scores)}')
            best_one = np.argmax(scores)
            train_indices, test_indices = indices[best_one]

    def get_seed(self, *args, **kwargs):
        """"""
        if args or kwargs.get('optimize'):
            scores = []
            for i in tqdm(range(10000)):
                scores.append(
                    cross_val_score(self.estimator, self.scaled_X, self.scaled_y,
                                    cv=KFold(shuffle=True, random_state=i), n_jobs=-1).mean()
                )

            print(np.argmax(np.array(scores)), print(scores[np.argmax(np.array(scores))]))
            return np.argmax(np.array(scores))
        else:
            return np.random.randint(1000000)

    def determine_feature_target_pearson(self, *args, **kwargs):
        """ Discover the pearson correlation between features and target """
        if not self.picture_dir.joinpath('Pearson').exists():
            self.picture_dir.joinpath('Pearson').mkdir()

        pearson_dir = self.picture_dir.joinpath('Pearson')
        for col, f_name in tqdm(zip(range(self.X.shape[1]), self.feature_names), 'Make Pearson plots'):
            x = self.X[:, col]
            fig, ax = SciPlotter(Pearson([x, self.y], f_name, self.target_name))()

            fig.savefig(pearson_dir.joinpath(f'{f_name}.png'))

    def get_sample(self, index, get_all=False):
        """"""
        index = self.samples_index == index
        X = self.X[index]
        y = self.y[index]
        if not get_all and self.feature_drop_index and self.necessary_features is not None:
            X = X[:, self.feature_drop_index:]
            feature_names = self.necessary_features
        else:
            feature_names = self.feature_names

        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        return pd.DataFrame(
            data, columns=feature_names.tolist() + [self.target_name], index=self.samples_index[index])

    def preprocessing(self):
        """"""
        self.xscalar = self.kwargs.get('xscalar', MinMaxScaler())
        self.yscaler = self.kwargs.get('yscaler')

        if self.xscalar:
            self.xscalar.fit(self.X)
            self.scaled_X = self.xscalar.transform(self.X)
        else:
            self.scaled_X = copy(self.X)

        if self.yscaler:
            self.yscaler.fit(self.y.reshape(-1, 1)).flatten()
            self.scaled_y = self.yscaler.transform(self.y.reshape(-1, 1)).flatten()
        else:
            self.scaled_y = copy(self.y)

    @staticmethod
    def _sorted_features_by_importance(feature_names: np.ndarray, importance, ascending=True):
        sorted_indices = np.argsort(importance)
        if not ascending:
            sorted_indices = sorted_indices[::-1]

        feature_names = feature_names[sorted_indices]
        importance = importance[sorted_indices]

        return feature_names, importance, sorted_indices

    def determine_feature_importance(
            self,
            sort_f_by_imp=True,
            ascending=True,
            make_plot=True,
            *args, **kwargs
    ):
        """
        Estimate the feature importance by a given sklearn estimator with fit() method and feature_importance_ attr
        Args:
            estimator: estimator with fit() and feature_importance.
            sort_f_by_imp: whether to sort the feature according to the feature importance, by descending order.
            ascending: if sort_f_by_imp is true, sort features by ascending order.
            make_plot: whether to make plots for showing feature importance and save them in the output_dir/pictures dir

        Keyword Args:
            force_show_label: whether to force show the labels of the feature labels in importance plot
        """
        estimator = clone(self.feature_measurer)
        estimator.fit(self.scaled_X, self.scaled_y)

        feature_names, importance = self.feature_names, estimator.feature_importances_
        if sort_f_by_imp:
            self.feature_names, self.feature_importance, sorted_indices = \
                self._sorted_features_by_importance(feature_names, importance)

            self.X, self.scaled_X = self.X[:, sorted_indices], self.scaled_X[:, sorted_indices]
        else:
            self.feature_importance = estimator.feature_importances_

        # Sort the feature_names X values and scaled_X values
        if make_plot:
            sciplot = SciPlotter(
                FeatureImportance(
                    self.feature_names[::-1],
                    self.feature_importance[::-1],
                    force_show_label=kwargs.get('force_show_label', False)
                )
            )
            fig, ax = sciplot()
            fig.savefig(self.work_dir.joinpath('pictures', f'feature_importance.png'))

        # Save feature importance to sheet
        df = pd.DataFrame([self.feature_names, self.feature_importance], columns=range(len(self.feature_names))).T
        df.to_excel(self.sheet_dir.joinpath('feature_importance.xlsx'))

    def _filter_feature_by_importance(self, imp_threshold=1e-3):
        """ filter raw features according to given threshold of feature importance """
        filtered_indices = self.feature_importance >= imp_threshold
        filtered_feature_names = self.feature_names[filtered_indices]
        filtered_scalered_X = self.scaled_X[:, filtered_indices]
        filtered_X = self.X[:, filtered_indices]

        estimator = clone(self.feature_measurer)
        estimator.fit(filtered_scalered_X, self.scaled_y)

        filtered_feature_names, filtered_feature_importance, sorted_indices = self._sorted_features_by_importance(
            filtered_feature_names, estimator.feature_importances_)

        self.feature_names = filtered_feature_names
        self.feature_importance = filtered_feature_importance
        self.scaled_X = filtered_scalered_X[:, sorted_indices]
        filtered_X = filtered_X[:, sorted_indices]

        # Save filtered feature sheets to disk
        with pd.ExcelWriter(self.sheet_dir.joinpath('filtered_X_by_importance.xlsx')) as writer:
            df_importance = pd.DataFrame(
                [self.feature_names, self.feature_importance], columns=range(len(self.feature_names))).T
            df_X = pd.DataFrame(filtered_X, columns=filtered_feature_names, index=self.samples_index)
            df_scaled_X = pd.DataFrame(self.scaled_X, columns=filtered_feature_names, index=self.samples_index)

            df_importance.to_excel(writer, sheet_name='importance')
            df_X.to_excel(writer, sheet_name='X')
            df_scaled_X.to_excel(writer, sheet_name='scaled_X')

    def _determine_features_by_performance_dropdown(self):
        """ sorted features by performance dropdown """
        base_score = np.mean([cross_val_score(self.estimator, self.scaled_X, self.scaled_y, n_jobs=-1) for _ in range(5)])

        performance_dropdown = []
        for i, feature_name in tqdm(enumerate(self.feature_names), 'sort feature by performance dropdown'):
            f = np.delete(self.scaled_X, i, axis=1)
            logging.info(f'the feature dropped {i} column with a shape {f.shape}')

            performance_dropdown.append(
                np.mean([
                    cross_val_score(
                        self.estimator, f, self.scaled_y, n_jobs=-1, cv=KFold(n_splits=5, shuffle=True, random_state=self.seed)
                    ) - base_score
                    for _ in range(5)]
                ))

        self.performance_dropdown = np.array(performance_dropdown)

        # Save to sheet
        series = pd.Series(performance_dropdown, index=self.feature_names, name='feature_dropdown')
        series.to_excel(self.sheet_dir.joinpath('feature_performance_dropdown.xlsx'))

    @staticmethod
    def pearson_matrix(
            X: np.ndarray, savefig_path: Union[str, os.PathLike] = None, feature_names: Union[np.ndarray] = None,
    ) -> (np.ndarray, float):
        """ calculate pearson correlation coefficient among all features and make the matrix plot """
        if feature_names is None:
            feature_names = [str(i) for i in range(X.shape[1])]

        # Make plot
        if savefig_path:
            sciplot = SciPlotter(PearsonMatrix(X, feature_names))
            fig, ax = sciplot()
            fig.savefig(savefig_path)

        p_mat = np.corrcoef(X, rowvar=False)
        # Calculate the mean absolute correlation.
        abs_p_mat = np.abs(p_mat)
        np.fill_diagonal(abs_p_mat, 0)
        mean_abs_corr = abs_p_mat.sum() / (abs_p_mat.size - abs_p_mat.shape[0])
        print(f'Mean absolute correlation coefficient: {mean_abs_corr: 4f}')

        return np.corrcoef(X, rowvar=False), mean_abs_corr

    @staticmethod
    def hierarchical_tree(
            X: np.ndarray,
            savefig_path: Union[str, os.PathLike] = None,
            feature_names: Union[list, np.ndarray] = None,
            threshold: float = None
    ) -> np.ndarray:
        """
        Carry out the hierarchical clustering and visualize the hierarchical clustering tree.
        Args:
            X: input data
            savefig_path: path to save the hierarchical tree plot
            feature_names:
            threshold: threshold for clustering

        Returns:
            clustering labels.
        """
        if feature_names is None:
            feature_names = [str(i) for i in range(X.shape[1])]

        threshold = threshold if threshold else np.sqrt(pow(0.1, 2) * len(feature_names))

        # make hierarchical_tree plot
        if savefig_path:
            tree_kwargs = {
                'figwidth': 20,
                'figheight': 6.4,
                'axes_position': (0.05, 0.1, 0.925, 0.85),
                'ticklabels_fontsize': 22
            }
            sciplot = SciPlotter(HierarchicalTree(X, range(len(feature_names)), threshold),
                                 **tree_kwargs)
            fig, ax = sciplot()
            fig.savefig(savefig_path)

        abs_correlation_mat = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
        clustering.fit(abs_correlation_mat)

        return clustering.labels_

    def _cluster_by_hierarchical_tree(self, threshold=None):
        """ clusters by hierarchical tree """
        self.pearson_mat, _ = self.pearson_matrix(
            self.scaled_X, self.work_dir.joinpath('pictures', 'filtered_pearson_mat.png'), self.feature_names)

        self.feature_cluster = self.hierarchical_tree(
            self.scaled_X, self.work_dir.joinpath('pictures', 'filtered_hierarchical_tree.png'),
            self.feature_names, threshold
        )

    def pca_dimension_reduction(self):
        """ reduce dimension for features in same cluster by principal component analysis """
        cluster_indices = self.feature_cluster
        scaled_X = self.scaled_X
        feature_names = self.feature_names

        uni_clt, clt_counts = np.unique(cluster_indices, return_counts=True)

        clt2features = {}
        pca_X = []
        clt_Xidx = []
        for clt, count in zip(uni_clt, clt_counts):
            if count > 1:
                x_idx_clt = np.nonzero(np.int_(cluster_indices == clt))[0]
                clt_Xidx.append(x_idx_clt)

                clt_x = scaled_X[:, x_idx_clt]
                clt_name = feature_names[x_idx_clt]

                pca = PCA(n_components=1)
                pca_x = pca.fit_transform(clt_x).flatten()

                pca_X.append(pca_x)
                clt2features[f'c{len(clt2features)}'] = np.array([clt_name, pca.components_.flatten()])

        pca_X = np.array(pca_X).T
        clt_Xidx = np.hstack(clt_Xidx)
        nonclt_Xidx = np.isin(np.arange(len(cluster_indices)), clt_Xidx, invert=True)

        nonclt_X = scaled_X[:, nonclt_Xidx]
        nonclt_feature_names = feature_names[nonclt_Xidx]

        update_scaled_X = np.hstack((nonclt_X, pca_X))
        update_feature_name = np.hstack((nonclt_feature_names, list(clt2features.keys())))

        self.scaled_X = update_scaled_X
        self.feature_names = update_feature_name
        self.cluster2feature = clt2features

        # Calculate the covariance matrix for scaled_X
        self.cov4scaledX = np.cov(self.scaled_X.T)

        # Save cluster to feature mapping
        with pd.ExcelWriter(self.sheet_dir.joinpath('cluster2feature.xlsx')) as writer:
            for clt_name, (feature_name, components) in self.cluster2feature.items():
                series = pd.Series(np.float_(components), index=feature_name, name='components')
                series.to_excel(writer, sheet_name=clt_name)

    def _determine_necessary_features(self, improve_threshold=-0.01):
        """ A custom method to select the final features set """
        # Re-determining the feature importance by estimator
        feature_measurer = clone(self.feature_measurer)
        feature_measurer.fit(self.scaled_X, self.scaled_y)
        self.feature_importance = feature_measurer.feature_importances_

        # get the model base_line
        baseline = np.mean([
            cross_val_score(
                self.estimator, self.scaled_X, self.scaled_y,
                cv=KFold(n_splits=5, shuffle=True, random_state=self.seed), n_jobs=-1
            )
            for _ in range(5)
        ])

        # most importance features should be in
        print(np.isin(self.feature_names, self.kwargs.get('required_features')).any())
        nece_feat_indices = np.logical_or(
            self.feature_importance > 0.05,
            np.isin(self.feature_names, self.kwargs.get('required_features'))
        )
        # nece_feat_indices = np.logical_or(self.performance_dropdown < -0.05, self.feature_importance > 0.05)

        # Iteratively add the features that have the highest improvement to the model to the necessary feature set
        # and remove the features that have limited improvement
        identified_feature_indices = np.nonzero(np.logical_not(nece_feat_indices))[0].tolist()
        while identified_feature_indices:
            score_improve = []
            for fi in identified_feature_indices:
                try_feature_indices = np.copy(nece_feat_indices)
                try_feature_indices[fi] = True

                f = self.scaled_X[:, try_feature_indices]

                score = np.mean([
                    cross_val_score(
                        self.estimator, f, self.scaled_y, n_jobs=-1,
                        cv=KFold(n_splits=5, shuffle=True, random_state=self.seed)
                    )
                    for _ in range(5)
                ])

                score_improve.append(score - baseline)

            if (max(score_improve) < improve_threshold or
                    len(np.nonzero(nece_feat_indices)[0]) >= self.kwargs.get('max_feature', len(nece_feat_indices))):
                break
            else:
                adding_feature_indices = identified_feature_indices[np.argmax(score_improve)]
                nece_feat_indices[adding_feature_indices] = True
                print(score_improve)
                print(nece_feat_indices.nonzero()[0])

                identified_feature_indices = [
                    i for i, s in zip(identified_feature_indices, score_improve)
                    if s > improve_threshold and i != adding_feature_indices
                ]

        self.necessary_feature_indices = nece_feat_indices
        print(self.feature_names)
        self.necessary_features = self.feature_names[nece_feat_indices]
        self.necessary_X = self.scaled_X[:, nece_feat_indices]

        # Calculate covariance matrix for necessary X
        self.cov4necsX = np.cov(self.necessary_X.T)

    def _determine_final_features(self, tol=0.01):
        """"""
        sfs = SFS(self.feature_measurer, tol=tol)
        sfs.fit(self.scaled_X, self.scaled_y)

        self.necessary_features = self.feature_names[sfs.get_support()]
        self.necessary_X = sfs.transform(self.scaled_X)

        # Calculate covariance matrix for necessary X
        self.cov4necsX = np.cov(self.necessary_X.T)

    def make_imp_plot_for_necessary_feat(self):
        """ Determine the feature importance for each necessary feature. """
        if isinstance(self.necessary_features, np.ndarray) and isinstance(self.necessary_X, np.ndarray):
            reg = GradientBoostingRegressor()
            reg.fit(self.necessary_X, self.scaled_y)

            sorted_index = np.argsort(reg.feature_importances_)
            fi = self.necessary_feature_importance = reg.feature_importances_
            fi = fi[sorted_index]
            fn = np.array(self.necessary_features)[sorted_index].tolist()

            plotter = SciPlotter(
                FeatureImportance(fn, fi, force_show_label=True),
                tick_fontsize=20
            )
            fig, ax = plotter()
            fig.savefig(self.work_dir.joinpath('pictures', 'nece_fea_imp.png'))

            series = pd.Series(fi[::-1], index=fn[::-1], name='importance')
            series.to_excel(self.sheet_dir.joinpath('nece_fea_imp.xlsx'))

    def show_feature_hierarchical_tree(
            self, for_necessary_feature=True,
            threshold: float = None,
            tree_plot_kwargs: dict = None,
            *args, **kwargs
    ):
        """"""
        tree_kwargs = {
            'figwidth': 20,
            'figheight': 6.4,
            'axes_position': (0.05, 0.1, 0.925, 0.85)
        }
        tree_kwargs.update(tree_plot_kwargs if isinstance(tree_plot_kwargs, dict) else {})

        X = self.necessary_X if for_necessary_feature else self.scaled_X
        feature_names = self.necessary_features if for_necessary_feature else self.feature_names

        # Calculate clustering labels
        threshold = threshold if threshold else np.sqrt(pow(0.1, 2) * len(feature_names))
        abs_correlation_mat = np.abs(np.nan_to_num(np.corrcoef(X.T)))
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
        clustering.fit(abs_correlation_mat)
        self.feature_cluster = clustering.labels_

        print('clustering counts: \n', np.unique(clustering.labels_, return_counts=True))

        # Make plot:
        sciplot = SciPlotter(HierarchicalTree(X, range(len(feature_names)), threshold), **tree_kwargs)
        fig, ax = sciplot()
        fig.savefig(self.work_dir.joinpath('pictures', 'hierarchical_tree.png'))

    def cross_validation(self, make_plot=True, *args, **kwargs):
        print(self.necessary_X.shape, self.y.shape)
        self.p_cross = cross_val_predict(
            self.estimator, self.necessary_X, self.y,
            cv=KFold(n_splits=5, shuffle=True, random_state=self.seed)
        )

        if self.yscaler:
            self.p_cross = self.yscaler.inverse_transform(self.p_cross)

        if make_plot:
            sciplot = SciPlotter(R2Regression([self.y, self.p_cross]))
            fig, ax = sciplot()
            fig.savefig(self.work_dir.joinpath('pictures', 'R2_validation.png'))

        if self.kwargs.get('save_sheet', kwargs.get('save_sheet')):
            df_pt = pd.DataFrame([self.y, self.p_cross], index=['target', 'prediction'], columns=self.samples_index).T
            if isinstance(self.other_info, pd.DataFrame):
                df_pt = pd.concat([self.other_info, df_pt], axis=1)

            with pd.ExcelWriter(self.sheet_dir.joinpath('cross_validation.xlsx')) as writer:
                df_pt.to_excel(writer, sheet_name='pt')

    def _split_train_test(self):
        """ split train and test dataset """
        def _get_stratify():
            if self.split_stratify:
                y_min, y_max = self.y.min(), self.y.max()
                if self.stratify_style == 'logarithm':
                    y_min, y_max = np.log(y_min), np.log(y_max)

                stratify = np.zeros_like(self.y)
                split_point = np.linspace(y_min, y_max, self.split_stratify+1).reshape(-1, 1)
                split_indices = np.logical_and(split_point[:-1] <= self.y, self.y <= split_point[1:])

                for i, si in enumerate(split_indices, 1):
                    stratify[si] = i

                return stratify

        test_indices = self.kwargs.get('test_indices')
        if isinstance(test_indices, (Sequence, np.ndarray)):
            train_indices = np.isin(np.arange(len(self.X)), test_indices, invert=True)

        else:
            test_size = self.kwargs.get('test_size')
            sample_indices = np.arange(len(self.scaled_y))
            train_indices, test_indices = train_test_split(
                sample_indices, shuffle=True, test_size=test_size, stratify=_get_stratify()
            )

        self.idx_train, self.idx_test = train_indices, test_indices
        self.sample_idx_train, self.sample_idx_test = self.samples_index[train_indices], self.samples_index[test_indices]
        self.X_train, self.X_test = self.necessary_X[train_indices], self.necessary_X[test_indices]
        self.y_train, self.y_test = self.scaled_y[train_indices], self.scaled_y[test_indices]

    def split_train_test(self, *args, **kwargs):
        """split train and test dataset"""
        def _get_stratify():
            if self.split_stratify:
                y_min, y_max = self.y.min(), self.y.max()
                if self.stratify_style == 'logarithm':
                    y_min, y_max = np.log(y_min), np.log(y_max)

                stratify = np.zeros_like(self.y)
                split_point = np.linspace(y_min, y_max, self.split_stratify+1).reshape(-1, 1)
                split_indices = np.logical_and(split_point[:-1] <= self.y, self.y <= split_point[1:])

                for i, si in enumerate(split_indices, 1):
                    stratify[si] = i

                return stratify

        test_indices = kwargs.get('test_indices') or self.kwargs.get('test_indices')
        if isinstance(test_indices, (Sequence, np.ndarray)):
            train_indices = np.isin(np.arange(len(self.X)), test_indices, invert=True)
            # test_indices = np.isin(self.samples_index, test_indices)
        else:
            test_size = self.kwargs.get('test_size', kwargs.get('test_size', 0.2))
            sample_indices = np.arange(len(self.scaled_y))
            train_indices, test_indices = train_test_split(
                sample_indices, shuffle=True, test_size=test_size, stratify=_get_stratify()
            )

        self.idx_train, self.idx_test = train_indices, test_indices
        self.sample_idx_train, self.sample_idx_test = self.samples_index[train_indices], self.samples_index[test_indices]
        self.X_train, self.X_test = self.necessary_X[train_indices], self.necessary_X[test_indices]
        self.y_train, self.y_test = self.scaled_y[train_indices], self.scaled_y[test_indices]

    @staticmethod
    def test_model(estimator: BaseEstimator, X_test, y_test, X_train=None, y_train=None, prefix: str = ''):
        """ test model in test set and report accuracy """
        p_test = estimator.predict(X_test)
        print(f'{prefix} test R2 = {r2_score(y_test, p_test)};')
        if isinstance(X_train, np.ndarray) and isinstance(y_train, np.ndarray):
            p_train = estimator.predict(X_train)
            print(f'{prefix} train R2 = {r2_score(y_train, p_train)}.')

        return p_test, locals().get('p_train')

    def make_prediction_target_plot(self, *args, **kwargs):
        """"""
        self.split_train_test(*args, **kwargs)

        self.estimator.fit(self.X_train, self.y_train)
        self.p_test, self.p_train = self.test_model(self.estimator, self.X_test, self.y_test, self.X_train, self.y_train)

        self.pt_diff_train = self.p_train - self.y_train
        self.pt_diff_test = self.p_test - self.y_test

        # Make plot
        sciplot = SciPlotter(
            R2Regression(
                [self.y_train, self.p_train],
                [self.y_test, self.p_test],
                s1=100, show_mae=self.kwargs.get('show_mae'),
                show_rmse=self.kwargs.get('show_rmse')
            )
        )
        fig, ax = sciplot()
        fig.savefig(self.work_dir.joinpath('pictures', 'pred_target.png'))

        # Save sheet
        X_train = pd.DataFrame(self.X_train, columns=self.necessary_features, index=self.sample_idx_train)
        X_test = pd.DataFrame(self.X_test, columns=self.necessary_features, index=self.sample_idx_test)
        train_pt = pd.DataFrame([self.y_train, self.p_train], index=['y_train', 'p_train'], columns=self.sample_idx_train).T
        test_pt = pd.DataFrame([self.y_test, self.p_test], index=['y_test', 'p_test'], columns=self.sample_idx_test).T

        if isinstance(self.other_info, pd.DataFrame):
            train_pt = pd.concat([self.other_info.loc[self.sample_idx_train, :], train_pt], axis=1)
            test_pt = pd.concat([self.other_info.loc[self.sample_idx_test, :], test_pt], axis=1)

        with pd.ExcelWriter(self.sheet_dir.joinpath('pred_target.xlsx')) as writer:
            train_pt.to_excel(writer, sheet_name='train_pt')
            test_pt.to_excel(writer, sheet_name='test_pt')
            X_train.to_excel(writer, sheet_name='train_X')
            X_test.to_excel(writer, sheet_name='test_X')

        self.project_train_test_data_to_2d_embedding(
            self.X_train, self.X_test,
            self.y_train, self.y_test,
            self.p_train, self.p_test
        )

    def project_train_test_data_to_2d_embedding(
            self,
            X_train, X_test,
            y_train, y_test,
            p_train, p_test,
            embedding=MDS()
    ):
        """"""
        def _map_y(ax, sciplot, *args, **kwargs):
            mappable = ax.scatter(emb_X[0], emb_X[1], c=total_y, s=100)

            sciplot.ax_modifier[ax] = SciPlotter.axes_modifier_container(
                (sciplot.add_axes_colorbar, {'mappable': mappable}))

        def _map_error(ax, sciplot, *args, **kwargs):
            error = p_test - y_test
            ax.scatter(emb_Xtrain[0], emb_Xtrain[1], c='#cccccc', s=100)
            mappable = ax.scatter(emb_Xtest[0], emb_Xtest[1], c=error, s=100, cmap='coolwarm')

            sciplot.ax_modifier[ax] = SciPlotter.axes_modifier_container(
                (sciplot.add_axes_colorbar, {'mappable': mappable}))

        total_X = np.vstack((X_train, X_test))
        total_y = np.hstack((y_train, y_test))

        weighted_X = total_X * self.necessary_feature_importance

        emb_X = embedding.fit_transform(weighted_X).T

        emb_Xtrain, emb_Xtest = emb_X[:, :len(X_train)], emb_X[:, len(X_train):]

        sci_plotter = SciPlotter([_map_y, _map_error])
        fig, ax = sci_plotter()

        fig.savefig(self.picture_dir.joinpath('data_embedding.png'))

    def generate_test_X(
            self,
            template_X: np.ndarray = None,
            independent: bool = False,
            norm_uniform: Literal['norm', 'uniform'] = 'uniform',
            min_offset: Union[float, np.ndarray] = 0.,
            max_offset: Union[float, np.ndarray] = 0.,
            sample_num: int = 1000,
            seed_: int = None
    ):
        """
        Generates a hypothetical test X with identical covariance of features as template X.
        Args:
            template_X: template X to define correlations (matrix). if not given, workflow.necessary is applied.
            independent: whether to suppose each feature is independent with the other, defaults to False.
            norm_uniform: generate from a uniform distribution or a normal distribution, defaults to 'uniform'
            min_offset: the proportion between the offset lower than min value and the diff from min value to max value,
                        the value should be a float or numpy array. if a float is provided, the same offset value will
                        act on all feature; if a numpy array is provided, the length of array should be equal to the
                        X.shape[1], in this case, different offset will assign to different features.
            max_offset: the proportion between the offset higher than max value and the diff from min value to max value
                        the value should be a float or numpy array. if a float is provided, the same offset value will
                        act on all feature; if a numpy array is provided, the length of array should be equal to the
                        X.shape[1], in this case, different offset will assign to different features.
            sample_num: number of samples to be generated, i.e., X.shape[0] == sample_num.
            seed_: random state for distribute generation

        Returns:
            generated test X with identical covariance of features as template X.
        """
        def set_correlation(data):
            trans_mat = np.linalg.cholesky(np.corrcoef(template_X, rowvar=False))
            return np.dot(data, trans_mat.T)

        if not isinstance(template_X, np.ndarray):
            template_X = self.necessary_X

        np.random.seed(seed_)
        if norm_uniform == 'norm':
            mu = np.mean(template_X, axis=0)
            std = np.std(template_X, axis=0)

            gen_X = np.random.normal(mu, std, size=(sample_num, template_X.shape[1]))

            if not independent:
                gen_X = set_correlation(gen_X)

        elif norm_uniform == 'uniform':
            X_min, X_max = np.min(template_X, axis=0), np.max(template_X, axis=0)
            X_diff = X_max - X_min
            X_min = X_min - min_offset * X_diff
            X_max = X_max + max_offset * X_diff

            if np.any(X_max <= X_min):
                error_dim = np.nonzero(np.int_(X_max <= X_min))[0].tolist()
                raise ValueError(f'the maximum value is smaller than the minimum values in X dimensions of {error_dim}')

            gen_X = np.random.uniform(0, 1, size=(sample_num, template_X.shape[1]))

            if not independent:
                gen_X = set_correlation(gen_X)

            for i, (x_min, x_max) in enumerate(zip(X_min, X_max)):
                gen_X[:, i] = minmax_scale(gen_X[:, i], (x_min, x_max))
        else:
            raise ValueError('Unrecognized value for norm_uniform argument, use "norm" or "uniform"')

        return gen_X

    def train_surrogate_tree(
            self,
            estimator: BaseEstimator=None,
            X: np.ndarray = None,
            surrogate_X: np.ndarray = None,
            feature_names=None,
            min_offset=0., max_offset=0.,
            surrogate: BaseEstimator = DecisionTreeRegressor(),
            X_test: np.ndarray = None, y_test: np.ndarray = None,
            gen_sample_num: int = None,
            make_tree_plot: bool = True,
            **kwargs
    ):
        """
        Train a surrogate student decision tree based on teacher estimator.
        Args:
            estimator(BaseEstimator): sklearn estimator, if not given, the trained estimator saved in Workflow will be applied.
            X(numpy.ndarray): input data to generate estimator prediction and training the surrogate student decision,
                if not given, the workflow.necessary_X will be used to be.

                When an integer is passed to the gen_sample_num, a pseudo X will be generated by
                workflow.generate_test_sample function, where the generated X should be with same dimensions as the
                applied X in the axis=1 and the sample number is equal to the gen_sample_num;
                the min_limit and max_limit in each dimension of generated X will be
                [X_min-min_offset, X_max+max_offset], respectively

            surrogate_X: Specify the X input to train the surrogate student decision. if not given, the X to get
                the prediction from teacher estimator will be used as. This argument would be useful to improve
                the interpretability of model when the original X input have been normalized or clustering.

            min_offset(float|np.ndarray): offset lower than the minimum values of applied X. if a float is given
                all dimensions will offset same value; if a np.ndarray is given, the length of min_offset should be
                equal to the applied X.shape[1], and the ith dimension will be offset min_offset[i].
            max_offset(float|np.ndarray): offset higher than the maximum values of applied X. if a float is given
                all dimensions will offset same value; if a np.ndarray is given, the length of max_offset should be
                equal to the applied X.shape[1], and the ith dimension will be offset max_offset[i].

            gen_sample_num(int): number of generated samples to train student surrogate model.

            feature_names: list of feature name for X, it should be with shape of X.shape[1], if not given and the
                X is not given too, the workflow.necessary_features will be used to be.

            surrogate(BaseEstimator): a student surrogate estimator, a DecisionTreeRegressor() is default.
            X_test(np.ndarray): the test X to test the surrogate estimator performance.
            y_test(np.ndarray): the test y to test the surrogate estimator performance.
            make_tree_plot(bool): whether to make a tree plot for surrogate.
            **kwargs: keyword arguments to sklearn plot_tree function

        Returns:

        """
        # Prepare input X to predict corresponding y by estimator, and the train the surrogate.
        if not estimator:
            estimator = self.estimator

        X = X if isinstance(X, np.ndarray) else self.necessary_X

        # Replace the X input to this when train the surrogate tree,
        # the replacement may useful to improve the interpretability when
        # the original X have been normalized or clustering.
        if isinstance(gen_sample_num, int):
            X = self.generate_test_X(
                X,
                min_offset=min_offset,
                max_offset=max_offset,
                sample_num=gen_sample_num
            )

        if isinstance(surrogate_X, np.ndarray) and X.shape != surrogate_X.shape:
            raise ValueError("The shape of X and surrogate_X must be the identical when surrogate_X is given!")
        elif not isinstance(surrogate_X, np.ndarray):
            surrogate_X = X

        p = estimator.predict(X)
        surrogate.fit(surrogate_X, p)

        if (not isinstance(X_test, np.ndarray) or not (y_test, np.ndarray)) and not isinstance(surrogate_X, np.ndarray):
            X_test, y_test = self.X_test, self.y_test

        if isinstance(X_test, np.ndarray) or (y_test, np.ndarray):
            self.test_model(surrogate, X_test, y_test, prefix="student surrogate")
        else:
            raise UserWarning("the trained surrogate does not to be tested!!")

        if make_tree_plot and isinstance(surrogate, BaseDecisionTree):
            fig, ax = plt.subplots()
            tree_ = plot_tree(surrogate, feature_names=feature_names, **kwargs, ax=ax)
            return surrogate, fig, ax

        return surrogate

    def partial_dependence(
            self,
            *feature_index: int,
            X: np.ndarray = None,
            estimator: BaseEstimator = None,
            x_label: str = None,
            y_label: str = None,
            savefig_path: [str, os.PathLike] = None,
            **kwargs
    ) -> (plt.Figure, plt.Axes):
        """
        Make partial dependence analysis (PDA) plot
        Args:
            *feature_index: feature indices in total feature set
            X: dataset. if not given, workflow.necessary_X will be
            estimator: trained estimator
            x_label: xlabel in PDA plot
            y_label: ylabel in PDA plot
            savefig_path: if given, save the figure to specified path.
            **kwargs: other arguments passed to hotpot.plots.SciPlotter.

        Returns:
            plt.Figure and plt.Axes objects
        """
        def _draw(ax: plt.Axes, sciplot: SciPlotter = None, *args, **kwargs):
            PartialDependenceDisplay.from_estimator(estimator, X, feature_index, ax=ax, kind=kind,)

            ax = plt.gca()
            ax.set_xlabel(x_label, font='Arial', fontweight='bold', fontsize=22)
            ax.set_ylabel(y_label, font='Arial', fontweight='bold', fontsize=22)

            sciplot.set_ticks(ax)

        if not 1 <= len(feature_index) <= 2:
            raise ValueError("the number of features index must be 1 or 2")
        elif len(feature_index) == 2:
            kind = 'average'
            feature_index = [feature_index]
        else:
            kind = 'individual'

        if not isinstance(X, np.ndarray):
            X = self.necessary_X
            x_label = self.necessary_features[feature_index[0]]
            try:
                y_label = self.necessary_features[feature_index[1]]
            except IndexError:
                y_label = 'Target'

        plotter = SciPlotter(_draw, **kwargs)
        fig, ax = plotter()

        if savefig_path:
            fig.savefig(savefig_path)

        return fig, ax

    def partial_dependence_analysis(
            self,
            pda_dir: Union[str, os.PathLike] = None,
            origin_data: bool = False,
            **kwargs
    ):
        """
        Automatic partial dependence analysis (PDA)
        Args:
            origin_data: whether to use origin data as test samples.
            pda_dir: path to save PDA pictures
            **kwargs: keyword arguments for workflow.generate_test_X method.

        Returns:
            Notes
        """
        if origin_data:
            X = self.necessary_X
        else:
            X = self.generate_test_X(self.necessary_X, **kwargs)

        if pda_dir is None:
            pda_dir = self.picture_dir.joinpath('PDA')
        else:
            pda_dir = Path(pda_dir)

        # Single feature PDA
        if not pda_dir.exists():
            pda_dir.mkdir()

        for i, f_name in enumerate(self.necessary_features):
            fig, ax = self.partial_dependence(
                i, X=X, estimator=self.estimator,
                x_label=f_name, y_label=str(self.target_name),
                savefig_path=pda_dir.joinpath(f'1_PDA_{f_name}.png')
            )
            plt.close(fig)

        for (i1, f1), (i2, f2) in combinations(zip(range(self.necessary_features.size), self.necessary_features), 2):
            fig, ax = self.partial_dependence(
                i1, i2, X=X, estimator=self.estimator,
                x_label=f1, y_label=f2,
                savefig_path=pda_dir.joinpath(f'2_PDA_{f1}_{f2}.png')
            )
            plt.close(fig)

    def calc_shap_values(
            self, estimator, X, y,
            feature_names: Union[Sequence, np.ndarray] = None,
            sample_size: int = 1000,
            X_test: np.ndarray = None,
            test_size: int = 1000,
            explainer_cls: shap.Explainer = shap.TreeExplainer,
            shap_values_save_path: Union[str, os.PathLike] = None,
            **kwargs
    ):
        """
        calculate shap values.
        Args:
            estimator: estimator to be explained by SHAP analysis.
            X: input data to train estimator
            y: target data to train estimator
            feature_names: feature names for each column of input data
            sample_size: how many samples sample from input data to carry out SHAP analysis
            X_test: data to be explained by SHAP explainer. if not given, test data is generated by
                    workflow.generate_test_X method with a 'test_size' size
            test_size: when the X_test is None, this arg is used to control the number of generated test samples
            explainer_cls: class of SHAP explainer, defaults to shap.TreeExplainer.
            shap_values_save_path: path to save the calculated SHAP results, should be an EXCEL file.
            **kwargs: keyword arguments for workflow.generate_test_X method.

        Returns:
            SHAP explainer, SHAP values.
        """
        if explainer_cls is None:
            explainer_cls = shap.TreeExplainer

        estimator.fit(X, y)
        if len(X) < 1000:
            explainer = explainer_cls(estimator, X)
        else:
            sample_X = shap.sample(X, sample_size)
            explainer = explainer_cls(estimator, sample_X)

        if X_test is None:
            X_test = self.generate_test_X(X, sample_num=test_size, **kwargs)

        shap_values = explainer(X_test)
        if isinstance(feature_names, (list, np.ndarray)):
            assert len(feature_names) == X.shape[1]
            shap_values.feature_names = feature_names

        # Save result, base, data value and SHAP value
        if shap_values_save_path:
            series_base = pd.Series(shap_values.base_values, name='base_values')
            df_shap_value = pd.DataFrame(shap_values.values, columns=self.necessary_features)
            df_data_value = pd.DataFrame(shap_values.data, columns=self.necessary_features)

            with pd.ExcelWriter(shap_values_save_path) as writer:
                series_base.to_excel(writer, sheet_name='base')
                df_shap_value.to_excel(writer, sheet_name='shap')
                df_data_value.to_excel(writer, sheet_name='data')

        return explainer, shap_values

    @staticmethod
    def make_shap_bar_beeswarm(
            shap_values: shap.Explanation,
            max_display: int = 15,
            savefig_path: Union[str, os.PathLike] = None,
            **kwargs
    ):
        """ Make a bar and a beeswarm plot for given shap values """
        sciplot = SciPlotter(
            [
                SHAPlot(shap_values, max_display=max_display),
                SHAPlot(shap_values, plot_type='beeswarm', max_display=max_display)
            ], ticklabels_fontsize=16, superscript_position=(-0.075, 1.075), **kwargs
        )

        fig, ax = sciplot()
        fig.savefig(savefig_path)

    def shap_analysis(self):
        """ calculate shap values and make resulted plots in an automatic workflow """
        self.explainer, self.shap_value = self.calc_shap_values(
            self.estimator, self.necessary_X, self.scaled_y, self.necessary_features,
            explainer_cls=self.kwargs.get('shap_explainer_cls'),
            shap_values_save_path=self.sheet_dir.joinpath('shap.xlsx')
        )

        # Make and save SHAP bar and beeswarm plot.
        self.make_shap_bar_beeswarm(self.shap_value, savefig_path=self.work_dir.joinpath('pictures', 'shap.png'))

        # Save pickled shap value
        if self.work_dir.exists():
            with open(self.work_dir.joinpath('shap.pkl'), 'wb') as writer:
                pickle.dump(self.shap_value, writer)
        else:
            raise FileNotFoundError(f'the output directory {self.work_dir} does not exist!')

    def train_final_model(self):
        """ Train final model by all samples """
        self.estimator.fit(self.necessary_X, self.scaled_y)
        if self.work_dir.exists():
            with open(self.work_dir.joinpath('final_model.pkl'), 'wb') as writer:
                pickle.dump(self.estimator, writer)

    def pickle_workflow(self):
        with open(self.work_dir.joinpath('workflow.pkl'), 'wb') as writer:
            pickle.dump(self, writer)

    @classmethod
    def load_workflow(cls, pkl_path):
        with open(pkl_path, 'rb') as file:
            return pickle.load(file)


def to_onehot(data: pd.DataFrame, encode_col_name: Union[str, list[str]]) -> (pd.DataFrame, list[str]):
    encode_value = data.loc[:, [encode_col_name] if isinstance(encode_col_name, str) else encode_col_name].values
    onehot_encoder = OneHotEncoder()
    encode_value = pd.DataFrame(
        onehot_encoder.fit_transform(encode_value).toarray(),
        columns=onehot_encoder.categories_[0].tolist(), index=data.index)

    return pd.concat([data, encode_value], axis=1), onehot_encoder.categories_[0].tolist()


def show_feature_map(X, method=TSNE(), scalar=MinMaxScaler(), loss: np.ndarray = None):
    X = scalar.fit_transform(X)
    X = method.fit_transform(X)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=loss)

    fig.show()

    return fig, ax

