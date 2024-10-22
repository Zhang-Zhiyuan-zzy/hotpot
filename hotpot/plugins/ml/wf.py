"""
python v3.9.0
@Project: hp5
@File   : _wf
@Auther : Zhiyuan Zhang
@Data   : 2024/7/30
@Time   : 9:41
"""
import logging
import os
import shutil
import pickle
import time
from pathlib import Path
from typing import Union, Sequence, Literal, Callable
from multiprocessing import Process, Queue
from copy import copy
from itertools import combinations
from functools import wraps

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm

from scipy.cluster.hierarchy import dendrogram

import sklearn
from sklearn.base import clone, BaseEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, minmax_scale
from sklearn.model_selection import LeaveOneOut, train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, BaseDecisionTree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector as SFS, RFECV
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor
import lightgbm as lgb

import shap
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors

from hotpot.plots import SciPlotter, R2Regression, PearsonMatrix, SHAPlot, scale_axes
from hotpot.utils.types import ModelLike


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


def _cross_val(estimator, X, y, cv=None, score_metric: Literal['R2', 'MAE', 'RMSE'] = 'R2', *args, **kwargs):
    if cv is None:
        cv = KFold(shuffle=True)

    pred = cross_val_predict(estimator, X, y, cv=cv, *args, **kwargs)

    if score_metric == 'R2':
        score = r2_score(y, pred)
    elif score_metric == 'MAE':
        score = mean_absolute_error(y, pred)
    elif score_metric == 'RMSE':
        score = np.sqrt(mean_squared_error(y, pred))
    else:
        raise NotImplementedError(f"score_metric {score_metric} not implemented")

    return score, pred, y
    # if not cv:
    #     return (
    #         cv_score(estimator, X, y, cv=KFold(shuffle=True), *args, **kwargs),
    #         cv_pred(estimator, X, y, cv=KFold(shuffle=True), *args, **kwargs), y
    #     )

    # else:
    #     estimator.fit(X, y)
    #
    #     valid_score, valid_pred, valid_true = [], [], []
    #     for train_index, test_index in cv.split(X, y):
    #         X_train, X_test = X[train_index], X[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #
    #         clone(estimator).fit(X_train, y_train)
    #         score = estimator.score(X_test, y_test)
    #         pred = estimator.predict(X_test)
    #
    #         valid_score.append(score)
    #         valid_pred.append(pred)
    #         valid_true.append(y_test)
    #
    #     return np.array(valid_score), np.hstack(valid_pred), np.hstack(valid_true)


class LinearAddLightGBM(BaseEstimator):
    def __init__(self):
        """"""
        self.linear = sklearn.linear_model.LinearRegression()
        # self.lightgbm = lgb.LGBMRegressor()
        self.lightgbm = GradientBoostingRegressor()
        self.linear_feature_index = None
        self.lightgbm_feature_index = None

        self.delta_y = None

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def fit(
            self, X, y=None,
            linear_feature_index: Union[Sequence[int], np.ndarray] = None,
            lightgbm_feature_index: Union[Sequence[int], np.ndarray] = None
    ):
        """"""
        if linear_feature_index is None:
            self.linear_feature_index = np.arange(X.shape[1])
        else:
            self.linear_feature_index = np.array(linear_feature_index)

        if lightgbm_feature_index is None:
            self.lightgbm_feature_index = np.arange(X.shape[1])
        else:
            self.lightgbm_feature_index = np.array(lightgbm_feature_index)


        self.linear.fit(X[:, self.linear_feature_index], y)
        self.delta_y = y - self.linear.predict(X[:, self.linear_feature_index])
        self.lightgbm.fit(X[:, self.lightgbm_feature_index], self.delta_y)

    def predict(self, X):
        return self.linear.predict(X[:, self.linear_feature_index]) + self.lightgbm.predict(X[:, self.lightgbm_feature_index])

    def score(self, X, y=None):
        """"""
        return r2_score(y, self.predict(X))

    @property
    def linear_coeff_(self):
        return self.linear.coef_


class CrossValidation:
    def __init__(self, X, y, cv=None):
        self.X, self.y = X, y
        self.cv = cv
        self.train_indices = None
        self.valid_indices = None

    def __call__(self, *args, **kwargs):
        ...


class MachineLearning:
    """
    A Standard procedure for machine learning.
    """
    def __init__(
            self,
            work_dir: Union[Path, str],
            data: pd.DataFrame,
            features: Union[Sequence, np.ndarray],
            target,
            estimator: BaseEstimator = GradientBoostingRegressor(),
            **kwargs
    ):
        """
        Args:
            work_dir (Path|str): Path to save results
            data (pd.DataFrame): Raw dataset, including features, target, and other correlation information
            features (Sequence|np.ndarray): feature columns in data DataFrame
            target: target column in data DataFrame
            estimator (BaseEstimator, optional): the algorithm class to train model

        Keyword Args:
            xscaler (sklearn.preprocessing.BaseScaler):
            yscaler (sklearn.preprocessing.BaseScaler):
            test_indices (Sequence[int], optional):
            test_size (float, optional):
            data_shuffle (bool, optional):
            data_stratify (bool, optional):
            stratify_log (bool, optional):
            feat_imp_measurer (Callable, optional):
            feat_imp_threshold (float, optional):
            recur_addi_threshold (float, optional):
            recur_addi_prev_threshold (float, optional):
            essential_features (Sequence[int|str], optional): the threshold of feature importance, which is
                evaluated by a given measurer ML algorithm, like RF, where a feature with an importance larger
                than this threshold will be added into the initial essential feature collection.
        """
        self.work_dir = Path(work_dir)
        self.picture_dir = self.work_dir.joinpath('pictures')
        self.sheet_dir = self.work_dir.joinpath('sheet')

        self.data = data
        self.features = list(features) if isinstance(features, Sequence) else features.tolist()
        self.target = target
        self.estimator = estimator

        self.sample_num = len(self.data)
        self.X = data.loc[:, features]
        self.y = data[target]
        self.other_info = data.drop(target, axis=1).drop(features, axis=1)

        self.xscaler = kwargs.get('xscaler', None)
        self.yscaler = kwargs.get('yscaler', None)
        self.scaled_data = None

        # Args to control dataset split
        self.test_indices = kwargs.get('test_indices', None)
        self.train_indices = None
        self.test_size = kwargs.get('test_size', 0.2)
        self.data_shuffle = kwargs.get('data_shuffle', False)
        self.data_stratify = kwargs.get('data_stratify', False)
        self.stratify_log = kwargs.get('stratify_log', False)

        # Args to control Feature Engineering
        self.feat_imp_measurer = kwargs.get('feat_imp_measurer', None)
        self.feat_imp_threshold = kwargs.get('feat_imp_threshold', 1e-3)
        self.feat_cluster_threshold = kwargs.get('feat_cluster_threshold', None)
        self.recur_addi_threshold = kwargs.get('recur_addi_threshold', 0.0)
        # A float range from 0.0 to 1.0. If given, the importance of features
        # will be evaluated firstly by a model, say GradientBoostingRegressor.
        # Then, the features with importance more than the given threshold
        # will be added into the essential feature collection, before the
        # recursive addition procedure is running.
        self.recur_addi_prev_threshold = kwargs.get('recur_addi_prev_threshold')

        # Feature engineering results
        self.trivial_feat_indices = None
        self.ntri_feat_indices = None
        self.pearson_mat = None
        self.ntri_feat_cluster = None
        self.clustered_feat_indices = None
        self.clustered_map = None
        self.clt_X = None
        self._essential = None
        self.essential_feat_indices = self._get_essential_features(kwargs.get('essential_features'))
        self.cv_best_metric = None,
        self.how_to_essential = None

        self.feature_importance = {}
        self.optimal_hyperparams = {}
        self.valid_score = None
        self.valid_pred = None
        self.valid_true = None

        # Attributes reserved for SHAP analysis
        self.explainer = None
        self.shap_value = None

    def _get_essential_features(self, feat: Union[None, Sequence, np.ndarray]) -> (None, Sequence, np.ndarray):
        if feat is None:
            return None

        if isinstance(feat, np.ndarray):
            feat = feat.tolist()

        feat_ = []
        for f in feat:
            if isinstance(f, str):
                feat_.append(self.features.index(f))
            elif isinstance(f, int):
                feat_.append(f)

        self._essential = np.array(feat_)

    def work(self):
        if not self.work_dir.exists():
            self.work_dir.mkdir()
        elif os.listdir(self.work_dir):
            raise IOError('Work directory must be an empty directory')

        print("Creating work directory: {}".format(self.work_dir))
        self.picture_dir.mkdir()
        self.sheet_dir.mkdir()

        self.preprocess()
        self.train_test_split()

        if not isinstance(self.essential_feat_indices, np.ndarray):
            self.feature_engineering()
        else:
            self.trivial_feat_indices = np.array([])
            self.clustered_feat_indices = np.array([])
            self.clustered_map = {}

        self.cross_validation()
        self.train_model()
        self.shap_analysis()

    def preprocess(self):
        print("Preprocessing data...")
        self.scaled_data = self.data

        if self.xscaler is not None:
            self.xscaler.fit(self.scaled_data.loc[:, self.features])
            self.scaled_data.loc[:, self.features] = self.xscaler.transform(self.scaled_data.loc[:, self.features])

        if self.yscaler is not None:
            self.yscaler.fit(self.scaled_data.loc[:, self.target])
            self.yscaler.loc[:, self.target] = self.yscaler.transform(self.yscaler.loc[:, self.target])

    @property
    def scaled_X(self) -> np.ndarray:
        return self.scaled_data[self.features].values

    @property
    def scaled_y(self):
        return self.scaled_data[self.target].values

    def _get_stratify(self):
        if self.data_stratify:
            y_min, y_max = self.y.min(), self.y.max()
            if self.stratify_log is True:
                y_min, y_max = np.log(y_min), np.log(y_max)

            stratify = np.zeros_like(self.y)
            split_point = np.linspace(y_min, y_max, self.data_stratify + 1).reshape(-1, 1)
            split_indices = np.logical_and(split_point[:-1] <= self.y, self.y <= split_point[1:])

            for i, si in enumerate(split_indices, 1):
                stratify[si] = i

            return stratify

    def train_test_split(self):
        print('Splitting data...')
        if self.test_indices is None:
            self.train_indices, self.test_indices = train_test_split(
                np.arange(self.sample_num),
                shuffle=self.data_shuffle,
                test_size=self.test_size,
                stratify=self._get_stratify()
            )

        else:
            self.train_indices = np.isin(np.arange(len(self.X)), self.test_indices, invert=True)

    @property
    def sXtr(self):
        return self.scaled_X[self.train_indices]

    @property
    def sYtr(self):
        return self.scaled_y[self.train_indices]

    @property
    def sXte(self):
        return self.scaled_X[self.test_indices]

    @property
    def sYte(self):
        return self.scaled_y[self.test_indices]

    @property
    def ntri_features(self):
        return np.delete(np.array(self.features), self.trivial_feat_indices.tolist()).tolist()

    @property
    def ntri_X(self):
        return np.delete(self.X, self.trivial_feat_indices.tolist(), axis=1)

    @property
    def ntri_sX(self):
        return np.delete(self.scaled_X, self.trivial_feat_indices.tolist(), axis=1)

    @property
    def ntri_Xtr(self):
        return self.ntri_X[self.train_indices]

    @property
    def ntri_Xte(self):
        return self.ntri_X[self.test_indices]

    @property
    def ntri_sXtr(self):
        return self.ntri_sX[self.train_indices]

    @property
    def ntri_sXte(self):
        return self.ntri_sX[self.test_indices]

    def feature_engineering(self):
        """ Performing feature engineering """
        print('Feature engineering...')
        self.quickly_feature_selection()
        print('Dimensionality Reduction and Redundancy Elimination')
        self.calc_pearson_matrix()
        self.feat_hiera_clustering()
        self.pca_dimension_reduction()
        # self.recursion_feature_addition()
        # self.recursive_feature_elimination()
        self.recursive_feature_selection()

    def _get_feature_measurer(self):
        """ Getting an estimator to measure the feature importance """
        if self.feat_imp_measurer:
            return self.feat_imp_measurer

        elif hasattr(self.estimator, 'feature_importances_') or hasattr(self.estimator, 'coef_'):
            return clone(self.estimator)

        else:
            return GradientBoostingRegressor()

    def quickly_feature_selection(self):
        print('Quickly feature selection...')
        feat_imp_measurer = self._get_feature_measurer()

        feat_imp_measurer.fit(self.sXtr, self.sYtr)
        if hasattr(feat_imp_measurer, 'feature_importances_'):
            feat_imp = feat_imp_measurer.feature_importances_

        elif hasattr(feat_imp_measurer, 'coef_'):
            feat_imp = feat_imp_measurer.coef_

        else:
            raise AttributeError('the given feat_imp_measurer has not attribute `feature_importances_` or `coef_`')

        self.trivial_feat_indices = np.where(feat_imp < self.feat_imp_threshold)[0]
        self.ntri_feat_indices = np.delete(np.arange(len(self.features)), self.trivial_feat_indices.tolist())

    @staticmethod
    def calc_pearson_coef(X):

        return np.nan_to_num(np.corrcoef(X, rowvar=False))

    def calc_pearson_matrix(self):
        """ calculate Pearson matrix for all non-trivial features """
        self.pearson_mat = self.calc_pearson_coef(self.ntri_sX)
        self.make_pearson_matrix_plot()

    def make_pearson_matrix_plot(self):
        """ draw Pearson matrix plot and save it to `work_dir/picture_dir` """
        plotter = SciPlotter(PearsonMatrix(self.pearson_mat, self.ntri_features, is_matrix=True))
        fig, ax = plotter()
        fig.savefig(self.picture_dir.joinpath('Pearson_mat.png'))

    def feat_hiera_clustering(self):
        if not self.feat_cluster_threshold:
            self.feat_cluster_threshold = np.sqrt(pow(0.1, 2) * len(self.ntri_feat_indices))

        abs_pearson_mat = np.abs(self.pearson_mat)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.feat_cluster_threshold)
        clustering.fit(abs_pearson_mat)

        self.ntri_feat_cluster = {}
        for cluster_label, feat_idx in zip(clustering.labels_, self.ntri_feat_indices):
            lst_feat_idx = self.ntri_feat_cluster.setdefault(cluster_label, [])
            lst_feat_idx.append(feat_idx)

        fig, ax = self.make_hierarchical_tree(clustering, self.feat_cluster_threshold)

        fig.savefig(self.picture_dir.joinpath('hiera_tree.png'))

    @staticmethod
    def make_hierarchical_tree(clustering, threshold):
        """ make a hierarchical tree plot return the Figure and Axes object """
        # Number of samples
        n_samples = len(clustering.labels_)

        # Create the linkage matrix
        counts = np.zeros(clustering.children_.shape[0])
        n_clusters = np.arange(n_samples, n_samples + clustering.children_.shape[0])

        # Calculate the number of samples in each cluster
        for i, merge in enumerate(clustering.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # original sample
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        # Create linkage matrix
        linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)

        fig, ax = plt.subplots()
        dendrogram(linkage_matrix, ax=ax, color_threshold=threshold)

        return fig, ax

    @property
    def clustered_feat_names(self):
        features = np.array(self.features)
        return {
            clt_idx: features[feat_idx].tolist()
            for clt_idx, feat_idx in self.ntri_feat_cluster.items()
        }

    def pca_dimension_reduction(self):
        pcaX = []
        clustering_map = {}
        clustered_feat_idx = []
        for clt_idx, lst_feat_idx in self.ntri_feat_cluster.items():

            if len(lst_feat_idx) > 1:
                pca = PCA(n_components=1)
                pca_x = pca.fit_transform(self.scaled_X[:, lst_feat_idx]).flatten()
                pcaX.append(pca_x)

                clustered_feat_idx.extend(lst_feat_idx)
                clustering_map[f'c{len(clustering_map)}'] = lst_feat_idx

        self.clt_X = np.array(pcaX).T
        self.clustered_map = clustering_map
        self.clustered_feat_indices = np.array(clustered_feat_idx)

    @property
    def reduced_features(self):
        return np.delete(
            np.array(self.features),
            np.hstack([self.trivial_feat_indices, self.clustered_feat_indices]).tolist(),
        ).tolist() + [f'c{i}' for i in range(len(self.clustered_map))]

    @property
    def reduced_sX(self):
        if isinstance(self.clt_X, np.ndarray):
            return np.concatenate(
                [
                    np.delete(
                        self.scaled_X,
                        # Drop the trivial and clustered features
                        np.hstack([self.trivial_feat_indices, self.clustered_feat_indices]).tolist(),
                        axis=1),
                    self.clt_X
                ],
                axis=1
            )
        else:
            return self.ntri_sX

    @property
    def reduced_sXtr(self):
        """ Scaled train X after redundant reduction """
        return self.reduced_sX[self.train_indices]

    @property
    def reduced_sXte(self):
        """ Scaled test X after redundant reduction """
        return self.reduced_sX[self.test_indices]

    def _initialize_feature_collection(self):
        """ Adding significant features to the final essential feature collection initially """
        if not self.recur_addi_prev_threshold:
            return []
        elif isinstance(self.recur_addi_prev_threshold, float) and 0.0 < self.recur_addi_prev_threshold < 1.0:
            feat_measurer = self._get_feature_measurer()

            feat_measurer.fit(self.reduced_sXtr, self.sYtr)
            return np.where(feat_measurer.feature_importances_ > self.recur_addi_prev_threshold)[0].tolist()
        elif not isinstance(self.recur_addi_prev_threshold, float):
            raise TypeError(f"recur_addi_prev_threshold should be a float, but got {type(self.recur_addi_prev_threshold)}")
        else:
            raise ValueError("recur_addi_prev_threshold need to larger than 0. and less than 1.0")

    def recursive_feature_addition(self):
        """ determine the final essential features by a recursive addition procedure """
        selected_feat_indices = self._initialize_feature_collection()

        reduced_sXtr = self.reduced_sXtr

        best_metric = None
        cycle_best_metric = None
        cycle_best_index = None
        while best_metric is None or len(selected_feat_indices) < len(self.reduced_features):

            if cycle_best_metric is not None:
                if (
                        best_metric is None or
                        best_metric + self.recur_addi_threshold < cycle_best_metric
                ):
                    best_metric = cycle_best_metric
                    selected_feat_indices.append(cycle_best_index)
                    cycle_best_metric = None
                else:
                    break

            for i in range(len(self.reduced_features)):
                if i in selected_feat_indices:
                    continue

                current_indices = selected_feat_indices + [i]
                current_sXtr = reduced_sXtr[:, current_indices]

                t1 = time.time()
                metrix, pred, true = cross_valid(self.estimator, current_sXtr, self.sYtr)
                metrix = metrix.mean()
                t2 = time.time()

                if t2 - t1 >= 500:
                    raise TimeoutError('Cross validation timed out after 500 second')
                else:
                    logging.debug(f'time to run cross val for {self.estimator} '
                          f'and {np.array(self.reduced_features)[current_indices]} '
                          f'in RFA is {t2 - t1}')

                if cycle_best_metric is None or metrix > cycle_best_metric:
                    cycle_best_metric = metrix
                    cycle_best_index = i

        return np.array(selected_feat_indices), best_metric
        # self.cv_best_metric = best_metric

    def recursive_feature_elimination(self):
        """"""
        if hasattr(self.estimator, 'feature_importances_') or hasattr(self.estimator, 'coef_'):
            measurer = clone(self.estimator)
        elif self.feat_imp_measurer is not None:
            measurer = self.feat_imp_measurer
        else:
            measurer = GradientBoostingRegressor()

        selector = RFECV(estimator=measurer)
        selector.fit(self.reduced_sXtr, self.sYtr)

        essential_feat_indices = np.nonzero(selector.support_)[0]
        # print(essential_feat_indices)
        # print(np.array(self.reduced_features)[essential_feat_indices].tolist())

        return (essential_feat_indices,
                cross_val_score(clone(measurer), self.reduced_sXtr[:, essential_feat_indices], self.sYtr).mean())

    def recursive_feature_selection(self):
        """ Recursive feature selection """
        print('Recursive feature selection...')
        feat_indices_eli, r2_score_eli = self.recursive_feature_elimination()
        feat_indices_add, r2_score_add = self.recursive_feature_addition()

        if (diff := r2_score_add - r2_score_eli) >= 0.01:
            self.essential_feat_indices = feat_indices_add
            self.cv_best_metric = r2_score_add
            self.how_to_essential = 'Recursive addition'
        elif diff > -0.01:
            if len(feat_indices_add) <= len(feat_indices_eli):
                self.essential_feat_indices = feat_indices_add
                self.cv_best_metric = r2_score_add
                self.how_to_essential = 'Recursive addition'
            else:
                self.essential_feat_indices = feat_indices_eli
                self.cv_best_metric = r2_score_eli
                self.how_to_essential = 'Recursive elimination'
        else:
            self.essential_feat_indices = feat_indices_eli
            self.cv_best_metric = r2_score_eli
            self.how_to_essential = 'Recursive elimination'

        print(self.how_to_essential)
        print(f'the essential features are:')
        for feat in self.essential_features:
            print(feat)

    @staticmethod
    def permutate_importance(estimator, *args, **kwargs):
        estimator.fit(X, y)
        return permutation_importance(estimator, *args, **kwargs)

    @staticmethod
    def gini_importance(estimator, X, y):
        """"""
        if not hasattr(estimator, 'feature_importances_') or hasattr(estimator, 'coef_'):
            estimator = RandomForestRegressor()

        estimator.fit(X, y)

        try:
            return estimator.feature_importances_
        except AttributeError:
            return estimator.coef_

    def _get_dataset(
            self,
            feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
            sample_type: Literal['train', 'test', 'all'] = 'all'
    ):
        """ Get dataset """
        if sample_type == 'all':
            y = self.scaled_y
            if feature_type == 'scaled':
                X = self.scaled_X
            elif feature_type == 'reduced':
                X = self.reduced_sX
            elif feature_type == 'essential':
                X = self.essential_sX
            else:
                raise ValueError('The feature type must be either "essential" or "reduced" or "scaled"')

        elif sample_type == 'train':
            y = self.sYtr
            if feature_type == 'scaled':
                X = self.sXtr
            elif feature_type == 'reduced':
                X = self.reduced_sXtr
            elif feature_type == 'essential':
                X = self.essential_sXtr
            else:
                raise ValueError('The feature type must be either "essential" or "reduced" or "scaled"')

        elif sample_type == 'test':
            y = self.sYte
            if feature_type == 'scaled':
                X = self.sXte
            elif feature_type == 'reduced':
                X = self.reduced_sXte
            elif feature_type == 'essential':
                X = self.essential_sXte
            else:
                raise ValueError('The feature type must be either "essential" or "reduced" or "scaled"')

        else:
            raise ValueError('The sample type must be either "all" or "train" or "test"')

        return X, y

    def cross_validation(
            self,
            feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
            sample_type: Literal['train', 'test', 'all'] = 'train'
    ):
        """"""
        X, y = self._get_dataset(feature_type, sample_type)
        self.valid_score, self.valid_pred, self.valid_true = cross_valid(
            self.estimator.__class__(**self.optimal_hyperparams), X, y,
            cv=KFold(n_splits=5, shuffle=True)
        )

        print(f'The performance in 5fold cross validation:')
        print(f'\t{self.valid_score}')

        r2_plot = R2Regression([self.valid_true, self.valid_pred], to_cv=True)
        sciplot = SciPlotter(r2_plot)

        fig, axs = sciplot()
        fig.savefig(self.picture_dir.joinpath('cross_valid.png'))

    def train_model(self):
        """"""
        self.estimator.fit(self.essential_sXtr, self.sYtr)

        pred_train = self.estimator.predict(self.essential_sXtr)
        pred_test = self.estimator.predict(self.essential_sXte)

        r2_train = r2_score(self.sYtr, pred_train)
        mae_train = mean_absolute_error(self.sYtr, pred_train)
        rmse_train = mean_squared_error(self.sYtr, pred_train)

        r2_test = r2_score(self.sYte, pred_test)
        mae_test = mean_absolute_error(self.sYte, pred_test)
        rmse_test = np.sqrt(mean_squared_error(self.sYte, pred_test))

        print(f'trained model performance:\n'
              f'\ttrain set:\tR^2={r2_train}\tMAE={mae_train}\tRMSE={rmse_train}\n'
              f'\ttest set:\tR^2={r2_test}\tMAE={mae_test}\tRMSE={rmse_test}\n')

        # Make pictures
        plotter = SciPlotter(R2Regression([self.sYtr, pred_train], [self.sYte, pred_test]))
        fig, axs = plotter()
        fig.savefig(self.picture_dir.joinpath('train_pred.png'))

    def determine_importance(
            self,
            feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
            sample_type: Literal['train', 'test', 'all'] = 'all'
    ):
        X, y = self._get_dataset(feature_type, sample_type)
        self.feature_importance = {
            "perm": self.permutate_importance(clone(self.estimator), X, y),
            "gini": self.gini_importance(self.estimator, X, y)
        }

    @property
    def essential_features(self):
        if self._essential is not None:
            return np.array(self.features)[self._essential].tolist()
        return np.array(self.reduced_features)[self.essential_feat_indices].tolist()

    @property
    def essential_sX(self):
        if self._essential is not None:
            return self.scaled_X[:, self._essential]
        return self.reduced_sX[:, self.essential_feat_indices]

    @property
    def essential_sXtr(self):
        return self.essential_sX[self.train_indices]

    @property
    def essential_sXte(self):
        return self.essential_sX[self.test_indices]

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
        Generates a hypothetical test X with similar covariance of features as template X.
        Args:
            template_X: template X to define covariance (matrix). if not given, Essential_sX will be applied.
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
            template_X = self.essential_sX

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
            df_shap_value = pd.DataFrame(shap_values.values, columns=self.essential_features)
            df_data_value = pd.DataFrame(shap_values.data, columns=self.essential_features)

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
            self.estimator, self.essential_sX, self.scaled_y, self.essential_features,
            # explainer_cls=self.kwargs.get('shap_explainer_cls'),
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


class MachineLearning_:
    """
    A Standard procedure for machine learning.
    """
    def __init__(
            self,
            work_dir: Union[Path, str],
            data: pd.DataFrame,
            features: Sequence[str],
            target,
            estimator: BaseEstimator = GradientBoostingRegressor(),
            **kwargs
    ):
        """
        Args:
            work_dir (Path|str): Path to save results
            data (pd.DataFrame): Raw dataset, including features, target, and other correlation information
            features (Sequence|np.ndarray): feature columns in data DataFrame
            target: target column in data DataFrame
            estimator (BaseEstimator, optional): the algorithm class to train model

        Keyword Args:
            xscaler (sklearn.preprocessing.BaseScaler):
            yscaler (sklearn.preprocessing.BaseScaler):
            test_indices (Sequence[int], optional):
            test_size (float, optional):
            data_shuffle (bool, optional):
            data_stratify (bool, optional):
            stratify_log (bool, optional):
            feat_imp_measurer (Callable, optional): An object with methods `fit()` and one of `coef_` or
                `feature_importance_`, which is applied to measure the feature importance in feature engineering.
            feat_imp_cutoff (float, optional): A feature importance cutoff to drop out the feature with less
                importance from feature collection.
            recur_addi_threshold (float, optional):
            recur_addi_prev_threshold (float, optional):
            essential_features (Sequence[int|str], optional): the threshold of feature importance, which is
                evaluated by a given measurer ML algorithm, like RF, where a feature with an importance larger
                than this threshold will be added into the initial essential feature collection.
            skip_feature_reduce (bool, optional): Whether to skip feature reduction procedure
            hyper_optimizer:
            dir_exists (bool, optional): whether to allow stored data in an existing directory.
        """
        self.work_dir = Path(work_dir)
        self.picture_dir = self.work_dir.joinpath('pictures')
        self.sheet_dir = self.work_dir.joinpath('sheet')

        self.data = data
        self.target = target
        self.estimator = estimator

        self.sample_num = len(self.data)
        self.other_info = data.drop(target, axis=1).drop(features, axis=1)

        self.xscaler = kwargs.get('xscaler', None)
        self.yscaler = kwargs.get('yscaler', None)

        # Args to control dataset split
        self.test_indices = kwargs.get('test_indices', None)
        self.train_indices = None
        self.test_size = kwargs.get('test_size', 0.2)
        self.data_shuffle = kwargs.get('data_shuffle', True)
        self.data_stratify = kwargs.get('data_stratify', False)
        self.stratify_log = kwargs.get('stratify_log', False)

        # Args to control Feature Engineering
        self.feat_imp_measurer = kwargs.get('feat_imp_measurer', None)
        self.feat_imp_cutoff = kwargs.get('feat_imp_cutoff', 1e-3)
        self.feat_cluster_threshold = kwargs.get('feat_cluster_threshold', None)
        self.recur_addi_threshold = kwargs.get('recur_addi_threshold', 0.0)
        # A float range from 0.0 to 1.0. If given, the importance of features
        # will be evaluated firstly by a model, say GradientBoostingRegressor.
        # Then, the features with importance more than the given threshold
        # will be added into the essential feature collection, before the
        # recursive addition procedure is running.
        self.recur_addi_prev_threshold = kwargs.get('recur_addi_prev_threshold')

        # Feature engineering results
        self.pearson_mat_dict = None
        self.clustered_maps = None
        self.clustering = None
        self.pca_features = None
        self.pca_X = None
        self.cv_best_metric = None,
        self.how_to_essential = None

        self.feature_importance = {}
        self.valid_score = None
        self.valid_pred = None
        self.valid_true = None

        # Attributes reserved for SHAP analysis
        self.explainer = None
        self.shap_value = None

        self._feature_names = [self._check_feature_names(features)]
        self._XS = [data.loc[:, features].values]
        self._ys = [data[target].values]
        self._feature_types = ['original']

        self.kwargs = kwargs

    @property
    def X(self):
        return self._XS[-1]

    @property
    def y(self):
        return self._ys[-1]

    @property
    def X_train(self):
        return self.X[self.train_indices]

    @property
    def y_train(self):
        return self.y[self.train_indices]

    @property
    def X_test(self):
        return self.X[self.test_indices]

    @property
    def y_test(self):
        return self.y[self.test_indices]

    @property
    def features(self):
        return self._feature_names[-1]

    @property
    def feature_type(self):
        return self._feature_types[-1]

    def _check_feature_names(self, features: Sequence[str]) -> list[str]:
        sheet_title = self.data.columns.tolist()

        for name in features:
            if name not in sheet_title:
                raise ValueError(f"The feature {name} not in data")

        return list(features)

    def _feature_type_to_index(self, type_name: str):
        return self._feature_types.index(type_name)

    def _get_feature_and_X(self, type_name: str):
        i = self._feature_type_to_index(type_name)
        feature_names = self._feature_names[i]
        X = self._XS[i]

        return feature_names, X

    @staticmethod
    def feature_names_to_indices_(feature_list: list, get_names: list):
        return np.array([feature_list.index(n) for n in get_names])

    def _feature_names_to_indices(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self.features.index(n) for n in names])

    def _get_essential_features(self):
        essential_features = self.kwargs.get('essential_features')
        if isinstance(essential_features, (Sequence, np.ndarray)):
            feat_index = self._feature_names_to_indices(essential_features)
            self._update(self.X[:, feat_index], self.y, essential_features, 'essential')

    def work(self):
        if not self.work_dir.exists():
            print("Creating work directory: {}".format(self.work_dir))
            self.work_dir.mkdir()
        elif os.listdir(self.work_dir):
            if self.kwargs.get('dir_exists'):
                for path in self.work_dir.glob("*"):
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            else:
                raise IOError('Work directory must be an empty directory')

        self.picture_dir.mkdir()
        self.sheet_dir.mkdir()

        self.preprocess()
        self.train_test_split()
        self._get_essential_features()

        self.feature_engineering()

        self.cross_validation()
        self.train_model()
        self.export_feature_importance(True)

        self.shap_analysis()

    @staticmethod
    def preprocess_(X, y, xscaler=None, yscaler=None) -> (np.ndarray, np.ndarray):
        scaled_X, scaled_y = X, y

        if xscaler is not None:
            xscaler.fit(X)
            scaled_X = xscaler.transform(X)

        if yscaler is not None:
            yscaler.fit(y)
            scaled_y = yscaler.transform(y)

        return scaled_X, scaled_y

    def preprocess(self):
        print("Preprocessing data...")
        if not self.kwargs.get('skip_preprocess', None):
            scaled_X, scaled_y = self.preprocess_(self.X, self.y, self.xscaler, self.yscaler)
        else:
            scaled_X, scaled_y = self.X, self.y

        self._update(scaled_X, scaled_y, self.features, 'scaled')

    def invert_scaling_X(
            self,
            X,
            features: list[str]=None,
            recovery_cluster_to_original: bool = False
    ):
        """
        Inversely transforms the scaled data `X` back to its original scale.

        This method accounts for both original features and features derived from clustering methods
        such as PCA. It can recover original features from clustered features if requested.

        Parameters
        ----------
        X : numpy.ndarray
            The data matrix to be inverse-transformed.
        features : list of str, optional
            List of feature names corresponding to the columns in `X`. If `None`, it is assumed
            that `X` contains all original features in order.
        recovery_cluster_to_original : bool, default False
            If True, recovers original features from clustered features (e.g., PCA components).

        Returns
        -------
        numpy.ndarray
            The data `X` transformed back to its original scale. If `recovery_cluster_to_original`
            is False, PCA features are appended without inversion.

        Raises
        ------
        ValueError
            If the length of `features` does not match the number of columns in `X`.
        ValueError
            If a given feature name is neither an original feature nor a clustered feature.
        ValueError
            If feature indices are not unique after processing.
        """

        # If features are not provided, assume X contains all original features in order
        if features is None:
            return self.xscaler.inverse_transform(X), self._feature_names[0]

        else:
            if X.shape[1] != len(features):
                raise ValueError('The length of features must be the same as the X.shape[1]')

            ori_features, _ = self._get_feature_and_X('original')

            # Initialize lists for indices and data
            inv_features = []
            indices_in_ori = []

            # Store feature names and value matrix after PCA dimensionality reduction
            pca_features = []
            pca_X = []

            # Store recovered features and values matrix from PCA-reduced features
            recovered_features = []
            recovered_X = []

            drop_cols = []
            for i, f in enumerate(features):
                try: # Check if the feature name (f) is an original name?
                    indices_in_ori.append(ori_features.index(f))
                    inv_features.append(f)

                # If the given feature name (f) is not an original name, ...
                except ValueError:
                    drop_cols.append(i)
                    _, pca_x, form_feature, form_X = self.get_clustering_feat_and_X_by_clustering_name(f)
                    if form_feature:
                        if recovery_cluster_to_original:
                            recovered_features.extend(form_feature)

                            recovered_x = self.invert_pca_x_(X[:, i], form_X, pca_x)
                            recovered_X.append(recovered_x)

                        else:
                            pca_features.append(f)
                            pca_X.append(X[:, i].reshape(-1, 1))

                    # If the feature name neither an original name nor clustered name, raise Error
                    else:
                        raise ValueError(f'The feature {f} is neither original nor clustered features')

            # Remove columns in X that are not original features
            X = np.delete(X, drop_cols, axis=1)

            # Convert the feature name to form clusters to its index in the original feature list
            indices_to_form_cluster = [ori_features.index(f) for f in recovered_features]

            # Merge
            indices_in_ori = np.array(indices_in_ori + indices_to_form_cluster)
            X = np.hstack([X]+recovered_X)

            # Make sure each index is unique.
            if len(np.unique(indices_in_ori)) != len(indices_in_ori):
                raise ValueError(f'The given feature nams cannot make sure indices unique')

            X_pad = np.zeros((X.shape[0], len(ori_features)))

            logging.debug(f"X_pad shape is {X_pad.shape}")
            logging.debug(f"indices_in_ori is {indices_in_ori}")
            X_pad[:, indices_in_ori] = X

            X_inv = self.xscaler.inverse_transform(X_pad)
            X_inv = X_inv[:, indices_in_ori]
            inv_features += recovered_features

            # Append the PCA feature values without inverse.
            if pca_X:
                return np.hstack((X_inv, np.hstack(pca_X))), inv_features + pca_features
            else:
                return X_inv, inv_features

    def get_clustering_feat_and_X_by_clustering_name(self, clustering_name: str, which_stage: str = 'non_trivial'):
        """ Get the names and value matrix of the features to form a specific cluster """
        try:
            features_to_form_cluster = self.clustered_maps[which_stage][clustering_name]
        except KeyError:
            return None, None, None, None

        # Get pca_x
        clustering_index = self.pca_features.index(clustering_name)
        pca_x = self.pca_X[:, clustering_index]

        feature_list_before_cluster, X_before_feature = self._get_feature_and_X(which_stage)

        feat_index = [feature_list_before_cluster.index(f_name) for f_name in features_to_form_cluster]
        X_form_cluster = X_before_feature[:, feat_index]

        return clustering_name, pca_x, features_to_form_cluster, X_form_cluster

    @staticmethod
    def invert_pca_x_(inverted_x, X_clt, x_pca, show_cv_results=True):
        """"""
        if show_cv_results:
            print(cross_val_score(MLPRegressor(activation='relu'), x_pca.reshape(-1, 1), X_clt))

        network = MLPRegressor(activation='relu')
        network.fit(x_pca.reshape(-1, 1), X_clt)

        return network.predict(inverted_x.reshape(-1, 1))

    def invert_clustering_x(self, X_clt, clustering_names: Union[str, list[str]]):
        """"""

    def _get_stratify(self):
        if self.data_stratify:
            y_min, y_max = self.y.min(), self.y.max()
            if self.stratify_log is True:
                y_min, y_max = np.log(y_min), np.log(y_max)

            stratify = np.zeros_like(self.y)
            split_point = np.linspace(y_min, y_max, self.data_stratify + 1).reshape(-1, 1)
            split_indices = np.logical_and(split_point[:-1] <= self.y, self.y <= split_point[1:])

            for i, si in enumerate(split_indices, 1):
                stratify[si] = i

            return stratify

    def train_test_split(self):
        print('Splitting data...')
        if self.test_indices is None:
            self.train_indices, self.test_indices = train_test_split(
                np.arange(self.sample_num),
                shuffle=self.data_shuffle,
                test_size=self.test_size,
                stratify=self._get_stratify()
            )

        else:
            self.test_indices = np.array(self.test_indices).flatten()
            self.train_indices = np.where(np.isin(np.arange(len(self.X)), self.test_indices, invert=True))[0]

    def feature_engineering(self):
        """ Performing feature engineering """
        print('Feature engineering...')

        if self.feature_type == 'scaled':
            self.quickly_feature_selection()

        if not self.kwargs.get('skip_feature_reduce') and self.feature_type == 'non_trivial':
            print('Dimensionality Reduction and Redundancy Elimination')
            self.calc_pearson_matrix()
            self.feat_hiera_clustering()
            self.pca_dimension_reduction()

        if (
                self.feature_type == 'reduced'
                or (self.kwargs.get('skip_feature_reduce') and self.feature_type == 'non_trivial')
        ):
            self.recursive_feature_selection()

    @staticmethod
    def _get_feature_measure_(measurer):
        return measurer if hasattr(measurer, 'feature_importances_') else GradientBoostingRegressor()

    def _get_feature_measurer(self):
        """ Getting an estimator to measure the feature importance """
        if self.feat_imp_measurer:
            return self.feat_imp_measurer

        elif hasattr(self.estimator, 'feature_importances_') or hasattr(self.estimator, 'coef_'):
            return clone(self.estimator)

        else:
            return GradientBoostingRegressor()

    @staticmethod
    def quickly_feature_selection_(X, y, feat_imp_measurer, feat_imp_threshold=1e-4):
        feat_imp_measurer = MachineLearning_._get_feature_measure_(feat_imp_measurer)
        feat_imp_measurer.fit(X, y)

        if hasattr(feat_imp_measurer, 'feature_importances_'):
            feat_imp = feat_imp_measurer.feature_importances_

        elif hasattr(feat_imp_measurer, 'coef_'):
            feat_imp = feat_imp_measurer.coef_

        else:
            raise AttributeError(f'the given feat_imp_measurer {feat_imp_measurer} has '
                                 f'not attribute `feature_importances_` or `coef_`')

        non_trivial_feat_indices = np.where(feat_imp >= feat_imp_threshold)[0]
        return non_trivial_feat_indices

    def quickly_feature_selection(self):
        print('Quickly feature selection...')
        feat_imp_measurer = self._get_feature_measurer()

        non_trivial_feat_indices = (
            self.quickly_feature_selection_(self.X, self.y, feat_imp_measurer, self.feat_imp_cutoff))

        self._update(
            self.X[:, non_trivial_feat_indices],
            self.y,
            np.array(self.features)[non_trivial_feat_indices].tolist(),
            'non_trivial'
        )

    @property
    def pearson_mat(self) -> np.ndarray:
        return self.pearson_mat_dict[self.feature_type]

    @staticmethod
    def calc_pearson_matrix_(X):
        return np.nan_to_num(np.corrcoef(X, rowvar=False))

    def calc_pearson_matrix(self):
        """ calculate Pearson matrix for all non-trivial features """
        self.pearson_mat_dict = {self.feature_type: self.calc_pearson_matrix_(self.X)}
        self.make_pearson_matrix_plot()

    def make_pearson_matrix_plot(self):
        """ draw Pearson matrix plot and save it to `work_dir/picture_dir` """
        plotter = SciPlotter(PearsonMatrix(self.pearson_mat, self.features, is_matrix=True))
        fig, ax = plotter()
        fig.savefig(self.picture_dir.joinpath(f'Pearson_mat_{self.feature_type}.png'))

    @staticmethod
    def make_hierarchical_tree(clustering, threshold):
        """ make a hierarchical tree plot return the Figure and Axes object """
        # Number of samples
        n_samples = len(clustering.labels_)

        # Create the linkage matrix
        counts = np.zeros(clustering.children_.shape[0])
        n_clusters = np.arange(n_samples, n_samples + clustering.children_.shape[0])

        # Calculate the number of samples in each cluster
        for i, merge in enumerate(clustering.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # original sample
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        # Create linkage matrix
        linkage_matrix = np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)

        fig, ax = plt.subplots()
        dendrogram(linkage_matrix, ax=ax, color_threshold=threshold)

        return fig, ax

    @staticmethod
    def feat_hiera_clustering_(X, features, y=None, cluster_threshold=None):
        if not cluster_threshold:
            cluster_threshold = np.sqrt(pow(0.1, 2) * X.shape[-1])

        abs_pearson_mat = np.abs(MachineLearning_.calc_pearson_matrix_(X))
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=cluster_threshold)
        clustering.fit(abs_pearson_mat)

        mapping_dict = {}
        for i, cluster_label in enumerate(clustering.labels_):
            lst_feat_names = mapping_dict.setdefault(cluster_label, [])
            lst_feat_names.append(features[i])

        clustering_map = {}
        for form_list in mapping_dict.values():
            if len(form_list) > 1:
                clustering_map[f"c{len(clustering_map)}"] = form_list

        return cluster_threshold, clustering_map, clustering

    def feat_hiera_clustering(self):
        self.clustered_maps = {}

        (
            self.feat_cluster_threshold,
            self.clustered_maps[self.feature_type],
            self.clustering

        ) = self.feat_hiera_clustering_(

            self.X,
            self.features,
            self.y,
            self.feat_cluster_threshold
        )

        fig, ax = self.make_hierarchical_tree(self.clustering, self.feat_cluster_threshold)
        fig.savefig(self.picture_dir.joinpath('hiera_tree.png'))

    @staticmethod
    def pca_dimension_reduction_(
            X: np.ndarray,
            features: list[str],
            clustering_map: dict = None,
            clustering_threshold: float = None
    ):
        """
        Reduce groups of features to ones of reduced features, by Principal Component Analysis (PCA).
        The groups are given by the dict argument `clustering_map`.

        Args:
            X: the orignal feature matrix before dimension reduction,
            features: the original feature names, with a same order as the given `X`.
            clustering_map(dict): the mapping between the reduced feature names (*key*) to the groups of
                feature names (*value*, list of original features).

        Return:
            pcaX: the feature matrix after undergoing PCA dimensionality reduction does not include features that
                were not subjected to dimensionality reduction.
            pca_features: the names of features from pca dimensionality reduction.
            reduced_X: the feature matrix after undergoing PCA dimensionality reduction, including features that
                were not subjected to dimensionality reduction.
            reduced_features: list of feature name, with a same order in `reduced_X`
        """
        if not clustering_map:
            _, clustering_map, _ = MachineLearning_.feat_hiera_clustering_(
                X,
                features,
                None,
                clustering_threshold)

        pcaX = []
        clustered_feat_idx = []
        for clt_idx, clt_feats in clustering_map.items():

            if len(clt_feats) > 1:
                pca = PCA(n_components=1)
                clt_feat_idx = [features.index(name) for name in clt_feats]
                pca_x = pca.fit_transform(X[:, clt_feat_idx]).flatten()
                pcaX.append(pca_x)

                clustered_feat_idx.extend(clt_feat_idx)

        pcaX = np.array(pcaX).T
        pca_features = [c_name for c_name in clustering_map]
        reduced_X = np.concatenate((np.delete(X, clustered_feat_idx, axis=1), pcaX), axis=1)
        reduced_features = (np.delete(np.array(features), clustered_feat_idx).tolist() + pca_features)

        return pcaX, pca_features, reduced_X, reduced_features

    def pca_dimension_reduction(self):
        """ Merging similar feature to a new clustering feature by principal component analysis (PCA) method, the
        similarity or distance determined by Pearson matrix and hierarchical tree."""
        self.pca_X, self.pca_features, reduced_X, reduced_features = self.pca_dimension_reduction_(
            self.X, self.features, self.clustered_maps['non_trivial']
        )

        self._update(reduced_X, self.y, reduced_features, 'reduced')

    def _update(self, X, y, feature_name, feature_type):
        """ Update the feature dictionary. """
        self._XS.append(X)
        self._ys.append(y)
        self._feature_names.append(feature_name)
        self._feature_types.append(feature_type)

    @staticmethod
    def _initialize_feature_collection(
            X, y, feat_measurer,
            recur_addi_prev_threshold: float = None,
    ):
        """ Adding significant features to the final essential feature collection initially """
        if not recur_addi_prev_threshold:
            return []

        elif isinstance(recur_addi_prev_threshold, float) and 0.0 < recur_addi_prev_threshold < 1.0:
            feat_measurer.fit(X, y)

        elif not isinstance(recur_addi_prev_threshold, float):
            raise TypeError(f"recur_addi_prev_threshold should be a float, but got {type(recur_addi_prev_threshold)}")

        else:
            raise ValueError("recur_addi_prev_threshold need to larger than 0. and less than 1.0")

        return np.where(feat_measurer.feature_importances_ > recur_addi_prev_threshold)[0].tolist()

    @staticmethod
    def recursive_feature_addition_(
            X_train: np.ndarray,
            y_train: np.ndarray,
            feature_measurer,
            prev_addi_threshold: float = 0.2,
            recur_addi_threshold: float = 0.
    ):
        """"""
        selected_feat_indices = MachineLearning_._initialize_feature_collection(
            X_train, y_train, feature_measurer, prev_addi_threshold
        )

        best_metric = None
        cycle_best_metric = None
        cycle_best_index = None
        while best_metric is None or len(selected_feat_indices) < X_train.shape[-1]:

            # Add the best feature to improve model performance
            if cycle_best_metric is not None:
                if (
                        best_metric is None or
                        best_metric + recur_addi_threshold < cycle_best_metric
                ):
                    best_metric = cycle_best_metric
                    selected_feat_indices.append(cycle_best_index)
                    cycle_best_metric = None
                else:
                    break

            # Screening feature in single cycle
            for i in range(X_train.shape[-1]):
                if i in selected_feat_indices:
                    continue

                current_indices = selected_feat_indices + [i]
                current_X = X_train[:, current_indices]

                metrix, pred, true = cross_valid(feature_measurer, current_X, y_train)
                metrix = metrix.mean()

                if cycle_best_metric is None or metrix > cycle_best_metric:
                    cycle_best_metric = metrix
                    cycle_best_index = i

        return selected_feat_indices, best_metric

    def recursive_feature_addition(self):
        """ determine the final essential features by a recursive addition procedure """
        return self.recursive_feature_addition_(
            self.X_train,
            self.y_train,
            clone(self.estimator),
            self.recur_addi_prev_threshold,
            self.recur_addi_threshold
        )

    @staticmethod
    def recursive_feature_elimination_(
        X_train: np.ndarray,
        y_train: np.ndarray,
        feat_measurer,
    ):
        selector = RFECV(estimator=feat_measurer)
        selector.fit(X_train, y_train)

        feat_indices = np.nonzero(selector.support_)[0]

        return feat_indices, cross_val_score(clone(feat_measurer), X_train[:, feat_indices], y_train).mean()

    def recursive_feature_elimination(self):
        """"""
        if (hasattr(self.estimator, 'feature_importances_')
                # or hasattr(self.estimator, 'coef_')
        ):
            measurer = clone(self.estimator)
        elif self.feat_imp_measurer is not None:
            measurer = self.feat_imp_measurer
        else:
            measurer = GradientBoostingRegressor()

        return self.recursive_feature_elimination_(self.X_train, self.y_train, measurer)

    @staticmethod
    def recursive_feature_selection_(
            X, features,
            feat_indices_eli, score_eli,
            feat_indices_add, score_add
    ):
        """"""
        if (diff := score_add - score_eli) >= 0.01:
            features = np.array(features)[feat_indices_add]
            X = X[:, feat_indices_add]
            cv_best_metric = score_add
            select_which = 'Recursive addition'
        elif diff > -0.01:
            if len(feat_indices_add) <= len(feat_indices_eli):
                features = np.array(features)[feat_indices_add]
                X = X[:, feat_indices_add]
                cv_best_metric = score_add
                select_which = 'Recursive addition'
            else:
                features = np.array(features)[feat_indices_eli]
                X = X[:, feat_indices_eli]
                cv_best_metric = score_eli
                select_which = 'Recursive elimination'
        else:
            features = np.array(features)[feat_indices_eli]
            X = X[:, feat_indices_eli]
            cv_best_metric = score_eli
            select_which = 'Recursive elimination'

        return X, features, cv_best_metric, select_which

    def recursive_feature_selection(self):
        """ Recursive feature selection """
        print('Recursive feature selection...')
        feat_indices_eli, r2_score_eli = self.recursive_feature_elimination()
        feat_indices_add, r2_score_add = self.recursive_feature_addition()

        X, features, cv_best_metric, how_to_essential = self.recursive_feature_selection_(
            self.X, self.features,
            feat_indices_eli, r2_score_eli,
            feat_indices_add, r2_score_add
        )

        self.cv_best_metric = cv_best_metric
        self.how_to_essential = how_to_essential
        self._update(X, self.y, features, 'essential')

        print(self.how_to_essential)
        print(f'the essential features are:')
        for feat in self.features:
            print(feat)

    @staticmethod
    def has_fitted(estimator: BaseEstimator):
        try:
            check_is_fitted(estimator)
            return True
        except AttributeError:
            return False

    @staticmethod
    def permutate_importance(estimator, X, y, features, *args, **kwargs):
        return features, permutation_importance(estimator, X, y, *args, **kwargs)['importances']

    @staticmethod
    def gini_importance(estimator, X, y, features: list[str]):
        """"""
        if X.shape[1] != len(features):
            raise ValueError('the columns must have the same number of features')

        if not hasattr(estimator, 'feature_importances_') or hasattr(estimator, 'coef_'):
            estimator = RandomForestRegressor()

        if not MachineLearning_.has_fitted(estimator):
            raise AttributeError('the estimator is not fitted')

        try:
            return features, estimator.feature_importances_
        except AttributeError:
            return features, estimator.coef_

    def determine_importance(
            self,
            feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
            sample_type: Literal['train', 'test', 'all'] = 'all'
    ):
        X, y = self._get_dataset(feature_type, sample_type)
        self.feature_importance = {
            "perm": self.permutate_importance(self.estimator, X, y, self.features),
            "gini": self.gini_importance(self.estimator, X, y, self.features)
        }

    def export_feature_importance(self, to_determine: bool = False):
        if to_determine:
            self.determine_importance()

        # Save feature importance to sheet
        with pd.ExcelWriter(self.sheet_dir.joinpath('feat_imp.xlsx')) as writer:
            for imp_type, (feat, imp) in self.feature_importance.items():
                if len(imp.shape) == 1:
                    data = pd.Series(imp, index=feat, name=imp_type)
                else:
                    data = pd.DataFrame(imp, index=feat)

                data.to_excel(writer, sheet_name=imp_type)

        # Save importance as picture
        def _draw_imp(ax: plt.Axes, sci_plotter):
            nonlocal imp_, feat_, imp_type_

            if len(imp_.shape) == 1:
                sort_idx = np.argsort(imp_)[::-1]
                feat_ = np.array(feat_)[sort_idx].tolist()
                imp_ = imp_[sort_idx]

                ax.bar(x=feat_, height=imp_)
            elif len(imp_.shape) == 2:
                sort_idx = np.argsort(imp_.mean(axis=1))[::-1]
                feat_ = np.array(feat_)[sort_idx].tolist()
                imp_ = imp_[sort_idx]

                ax.boxplot(imp_.T, positions=np.arange(len(imp_)))
            else:
                raise AttributeError("imp_ must have 1 or 2 dimensions")

            ax.set_xticks(ax.get_xticks(), feat_)

            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')

            ax.set_title(imp_type_)

        for imp_type_, (feat_, imp_) in self.feature_importance.items():
            plotter = SciPlotter(_draw_imp)
            fig, ax = plotter()

            fig.savefig(self.picture_dir.joinpath(f'imp_{imp_type_}.png'))

    def _get_dataset(
            self,
            feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
            sample_type: Literal['train', 'test', 'all'] = 'all'
    ):
        feature_type_idx = self._feature_types.index(feature_type)
        X, y = self._XS[feature_type_idx], self._ys[feature_type_idx]

        if sample_type == 'all':
            return X, y
        elif sample_type == 'train':
            return X[self.train_indices], y[self.train_indices]
        elif sample_type == 'test':
            return X[self.test_indices], y[self.test_indices]
        else:
            raise ValueError(f'the sample type {sample_type} is not supported')

    def _hyper_optimize_cv(
            self,
            hypers_dict: dict,
            feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
            sample_type: Literal['train', 'test', 'all'] = 'train',
            cv=None
    ):
        """ Cross validation for measure the model performance and optimize hyperparameters """
        X, y = self._get_dataset(feature_type, sample_type)
        return cross_val_score(self.estimator.__class__(**hypers_dict), X, y, cv=cv).mean()

    def cross_validation(
            self,
            feature_type: Literal['essential', 'reduced', 'scaled'] = 'essential',
            sample_type: Literal['train', 'test', 'all'] = 'train'
    ):
        """"""
        X, y = self._get_dataset(feature_type, sample_type)
        self.valid_score, self.valid_pred, self.valid_true = cross_valid(
            self.estimator,
            X, y,
            cv=KFold(n_splits=5, shuffle=True)
        )

        print(f'The performance in 5fold cross validation:')
        print(f'\t{self.valid_score}')

        r2_plot = R2Regression([self.valid_true, self.valid_pred], to_sparse=True)
        sciplot = SciPlotter(r2_plot)

        fig, axs = sciplot()
        fig.savefig(self.picture_dir.joinpath('cross_valid.png'))

    def train_model(self):
        """"""
        self.estimator.fit(self.X_train, self.y_train)

        pred_train = self.estimator.predict(self.X_train)
        pred_test = self.estimator.predict(self.X_test)

        r2_train = r2_score(self.y_train, pred_train)
        mae_train = mean_absolute_error(self.y_train, pred_train)
        rmse_train = mean_squared_error(self.y_train, pred_train)

        r2_test = r2_score(self.y_test, pred_test)
        mae_test = mean_absolute_error(self.y_test, pred_test)
        rmse_test = np.sqrt(mean_squared_error(self.y_test, pred_test))

        print(f'trained model performance:\n'
              f'\ttrain set:\tR^2={r2_train}\tMAE={mae_train}\tRMSE={rmse_train}\n'
              f'\ttest set:\tR^2={r2_test}\tMAE={mae_test}\tRMSE={rmse_test}\n')

        # Make pictures
        xy1_highlight_indices = np.where(np.isin(self.train_indices, self.kwargs.get('highlight_sample_indices')))[0]
        xy2_highlight_indices = np.where(np.isin(self.test_indices, self.kwargs.get('highlight_sample_indices')))[0]

        plotter = SciPlotter(R2Regression(
            [self.y_train, pred_train],
            [self.y_test, pred_test],
            show_mae=True,
            show_rmse=True,
            xy1_highlight_indices=xy1_highlight_indices,
            xy2_highlight_indices=xy2_highlight_indices,
        ))
        fig, axs = plotter()
        fig.savefig(self.picture_dir.joinpath('train_pred.png'))

        # Save true-pred sheet
        df_train = pd.DataFrame([self.y_train, pred_train], index=['true', 'pred'], columns=self.train_indices).T
        df_test = pd.DataFrame([self.y_test, pred_test], index=['true', 'pred'], columns=self.test_indices).T

        with pd.ExcelWriter(self.sheet_dir.joinpath('true_pred_data.xlsx')) as writer:
            df_train.to_excel(writer, sheet_name='train')
            df_test.to_excel(writer, sheet_name='test')

            if len(indices := self.kwargs.get('highlight_sample_indices', [])):
                pd.Series(indices).to_excel(writer, sheet_name='highlight_samples')

        with open(self.sheet_dir.joinpath('essential_features.csv'), 'w') as writer:
            writer.write('\n'.join(self.features))

    # @staticmethod
    # def save_data_to_excel_(
    #         data: Union[pd.DataFrame, np.ndarray],
    #         test_indices: Union[Sequence[int], np.ndarray],
    #         highlight_indices: Union[Sequence[int], np.ndarray] = None,
    #         columns: Sequence = None
    # ):
    #     """"""
    #     if isinstance(data, np.ndarray):
    #         if isinstance(columns, Sequence):
    #             data = pd.DataFrame(data, columns=list(columns))
    #         else:
    #             data = pd.DataFrame(data, columns=[f'f{i}' for i in range(data.shape[1])])
    #
    #     elif isinstance(data, pd.DataFrame):
    #         if isinstance(columns, Sequence):
    #             data.columns = columns
    #
    #     else:
    #         raise TypeError('the data should be a pandas dataframe or a numpy array!')
    #
    #     if highlight_indices is not None:
    #         if len(np.intersect1d(test_indices, highlight_indices)):
    #             raise IndexError('')

    @staticmethod
    def partial_dependence_(
            estimator: ModelLike,
            features: Sequence[str],
            X: np.ndarray,
            feature_indices: Sequence[int],
            target_name: str = "Target",
            figsave_path: Union[str, os.PathLike]=None,
            **kwargs
    ):
        """"""
        def _draw_partial_dependence(ax: plt.Axes, sciplotter: SciPlotter = None, *args, **kw):
            nonlocal display
            display = PartialDependenceDisplay.from_estimator(estimator, X, feature_indices, ax=ax, kind=kind,)

            ax = plt.gca()
            ax.set_xlabel(x_label, fontname='Arial', fontweight='bold', fontsize=22)
            ax.set_ylabel(y_label, fontname='Arial', fontweight='bold', fontsize=22)

            sciplotter.set_ticks(ax)

        # Determine the xy labels in partial dependence plot
        x_label = features[feature_indices[0]]
        try:
            y_label = features[feature_indices[1]]
        except IndexError:
            y_label = target_name

        if not 1 <= len(feature_indices) <= 2:
            raise ValueError("the number of features index must be 1 or 2")
        elif len(feature_indices) == 2:
            kind = 'average'
            feature_indices = [feature_indices]
        else:
            kind = 'individual'

        # Make plots
        display: Union[sklearn.inspection.PartialDependenceDisplay, None] = None
        plotter = SciPlotter(_draw_partial_dependence, **kwargs)
        fig, ax = plotter()

        if figsave_path:
            fig.savefig(figsave_path)

        return fig, ax, display

    def partial_dependence(
            self,
            estimator: ModelLike = None,
            feature_types: Literal['original', 'non_trivial', 'reduced', 'essential'] = None,
            recover_to_original_scale: bool = False,
            analyze_in_original_data: bool = False,
            **kwargs
    ):
        """"""
        if not estimator:
            estimator = self.estimator

        if isinstance(feature_types, str):
            feature_index = self._feature_types.index(feature_types)
            features = self._feature_names[feature_index]
            X = self._XS[feature_index]
        else:
            features = self.features
            X = self.X

        if not analyze_in_original_data:
            X = self.generate_test_X_(X, **kwargs)

        if recover_to_original_scale:
            invert_X, invert_features = self.invert_scaling_X(X, features)

        # Make plots
        pda_dir = self.picture_dir.joinpath('pda')

        # Analyze single variables
        single_pda_dir = pda_dir.joinpath('single')
        if not single_pda_dir.exists():
            single_pda_dir.mkdir(parents=True)
        for i in range(len(features)):
            fig, ax, display = self.partial_dependence_(
                estimator, features, X,
                feature_indices=(i,),
                target_name=self.target
            )

            if recover_to_original_scale:
                fi_name = features[i]
                inv_fi_idx = invert_features.index(fi_name)
                inv_X = invert_X[:, inv_fi_idx].reshape(-1, 1)
                min_inv_X, max_inv_X = inv_X.min(axis=0), inv_X.max(axis=0)

                scale_axes(ax[0][0], xaxis_range=(min_inv_X, max_inv_X))

            fig.savefig(single_pda_dir.joinpath(f"{features[i].replace('/', '_')}.png"))

        # Analyze pair variables
        pair_pda_dir = pda_dir.joinpath('pair')
        if not pair_pda_dir.exists():
            pair_pda_dir.mkdir()
        for i, j in combinations(range(len(features)), 2):
            fig, ax, display = self.partial_dependence_(
                estimator, features, X,
                feature_indices=(i, j),
                target_name=self.target
            )

            if recover_to_original_scale:
                fi_name, fj_name = features[i], features[j]
                inv_fi_idx, inv_fj_idx = invert_features.index(fi_name), invert_features.index(fj_name)
                inv_X = invert_X[:, [inv_fi_idx, inv_fj_idx]]
                min_inv_X, max_inv_X = inv_X.min(axis=0), inv_X.max(axis=0)

                scale_axes(ax[0][0], xaxis_range=(min_inv_X[0], max_inv_X[0]), yaxis_range=(min_inv_X[1], max_inv_X[1]))

            fig.savefig(pair_pda_dir.joinpath(f"{features[i]}_{features[j]}.png".replace('/', '_')))

    @staticmethod
    def train_surrogate_tree_(
            teacher_model: ModelLike,
            X: np.ndarray,
            feature_names,
            surrogate_model: ModelLike = DecisionTreeRegressor(),
            validate_surrogate: bool = True,
            generate_train_data: bool = False,
            generate_valid_data: bool = True,
            generator_kwargs: dict = None,
            tree_plot_kwargs: dict = None
    ):
        """
        Trains a surrogate decision tree model to approximate the predictions of a teacher model.

        Args:
            teacher_model (ModelLike): The pre-trained model whose behavior to mimic.
            X (np.ndarray): Input features for training or as a template for data generation.
            feature_names (list): Names of the features used for plotting the decision tree.
            surrogate_model (ModelLike, optional): The surrogate model to train. Defaults to `DecisionTreeRegressor()`.
            validate_surrogate (bool, optional): If `True`, validates the surrogate model on a validation set. Defaults to `True`.
            generate_train_data (bool, optional): If `True`, generates new training data using `generator_kwargs`. Defaults to `False`.
            generate_valid_data (bool, optional): If `True`, generates new validation data using `generator_kwargs`. Defaults to `True`.
            generator_kwargs (dict, optional): Arguments for the data generation function. Defaults to `None`.
            tree_plot_kwargs (dict, optional): Additional arguments for plotting the decision tree. Defaults to `None`.

        Returns:
            surrogate_model (ModelLike): The trained surrogate model.
            fig (matplotlib.figure.Figure): The figure object of the plotted decision tree.
            ax (matplotlib.axes.Axes): The axes object of the plot.
            tree: The decision tree plot object.

        """
        if not generator_kwargs:
            generator_kwargs = {'template_X': X}
        else:
            generator_kwargs['template_X'] = X

        if generate_train_data:
            X_train = MachineLearning_.generate_test_X_(**generator_kwargs)
        else:
            X_train = X

        # Train surrogate
        check_is_fitted(teacher_model)
        teacher_pred = teacher_model.predict(X_train)
        surrogate_model.fit(X_train, teacher_pred)

        if validate_surrogate:
            if generate_valid_data:
                X_valid = MachineLearning_.generate_test_X_(**generator_kwargs)
            else:
                X_valid = X

            teacher_pred_val = teacher_model.predict(X_valid)
            surrogate_pred_val = surrogate_model.predict(X_valid)
            val_score = r2_score(teacher_pred_val, surrogate_pred_val)
            print("Surrogate model validation R^2 score:", val_score)

            # Make tree plot
        tree_plot_kwargs = tree_plot_kwargs or {}
        fig, ax = plt.subplots()
        tree = plot_tree(surrogate_model, feature_names=feature_names, ax=ax, **tree_plot_kwargs)

        return surrogate_model, fig, ax, tree

    def train_surrogate_tree(
            self,
            teacher_model: ModelLike = None,
            X: np.ndarray = None,
            feature_names: Sequence[str] = None,
            surrogate_model: ModelLike = DecisionTreeRegressor(),
            validate_surrogate: bool = True,
            generate_train_data: bool = False,
            generate_valid_data: bool = True,
            generator_kwargs: dict = None,
            tree_plot_kwargs: dict = None
    ):
        """
        Args:
            teacher_model (ModelLike): The pre-trained model whose behavior to mimic.
            X (np.ndarray): Input features for training or as a template for data generation.
            feature_names (list): Names of the features used for plotting the decision tree.
            surrogate_model (ModelLike, optional): The surrogate model to train. Defaults to `DecisionTreeRegressor()`.
            validate_surrogate (bool, optional): If `True`, validates the surrogate model on a validation set. Defaults to `True`.
            generate_train_data (bool, optional): If `True`, generates new training data using `generator_kwargs`. Defaults to `False`.
            generate_valid_data (bool, optional): If `True`, generates new validation data using `generator_kwargs`. Defaults to `True`.
            generator_kwargs (dict, optional): Arguments for the data generation function. Defaults to `None`.
            tree_plot_kwargs (dict, optional): Additional arguments for plotting the decision tree. Defaults to `None`.

        Returns:
            surrogate_model (ModelLike): The trained surrogate model.
            fig (matplotlib.figure.Figure): The figure object of the plotted decision tree.
            ax (matplotlib.axes.Axes): The axes object of the plot.
            tree: The decision tree plot object.
        """
        arguments = locals().copy()
        arguments.pop('self')

        arguments['teacher_model'] = teacher_model or self.estimator
        arguments['X'] = X or self.X
        arguments['feature_names'] = feature_names or self.features

        surrogate, fig, ax, tree = self.train_surrogate_tree_(**arguments)

        fig.savefig(self.picture_dir.joinpath('surrogate_tree.png'))

        return surrogate, fig, ax, tree

    @staticmethod
    def generate_test_X_(
            template_X,
            independent: bool = False,
            norm_uniform: Literal['norm', 'uniform'] = 'uniform',
            min_offset: Union[float, np.ndarray] = 0.,
            max_offset: Union[float, np.ndarray] = 0.,
            X_scale: Union[float, np.ndarray] = None,
            sample_num: int = 1000,
            seed_: int = None
    ):
        """
        Generates a hypothetical test X with similar covariance of features as template X.
        Args:
            template_X: template X to define covariance (matrix).
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

            if np.any(X_max < X_min):
                error_dim = np.nonzero(np.int_(X_max <= X_min))[0].tolist()
                raise ValueError(f'the maximum value is smaller than the minimum values in X dimensions of {error_dim}')

            # TODO: requires modify
            const_dim = np.where(X_min == X_max)[0]
            for dim in const_dim:
                X_min[dim] -= 1e-6
                X_max[dim] += 1e-6

            gen_X = np.random.uniform(0, 1, size=(sample_num, template_X.shape[1]))

            if not independent:
                gen_X = set_correlation(gen_X)

            for i, (x_min, x_max) in enumerate(zip(X_min, X_max)):
                gen_X[:, i] = minmax_scale(gen_X[:, i], (x_min, x_max))
        else:
            raise ValueError('Unrecognized value for norm_uniform argument, use "norm" or "uniform"')

        return gen_X

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
        Generates a hypothetical test X with similar covariance of features as template X.
        Args:
            template_X: template X to define covariance (matrix). if not given, Essential_sX will be applied.
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
        kwargs = locals().copy()
        kwargs.pop('self')
        if template_X is not None:
            kwargs['template_X'] = template_X
        else:
            kwargs['template_X'] = self.X

        return self.generate_test_X_(**kwargs)

    @staticmethod
    def shap_analysis_(
            estimator, X, y,
            feature_names: Union[Sequence, np.ndarray] = None,
            gen_X_train: bool = False,
            sample_size: int = 1000,
            X_test: np.ndarray = None,
            test_size: int = 1000,
            explainer_cls: type = shap.TreeExplainer,
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

        if MachineLearning_.has_fitted(estimator):
            estimator.fit(X, y)

        if gen_X_train:
            X_train = MachineLearning_.generate_test_X_(X, sample_num=sample_size, **kwargs)
        else:
            X_train = X

        if len(X_train) <= sample_size:
            explainer = explainer_cls(estimator, X_train)
        else:
            sample_X = shap.sample(X_train, sample_size)
            explainer = explainer_cls(estimator, sample_X)

        if X_test is None:
            X_test = MachineLearning_.generate_test_X_(
                X, sample_num=test_size, **kwargs
            )

        shap_values = explainer(X_test)
        if isinstance(feature_names, (list, np.ndarray)):
            assert len(feature_names) == X.shape[1]
            shap_values.feature_names = feature_names

        # Save result, base, data value and SHAP value
        if shap_values_save_path:
            series_base = pd.Series(shap_values.base_values, name='base_values')
            df_shap_value = pd.DataFrame(shap_values.values, columns=feature_names)
            df_data_value = pd.DataFrame(shap_values.data, columns=feature_names)

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
        self.explainer, self.shap_value = self.shap_analysis_(
            self.estimator, self.X, self.y, self.features,
            # explainer_cls=self.kwargs.get('shap_explainer_cls'),
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


def linear_leave_one_out_analysis(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    loo = LeaveOneOut()
    coef, intercept = [], []
    for train_index, test_index in loo.split(X, y):
        X_train, y_train = X[train_index], y[train_index]

        lin_model = sklearn.linear_model.LinearRegression()
        lin_model.fit(X_train, y_train)

        coef.append(lin_model.coef_)
        intercept.append(lin_model.intercept_)

    return np.array(coef), np.array(intercept)



cross_valid = _cross_val
if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target

    # cvs, cvp = cross_valid(GradientBoostingRegressor(), X, y, cv=KFold(shuffle=True))
