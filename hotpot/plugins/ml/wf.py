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

from sklearn.base import clone, BaseEstimator
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
from xgboost import XGBRegressor

import shap
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors
from xgboost.spark import estimator

from hotpot.plots import SciPlotter, R2Regression, PearsonMatrix, HierarchicalTree, FeatureImportance, SHAPlot, Pearson


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


def _cross_val(estimator, X, y, cv=None, *args, **kwargs):
    from sklearn.model_selection import cross_val_score as cv_score, cross_val_predict as cv_pred
    if not cv:
        return (
            cv_score(estimator, X, y, cv=KFold(shuffle=True), *args, **kwargs),
            cv_pred(estimator, X, y, cv=KFold(shuffle=True), *args, **kwargs), y
        )

    else:
        estimator.fit(X, y)

        valid_score, valid_pred, valid_true = [], [], []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clone(estimator).fit(X_train, y_train)
            score = estimator.score(X_test, y_test)
            pred = estimator.predict(X_test)

            valid_score.append(score)
            valid_pred.append(pred)
            valid_true.append(y_test)

        return np.array(valid_score), np.hstack(valid_pred), np.hstack(valid_true)


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

        r2_plot = R2Regression([self.valid_true, self.valid_pred])
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


cross_valid = _cross_val
if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target

    # cvs, cvp = cross_valid(GradientBoostingRegressor(), X, y, cv=KFold(shuffle=True))