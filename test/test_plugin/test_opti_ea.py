"""
python v3.9.0
@Project: hotpot
@File   : test_opti_ea
@Auther : Zhiyuan Zhang
@Data   : 2024/8/10
@Time   : 17:13
"""
from pathlib import Path
import os, sys, logging
import unittest as ut

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from hotpot.dataset import load_dataset
import hotpot.plugins.opti.ev.core as ev

logging.basicConfig(level=logging.INFO)


class TestOptiEa(ut.TestCase):

    def test_sklearn_optimizer(self):
        print(sys.platform)
        print(os.getcwd())

        data = load_dataset("filled_electric")

        data = data.drop(
            ['Volume_ratio_Ar', '_Dielectric_constant_imag', '_Dielectric_loss_Angle_tangent_value'],
            axis=1
        )
        y = data['_Dielectric_constant_real']
        X = data.drop(['_Dielectric_constant_real'], axis=1)

        spaces = {
            'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
            'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            'n_estimators': [25, 50, 75, 100, 125, 150, 200, 300, 400, 500],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'criterion': ['friedman_mse', 'squared_error'],
            'min_samples_split': [2, 3, 4, 5, 8, 10],
            'min_samples_leaf': [1, 2, 3, 5],
            'max_depth': [None, 2, 3, 5, 7, 10]
        }

        optimizer = ev.MLOptimizer(
            model=GradientBoostingRegressor,
            X_train=X, y_train=y,
            params_space=spaces,
            init_params=5,
            living_space=20,
            survival_rate=0.6
        )
        optimizer.optimize()

    def test_imp(self):
        data = load_dataset("filled_electric")

        data = data.drop(
            ['Volume_ratio_Ar', '_Dielectric_constant_imag', '_Dielectric_loss_Angle_tangent_value'],
            axis=1
        )
        y = data['_Dielectric_constant_real']
        X = data.drop(['_Dielectric_constant_real'], axis=1)

        estimator = GradientBoostingRegressor()
        estimator.fit(X, y)
        print(estimator.feature_importances_)

    def test_machine_learning_ea_with_feature_engineering(self):
        data_path = Path(__file__).parents[2].joinpath('hotpot/dataset/ChemData/examples/logβ.xlsx')

        with pd.ExcelFile(data_path) as excel:
            data = excel.parse(sheet_name='ChemData', index_col=0)
            test_indices = excel.parse(sheet_name='test_indices', index_col=0)

        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        features = data.columns.tolist()[:-1]

        # Species (algorithm)
        models = {
            'GBDT': GradientBoostingRegressor,
            'RF': RandomForestRegressor,
            'KRR': KernelRidge,
            'RR': Ridge,
            'XGBoost': XGBRegressor,
        }

        # Gene Space(hyperparameters)
        hypers = {
            'GBDT': {
                'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                'n_estimators': [25, 50, 75, 100, 125, 150, 200, 300, 400, 500],
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'criterion': ['friedman_mse', 'squared_error'],
                'min_samples_split': [2, 3, 4, 5, 8, 10],
                'min_samples_leaf': [1, 2, 3, 5],
                'max_depth': [None, 2, 3, 5, 7, 10, 16]
            },
            'RF': {
                'n_estimators': [25, 50, 75, 100, 125, 150, 200, 300, 400, 500],
                'criterion': ['friedman_mse', 'squared_error', 'absolute_error'],
                'max_depth': [None, 2, 3, 5, 7, 10, 16, 32],
                'min_samples_split': [2, 3, 4, 5, 8, 10],
                'min_samples_leaf': [1, 2, 3, 5],
                'max_features': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            },
            'KRR': {
                'alpha': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid', 'laplacian']
            },
            'RR': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.],
                'tol': [1e-5, 5e-5, 1e-4, 5e-3]
            },
            'XGBoost': {
                'n_estimators': [25, 50, 75, 100, 125, 150, 200, 300, 400, 500],
                'max_depth': [None, 2, 3, 5, 7, 10, 16, 32],
                'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                'booster': ['gbtree', 'gblinear', 'dart'],
                'gamma': [0., 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            }
        }

        print(os.cpu_count())
        results = ev.ml_optimizer(
            models=models,
            hypers=hypers,
            X=X,
            y=y,
            features=features,
            feature_engineering=True,
            which_return='top',
            nproc=16,
            n_iters=3,
            timeout=3600
        )

        print(results)
