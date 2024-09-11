"""
python v3.9.0
@Project: hotpot
@File   : test_opti_ea
@Auther : Zhiyuan Zhang
@Data   : 2024/8/10
@Time   : 17:13
"""
import os, sys, logging
import unittest as ut

from sklearn.ensemble import GradientBoostingRegressor

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

        optimizer = ev.SklearnOptimizer(
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
