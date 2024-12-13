"""
python v3.9.0
@Project: hotpot
@File   : test_doc_code
@Auther : Zhiyuan Zhang
@Data   : 2024/9/12
@Time   : 15:14
"""
import logging
import unittest as ut


class TestDocCode(ut.TestCase):
    """ Test example codes given by hotpot documentation """
    def test_evolution_algorithm(self):
        from sklearn.linear_model import Ridge
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from xgboost import XGBRegressor

        from sklearn.model_selection import cross_val_score, KFold

        from hotpot.dataset import load_dataset
        from hotpot.plugins.opti.ev import Earth, Individual

        # define fitness func
        data = load_dataset('logβ1')
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

        def fitness_function(individual: Individual) -> float:
            algor = models[individual.species_name]
            estimator = algor(**individual.gene_dict)

            logging.info(estimator)
            fitness = cross_val_score(estimator, X, y, cv=KFold(shuffle=True), n_jobs=-1).mean()

            return float(fitness)

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

        earth = Earth(
            fitness_function=fitness_function,  # set the fitness_function in global
            living_ratio=0.6,
            keep_ratio=0.2  # keep the genes for top 20% individual
        )

        # Add species to the earth
        for species_name, spaces in hypers.items():
            earth.create_species(species_name, spaces)

        earth.create_individuals(3)

        # progress 5 iteration
        for _ in range(5):
            earth.progress_time()
            for top_individual in earth.sorted_individuals(True)[:5]:
                print(top_individual, top_individual.fitness_)
