"""
python v3.9.0
@Project: hotpot
@File   : func
@Auther : Zhiyuan Zhang
@Data   : 2024/7/30
@Time   : 20:35
"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


class LivingFunc:

    @staticmethod
    def top_living(earth):
        individuals = earth.sorted_individuals()
        if earth.living_rate <= 0:
            raise ValueError('the living space should be greater than 0')

        if 0 < earth.living_rate < 1:
            death_number = int((1 - earth.living_rate) * len(individuals))
        else:
            death_number = len(individuals) - int(earth.living_rate)

        to_die = individuals[:death_number]

        return to_die

    @staticmethod
    def healthy_living(earth):
        to_die = []
        for individual in earth.individuals:
            if individual.fitness < earth.fitness_threshold:
                to_die.append(individual)

        return to_die


class MachineLearningR2Fitness:
    def __init__(self, X, y, algorithm: type):
        self.X, self.y = X, y
        self.algorithm = algorithm

    def __call__(self, individual):
        gene_dict = individual.gene_series.series_dict
        model = self.algorithm(**gene_dict)

        r2 = cross_val_score(model, X=self.X, y=self.y, n_jobs=-1, scoring=r2_score)

        return r2

