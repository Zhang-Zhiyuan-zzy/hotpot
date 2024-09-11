"""
python v3.9.0
@Project: hp5
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2024/7/29
@Time   : 10:21
"""
import time
import logging
import os
from abc import abstractmethod
import random
from typing import *
from itertools import combinations
from functools import wraps

import multiprocessing as mp

from tqdm import tqdm
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor

from .func import LivingFunc as LFunc

class AllSpeciesMeta(type):

    _registered_methods = {}

    def __new__(mcs, name, bases, namespace):
        for registered_name, method in mcs._registered_methods.items():
            namespace[registered_name] = mcs.bundle_operate(method)

        return super().__new__(mcs, name, bases, namespace)

    @classmethod
    def register(mcs, method):
        mcs._registered_methods[method.__name__] = method
        return method

    @staticmethod
    def bundle_operate(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):

            results = {}
            for species in self.species:
                res = method(species, *args, **kwargs)
                if res is not None:
                    results[species.name] = res

            if results:
                return results

        return wrapper


def get_params(param_name: str, *objs):
    for obj in objs:
        if param_value := getattr(obj, param_name, None):
            return param_value

    return getattr(UniversalParams, param_name)


class UniversalParams:
    """ A collection of parameters for the created world """
    mutation_rate: float = 0.1  # mutation rate for every genes,
    num_children: int = 2
    fitness_function: Callable = None


class GeneLibrary:
    """ This class is a singleton class """
    _instance = None
    _library = []
    _library_dict = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GeneLibrary, cls).__new__(cls)
            return cls._instance

        else:
            return cls._instance

    @property
    def library(self):
        return self._library

    @property
    def library_dict(self):
        return self._library_dict

    def append_gene_space(self, gene_space):
        self._library.append(gene_space)

    def clear(self):
        self._library = []
        self._library_dict = {}


gene_library = GeneLibrary()


class Species:
    """ Species is a collection of Individual who do not have reproductive
    isolation and can mate and reproduce with each other """
    def __init__(
            self, name: str,
            *gene_spaces: 'GeneSpace',
            mutation_rate: float = None,
            lifetime: Union[int, Callable] = None,
            fitness_function: Callable = None,
    ):
        self.name = name
        self.gene_spaces = gene_spaces
        self._lifetime = lifetime
        self.fitness_function = fitness_function

        self.individuals = []

        self.mutation_rate = None
        if isinstance(mutation_rate, float):
            if 0.0 <= mutation_rate <= 1.0:
                self.mutation_rate = mutation_rate
            else:
                raise ValueError('the mutation rate should be between 0.0 and 1.0')

        # assign species the gene belong to
        for gene_space in self.gene_spaces:
            gene_space.species = name

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    @property
    def lifetime(self) -> int:
        if isinstance(self._lifetime, int):
            return self._lifetime
        elif isinstance(self._lifetime, Callable):
            return self._lifetime()

    @AllSpeciesMeta.register
    def create_individual(self, **kwargs) -> "Individual":
        if not kwargs:
            individual = Individual(self, self.get_gene_series())
        else:
            if len(kwargs) != len(self.gene_spaces):
                raise ValueError('the given gene {name, value} pairs should have the same length as the gene space')

            list_gene = []
            for gene_name, gene_value in kwargs.items():
                if not self.is_fitted_gene(gene_name, gene_value):
                    raise ValueError(f'the given gene_name {gene_name} or value {gene_value} '
                                     f'is not fitted to the gene space of this species')
                gene = Gene(self.get_gene_space(gene_name))
                gene.value = gene_value
                list_gene.append(gene)

            individual = Individual(self, GeneSeries(*list_gene))

        return individual

    def get_gene_space(self, name: str) -> "GeneSpace":
        for gene_space in self.gene_spaces:
            if gene_space.name == name:
                return gene_space

        raise NameError('the given gene space name does not exist')

    @AllSpeciesMeta.register
    def create_individuals(self, nums: int = 2):
        return [self.create_individual() for _ in range(nums)]

    def get_gene_series(self) -> 'GeneSeries':
        return GeneSeries(*[gene_space.generate_gene() for gene_space in self.gene_spaces])

    @overload
    def is_fitted_gene(self, gene: "Gene") -> bool:
        """ Check if a gene is fitted to gene space of this species """

    @overload
    def is_fitted_gene(self, name: str, value) -> bool:
        """ Check if a gene is fitted to gene space of this species """

    def is_fitted_gene(self, *args, **kwargs) -> bool:
        """ Check if a gene is fitted to gene space of this species """
        if len(*args) == 1 and isinstance(args[0], Gene):
            gene = args[0]
            return any(gene.gene_space is gs and gene.value in gs for gs in self.gene_spaces)

        elif len(args) == 2 and isinstance(args[0], str):
            gene_name, gene_value = args
            return any(gene_name == gs.name and gene_value in gs for gs in self.gene_spaces)

        return False

    def add_individual(self, individual: 'Individual'):
        # Check whether its gene are legal
        if len(individual.gene_series) != len(self.gene_spaces):
            raise ValueError(
                'the length of gene_series of individual is not equal to the length of gene_space of this Species'
            )

        indiv_gene_dict = individual.gene_dict
        for gene_space in self.gene_spaces:
            if gene_space.name not in indiv_gene_dict:
                raise ValueError('An unexpected gene name is checked')
            elif indiv_gene_dict[gene_space.name] not in gene_space:
                raise ValueError('the gene value of individual is out of the range of the gene space')

        self.individuals.append(individual)

    @AllSpeciesMeta.register
    def reproduce(self):
        """ population of the species reproduce """
        new_generations = []
        for female, male in combinations(self.individuals, 2):
            children = female.mate(male)
            new_generations.extend(children)

        return new_generations


class Individual:
    """ Individual is determined by a set of genes and is living in a certain environment """
    class FitnessProcess:
        def __init__(self, individual, func, timeout=300):
            self.individual = individual
            self.func = func
            self.timeout = timeout
            self.fitness = None
            self.process = None
            self.queue = None

            self._start_time = None
        def start(self):
            """"""
            self.queue = mp.Queue()
            self.process = mp.Process(target=self._run_fitness_function, args=(self.queue,))
            self.process.start()

            self._start_time = time.time()

        def _run_fitness_function(self, queue):
            try:
                queue.put(self.func(self.individual))
            except Exception as e:
                print(e)
                queue.put(-8888888.)

        @property
        def is_timeout(self):
            return self.run_time > self.timeout

        @property
        def is_alive(self):
            return self.process.is_alive()

        @property
        def run_time(self):
            return time.time() - self._start_time

        def terminate(self):
            if not self.is_alive:
                self.process.terminate()
                # self.process.close()
                self.fitness = self.queue.get(10)
                self.process = None
                self.queue = None

            elif self.is_timeout:
                self.process.terminate()

                while self.is_alive:
                    pass

                self.process.close()
                self.fitness = -9999999
                self.process = None
                self.queue = None

            self.individual.fitness_ = self.fitness

            if self.fitness is not None:
                return True
            return False

        def force_terminate(self):
            if self.is_alive:
                self.fitness = self.queue.get()
            else:
                self.fitness = -8888888

            self.process.terminate()
            self.process.join()
            self.queue = None
            self.fitness = None
            self.individual.fitness_ = self.fitness

    def __init__(
            self,
            species: Species,
            gene_series: 'GeneSeries',
            fitness_function: Callable = None,
            mutation_rate: float = None
    ):
        self.species = species
        self.gene_series = gene_series
        self.fitness_ = None
        self.fitness_function = fitness_function
        self.fitness_process: Union[None, Individual.FitnessProcess] = None
        self.age = 0

        self.mutation_rate = get_params('mutation_rate', self.gene_series)
        if isinstance(mutation_rate, float):
            if 0.0 <= mutation_rate <= 1.0:
                self.mutation_rate = mutation_rate
            else:
                raise ValueError('the mutation rate should be between 0.0 and 1.0')

        # Add self into own species
        self.species.add_individual(self)

    def __repr__(self):
        return self.species_name

    @property
    def mean_lifetime(self):
        return self.species.lifetime

    @property
    def gene_dict(self) -> dict:
        return self.gene_series.series_dict

    @property
    def species_name(self) -> str:
        return self.species.name

    def gene_mutation_rate(self, gene: 'Gene') -> float:
        return get_params('mutation_rate', gene, self, self.species)

    def gene_mutate(self, temp_mutation_rate: float = None):
        mutation_rate = temp_mutation_rate or self.mutation_rate
        logging.debug(f'mutation rate: {mutation_rate}')

        for gene in self.gene_series:
            start_value = gene.value
            if random.uniform(0, 1) < mutation_rate:
                end_value = gene.mutate()
                logging.info(f"{self.species_name} {gene} gene mutate from {start_value} to {end_value}")

    @property
    def lifetime(self) -> int:
        return self.species.lifetime

    def mate(self, other: 'Individual'):
        if self.species_name != other.species_name:
            raise AttributeError('mating species should belong to the same species')
        if self.gene_series != other.gene_series:
            raise AttributeError('mating species should have identical gene series')

        num_children = get_params('num_children', self, other, self.species)

        children = []
        for i in range(num_children):
            reproduce_series = GeneSeries(
                *np.where(
                    np.random.randint(0, 2, len(self.gene_series)),
                    np.array(self.gene_series.series),
                    np.array(other.gene_series.series)
                ).tolist()
            )

            child = Individual(self.species, reproduce_series, mutation_rate=getattr(self, 'mutation_rate', None))
            child.gene_mutate()

            children.append(child)

        return children

    def get_fitness_function(self) -> Callable:
        return get_params('fitness_function', self, self.species)

    def calc_fitness(self, timeout=300):
        """ Calculates the fitness of the individuals """

        fitness_function = self.get_fitness_function()

        logging.debug(f"Calculating fitness for {self} ... by function {fitness_function}")

        if not isinstance(fitness_function, Callable):
            raise AttributeError('fitness function is not given to the individuals')

        self.fitness_process = self.FitnessProcess(self, fitness_function, timeout)
        self.fitness_process.start()

    def die(self):
        self.species.individuals.remove(self)
        return self


class Gene:
    """ Gene is a variable under a specified variable space.
    Gene can self mutate into other values in the variables space or
    be passed on to the next generation """
    def __init__(
            self,
            gene_space: "GeneSpace",
            mutation_rate: float = None,
            value=None
    ):
        self.gene_space = gene_space
        self.value = value

        self.mutation_rate = None
        if isinstance(mutation_rate, float):
            if 0.0 <= mutation_rate <= 1.0:
                self.mutation_rate = mutation_rate
            else:
                raise ValueError('the mutation rate should be between 0.0 and 1.0')

    def __gt__(self, other):
        return self.name >= other.name

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    @property
    def name(self) -> str:
        return self.gene_space.name

    @property
    def species(self) -> str:
        return self.gene_space.species

    def mutate(self):
        self.value = self.gene_space.get_random_gene()
        return self.value


class GeneSeries:
    """ A series of genes """
    def __init__(self, *gene):
        self._genes = list(gene)

    def __repr__(self):
        return ' --> '.join(f"{gene.name}({gene.value})" for gene in self.series)

    def __iter__(self):
        return iter(self.series)

    def __eq__(self, other):
        return all(sg.name == og.name for sg, og in zip(self.series, other.series))

    def __len__(self):
        return len(self._genes)

    @property
    def series(self) -> list['Gene']:
        return [gen for gen in sorted(self._genes)]

    @property
    def series_dict(self) -> dict:
        return {gen.name: gen.value for gen in self._genes}


class GeneSpace:
    """ GeneSpace is a variable space of Gene """
    def __new__(cls, *args, **kwargs):
        obj = super(GeneSpace, cls).__new__(cls)
        gene_library.append_gene_space(obj)

        return obj

    def __init__(self, name: str, species: str = None):
        self.name = name
        self.species = species

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def generate_gene(self):
        gene = Gene(self)
        gene.mutate()

        return gene

    @abstractmethod
    def get_random_gene(self):
        raise NotImplementedError


class DiscreteGenSpace(GeneSpace):
    """ a specified GeneSpace contains a list of discrete variable """
    def __init__(self, name: str, gen_vals: Sequence):
        super().__init__(name)
        self.gen_vals = gen_vals

    def __contains__(self, item):
        return item in self.gen_vals

    def get_random_gene(self):
        return random.choice(self.gen_vals)


class ContinuousGenSpace(GeneSpace):
    """ a specified GeneSpace contains continuous variable (Real numbers) in a range """
    def __init__(self, name: str, min_val: int, max_val: int):
        super().__init__(name)
        self.min_val = min_val
        self.max_val = max_val

    def __contains__(self, item):
        return self.min_val <= item <= self.max_val

    def get_random_gene(self):
        return random.uniform(self.min_val, self.max_val)


class AllSpecies(metaclass=AllSpeciesMeta):
    """ A collection of all species in Earth """

    def __init__(self):
        self.species = []

    def __iter__(self):
        return iter(self.species)

    def __repr__(self):
        return f"AllSpecies({len(self)})"

    def __len__(self):
        return len(self.species)

    def __contains__(self, item):
        return item in self.species

    def __getitem__(self, item: int) -> Species:
        return self.species[item]

    def add(self, species: Species):
        self.species.append(species)

    def remove(self, species: Species):
        self.species.remove(species)


class Earth:
    """ Earth is an environment that all Individuals and Species survive, compete, and reproduce """
    def __init__(
            self,
            survive_func,
            living_rate: Union[int, float] = 0.25,
            living_threshold: float = 0.8,
            max_living_space: int = 100,
            **kwargs
    ):
        self.all_species = AllSpecies()

        self.survive_func = survive_func
        self.living_rate = living_rate
        self.living_threshold = living_threshold
        self.max_living_space = max_living_space
        self.time = 0

        # other parameters
        self.env_func = kwargs.get('env_func', None)  # func to control environment

    @property
    def living_species(self) -> set[Species]:
        return set(individual.species for individual in self.individuals)

    def is_extinct(self, species: Species) -> bool:
        return species not in self.living_species

    @overload
    def create_species(self, species: Species) -> Species:
        """"""

    @overload
    def create_species(
            self,
            name: str,
            gene_species: GeneSpace,
            mutation_rate: float = None,
            lifetime: Union[int, Callable] = None,
            **kwargs
    ) -> Species:
        """"""

    def create_species(self, *args, **kwargs) -> Species:
        """"""
        if len(args) == 1:
            species = args[0]
            if not isinstance(species, Species):
                raise TypeError(f"Invalid species type: {type(species)}, must be Species")

        else:
            species = Species(*args, **kwargs)

        self.all_species.add(species)

        return species

    def calc_fitness(
            self, nproc: int = None,
            timeout: int = 300,
            not_calc_existed_fitness: bool = False,
            false_to_mutate: bool = False,
    ):
        def _running_directly():
            for individual in self.individuals:
                fit_func = individual.get_fitness_function()
                if not not_calc_existed_fitness or individual.fitness_ is not None:
                    individual.fitness_ = fit_func(individual)
                    logging.info(f'Fitness of {individual} is {individual.fitness_}')

        def _running_multi_processes():
            waiting_individual = self.individuals
            running_process = []
            while waiting_individual or running_process:

                if waiting_individual and len(running_process) < nproc:
                    individual = waiting_individual.pop(0)

                    if not_calc_existed_fitness and individual.fitness_ is not None:
                        continue

                    individual.calc_fitness(timeout)
                    running_process.append(individual)

                terminated_individuals = []
                for running_idd in running_process:
                    if running_idd.fitness_process.terminate():
                        if false_to_mutate and running_idd.fitness_ < 0.:
                            running_idd.gene_mutate(0.8)
                            running_idd.fitness_ = None
                            waiting_individual.append(running_idd)

                            logging.info(f"{running_idd} False to calculate and mutate...")

                        else:
                            logging.debug(f'Fitness of {running_idd} is {running_idd.fitness_}')

                        terminated_individuals.append(running_idd)

                if running_process and not terminated_individuals:
                    time.sleep(10)
                    logging.info(f'{len(waiting_individual)} or {len(running_process)}')

                for terminated_individual in terminated_individuals:
                    running_process.remove(terminated_individual)

        if nproc is None:
            _running_directly()

        elif nproc < 0:
            nproc = os.cpu_count()

        if isinstance(int(nproc), int):
            _running_multi_processes()

        logging.debug(f'Calculation terminated')

    @property
    def individuals(self) -> list[Individual]:
        return [individual for species in self.all_species for individual in species.individuals]

    def sorted_individuals(self, reverse: bool = False) -> list[Individual]:
        individuals = self.individuals
        random.shuffle(individuals)
        return sorted(tqdm(individuals), key=lambda x: x.fitness_, reverse=reverse)

    def progress_time(
            self, nproc=None,
            timeout=300,
            not_calc_existing_fitness: bool = True,
            false_to_mutate: bool = False
    ):
        """
        Time progression over all species.
        Life of all species born, reproduced, fought for living, and to die
        """
        # Exclude older individual:
        for individual in self.individuals:
            if individual.age and individual.mean_lifetime and individual.age > individual.mean_lifetime:
                individual.die()

        # Fighting for living
        self.calc_fitness(
            nproc=nproc,
            timeout=timeout,
            not_calc_existed_fitness=not_calc_existing_fitness,
            false_to_mutate=false_to_mutate
        )
        to_die = self.survive_func(self)
        for dying_individual in to_die:
            dying_individual.die()

        # old individual naturally die
        for individual in self.individuals:
            if individual.lifetime is not None and individual.age > individual.lifetime:
                individual.die()

            individual.age += 1

        individuals = list(self.sorted_individuals(reverse=True))
        for individual in individuals[self.max_living_space:]:
            individual.die()

        # reproducing
        for species in self.all_species:
            species.reproduce()

        self.time += 1


class EvoluteOptimizer:
    def __init__(self, gene_series, pop_size, ):
        ...


class SklearnOptimizer:
    def __init__(
            self,
            model: type,
            X_train: np.ndarray,
            y_train: np.ndarray,
            params_space: dict[str, Union[list, tuple]],
            cv: Union[int, Callable] = 5,
            shuffle: bool = True,
            *,
            mutation_rate: float = 0.2,
            init_params: Union[int, list[dict]] = 3,
            living_space: int = 10,
            survival_func: Callable = None,
            survival_rate: float = 0.4,
            survival_threshold: float = 0.8,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.params_space = params_space
        self.cv = cv
        self.shuffle = shuffle

        self.init_params = init_params
        self.living_space = living_space
        self.mutation_rate = mutation_rate

        self.survival_func = survival_func if isinstance(survival_func, Callable) else LFunc.top_living
        self.survival_rate = survival_rate
        self.survival_threshold = survival_threshold

        self._fitness_individuals = None

    def optimize(self, iter_num: int = 5, nproc: int = -1):
        """"""
        earth = Earth(
            self.survival_func,
            self.survival_rate,
            self.survival_threshold,
            self.living_space
        )

        list_gene_space = []
        for param_name, param_space in self.params_space.items():
            if isinstance(param_space, tuple):
                if len(param_space) != 2:
                    raise ValueError('The continuous parameters space must give its [min_limit, max_limit]')
                if any(not isinstance(v, (float, int)) for v in param_space):
                    raise ValueError('The limit of continuous parameters space must be either an integer or a float')

                list_gene_space.append(ContinuousGenSpace(param_name, param_space[0], param_space[1]))

            elif isinstance(param_space, list):
                list_gene_space.append(DiscreteGenSpace(param_name, param_space))

            else:
                raise TypeError('the value of params spaces must be either a list or a tuple')

        earth.create_species(
            self.model.__name__,
            *list_gene_space,
            mutation_rate=self.mutation_rate,
            lifetime=2,
            fitness_function=self._fitness_function
        )

        if isinstance(self.init_params, int):
            earth.all_species.create_individuals(self.init_params)
        elif isinstance(self.init_params, list) and all(isinstance(p, dict) for p in self.init_params):
            for init_param in self.init_params:
                earth.all_species[0].create_individual(**init_param)
        else:
            raise NotImplementedError

        for n in range(iter_num):
            earth.progress_time(nproc, timeout=3000)

        self._fitness_individuals = [i for i in earth.individuals if isinstance(i.fitness_, float)]

    def _fitness_function(self, individual):
        estimator = self.model(**individual.gene_dict)
        cv = KFold(n_splits=self.cv, shuffle=self.shuffle) if isinstance(self.cv, int) else self.cv
        score = cross_val_score(
            estimator,
            self.X_train,
            self.y_train,
            cv=cv,
        ).mean()

        return score

    @property
    def surviving_individuals(self):
        return sorted(self._fitness_individuals, key=lambda x: x.fitness_)

    @property
    def best_individual(self) -> Individual:
        return self.surviving_individuals[0]

    @property
    def best_params(self) -> dict:
        return self.best_individual.gene_dict

    @property
    def best_fitness(self) -> float:
        return self.best_individual.fitness_

    @property
    def best_estimator(self) -> BaseEstimator:
        model = self.model(**self.best_params)
        model.fit(self.X_train, self.y_train)
        return model


if __name__ == "__main__":
    from hotpot.plugins import load_dataset
    data = load_dataset("filled_electric")

    data = data.drop(['Volume_ratio_Ar', '_Dielectric_constant_imag', '_Dielectric_loss_Angle_tangent_value'])
    y = data['_Dielectric_constant_real']
    X = data.drop(['_Dielectric_constant_real'])

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


    optimizer = SklearnOptimizer(
        model=GradientBoostingRegressor,
        X_train=X, y_train=y,
        params_space=spaces
    )
    optimizer.optimize()
