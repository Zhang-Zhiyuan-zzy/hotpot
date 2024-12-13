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
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor

from .func import LivingFunc as LFunc
from hotpot.utils.types import ModelLike
from hotpot.plugins.ml.wf import MachineLearning_ as ML


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


class Individual:
    """ Individual is determined by a set of genes and is living in a certain environment """
    class FitnessProcess:
        def __init__(self, individual, func, timeout=None):
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
            if not self.timeout:
                return False

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

                print(RuntimeWarning(f'Calculate fitness timeout in {self.timeout} seconds with fitness {self.fitness}'))

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
            species: "Species",
            gene_series: 'GeneSeries',
            fitness_function: Callable = None,
            mutation_rate: float = 0.5
    ):
        self.species = species
        self.gene_series = gene_series
        self.fitness_ = None
        self.fitness_function = fitness_function
        self.fitness_process: Union[None, Individual.FitnessProcess] = None
        self.age = 0

        if isinstance(mutation_rate, float):
            if 0.0 <= mutation_rate <= 1.0:
                self.mutation_rate = mutation_rate
            else:
                raise ValueError('the mutation rate should be between 0.0 and 1.0')

        # Add self into own species
        self.species.add_individual(self)
        self.results = {}

    def __repr__(self):
        return self.species_name

    @property
    def earth(self):
        return self.species.earth

    @property
    def expected_lifetime(self):
        """ The expected lifetime of the species of the individual. """
        return self.species.expected_lifetime

    @property
    def gene_dict(self) -> dict:
        return self.gene_series.series_dict

    @property
    def species_name(self) -> str:
        return self.species.name

    def _get_mutation_rate(self, gene) -> float:
        return get_params('mutation_rate', gene, self, self.species, self.earth)

    def gene_mutate(self):
        for gene in self.gene_series:
            start_value = gene.value
            if random.uniform(0, 1) < self._get_mutation_rate(gene):
                end_value = gene.mutate()
                logging.info(f"{self.species_name} {gene} gene mutate from {start_value} to {end_value}")

    def mate(self, other: 'Individual', gene_mutate: bool = True):
        if self.species_name != other.species_name:
            raise AttributeError('mating species should belong to the same species')
        if self.gene_series != other.gene_series:
            raise AttributeError('mating species should have identical gene series')

        num_children = get_params('num_children', self, other, self.species)

        children = []
        for i in range(num_children):

            # Exchange genes
            reproduce_series = GeneSeries(
                *np.where(
                    np.random.randint(0, 2, len(self.gene_series)),
                    np.array(self.gene_series.series),
                    np.array(other.gene_series.series)
                ).tolist()
            )

            # This code will add the created Individual to the specified species automatically.
            child = Individual(self.species, reproduce_series, mutation_rate=getattr(self, 'mutation_rate', None))

            if gene_mutate:
                child.gene_mutate()

            children.append(child)

        return children

    def get_fitness_function(self) -> Callable:
        return get_params('fitness_function', self, self.species, self.earth)

    def calc_fitness(self, timeout=None):
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

    def gene_mutation_rate(self) -> float:
        return get_params('mutation_rate', self, self.species)


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
    def __init__(self, name: str, *gen_vals: Sequence):
        super().__init__(name)
        self.gen_vals = list(gen_vals)

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


def formalize_params_space(name, space: Sequence):
    """
    Based on the type of space, assign a suitable GeneSpace form to contain it.
    If the given space is a tuple, with 2 numbers [int, float], and the first number is small
    than the second, the ContinuousGenSpace will be used to form the space. Otherwise, the
    DiscreteGenSpace will be used to form the space
    Args:
        name: the Gene name.
        space: the value range of the Gene
    """
    space_cls = (
        ContinuousGenSpace
        if (
            isinstance(space, tuple)
            and len(space) == 2
            and isinstance(space[0], (int, float))
            and isinstance(space[1], (int, float))
            and space[0] < space[1]
        )
        else DiscreteGenSpace
    )

    return space_cls(name, *space)


class _AllSpeciesMeta(type):

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

        self.earth = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    @property
    def expected_lifetime(self) -> int:
        """ The expected lifetime. """
        if isinstance(self._lifetime, int):
            return self._lifetime
        elif isinstance(self._lifetime, Callable):
            return self._lifetime()

    @_AllSpeciesMeta.register
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

            # This code will add the creating Individual into self(species)
            individual = Individual(self, GeneSeries(*list_gene))

        return individual

    def get_gene_space(self, name: str) -> "GeneSpace":
        for gene_space in self.gene_spaces:
            if gene_space.name == name:
                return gene_space

        raise NameError('the given gene space name does not exist')

    @_AllSpeciesMeta.register
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

    @_AllSpeciesMeta.register
    def reproduce(self):
        """ population of the species reproduce """
        new_generations = []
        for female, male in combinations(self.individuals, 2):

            # The Individual.mate will initialize children,
            # the individual.__init__() method will include
            # the creating child to its own species.
            children = female.mate(male)
            new_generations.extend(children)

        return new_generations

    @_AllSpeciesMeta.register
    def clear_population(self):
        self.individuals = []


class _AllSpecies(metaclass=_AllSpeciesMeta):
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

    def __getitem__(self, item: int) -> "Species":
        return self.species[item]

    def add(self, species: "Species"):
        self.species.append(species)

    def remove(self, species: "Species"):
        self.species.remove(species)


class Earth:
    """ Earth is an environment that all Individuals and Species survive, compete, and reproduce """
    def __init__(
            self,
            survival_func: Callable = None,
            living_ratio: float = 0.6,
            fitness_threshold: float = 0.8,
            max_living_space: int = 25,
            fitness_function: Callable = None,
            *,
            keep_ratio: float = None,
            mutation_rate: float = None,
            **kwargs
    ):
        self._all_species = _AllSpecies()

        self.survival_func = survival_func if isinstance(survival_func, Callable) else LFunc.top_living
        self.living_rate = (
            living_ratio
            if 0 < living_ratio < 1
            else 0
            if living_ratio <= 0
            else 1
        )

        if not keep_ratio:
            self.keep_ratio = 0.
        elif isinstance(keep_ratio, float) and keep_ratio < living_ratio:
            self.keep_ratio = keep_ratio
        elif isinstance(keep_ratio, float):
            raise ValueError('keep_ratio must be smaller than living_ratio')
        else:
            raise TypeError('keep_ratio must be float')

        self.fitness_threshold = fitness_threshold
        self.fitness_function = fitness_function
        self.max_living_space = max_living_space
        self.mutation_rate = mutation_rate
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
        """ Give the species directly """

    @overload
    def create_species(
            self,
            name: str,
            *gene_species: GeneSpace,
            mutation_rate: float = None,
            lifetime: Union[int, Callable] = None,
            fitness_function: Callable = None,
            **kwargs
    ) -> Species:
        """ Give the species name and organized GeneSpace """

    @overload
    def create_species(
            self,
            name: str,
            params_dict: Dict[str, Sequence]
    ):
        """ Give the name of Species, the GeneSpace is given by a dictionary """

    def create_species(self, *args, **kwargs) -> Species:
        """"""
        # Give the species directly
        if len(args) == 1:
            species = args[0]
            if not isinstance(species, Species):
                raise TypeError(f"Invalid species type: {type(species)}, must be Species")

        # Give the species name and its GeneSpace
        elif isinstance(args[0], str) and isinstance(args[1], GeneSpace):
            species = Species(*args, **kwargs)

        elif isinstance(args[0], str) and isinstance(args[1], dict):
            species = Species(
                args[0],
                *[formalize_params_space(n, v) for n, v in args[1].items()]
            )

        else:
            raise ValueError(f"Got an unrecognized argument pattern: {args}")

        species.earth = self  # recording the earth
        self._all_species.add(species)

        return species

    def clear_population(self):
        self._all_species.clear_population()

    @property
    def species(self) -> list[Species]:
        return list(self._all_species)

    @property
    def num_species(self) -> int:
        return len(self._all_species)

    def create_individuals(self, num_individuals: Union[Sequence[int], int]):
        """
        Create individuals for all species
        Args:
            num_individuals: How many individuals to create. If a single integer is given, all species create
                             a sample number of individuals equal to `num_individuals`. Otherwise, the
                             `num_individuals` should give a sequence of integers with the same length as
                             the number of species, num_individuals[i] individuals will be created for
                             i-th species.
        """
        if isinstance(num_individuals, int):
            self._all_species.create_individuals(num_individuals)

        elif isinstance(num_individuals, Sequence):
            if len(num_individuals) != self.num_species:
                raise ValueError(f"The length of num_individuals should be equal to the number of species")

            for species, num in zip(self.species, num_individuals):
                species.create_individuals(num)

        else:
            raise TypeError('num_individuals must be an integer or a sequence of integers')

    def calc_fitness(
            self, nproc: int = None,
            timeout: int = None,
            not_calc_existed_fitness: bool = False,
            false_to_mutate: bool = False,
    ):
        def _running_directly():
            for individual in self.individuals:
                fit_func = individual.get_fitness_function()
                if not not_calc_existed_fitness or individual.fitness_ is None:
                    individual.fitness_ = fit_func(individual)
                    logging.info(f'Fitness of {individual} is {individual.fitness_}')
                    # individual.calc_fitness(timeout)

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
                            running_idd.gene_mutate()
                            running_idd.fitness_ = None
                            waiting_individual.append(running_idd)

                            logging.info(f"{running_idd} False to calculate and mutate...")

                        else:
                            logging.debug(f'Fitness of {running_idd} is {running_idd.fitness_}')

                        terminated_individuals.append(running_idd)

                if running_process and not terminated_individuals and len(running_process) >= nproc:
                    time.sleep(10)
                    logging.info(f'{len(waiting_individual)} or {len(running_process)}')

                for terminated_individual in terminated_individuals:
                    running_process.remove(terminated_individual)

        if nproc is None:
            _running_directly()

        elif nproc < 0:
            nproc = os.cpu_count()

        if isinstance(nproc, int):
            _running_multi_processes()

        logging.debug(f'Calculation terminated')

    @property
    def individuals(self) -> list[Individual]:
        return [individual for species in self._all_species for individual in species.individuals]

    def sorted_individuals(self, decending: bool = False) -> list[Individual]:
        individuals = self.individuals
        random.shuffle(individuals)
        return sorted(
            tqdm(individuals),
            key=lambda x: (
                (x.fitness_, random.random())
                if x.fitness_ is not None
                else
                (-np.inf, random.random())
            ),
            reverse=decending
        )

    def progress_time(
            self, nproc=None,
            timeout=None,
            not_calc_existing_fitness: bool = True,
            false_to_mutate: bool = False
    ):
        """
        Time progression over all species.
        Life of all species born, reproduced, fought for living, and to die
        """
        # Exclude older individual:
        for individual in self.individuals:
            if individual.age and individual.expected_lifetime and individual.age > individual.expected_lifetime:
                individual.die()

        # Fighting for living
        self.calc_fitness(
            nproc=nproc,
            timeout=timeout,
            not_calc_existed_fitness=not_calc_existing_fitness,
            false_to_mutate=false_to_mutate
        )
        to_die = self.survival_func(self)
        for dying_individual in to_die:
            dying_individual.die()

        # old individual naturally die
        keep_number = len(self.individuals) * self.keep_ratio if isinstance(self.keep_ratio, int) else -1
        for i, individual in enumerate(self.sorted_individuals(decending=True)):
            if individual.expected_lifetime is not None and individual.age > individual.expected_lifetime:
                individual.die()

            if i >= keep_number:
                individual.age += 1
            else:
                individual.age = 0  # Refresh its age

        # Cutoff individual exceed the max living space
        individuals = list(self.sorted_individuals(decending=True))
        for individual in individuals[self.max_living_space:]:
            individual.die()

        # reproducing
        if isinstance(self.keep_ratio, float):
            # If none of individual need to keep
            for species in self._all_species:
                species.reproduce()

        self.time += 1


class EvoluteOptimizer:
    def __init__(self, gene_series, pop_size, ):
        ...



class _PerformanceFitness:
    """ Given an individual with models name and hyperparameters to calculate fitness  """
    def __init__(
            self,
            models: dict,
            X: np.ndarray,
            y: np.ndarray,
            cv: Union[int, KFold] = None
    ):
        self.models = models
        self.X = X
        self.y = y

    def __call__(self, individual: Individual):
        model = self.models[individual.species_name](**individual.gene_dict)


def ml_optimizer(
        models: dict[str, type(ModelLike)],
        hypers: dict[str, Sequence],
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        features: list[str] = None,
        fitness_function: Callable = None,
        initial_size: Union[int, list[int]] = 3,
        *,
        feature_engineering: bool = False,
        fe_kwargs: dict[str, Any] = None,
        which_return: Union[Literal['best', 'top'], int] = 'best',
        **kwargs
):
    optimizer = MachineLearningOptimizer(
        models=models,
        hypers=hypers,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        features=features,
        fitness_function=fitness_function,
        initial_size=initial_size,
        feature_engineering=feature_engineering,
        fe_kwargs=fe_kwargs,
        **kwargs
    )

    optimizer.optimize(**kwargs)

    if which_return == 'best':
        return optimizer.optimal_result
    elif which_return == 'top':
        count = max(int(0.1 * len(optimizer.earth.individuals)), 1)
        return optimizer.sorted_results[:count]
    elif isinstance(which_return, int):
        return optimizer.sorted_results[:which_return]
    else:
        raise TypeError(f"The 'which_return' argument is not an integer or 'best', 'top'")


class MachineLearningOptimizer:
    def __init__(
            self,
            models: dict[str, type(ModelLike)],
            hypers: dict[str, dict],
            X: np.ndarray,
            y: np.ndarray,
            X_test: np.ndarray = None,
            y_test: np.ndarray = None,
            features: list[str] = None,
            *,
            fitness_function: Callable = None,
            initial_size: Union[int, list[int]] = 3,
            feature_engineering: bool = False,
            fe_kwargs: dict[str, Any] = None,
            **kwargs
    ):
        """

        Args:
            models:
            hypers:
            X:
            y:
            X_test:
            y_test:
            fitness_function:
            initial_size:
            feature_engineering: Whether to perform feature engineering before evaluating the fitness
            fe_kwargs: Additional keyword arguments to the feature engineering processes
            **kwargs:
        """
        self.models = models
        self.hypers = hypers
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.features = self._get_feature_names(features)
        self.feature_engineering = feature_engineering
        self.fe_kwargs = fe_kwargs if fe_kwargs else {}

        if isinstance(fitness_function, Callable):
            self.fitness_function = fitness_function
        elif self.X_test is not None and self.y_test is not None:
            self.fitness_function = self.train_test_fitness
        else:
            self.fitness_function = self.cross_validation_fitness

        self.initial_size = initial_size
        self.earth = Earth(
            fitness_function=self.fitness_function,
            **kwargs
        )

        # Check the names in self.models match in self.hypers
        if len(self.models) != len(self.hypers) or any(name not in self.models for name in self.hypers):
            raise ValueError(f"the names of the models and hypers do not match")

        for species_name, params_dict in self.hypers.items():
            self.earth.create_species(species_name, params_dict)

    def _get_feature_names(self, features: list[str]) -> list[str]:
        if features is None:
            return [f'f{i}' for i in range(X.shape[1])]

        if len(features) != self.X.shape[1]:
            raise ValueError("The number of features is not equal to the number of columns of X")

        return features

    @staticmethod
    def _feature_engineering(estimator, X, y, features, skip_recur_addi: bool = True, **kwargs):
        """"""
        ntri_idx = ML.quickly_feature_selection_(X, y, estimator, **kwargs)

        X = X[:, ntri_idx]
        features = np.array(features)[ntri_idx].tolist()

        _, _, X, features = ML.pca_dimension_reduction_(X, features, **kwargs)

        if not skip_recur_addi:
            print('recursive addition')
            feat_indices_add, score_add, = \
                ML.recursive_feature_addition_(
                    X, y, estimator, **kwargs
                )
        else:
            feat_indices_add, score_add = [], -np.inf

        print('recursive elimination')
        feat_indices_eli, score_eli = \
            ML.recursive_feature_elimination_(
                X, y, estimator
            )

        X, features, cv_best_metric, how_to_essential = \
            ML.recursive_feature_selection_(
                X, features,
                feat_indices_eli, score_eli,
                feat_indices_add, score_add
            )

        return X, features

    def train_test_fitness(self, individual):
        """"""
        model_cls = self.models[individual.species_name]
        hyper = individual.gene_dict
        model = model_cls(**hyper)

        if self.feature_engineering:
            X, feature = self._feature_engineering(model, self.X, self.y, self.features, **self.fe_kwargs)
            feat_idx = np.where(np.isin(self.features, feature))[0]
            X_test = self.X_test[:, feat_idx]
            individual.results["features"] = feature
        else:
            X, feature = self.X, self.features
            X_test = self.X_test

        model.train(X, self.y)
        return model.score(X_test, self.y_test)

    def cross_validation_fitness(self, individual):
        """"""
        model_cls = self.models[individual.species_name]
        hyper = individual.gene_dict
        model = model_cls(**hyper)

        if self.feature_engineering:
            X, feature = self._feature_engineering(model, self.X, self.y, self.features, **self.fe_kwargs)
            individual.results["features"] = feature
        else:
            X, feature = self.X, self.features

        return cross_val_score(model, X, self.y).mean()

    def optimize(self, n_iters: int = 1, **kwargs):
        """"""
        self.earth.clear_population()
        self.earth.create_individuals(self.initial_size)

        for _ in range(n_iters):
            self.earth.progress_time(**kwargs)

            print("Top performance models:")
            for top_individual in self.earth.sorted_individuals():
                print(top_individual, top_individual.fitness_)

    @property
    def sorted_results(self) -> list[tuple[type(ModelLike), dict]]:
        return [
            (self.models[individual.species_name], individual.gene_dict)
            for individual in self.earth.sorted_individuals(True)
        ]

    @property
    def optimal_result(self) -> tuple[type(ModelLike), dict]:
        return self.sorted_results[0]

    @property
    def optimal_model(self) -> ModelLike:
        model_cls, hypers = self.optimal_result
        return model_cls(**hypers)


class MLOptimizer:
    def __init__(
            self,
            model: ModelLike,
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

        self.survival_func = None
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
                list_gene_space.append(DiscreteGenSpace(param_name, *param_space))

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
            earth._all_species.create_individuals(self.init_params)
        elif isinstance(self.init_params, list) and all(isinstance(p, dict) for p in self.init_params):
            for init_param in self.init_params:
                earth._all_species[0].create_individual(**init_param)
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


    optimizer = MLOptimizer(
        model=GradientBoostingRegressor,
        X_train=X, y_train=y,
        params_space=spaces
    )
    optimizer.optimize()
