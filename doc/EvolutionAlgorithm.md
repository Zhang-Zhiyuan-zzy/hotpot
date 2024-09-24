# Evolutionary Algorithm (EA)

This documentation introduce how to optimize parameters by Evolution algorithm (EA).

## Tutorial

We elucidate the parameter optimization process by EA with the example of optimizing the hyperparameter (HP)
of Machine learning model.

At first, import the EA module:

```python
from hotpot.plugins.opti.ev import Earth, Species, Individual
```

In the EA module, there are three basic class `Individual`, `Species`, and `Earth`:

- The `Individual` represents a specific sample, defined by its `Species` and its `GeneSeries`.
- The `Species` is defined by a group of Gene types and the range of values for those Gene types,
  namely `GeneSpace`. The `Species` also specify some attributes of individuals belonging to them,
  such as:
  - 'expected_lifetimes': the time allowed for its individual to survive, which is the maximum number
    of iterations from creation to deletion.
  - 'mutation_rate': the probability of gene mutation
- The `Earth` is the habitat for all `Individual` or all `Species`. It is the main control
  program of the EA. Once by invoking `Earth.progress_time()`, the age on Earth advances
  a time(i.e., an iteration of the evolutionary algorithm). Each time the time progresses,
  the Earth's environment (the conditions for the survival of species and individuals) changes.
  Individuals that are too old and weak will perish, while young and strong individuals survive and
  reproduce. The genes of individuals will also undergo mutations according to certain rules.

In the evolutionary process of evolutionary algorithms (EA), a fitness_function must be provided
to assess the fitness of `Individual`(`Individual.fitness_`). By inputting the fitness_function at
different levels, its scope of application can be set. if the fitness_function input is specified
for an `Individual`, it is used solely for evaluating that `Individual`'s fitness. When input for
a `Species`, the fitness_function is used to assess all individuals within that species. When input
for the `Earth`, the fitness_function is used for the evaluation of global individuals' fittness.
When fitness_functions for individuals, species, and the Earth are all set simultaneously, the final
fitness_function to be executed is chosen based on the priority order: individual over species over Earth.

The `fitness_function` is defined as the following format$(R^2)$:

```python
from hotpot.plugins.opti.ev import Individual
def fitness_function(individual: Individual) -> float:
    fitness = (
      ...
    )
    return fitness
```

In the following, we give a brief example for the implementation of the EA. The task is to find the
optimal machine learning(ML) algorithm and its optimal hyperparameters(HPs). We will carry out an
optimization across several ML algorithm with distinct

```python
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
```

The function `ml_optimizer()` and class `MachineLearningOptimizer` are convenient interfaces specifically
designed for machine learning algorithms and hyperparameters optimization.

```python
from hotpot.plugins.opti.ev import MachineLearningOptimizer, ml_optimizer

models = ...
hypers = ...
X, y = ..., ...
features = ...


results = ml_optimizer(
    models=models,
    hypers=hypers,
    X=X,
    y=y,
    features=features,
    which_return='top',
    nproc=8
)
```

To select the optimal machine learning algorithm and hyperparameters, evolutionary algorithms can be
combined with feature engineering to simultaneously optimize feature selection. To this end, specify
argument `feature_engineering=True` in `ml_optimizer`

> result = ml_optimizer(..., feature_engineering=True, ...)

## API Documentation


<h4><font color=#b8860s>**class**</font> Individual(species: Species, gene_series: GeneSeries, fitness_function: Callable, mutation_rate: float = 0.5):</h4>

The `Individual` class represents an individual entity within a given evolutionary environment. Each individual is characterized by a set of genes and interacts with its environment.


#### Attributes

* **species**: The species to which this individual belongs.
* **gene\_series**: Represents the individual's genetic code.
* **mutation\_rate**: The rate at which mutations occur in the individual's genes.
* **fitness\_**: Represents the fitness score of the individual.
* **fitness\_process**: An instance of the `FitnessProcess` responsible for managing the fitness calculation process for this individual.
* **age**: The age of the individual.


#### Methods

1. **`__init__(self, species, gene_series, mutation_rate=None)`**: Initializes an individual with the provided species and gene series. Optionally takes a mutation rate.
   * **Parameters**:
     * `species`: The species the individual belongs to.
     * `gene_series`: The series of genes that represent the individual's genetic code.
     * `mutation_rate`: The rate at which gene mutations can occur.
2. **`gene_mutate(self)`**: Mutates the individual's genes based on the mutation rate.
   * **How to call**:
     <pre class="code-fence" md-src-pos="1501..1551"><div class="code-fence-highlighter-copy-button" data-fence-content="aW5kaXZpZHVhbC5nZW5lX211dGF0ZSgp"><img class="code-fence-highlighter-copy-button-icon"/><span class="tooltiptext"></span></div><code class="language-python" md-src-pos="1501..1551"><span md-src-pos="1501..1511"></span><span md-src-pos="1511..1521">individual</span><span md-src-pos="1521..1522">.</span><span md-src-pos="1522..1533">gene_mutate</span><span md-src-pos="1533..1534">(</span><span md-src-pos="1534..1535">)</span><span md-src-pos="1535..1536">
     </span><span md-src-pos="1542..1551"></span></code></pre>
3. **`mate(self, other: 'Individual', gene_mutate: bool = True)`**: Allows the individual to mate with another individual from the same species, producing offspring.
   * **Parameters**:
     * `other`: Another individual of the same species with which to mate.
     * `gene_mutate`: Whether to apply gene mutations to the offspring.
   * **Returns**: A list of offspring individuals.
   * **How to call**:
     <pre class="code-fence" md-src-pos="1975..2046"><div class="code-fence-highlighter-copy-button" data-fence-content="b2Zmc3ByaW5nID0gaW5kaXZpZHVhbC5tYXRlKG90aGVyX2luZGl2aWR1YWwp"><img class="code-fence-highlighter-copy-button-icon"/><span class="tooltiptext"></span></div><code class="language-python" md-src-pos="1975..2046"><span md-src-pos="1975..1985"></span><span md-src-pos="1985..1995">offspring </span><span md-src-pos="1995..1996">=</span><span md-src-pos="1996..2007"> individual</span><span md-src-pos="2007..2008">.</span><span md-src-pos="2008..2012">mate</span><span md-src-pos="2012..2013">(</span><span md-src-pos="2013..2029">other_individual</span><span md-src-pos="2029..2030">)</span><span md-src-pos="2030..2031">
     </span><span md-src-pos="2037..2046"></span></code></pre>
4. **`get_fitness_function(self) -> Callable`**: Retrieves the fitness function for the individual.
   * **Returns**: A callable fitness function.
   * **How to call**:
     <pre class="code-fence" md-src-pos="2229..2307"><div class="code-fence-highlighter-copy-button" data-fence-content="Zml0bmVzc19mdW5jdGlvbiA9IGluZGl2aWR1YWwuZ2V0X2ZpdG5lc3NfZnVuY3Rpb24oKQ=="><img class="code-fence-highlighter-copy-button-icon"/><span class="tooltiptext"></span></div><code class="language-python" md-src-pos="2229..2307"><span md-src-pos="2229..2239"></span><span md-src-pos="2239..2256">fitness_function </span><span md-src-pos="2256..2257">=</span><span md-src-pos="2257..2268"> individual</span><span md-src-pos="2268..2269">.</span><span md-src-pos="2269..2289">get_fitness_function</span><span md-src-pos="2289..2290">(</span><span md-src-pos="2290..2291">)</span><span md-src-pos="2291..2292">
     </span><span md-src-pos="2298..2307"></span></code></pre>
5. **`calc_fitness(self, timeout=None)`**: Calculates the fitness of the individual using the fitness function.
   * **Parameters**:
     * `timeout`: The maximum time allowed for the fitness calculation.
   * **How to call**:
     <pre class="code-fence" md-src-pos="2549..2610"><div class="code-fence-highlighter-copy-button" data-fence-content="aW5kaXZpZHVhbC5jYWxjX2ZpdG5lc3ModGltZW91dD02MCk="><img class="code-fence-highlighter-copy-button-icon"/><span class="tooltiptext"></span></div><code class="language-python" md-src-pos="2549..2610"><span md-src-pos="2549..2559"></span><span md-src-pos="2559..2569">individual</span><span md-src-pos="2569..2570">.</span><span md-src-pos="2570..2582">calc_fitness</span><span md-src-pos="2582..2583">(</span><span md-src-pos="2583..2590">timeout</span><span md-src-pos="2590..2591">=</span><span md-src-pos="2591..2593">60</span><span md-src-pos="2593..2594">)</span><span md-src-pos="2594..2595">
     </span><span md-src-pos="2601..2610"></span></code></pre>
6. **`die(self)`**: Removes the individual from its species.
   * **Returns**: The individual itself.
   * **How to call**:
     <pre class="code-fence" md-src-pos="2748..2790"><div class="code-fence-highlighter-copy-button" data-fence-content="aW5kaXZpZHVhbC5kaWUoKQ=="><img class="code-fence-highlighter-copy-button-icon"/><span class="tooltiptext"></span></div><code class="language-python" md-src-pos="2748..2790"><span md-src-pos="2748..2758"></span><span md-src-pos="2758..2768">individual</span><span md-src-pos="2768..2769">.</span><span md-src-pos="2769..2772">die</span><span md-src-pos="2772..2773">(</span><span md-src-pos="2773..2774">)</span></code></pre>
