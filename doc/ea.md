# Evolution Algorithm (EA)
This documentation introduce how to optimize parameters by Evolution algorithm (EA).

## Tutorial
We elucidate the parameter optimization process by EA with the example of optimizing the hyperparameter (HP)
of Machine learning model.

At first, import the EA module:
```python
from hotpot.plugins.opti.ev import Earth, Species, Individual
```
In the EA module, there are three basic class `Individual`, `Species`, and `Earch`:
- The `Individual` represents a specific sample, defined by its `Species` and its `GeneSeries`