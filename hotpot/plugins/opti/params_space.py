import json
from typing import Any, Callable
from dataclasses import dataclass
import optuna


__all__ = [
    'ParamSpace',
    'optuna_optimize',
    'ParamSets'
]


@dataclass
class CategoricalParams:
    name: str
    params: list[Any]

@dataclass
class DiscreteUniformParams:
    name: str
    low: float
    high: float
    q: float

@dataclass
class FloatParams:
    name: str
    low: float
    high: float
    log: bool = False

@dataclass
class IntParams:
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False


class ParamSpace:
    def __init__(self):
        self.space = []

    def add_categorical_params(self, name, params):
        self.space.append(CategoricalParams(name, params))

    def add_discrete_uniform_params(self, name, low, high, q: float = 1.0):
        self.space.append(DiscreteUniformParams(name, low, high, q))

    def add_float_params(self, name, low, high, log=False):
        self.space.append(FloatParams(name, low, high, log))

    def add_int_params(self, name, low, high, step=1, log=False):
        self.space.append(IntParams(name, low, high, step, log))

    def copy_to_optuna_trial(self, trial: optuna.Trial):
        params = {}
        for param in self.space:
            if isinstance(param, CategoricalParams):
                value = trial.suggest_categorical(param.name, param.params)
            elif isinstance(param, DiscreteUniformParams):
                value = trial.suggest_discrete_uniform(param.name, param.low, param.high, param.q)
            elif isinstance(param, FloatParams):
                value = trial.suggest_float(param.name, param.low, param.high, log=param.log)
            elif isinstance(param, IntParams):
                value = trial.suggest_int(param.name, param.low, param.high, step=param.step, log=param.log)
            else:
                raise TypeError(f"Unknown parameter type: {type(param)}")

            params[param.name] = value

        return params


def optuna_optimize(
        response: Callable[[dict], float],
        space: ParamSpace,
        n_trials: int = 100,
        sampler: optuna.samplers.BaseSampler = optuna.samplers.GPSampler,
        return_study: bool = False,
):
    def objective(trial: optuna.Trial) -> float:
        params = space.copy_to_optuna_trial(trial)
        return response(params)

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    if return_study:
        return study
    else:
        return study.best_params


class ParamSets:
    """ A handle of hyperparameters. """
    def __init__(self, hparams: dict = None):
        self._hparams = {
            'lr': 1e-3,
            'weight_decay': 4e-5
        }
        if hparams is not None:
            self._hparams.update(hparams)

    def __getattr__(self, item):
        try:
            object.__getattribute__(self, item)
        except AttributeError as e:
            if item in self._hparams:
                return self._hparams[item]
            raise e

    def __setattr__(self, key, value):
        if key == '_hparams':
            object.__setattr__(self, key, value)
        else:
            self._hparams[key] = value

    def export(self, path):
        hparams = {}
        for name, value in self._hparams.items():
            if isinstance(value, (int, float, str)):
                hparams[name] = value
            elif isinstance(value, type):
                hparams[name] = value.__name__
            else:
                hparams[name] = value.__class__.__name__

        # Save the Hyper parameters dict to file
        with open(path, 'w') as f:
            json.dump(hparams, f, indent=4)

    @classmethod
    def from_dict(cls, hparams: dict):
        return cls(hparams)

    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as f:
            return cls.from_dict(json.load(f))


