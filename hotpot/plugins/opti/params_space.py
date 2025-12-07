import json
from typing import Any, Callable, Optional
from dataclasses import dataclass
import optuna
import torch


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
    """Hyperparameters as attributes, backed by a dict."""
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        hparams: dict[str, Any] = {
            "lr": 1e-3,
            "weight_decay": 4e-5,
        }
        if params:
            hparams.update(params)
        object.__setattr__(self, "_hparams", hparams)

    def __getattr__(self, key: str) -> Any:
        hparams = object.__getattribute__(self, "_hparams")
        if key in hparams:
            return hparams[key]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {key!r}")

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_hparams":
            object.__setattr__(self, key, value)
        else:
            self._hparams[key] = value

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._json_safe(x) for x in value]
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, type):
            return value.__name__
        return value.__class__.__name__

    def export(self, path: str) -> None:
        data = {k: self._json_safe(v) for k, v in self._hparams.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, path: str) -> "ParamSets":
        with open(path) as f:
            data = json.load(f)

        name = data.get("OPTIMIZER")
        if isinstance(name, str):
            data["OPTIMIZER"] = getattr(torch.optim, name)
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._hparams)
