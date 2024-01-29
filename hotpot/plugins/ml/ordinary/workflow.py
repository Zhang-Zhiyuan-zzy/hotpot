"""
python v3.9.0
@Project: hotpot
@File   : workflow
@Auther : Zhiyuan Zhang
@Data   : 2024/1/13
@Time   : 9:30
Notes: The convenient workflows for training ordinary(non-deep) machine learning models
"""
from typing import Union, Sequence, Literal, Callable
from multiprocessing import Process, Queue

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import norm


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import matplotlib.pyplot as plt

from hotpot.plots import SciPlot, R2Regression, PearsonMatrix, HierarchicalTree


def expectation_improvement(y, mu, sigma):
    ymax = np.max(y)

    diff = mu - ymax
    u = diff / sigma

    ei = diff * norm.cdf(u) + sigma * norm.pdf(u)
    ei[sigma <= 0.] = 0.
    return ei


class Params:
    def __init__(self, name: str, params: Sequence):
        self.name = name
        self.params = list(params)

        # transform discrete labels to integer index
        if any(isinstance(p, str) for p in self.params):
            self._param = [i for i in range(len(params))]
            self.is_labels = True
        else:
            self._param = self.params
            self.is_labels = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, N={len(self.params)})"

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        return self.params

    def __getitem__(self, item):
        return self.params[item]

    def invert(self, value):
        """ given the labels params, return the actual label """
        return self.params[self._param.index(value)]


class ParamsRange(Params):
    def __init__(self, name: str, start: Union[int, float], stop: Union[int, float], step: [int, float] = None):
        if (isinstance(start, int) and isinstance(stop, int)) or (isinstance(start, float) and isinstance(stop, float)):
            if start < stop:
                self.start = start
                self.stop = stop
            else:
                raise ValueError('The start must larger than stop!')
        else:
            raise TypeError('The start and stop should both int or both float!')

        if isinstance(self.start, int):
            if step is None:
                self.step = 1
            elif isinstance(step, int):
                self.step = step
            else:
                raise TypeError('The step should be int, same with start and stop')
        else:
            if step is None:
                self.step = (self.stop - self.start) / 10
            elif isinstance(step, float):
                self.step = step
            else:
                raise TypeError('The step should be float, same with start and stop')

        self.num_params = int((self.stop - self.start) // self.step)
        params = [self.start + i*self.step for i in range(self.num_params)]
        super().__init__(name, params)


class ParamsSpace:
    def __init__(self, *params: Params):
        self._params = params
        self.embed_params = self._embedding()

        self.tried_params = None
        self.tried_scores = None
        self.tried_indices = None

    def __getitem__(self, item) -> dict:
        embde_param = self.embed_params[item]
        return {p.name: p.invert(ep) for p, ep in zip(self._params, embde_param)}

    def _check(self):
        """ Check if the given params have duplicate names """

    def invert(self, param_values):
        return [p.invert(v) for p, v in zip(self._params, param_values)]

    @property
    def optimal_params(self) -> dict:
        return self.tried_params[np.argmax(self.tried_scores)]

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self._params]

    def sample(self, seed=None) -> (int, dict):
        np.random.seed(seed)
        idx = np.random.randint(self.size)
        return idx, self[idx]

    @property
    def size(self) -> int:
        return len(self.embed_params)

    def to_params_score_sheet(self) -> pd.DataFrame:
        df = pd.DataFrame(self.tried_params)
        df['Scores'] = self.tried_scores

        return df

    def _embedding(self):
        mesh_params = np.meshgrid(*[np.array(getattr(p, '_param')) for p in self._params])
        design_params = np.vstack([p.flatten() for p in mesh_params]).T

        return design_params


class WorkFlow:
    """ The convenient workflows for training ordinary(non-deep) machine learning models """
    def __init__(
            self, data: pd.DataFrame, scaler=StandardScaler, model=GradientBoostingRegressor,
    ):
        self.data = data
        self.Xscaler = scaler()
        self.yscaler = scaler()
        self._model = model

        self.features = None
        self.target = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def cross_validation(self, split_method=LeaveOneOut(), model_kwargs: dict = None) -> np.ndarray:
        if model_kwargs is None:
            model_kwargs = {}
        if self.X_train is None or self.y_train is None:
            self.X_train = self.X
            self.y_train = self.y

        pred, true = [], []
        for train_index, val_index in split_method.split(self.X_train):
            X_train, y_train = self.X_train[train_index], self.y_train[train_index]
            X_val, y_val = self.X_train[val_index], self.y_train[val_index]

            model = self._model(**model_kwargs)
            model.fit(X_train, y_train)
            pred.append(model.predict(X_val))
            true.append(y_val)

        pred, true = np.concatenate(pred), np.concatenate(true)
        return np.vstack([true, pred])

    def gaussian_process_cross_validation(self, split_method=LeaveOneOut(), model_kwargs: dict = None):
        if self.X_train is None or self.y_train is None:
            self.X_train = self.X
            self.y_train = self.y

        if model_kwargs is None:
            model_kwargs = {
                "kernel": ConstantKernel() * RBF(np.ones(self.X_train.shape[1]), (1e-6, 1e6)) + WhiteKernel(),
                "n_restarts_optimizer": 50
            }

        pred, true, list_sigma = [], [], []
        for train_index, val_index in split_method.split(self.X_train):
            X_train, y_train = self.X_train[train_index], self.y_train[train_index]
            X_val, y_val = self.X_train[val_index], self.y_train[val_index]

            model = GaussianProcessRegressor(**model_kwargs)
            model.fit(X_train, y_train)
            mu, sigma = model.predict(X_val,return_std=True)
            pred.append(mu)
            true.append(y_val)
            list_sigma.append(sigma)

        pred, true = np.concatenate(pred), np.concatenate(true)
        return np.vstack([true, pred]), np.array(list_sigma)

    def feature_engineering(self):
        pass

    @staticmethod
    def metric(pt: np.ndarray, work_type: Literal['reg', 'cls'] = 'reg'):
        metric_methods = {
            'reg': {'R2': r2_score, 'MAE': mean_absolute_error, 'RMSE': lambda p, t: np.sqrt(mean_squared_error(p, t))},
        }
        methods = metric_methods[work_type]

        metrics_results = {n: m(*pt) for n, m in methods.items()}
        for name, result in metrics_results.items():
            print(f"{name}: {result}")

        return metrics_results

    def params_optim(
            self, params_space: ParamsSpace, split_method=LeaveOneOut(), seed=None, score=r2_score,
            n_init: int = 10, n_iter: int = 10, n_job=10
    ):
        """"""
        def _initialize(queue):
            idx_, kwargs_ = params_space.sample(seed)
            queue.put((idx_, kwargs_, score(*self.cross_validation(split_method, model_kwargs=kwargs_))))

        def get_processes_data():
            for proc, queue in processes.items():
                proc.join()
                idx_, kwargs_, scores = queue.get()
                indices.append(idx_)
                list_kwargs.append(kwargs_)
                list_score.append(scores)

            processes.clear()

        processes = {}
        indices, list_kwargs, list_score = [], [], []
        for i in tqdm(range(n_init), "Initialize"):
            q = Queue()
            p = Process(name=str(i), target=_initialize, args=(q,))
            p.start()
            processes[p] = q

            if len(processes) % n_job == 0:
                get_processes_data()

        get_processes_data()

        for i in range(n_iter):
            kernel = ConstantKernel(1.0, (1e-6, 1e6)) * RBF(np.ones(len(params_space.param_names)), (1e-6, 1e6)) + \
                     WhiteKernel(1.0, (1e-6, 1e6))
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=50, normalize_y=True)

            gp.fit(params_space.embed_params[indices], np.array(list_score))
            mu, sigma = gp.predict(params_space.embed_params, return_std=True)
            ei = expectation_improvement(np.array(list_score), mu, sigma)

            idx = np.argmax(ei)
            model_kwargs = params_space[idx]
            pt = self.cross_validation(split_method, model_kwargs=model_kwargs)
            list_score.append(score(*pt))

            indices.append(idx)
            list_kwargs.append(model_kwargs)

            print(f"iter {i}: Score={list_score[-1]}")

        # Store the tried params
        params_space.tried_params, params_space.tried_scores, params_space.tried_indices = \
            list_kwargs, list_score, indices

        return indices, list_kwargs, list_score

    def plot_r2(self, pt1, pt2=None, err1=None, err2=None, show=True) -> (plt.Figure, plt.Axes):
        r2_plotter = R2Regression(pt1, pt2, err1=err1, err2=err2)
        sci_plot = SciPlot(r2_plotter)
        fig, ax = sci_plot()
        if show:
            fig.show()

        return fig, ax

    def plot_pearson_matrix_and_tree(
            self, X: np.ndarray, labels: list[str], show_values=False
    ) -> (plt.Figure, np.ndarray[plt.Axes]):
        pmat_plotter = PearsonMatrix(X, labels, show_values=show_values)
        tree_plotter = HierarchicalTree(X, labels)

        sci_plot = SciPlot([[pmat_plotter], [tree_plotter]])
        fig, axs = sci_plot()

        return fig, axs

    def preprocessing(
            self, Xoffset: int = 0, Xindex: Union[Sequence, np.ndarray] = None, y_index: int = -1
    ) -> (np.ndarray, np.ndarray):
        if Xindex is not None:
            X, y = self.data.iloc[:, Xindex], self.data.iloc[:, y_index]
        else:
            X, y = self.data.drop(columns=[self.data.columns[y_index]]).iloc[:, Xoffset:], self.data.iloc[:, y_index]

        self.features, self.target = X.columns, y.name
        self.Xscaler.fit(X.values)
        self.yscaler.fit(y.values.reshape(-1, 1))

        self.X = self.Xscaler.transform(X.values)
        self.y = self.yscaler.transform(y.values.reshape(-1, 1)).flatten()

        return self.X, self.y

    def split_train_test(self, test_size=0.2, random_state=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, model_kwargs: dict = None):
        if model_kwargs is None:
            model_kwargs = {}
        if self.X_train is None or self.y_train is None:
            self.X_train = self.X
            self.y_train = self.y

        model = self._model(**model_kwargs)
        model.fit(self.X_train, self.y_train)
        return model
