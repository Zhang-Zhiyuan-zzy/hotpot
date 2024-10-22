"""
python v3.9.0
@Project: hotpot
@File   : opti
@Auther : Zhiyuan Zhang
@Data   : 2024/1/3
@Time   : 11:27
"""
import os
from pathlib import Path
import logging
from typing import Union, Callable

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE, MDS

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel

from hotpot.plots import BayesDesignSpaceMap
from hotpot.plugins.plots import BayesDesignSpaceMap


class AcquisitionFunc:
    @staticmethod
    def expected_improvement(m, sigma, ymax):
        """Return the expected improvement.

        Arguments
        m     -- The predictive mean at the test points.
        sigma -- The predictive standard deviation at
                 the test points.
        ymax  -- The maximum observed value (so far).
        """
        diff = m - ymax
        u = diff / sigma
        ei = (diff * torch.distributions.Normal(0, 1).cdf(u) +
              sigma * torch.distributions.Normal(0, 1).log_prob(u).exp()
              )
        ei[sigma <= 0.] = 0.
        return ei


class GaussianProcess(gpytorch.models.ExactGP):
    """Exact Gaussian Process model.

    Arguments
    train_x     --  The training inputs.
    train_y     --  The training labels.
    mean_module --  The mean module. Defaults to a constant mean.
    covar_module--  The covariance module. Defaults to a RBF kernel.
    likelihood  --  The likelihood function. Defaults to Gaussian.
    """
    def __init__(
        self, train_x, train_y,
        mean_module=gpytorch.means.ConstantMean(),
        covar_module=ScaleKernel(RBFKernel()),
        likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.0)),

    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BayesianOptimizer:
    """ Implementing the Bayesian Optimization """
    def __init__(
            self, surrogate: gpytorch.models.ExactGP,
            acq_func=AcquisitionFunc.expected_improvement,
            batch_size: int = 1
    ):
        """
        Args:
            surrogate: surrogate model
            acq_func: acquisition function
        """
        self.surrogate = surrogate
        self.acq_func = acq_func
        self.batch_size = batch_size
        self.is_trained = False

    def __call__(self, X_design, n_iter=150, lr=0.1):
        if isinstance(X_design, np.ndarray):
            X_design = torch.tensor(X_design)

        logging.info('\n'.join([f'{name}, {p}' for name, p in self.surrogate.named_parameters()]))
        train_x, train_y = self.surrogate.train_inputs[0], self.surrogate.train_targets
        logging.info('\n'.join([f'{name}, {p}' for name, p in self.surrogate.named_parameters()]))

        X_optimal, mu_optimal, sigma_optimal, X_opti_idx = [], [], [], []
        for c in range(self.batch_size):
            self.gp_train(n_iter=n_iter, lr=lr)
            mu, sigma = self.gp_predict(X_design)
            acq_value = self.acq_func(mu, sigma, train_y.max())

            # Find best point to include
            i = torch.argmax(acq_value)
            X_opti_idx.append(i)
            X_optimal.append(X_design[i])
            mu_optimal.append(mu[i])
            sigma_optimal.append(sigma[i])

            # Update the train X and train y dataset
            # the new y is supposed to be the predicted mu by GP model
            train_x = torch.from_numpy(torch.vstack([train_x, X_optimal[-1]]).detach().numpy())
            train_y = torch.from_numpy(torch.hstack([train_y, mu_optimal[-1]]).detach().numpy())
            self.surrogate.set_train_data(train_x, train_y, strict=False)

        return torch.stack(X_optimal), torch.stack(mu_optimal), torch.stack(sigma_optimal), torch.stack(X_opti_idx)

    def gp_train(self, n_iter=100, lr=0.1, report_gap=None):
        """Train the model.

        Arguments
        n_iter  --  The number of iterations.
        """
        if report_gap is None:
            report_gap = n_iter

        self.surrogate.train()
        # optimizer = torch.optim.LBFGS(self.surrogate.parameters(), lr=lr)
        optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=lr)
        likelihood = self.surrogate.likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.surrogate)

        def closure():
            optimizer.zero_grad()
            output = self.surrogate(self.surrogate.train_inputs[0])
            lo = -mll(output, self.surrogate.train_targets)
            lo.backward()
            return lo

        for i in range(n_iter):
            loss = optimizer.step(closure)
            if (i + 1) % report_gap == 0:
                print(f'Iter {i + 1:3d}/{n_iter} - Loss: {loss.item():.3f}')
        self.surrogate.eval()

        self.is_trained = True
        
    def gp_predict(self, X):
        """ predict mu and sigma using build-in GP model """
        pred = self.surrogate(X)
        mu = pred.mean
        sigma2 = pred.variance
        sigma = torch.sqrt(sigma2)

        return mu, sigma

    def visualize_design_space(
            self,
            X_design,
            X_opti_idx=None,
            n_iter=150,
            lr=0.1,
            emb_method=TSNE(),
            figpath=None,
            emb_x=None,
            show_fig=False,
            y_scaler=None,
    ):
        if not self.is_trained:
            self.gp_train(n_iter, lr)

        if isinstance(X_design, np.ndarray):
            X_design = torch.from_numpy(X_design)

        mu, sigma = self.gp_predict(X_design)
        mu, sigma = mu.detach().numpy(), sigma.detach().numpy()
        if y_scaler:
            mu = y_scaler.inverse_transform(mu.reshape(-1, 1)).flatten()
            sigma = y_scaler.inverse_transform(sigma.reshape(-1, 1)).flatten()

        if emb_x is None:
            emb_x = emb_method.fit_transform(X_design)

        if show_fig or figpath:
            beyes_map = BayesDesignSpaceMap(emb_x, mu, sigma, X_opti_idx)
            fig, axs = beyes_map()

            if show_fig:
                print('show')
                fig.show()
            if figpath:
                fig.savefig(figpath)

        return emb_x, mu, sigma

    def make_2d_design_space_plots(
            self,
            emb_x: Union[list[np.ndarray], np.ndarray],
            mus: Union[list[np.ndarray], np.ndarray],
            sigmas: Union[list[np.ndarray], np.ndarray],
            mu_norm: tuple[float, float] = None,
            sigma_norm: tuple[float, float] = None,
    ):
        """
        Make 2D design space plots in a same colorbar scale.
        Args:
            emb_x:
            mus:
            sigmas:
            mu_norm:
            sigma_norm:

        Returns:

        """
        # Convert the Numpy Array to list of Array.
        for var_name in ['emb_x', 'mus', 'sigmas']:
            if isinstance(locals()[var_name], np.ndarray):
                locals()[var_name] = [locals()[var_name]]

        # Set up the colorbar normalization.
        if mu_norm is None:
            mu_norm = min(mu.min() for mu in mus), max(mu.max() for mu in mus)
        if sigma_norm is None:
            sigma_norm = min(sigma.min() for sigma in sigmas), max(sigma.max() for sigma in sigmas)


def beyes_run(X, y, X_design, batch_size=5):
    """
    Running the Bayesian iteration
    Args:
        X: the known parameters
        y: the known optimized target (or indicator)
        X_design: the  allowed design space of the optimizing procedure.
        batch_size: the number of samples to proposed in each optimization step

    Return:
        beyes(BayesianOptimizer): the Bayesian optimizer instance
        X_opti: the proposed optimal X parameters for next experiments
        mu_opti: the estimated mean target value of the proposed optimal X
        sigma_opti: the estimated standard deviation of target of the proposed optimal X
        X_idx: the index of the proposed optimal X in the whole design space
    """
    X, y, X_design = (torch.tensor(v) for v in [X, y, X_design])

    gp = GaussianProcess(X, y, covar_module=ScaleKernel(RBFKernel(ard_num_dims=X.shape[1])))
    optimizer = BayesianOptimizer(gp, batch_size=batch_size)
    X_opti, mu_opti, sigma_opti, X_idx = optimizer(X_design, n_iter=300)

    for param_name, param in optimizer.surrogate.named_parameters():
        print(f'Parameter name: {param_name:42} value = {param.detach().cpu().tolist()}')

    return optimizer, X_opti, mu_opti, sigma_opti, X_idx


def next_params(
        X, y,
        param_range: np.ndarray,
        param_names: list[str],
        next_param_path,
        mesh_counts: int = 20,
        figpath: Union[str, os.PathLike] = None,
        log_indices: Union[int, list[int]] = None,
):
    X = torch.tensor(X)
    y = torch.tensor(y)

    param_tran = ParamPreprocessor(
        param_range=param_range,
        param_names=param_names,
        logX_indices=log_indices,
        param_mesh_counts=mesh_counts
    )
    param_tran.fit(X, y)
    X_design = param_tran.get_X_design()
    X_scale, y_scale, X_design_scale = param_tran.transform(X, y, X_design)

    bayes, X_opti, mu_opti, sigma_opti, X_idx = beyes_run(X_scale, y_scale, X_design_scale, batch_size=5)

    bayes.visualize_design_space(X_design_scale, X_idx, figpath=figpath)

    # Inverse transform
    X_opti, (mu_opti, sigma_opti), _ = param_tran.inverse_transform(X_opti, mu_opti, sigma_opti)

    if isinstance(log_indices, list):
        for i in log_indices:
            X_opti[:, i] = np.power(10, X_opti[:, i])

    data = np.concatenate([X_opti, mu_opti, sigma_opti], axis=1)
    df = pd.DataFrame(data, columns=param_names + ['mu', 'sigma'])
    df.to_csv(next_param_path)


def draw_comics_map(
        X, y,
        init_index: int,
        batch_size: int,
        param_range: np.ndarray = None,
        param_names=None,
        mesh_counts: int = 20,
        log_indices=None,
        figpath_dir=None,
        emb_method=TSNE()
):
    """"""
    assert X.shape[0] == y.shape[0] > init_index

    param_tran = ParamPreprocessor(
        param_range=param_range,
        param_names=param_names,
        logX_indices=log_indices,
        param_mesh_counts=mesh_counts
    )
    param_tran.fit(X, y)
    X_design = param_tran.get_X_design()
    X_scale, y_scale, X_design_scale = param_tran.transform(X, y, X_design)
    emb_X_design = emb_method.fit_transform(X_design_scale)

    list_emb_x, mus, sigmas, opti_X_idx = [], [], [], []
    for iter_num, idx in enumerate(range(init_index, X.shape[0]+batch_size, batch_size), 1):
        X_batch, y_batch = X_scale[:idx], y_scale[:idx]
        bayes, X_opti, mu_opti, sigma_opti, X_idx = beyes_run(X_batch, y_batch, X_design_scale, batch_size=5)

        figpath = Path(figpath_dir).joinpath(f'{iter_num}.png') if figpath_dir else None
        emb_x, mu, sigma = bayes.visualize_design_space(
            X_design_scale, X_idx.detach().numpy(),
            # figpath=figpath,
            emb_method=None,
            emb_x=emb_X_design,
            y_scaler=param_tran.yscaler
        )
        list_emb_x.append(emb_x)
        mus.append(mu)
        sigmas.append(sigma)
        opti_X_idx.append(X_idx.detach().numpy())

    mu_norm = min(mu.min() for mu in mus), max(mu.max() for mu in mus)
    sigma_norm = min(sig.min() for sig in sigmas), max(sig.max() for sig in sigmas)

    for i, (emb_x, mu, sigma, X_idx) in enumerate(zip(list_emb_x, mus, sigmas, opti_X_idx)):
        bm = BayesDesignSpaceMap(
            emb_x, mu, sigma, X_idx,
            mu_norm=mu_norm, sigma_norm=sigma_norm,
            cmap_mu='viridis', cmap_sigma='Grays',
            superscript=False
        )
        fig, axs = bm()
        fig.savefig(Path(figpath_dir).joinpath(f'comics_{i}.png'))

    beyes_map = BayesDesignSpaceMap(list_emb_x, mus, sigmas, opti_X_idx, cmap='viridis')
    fig, axs = beyes_map()
    fig.savefig(Path(figpath_dir).joinpath(f'comics.png'))


class ParamPreprocessor:
    """
    Preprocessor for optimized parameters.
        1) scale the raw parameter values in a linear or logarithmic space.
        2) get a meshed parameter design space according to given parameters space.
    """
    def __init__(
            self,
            scaler: Callable = MinMaxScaler(),
            yscaler=MinMaxScaler((0, 10)),
            param_range: Union[torch.Tensor, np.ndarray] = None,
            param_names: list[str] = None,
            param_mesh_counts: int = 20,
            logX_indices: Union[int, list[int]] = None,
    ):
        """
        Args:
            scaler:
            yscaler:
            param_range:
            param_names:
            param_mesh_counts:
            logX_indices:
        """
        self.scaler = scaler
        self.yscaler = yscaler

        if param_range is not None and param_range.shape[1] != 2:
            raise ValueError('the length of param_range in dimension 1 should be 2')
        self.param_range = param_range

        if param_names:
            assert len(param_names) == len(param_range)
        self.param_names = param_names

        self.param_mesh_counts = param_mesh_counts

        if isinstance(logX_indices, int):
            self.logX_indices = [logX_indices]
        else:
            self.logX_indices = logX_indices

    def get_X_design(self, to_log=True):
        """ Get the design parameters according to the parameter range """
        if not isinstance(self.param_range, (torch.Tensor, np.ndarray)):
            raise AttributeError('the param_ranges are not given, cannot get X_design')

        param_range = self.param_range
        if to_log and isinstance(self.logX_indices, list):
            for i in self.logX_indices:
                param_range[i] = np.log10(param_range[i])

        param_axis = []
        for p_range in param_range:
            param_axis.append(torch.linspace(*p_range, self.param_mesh_counts))

        param_meshgrid = torch.meshgrid(param_axis)
        X_design = torch.vstack([pm.flatten() for pm in param_meshgrid]).T

        return X_design

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        self.scaler.fit(X)
        if y is not None:
            self.yscaler.fit(y.reshape([y.shape[0], 1]))

    def transform(self, X: torch.Tensor, y: torch.Tensor = None, X_design: torch.Tensor = None):
        """"""
        if isinstance(X_design, torch.Tensor):
            assert X_design.shape[1] == X.shape[1]

        if isinstance(self.logX_indices, list):
            for i in self.logX_indices:
                X[:, i] = torch.log10(X[:, i])

        X = self.scaler.transform(X)
        y = self.yscaler.transform(y.reshape(-1, 1)).flatten()

        if X_design is not None:
            X_design = self.scaler.transform(X_design)

        return X, y, X_design

    @staticmethod
    def to_numpy(*tensors: torch.Tensor):
        return (t.detach().numpy() if isinstance(t, torch.Tensor) else np.array(t) for t in tensors)

    @staticmethod
    def to_tensor(*arrays: np.ndarray):
        return (torch.tensor(a) for a in arrays)

    def inverse_transform(
            self,
            X: Union[np.ndarray, torch.Tensor],
            *ys: Union[np.ndarray, torch.Tensor],
            X_design=None
    ):
        X = next(self.to_numpy(X))
        ys = self.to_numpy(*ys)

        X = self.scaler.inverse_transform(X)
        ys = [self.yscaler.inverse_transform(y.reshape([y.shape[0], 1])) for y in ys]

        if isinstance(self.logX_indices, list):
            for i in self.logX_indices:
                X[:, i] = torch.log10(X[:, i])

        if X_design is not None:
            X_design = self.scaler.inverse_transform(X_design)
            for i in self.logX_indices:
                X_design = torch.log10(X_design[:, i])

        return X, ys, X_design


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    df = pd.read_excel('/mnt/c/Users/zhang/OneDrive/Papers/COF/data.xlsx', index_col=0)
    X = torch.tensor(df.iloc[:, :3].values)
    X[:, 1] = torch.log10(X[:, 1])
    y = torch.tensor(df.iloc[:, 3].values)

    temp_range = [X[:, 0].min(), X[:, 0].max()]
    log_ratio_range = [X[:, 1].min(), X[:, 1].max()]
    equiv_range = [X[:, 2].min(), X[:, 2].max()]

    temp_design_range = [-20., 150.]
    log_ratio_design_range = [-1., 1]
    equiv_design_range = [0.001, 0.1]

    temp_space = torch.linspace(*temp_design_range, 20)
    log_ratio_space = torch.linspace(*log_ratio_design_range, 20)
    equiv_space = torch.linspace(*equiv_design_range, 20)

    temp_space, log_ratio_space, equiv_space = torch.meshgrid([temp_space, log_ratio_space, equiv_space])

    X_design = torch.vstack([temp_space.flatten(), log_ratio_space.flatten(), equiv_space.flatten()]).T

    scaler = MinMaxScaler()
    yscaler = MinMaxScaler(feature_range=(0., 10.))

    scaler.fit(X)
    yscaler.fit(y.reshape([y.shape[0], 1]))

    X_scale = torch.from_numpy(scaler.transform(X))
    y_scale = torch.from_numpy(yscaler.transform(y.reshape([y.shape[0], 1]))).flatten()

    X_design_scale = scaler.transform(X_design)

    gp = GaussianProcess(X_scale, y_scale, covar_module=ScaleKernel(RBFKernel(ard_num_dims=3)))
    bayes = BayesianOptimizer(gp, batch_size=10)
    X_opti, mu_opti, sigma_opti, X_opti_idx = bayes(X_design_scale, 300)

    X_opti = scaler.inverse_transform(X_opti)
    X_opti[:, 1] = np.power(10, X_opti[:, 1])
    mu_opti = yscaler.inverse_transform(mu_opti.detach().numpy().reshape([mu_opti.shape[0], 1]))
    sigma_opti = yscaler.inverse_transform(sigma_opti.detach().numpy().reshape([sigma_opti.shape[0], 1]))

    data = np.concatenate([X_opti, mu_opti, sigma_opti], axis=1)
    df = pd.DataFrame(data, columns=['temp', 'ratio', 'cata. Equiv.', 'mu', 'sigma'])
    df.to_csv('/mnt/c/Users/zhang/OneDrive/Papers/COF/result2.csv')
    # bayes.visualize_design_space(X_design_scale, figpath='/home/zz1/proj/cof/data/vis.png', emb_method=TSNE())
    bayes.visualize_design_space(X_design_scale, X_opti_idx, emb_method=TSNE())
