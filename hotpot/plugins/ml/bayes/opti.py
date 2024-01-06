"""
python v3.9.0
@Project: hotpot
@File   : opti
@Auther : Zhiyuan Zhang
@Data   : 2024/1/3
@Time   : 11:27
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel


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


class BayesianOptimization:
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
        X_design = torch.tensor(X_design)

        logging.info('\n'.join([f'{name}, {p}' for name, p in self.surrogate.named_parameters()]))
        train_x, train_y = self.surrogate.train_inputs[0], self.surrogate.train_targets
        logging.info('\n'.join([f'{name}, {p}' for name, p in self.surrogate.named_parameters()]))

        X_optimal, mu_optimal, sigma_optimal = [], [], []
        for c in range(self.batch_size):
            self.train(n_iter=n_iter, lr=lr)
            f_design = self.surrogate(X_design)
            mu = f_design.mean
            sigma2 = f_design.variance
            sigma = torch.sqrt(sigma2)

            acq_value = self.acq_func(mu, sigma, train_y.max())

            # Find best point to include
            i = torch.argmax(acq_value)
            X_optimal.append(X_design[i])
            mu_optimal.append(mu[i])
            sigma_optimal.append(sigma[i])

            train_x = torch.from_numpy(torch.vstack([train_x, X_optimal[-1]]).detach().numpy())
            train_y = torch.from_numpy(torch.hstack([train_y, mu_optimal[-1]]).detach().numpy())
            self.surrogate.set_train_data(train_x, train_y, strict=False)

        return torch.stack(X_optimal), torch.stack(mu_optimal), torch.stack(sigma_optimal)

    def train(self, n_iter=100, lr=0.1):
        """Train the model.

        Arguments
        n_iter  --  The number of iterations.
        """
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
            if (i + 1) % 1 == 0:
                print(f'Iter {i + 1:3d}/{n_iter} - Loss: {loss.item():.3f}')
        self.surrogate.eval()

        self.is_trained = True

    def visualize_design_space(self, X_design, n_iter=150, lr=0.1, emb_method=TSNE(), figpath=None):
        if not self.is_trained:
            self.train(n_iter, lr)

        f_design = self.surrogate(torch.from_numpy(X_design))
        m = f_design.mean.detach().numpy()
        sigma2 = f_design.variance
        sigma = torch.sqrt(sigma2).detach().numpy()

        emb_x = emb_method.fit_transform(X_design)

        fig, axs = plt.subplots(1, 2)
        fig.set(figheight=6.4, figwidth=14)
        cmap = plt.colormaps["plasma"]

        axs[0].scatter(emb_x[:, 0], emb_x[:, 1], c=m, alpha=0.3)
        axs[1].scatter(emb_x[:, 0], emb_x[:, 1], c=sigma, alpha=0.3)
        # plt.colorbar()
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axs[0])
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axs[1])

        fig.show()
        if figpath:
            fig.savefig(figpath)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    df = pd.read_excel('/home/zz1/proj/cof/data/data.xlsx', index_col=0)
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
    bayes = BayesianOptimization(gp, batch_size=5)
    X_opti, mu_opti, sigma_opti = bayes(X_design_scale)

    X_opti = scaler.inverse_transform(X_opti)
    X_opti[:, 1] = np.power(10, X_opti[:, 1])
    mu_opti = yscaler.inverse_transform(mu_opti.detach().numpy().reshape([mu_opti.shape[0], 1]))
    sigma_opti = yscaler.inverse_transform(sigma_opti.detach().numpy().reshape([sigma_opti.shape[0], 1]))

    data = np.concatenate([X_opti, mu_opti, sigma_opti], axis=1)
    df = pd.DataFrame(data, columns=['temp', 'ratio', 'cata. Equiv.', 'mu', 'sigma'])
    df.to_csv('/home/zz1/proj/cof/data/result.csv')
    # bayes.visualize_design_space(X_design_scale, figpath='/home/zz1/proj/cof/data/vis.png', emb_method=TSNE())
