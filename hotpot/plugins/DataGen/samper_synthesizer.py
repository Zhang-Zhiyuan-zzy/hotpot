"""
@File Name:        samper_synthesizer
@Project:          HOTPOT
@Author:           Zhiyuan Zhang
@Created On:       2025/10/31 10:51
@Project:          Hotpot
"""
import numpy as np
from scipy.stats import t


def _y_true_generator(
        y_true=None,
        sample_num: int = 1000,
        low: float = 0, high: float = 1,
        mu: float = 0, sigma: float = 1,
        df: float = 1,
        rand_generator='uniform',
):
    if y_true is None:
        if rand_generator == 'uniform':
            return np.random.uniform(low=low, high=high, size=sample_num)
        elif rand_generator == 'normal':
            return np.random.normal(loc=mu, scale=sigma, size=sample_num)
        elif rand_generator == 't-distributed':
            return t.rvs(df, loc=mu, scale=sigma, size=sample_num)
        else:
            raise ValueError('Unrecognized random generator: {}'.format(rand_generator))
    else:
        return np.asarray(y_true).flatten()


def generate_predictions_with_r2(
        r2,
        sample_num: int = 1000,
        low: float = 0, high: float = 1,
        mu: float = 0, sigma: float = 1,
        df: int = 1.,
        y_true=None,
        rand_generator='uniform',
        random_state=None
):
    """
    Generate predictions y_pred such that R²(y_true, y_pred) ≈ r2.
    """
    if not (0 <= r2 <= 1):
        raise ValueError("R² must be between 0 and 1")

    rng = np.random.default_rng(random_state)

    y_true = _y_true_generator(y_true, sample_num, low, high, mu, sigma, df, rand_generator=rand_generator)
    var_true = np.var(y_true)

    # compute variance of noise
    noise_std = np.sqrt(var_true * (1 - r2) / r2)

    noise = rng.normal(0, noise_std, size=y_true.shape)
    y_pred = y_true + noise
    return y_pred, y_true


if __name__ == '__main__':
    from hotpot.plugins.plots import SciPlotter, R2Regression
    y_pred, y_true = generate_predictions_with_r2(0.78, mu=20, sigma=9, rand_generator='t-distributed', sample_num=150, df=20)
    xy = np.array([y_true, y_pred])

    plotter = SciPlotter(
        R2Regression(xy, c1='darkgreen', marker1='x', s1=150, target_name='True logK', prediction_name='Predicted logK',)
    )
    fig, axs = plotter()
    fig.show()
    fig.savefig('/mnt/d/zhang/OneDrive/Papers/BayesDesign/Illustrating picture/0.53.png')
