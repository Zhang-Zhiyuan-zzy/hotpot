"""
python v3.9.0
@Project: hotpot
@File   : utils
@Auther : Zhiyuan Zhang
@Data   : 2024/10/18
@Time   : 19:08
"""
from typing import *
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, MDS

import matplotlib.pyplot as plt


from hotpot.dataset import load_dataset


def scatter_discriminant_analysis(
        X: np.ndarray,
        y: np.ndarray,
        split_points: Union[np.ndarray, Sequence] = None
):
    """"""
    if split_points is None:
        split_points = [0.5]
    elif isinstance(split_points, np.ndarray):
        split_points = split_points.flatten().tolist()
    elif isinstance(split_points, Sequence):
        split_points = list(split_points)
    else:
        raise TypeError(f"split_points must be a numpy array or a sequence, not {type(split_points)}")

    sorted_indices = np.argsort(y)

    categories = np.zeros_like(y)

    lines = []
    for sp in split_points:
        if 0 < sp < 1:
            sp = int(len(y) * sp)
        else:
            sp = int(sp)

        categories[sorted_indices[:sp]] = 1

        clf = LinearDiscriminantAnalysis()
        clf.fit(X, categories)

        lines.append((clf.coef_.flatten(), clf.intercept_))

    return lines


def plot_scatter_discriminant_lines(X, y, lines, dim_method=TSNE(n_components=2)):
    """"""
    def plot(ax: plt.Axes, sciplotter):
        """"""
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_diff = X_max - X_min

        X_lin = np.linspace(X_min - 0.1 * X_diff, X_max + 0.1 * X_diff, 100)

        ax.scatter(X[:, 0], X[:, 1], c=y)
        for line in lines:
            y_lin = np.dot(line[0], X_lin) + line[1]


if __name__ == "__main__":
    data = load_dataset('logβ1')

    X, y = data[['dG', 'logK']].values, data['SMR_VSA1'].values

    lin = scatter_discriminant_analysis(X, y)

    # clf = LinearDiscriminantAnalysis()
    # clf.fit(X, y)

