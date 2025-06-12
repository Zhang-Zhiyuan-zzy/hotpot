# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : curve
 Created   : 2025/6/12 8:56
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def curve_interpolate(X, Y, method=PchipInterpolator, num: int = 10000):
    m = method(X, Y)

    x = np.linspace(min(X), max(X), num)
    y = m(x)

    return x, y


def plot_piecewise_colored_curve(
    X, Y,
    segment_bounds=None,
    color_cycle=None,
    show_points=False,
    xi=None, yi=None,
    point_kwargs=None,
    curve_kwargs=None,
    xlabel="x", ylabel="y", title=None,
    figsize=(10,7),
    axis_label_fontsize=14,
    axis_label_weight="regular",
    xtick_fontsize=12,
    ytick_fontsize=16,
    save_path: str = None
):
    """
    分段配色曲线绘图（更大更粗字，无网格）
    """
    if color_cycle is None:
        color_cycle = plt.get_cmap('tab20').colors
    if point_kwargs is None:
        point_kwargs = dict(marker='o', color='k', markersize=8, lw=0, zorder=5)
    if curve_kwargs is None:
        curve_kwargs = dict(linewidth=3, alpha=1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax = plt.gca()
    if segment_bounds is None:
        ax.plot(X, Y, color=color_cycle[0], **curve_kwargs)
    else:
        for j in range(len(segment_bounds)-1):
            i0, i1 = segment_bounds[j], segment_bounds[j+1]
            ax.plot(X[i0:i1], Y[i0:i1], color=color_cycle[j % len(color_cycle)], **curve_kwargs)
    if show_points and xi is not None and yi is not None:
        ax.plot(xi, yi, label="Points", **point_kwargs)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize, weight=axis_label_weight)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize, weight=axis_label_weight)
    # 坐标轴刻度
    ax.tick_params(axis='both', labelsize=xtick_fontsize, width=2)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # [t.set_fontweight('bold') for t in ax.get_xticklabels()]
    # [t.set_fontweight('bold') for t in ax.get_yticklabels()]


    if title:
        fig.title(title, fontsize=axis_label_fontsize+2, weight=axis_label_weight)
    # 去掉网格
    ax.grid(False)
    fig.tight_layout()
    fig.show()

    if save_path is not None:
        fig.savefig(save_path)

if __name__ == '__main__':
    origin_colors = [
        "#0000FF",  # 蓝
        "#228B22",  # 绿
        "#B22222",  # 棕红
        "#FF00FF"  # 品红
    ]


    # 三次样条插值示例
    # xi = [0,   10,   15,   20,   30,    50,   80,   120,  125,  130,  135,  140,  250,  375,  500,  1000, 1500, 2000]
    # yi = [101, 94.4, 94,   93.8, 93.7,  93.6, 93.5, 93.4, 93.2, 93.0, 93.1, 93.4, 94.5, 95.5, 97,   98.0, 98.5, 98.9]
    # yi = [100, 89.4, 88.9, 88.8, 88.7,  88.6, 88.5, 88.4, 88.2, 88.0, 88.1, 88.3, 88.9, 89.5, 90.5, 91.3, 91.5, 91.9] # NN-1@XGKCMU放大未优化
    # yi = [100, 92.4, 91.9, 91.8, 91.7,  91.6, 91.5, 91.3, 91.1, 91.0, 91.1, 91.4, 93.1, 93.9, 95.4, 96.3, 96.6, 97.0]  # NN-1@XGKCMU放大优化

    # 铝基NN-7@BUNKUS
    # name="铝基NN-7@BUNKUS"
    # xi = [0,   10,   15,   20,   30,    50,   80,   120,  125,  130,  135,  140,  250,  375,  500,  1000, 1500, 2000]
    # yi = [100, 84.4, 84.2, 84.1, 84.0,  83.95,83.9, 83.9, 83.8, 84.2, 85.4, 85.9, 87.6, 88.7, 89.7, 91.2, 91.3, 91.4]

    # # 铜基NN-4@XAFFUH
    # xi = [0,   10,   15,   20,   30,    50,    80,   120,  125,  130,  135,  140,  250,  375,  500,  1000, 1500, 2000]
    # yi = [100, 94.4, 94.2, 94.1, 94.0,  93.95, 93.9, 93.7, 93.4, 93.9, 94.1, 94.3, 95.1, 96.0, 96.8, 98.2, 98.9, 99.31]

    # # 铜基NN-4@XAFFUH
    # name = '铜基NN-4@XAFFUH'
    # xi = [0,   10,   15,   20,   30,    50,    80,   120,  125,  130,  135,  140,  250,  375,  500,  1000, 1500, 2000]
    # yi = [100, 90.4, 90.2, 90.1, 90.0,  89.95, 89.9, 89.6, 89.3, 89.7, 90.1, 90.3, 90.9, 91.4, 92.1, 93.6, 94.7, 94.8]

    # RN-5@AWUPOZ
    name = '锆基RN-5@AWUPOZ'
    xi = [0,   10,   15,   20,   30,    50,   80,   120,  125,  130,  135,  140,  250,  375,  500,  1000, 1500,  2000]
    yi = [100, 87.4, 87.2, 87.1, 87.0,  86.95,86.9, 86.9, 86.8, 87.2, 87.7, 88.4, 90.1, 91.2, 92.1, 93.1, 93.15, 93.17]


    # 铁基NN-3@DKIUFA
    # name = '铁基NN-3@DKIUFA'
    # xi = [0,   10,   15,   20,   30,    50,    80,   120,  125,  130,  135,  140,  250,  375,  500,  1000, 1500, 2000]
    # yi = [100, 88.4, 88.2, 88.1, 88.0,  87.95, 87.9, 87.6, 87.0, 87.7, 87.8, 87.9, 89.4, 89.6, 89.7, 89.71,89.70,89.71]

    # name = '镍基RN-5@VAGMEX'
    # xi = [0,   10,   15,   20,   30,    50,    80,   120,  125,  130,  135,  140,  250,  375,  500,  1000, 1500, 2000]
    # yi = [100, 92.4, 92.2, 92.1, 92.0,  91.90, 91.9, 91.4, 91.0, 91.2, 91.8, 92.3, 93.7, 95.1, 96.7, 97.8, 98.3, 98.8]

    X, Y = curve_interpolate(xi, yi)

    # 自定义分段点(以X中实际坐标分堆,例如以350,800,1500为分界)
    seg_x = [0, 12.5, 115, 130, 2000]
    idx_bounds = [np.searchsorted(X, xx, side='left') for xx in seg_x]
    plot_piecewise_colored_curve(
        X, Y,
        segment_bounds=idx_bounds,
        color_cycle=origin_colors,
        show_points=False,
        xi=xi, yi=yi,
        # title="Piecewise Colored Spline",
        xlabel="Time t (min)",
        ylabel="Weight (%)",
        save_path=f'/mnt/d/zhang/OneDrive/Desktop/{name}.png'
    )
