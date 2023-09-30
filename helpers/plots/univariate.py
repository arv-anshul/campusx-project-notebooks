from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from helpers import constants as C

_QQDist = Literal["norm", "uniform", "log"]
_WIDTH = 100


def univariate_eda(
    data: pd.Series,
    describe_plot: str = ...,
    *,
    visualise: bool = True,
    compare: bool = False,
    ecdf: bool = False,
    qqplot_kw: dict[Literal["dist"], _QQDist] | Literal[False] = False,
):
    if isinstance(describe_plot, str):
        print("+" * _WIDTH)
        print(describe_plot.center(_WIDTH))
        print("+" * _WIDTH)

    if visualise:
        visualise_feature(data)
    if compare:
        compare_plot(data)
    if ecdf:
        ecdf_plot(data)
    if qqplot_kw:
        qqplot(data, **qqplot_kw)


def visualise_feature(data: pd.Series) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    fig.suptitle(f"Visualize {data.name}", fontsize=18)

    sns.boxplot(x=data, ax=ax1).set_title("Boxplot")
    sns.histplot(data, kde=True, ax=ax2).set_title("Histplot")
    sns.stripplot(data, ax=ax3).set_title("Stripplot")

    plt.tight_layout()
    plt.show()


def qqplot(data: pd.Series, dist: _QQDist) -> None:
    if dist == "norm":
        theoretical_dist = stats.norm
    elif dist == "uniform":
        theoretical_dist = stats.uniform
    elif dist == "log":
        data = pd.Series(np.log1p(data), name=data.name)
        theoretical_dist = stats.norm
    else:
        raise ValueError

    fig = sm.qqplot(data, theoretical_dist, line="45")  # type: ignore
    fig.suptitle(f"{dist.title()} QQ-Plot of {data.name}")
    plt.show()


def compare_plot(data: pd.Series) -> None:
    data_log = np.log1p(data)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    fig.suptitle(f"Compare plot of {data.name}", fontsize=18)

    org_dist_str = "Original Distribution"
    trf_dist_str = "Log Transformed Distribution"

    # HistPlot
    sns.histplot(data, kde=True, color=C.SKY_BLUE, ax=ax1)
    ax1.set_title(org_dist_str)

    sns.histplot(data_log, kde=True, color=C.LIGHT_GREEN, ax=ax2)
    ax2.set_title(trf_dist_str)

    # BoxPlot
    sns.boxplot(x=data, color=C.SKY_BLUE, ax=ax3)
    sns.boxplot(x=data_log, color=C.LIGHT_GREEN, ax=ax4)

    plt.tight_layout()
    plt.show()


def ecdf_plot(data: pd.Series) -> None:
    ecdf = data.value_counts().sort_index().cumsum().div(data.shape[0])
    sns.lineplot(x=ecdf.index, y=ecdf, marker="o", linestyle="none")

    plt.xticks(rotation="vertical")
    plt.title(f"ECDF plot of {data.name}")
    plt.show()
