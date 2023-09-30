import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def count_plot(df: pd.DataFrame, columns: list[str], title: str = ...):
    num_plots = len(columns)

    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.tight_layout(pad=2)

    if isinstance(title, str):
        fig.suptitle(title)

    for i, col in enumerate(columns):
        ax: plt.Axes = axes[i // cols, i % cols] if num_plots > 1 else axes  # type: ignore

        sns.countplot(data=df, x=col, ax=ax)
        ax.set_ylabel("")

    plt.show()


def agg_plot(
    df: pd.DataFrame,
    x: list[str],
    y: list[str],
    agg: list[str] = ["mean"],
) -> None:
    num_plots = len(x) * len(y)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))

    for method in agg:
        plot_count = 0

        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        fig.suptitle(f"Method: {method.title()}", fontsize=15)

        for x_col in x:
            for y_col in y:
                ax: plt.Axes = (
                    axes[plot_count // cols, plot_count % cols]
                    if num_plots > 1
                    else axes
                )  # type: ignore

                sns.barplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    estimator=method,
                    ax=ax,
                    errorbar=("ci", 0),
                )
                ax.set_xlabel(x_col)

                plot_count += 1

    plt.tight_layout()
    plt.show()
