import pandas as pd
from matplotlib import pyplot as plt


def null_plot(df: pd.DataFrame):
    ax = (
        df.isnull()
        .sum()
        .div(len(df))
        .mul(100)
        .add(0.5)
        .round()
        .plot.bar(ylabel="Null Values (in %)", ylim=(0, 100), figsize=(12, 4))
    )

    for bar in ax.patches:
        plt.text(
            x=(bar.get_x() + (bar.get_width() // 2)),
            y=bar.get_height() + 2.5,
            s=str(round(bar.get_height())),
            rotation=90,
            fontsize=12,
        )
    plt.show()
