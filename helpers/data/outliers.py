import pandas as pd


def extract_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        .sort_values(by=col, ascending=False)
        .reset_index(drop=True)
    )

    return outliers
