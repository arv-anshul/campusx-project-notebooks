import pandas as pd


def custom_describe(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional summary statistics for a numeric feature.

    Parameters:
    -----------
        data (pd.Series or pd.DataFrame): The input data for which to calculate custom statistics.

    Returns:
    --------
        pd.DataFrame: A DataFrame containing custom statistics for the input data.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()

    describe_stats = []
    for col in data.columns:
        d: pd.Series = data[col]

        skew = d.skew()
        kurt = d.kurt()
        notnull = d.notnull().sum()
        isnull = d.isnull().sum()

        describe_stats.append([skew, kurt, notnull, isnull])

    columns = ["skew", "kurtosis", "notnull", "isnull"]
    more_describe = pd.DataFrame(describe_stats, index=data.columns, columns=columns).T

    return pd.concat(
        [
            data.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]),
            more_describe,
        ],
    ).round(3)


def compare_describe(
    data1: pd.Series,
    data2: pd.Series,
    *,
    reciprocal_ratio: bool = False,
) -> pd.DataFrame:
    desc1 = custom_describe(data1).iloc[:, 0].rename(f"{data1.name} as data1")
    desc2 = custom_describe(data2).iloc[:, 0].rename(f"{data2.name} as data2")

    sub_desc = desc1.sub(desc2).rename("data1 - data2")
    ratio_desc = desc1.div(desc2).rename("data1 ÷ data2")

    # Default return DataFrame consist
    to_concat = [desc1, desc2, sub_desc, ratio_desc]

    if reciprocal_ratio:
        to_concat.append(desc2.div(desc1).rename("data2 ÷ data1"))

    desc_df = pd.concat(to_concat, axis=1)
    return desc_df
