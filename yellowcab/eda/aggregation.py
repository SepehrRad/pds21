import pandas as pd


def agg_stats(
        group_series,
        data_series,
        agg_functions=None,
):
    """
    This function calculates (user specified) aggregated statistics of two input series & returns them as DataFrame.
    ----------------------------------------------
    :param
        group_series(pd.Series): Data you want to group by.
        data_series(pd.Series): Data you want aggregated.
        agg_functions(list[String] or String): Aggregates data based on given functions. Default is
        ["min", "max", "mean", "median","std", "var", "sem"]. For more information use pandas groupby documentation.
    :returns
        pd.DataFrame: DataFrame containing aggregated statistics.
    """
    if agg_functions is None:
        agg_functions = ["min", "max", "mean", "median", "std", "var", "sem"]
    time_series_name = group_series.name
    data_series_name = data_series.name
    df = pd.concat([group_series, data_series], axis=1)
    df = df.groupby(time_series_name)[data_series_name].agg(agg_functions)
    if isinstance(df, pd.Series):
        df = df.to_frame(name=agg_functions)
    df = df.add_suffix("_" + data_series_name)
    return df


def describe_stats(group_series, data_series):
    """
    This function calculates a set of aggregated statistics of two input series & returns them as DataFrame.
    ----------------------------------------------
    :param
        group_series(pd.Series): Data you want to group by.
        data_series(pd.Series): Data you want aggregated.
    :returns
        pd.DataFrame: DataFrame containing aggregated statistics.
    """
    group_series_name = group_series.name
    data_series_name = data_series.name
    df = pd.concat([group_series, data_series], axis=1)
    df = df.groupby(group_series_name)[data_series_name].describe()
    df = df.add_suffix("_" + data_series_name)
    return df
