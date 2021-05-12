import numpy as np
import pandas as pd
import datetime
from pyod.models.hbos import HBOS
from scipy.stats import zscore

column_description = {
    'cyclical_features': ['start_month', 'start_day', 'start_hour', 'end_hour', 'end_day', 'end_month'],
    'categorical_features': ['RatecodeID', 'payment_type'],
    'temporal_features': ['pickup_datetime', 'dropoff_datetime'],
    'spatial_features': ['PULocationID', 'DOLocationID']}


def __get_date_components(df):
    """
    This function splits the pickup- and dropoff-time columns into 3 columns each, divided by month, day and hour &
    returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    """
    for col in column_description.get('temporal_features'):
        name = col[:-9]
        df[f'{name}_month'] = df[col].dt.month
        df[f'{name}_day'] = df[col].dt.day
        df[f'{name}_hour'] = df[col].dt.hour


def __is_weekend(df):
    """
    This function adds a "weekend"-column to the given DataFrame. It indicates whether the the trip is on the
    weekend (TRUE) or not (FALSE) & returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    # get day of week from pickup
    df['weekend'] = df['pickup_datetime'].dt.dayofweek > 4


def __get_duration(df):
    """
    This function adds a "trip_duration_minutes"-column to the given DataFrame & returns the processed DataFrame

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    df['trip_duration_minutes'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60


def __replace_ids(df):
    """
    This function replaces the IDs of "rate_id_dict" and "payment_type_dict" & returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    rate_id_dict = {1.0: 'Standard rate', 2.0: 'JFK', 3.0: 'Newark', 4.0: 'Nassau/Westchester', 5.0: 'Negotiated fare',
                    6.0: 'Group ride'}
    payment_type_dict = {1.0: 'Credit card', 2.0: 'Cash', 3.0: 'No charge', 4.0: 'Dispute', 5.0: 'Unknown',
                         6.0: 'Voided trip'}
    df['RatecodeID'].replace(rate_id_dict, inplace=True)
    df['payment_type'].replace(payment_type_dict, inplace=True)
    df = df[pd.to_numeric(df['RatecodeID'], errors='coerce').isna()]
    df = df[pd.to_numeric(df['payment_type'], errors='coerce').isna()]
    return df


def __set_column_types(df):
    """
    Sets the column types for the given DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    categorical_cols = column_description.get('categorical_features')
    spatial_cols = column_description.get('spatial_features')
    df[categorical_cols] = df[categorical_cols].astype('category')
    df[spatial_cols] = df[spatial_cols].astype('str')


def __remove_invalid_numeric_data(df, verbose=False):
    df_numeric_view = df.select_dtypes(include='number')
    sum_negative_entries = 0
    for col in df_numeric_view.columns:
        if col in ['passenger_count', 'trip_distance']:
            negative_entries = df[df[col] <= 0]
        else:
            negative_entries = df[df[col] < 0]
        if verbose:
            print(f'--> {negative_entries.shape[0]} invalid entries found in {col}')
        sum_negative_entries += negative_entries.shape[0]
        df.drop(negative_entries.index, inplace=True)
    print(f'{sum_negative_entries} invalid entries have been successfully dropped!')
    return df.reset_index(drop=True)


def __remove_outliers(df, density_sensitive_cols, excluded_cols=None, n_bins=10,
                      zscore_threshold=4.5, verbose=False, contamination=0.1, tol=0.5, alpha=0.1):
    if n_bins == 'auto':
        n_bins = int(1 + 3.322 * np.log(df.shape[0]))  # Sturge’s Rule for detecting bin numbers automatically

    outlier_count = 0
    df_numeric_view = df.select_dtypes(include='number')

    for col in df_numeric_view.columns:
        if excluded_cols and col in excluded_cols:
            continue
        if col in density_sensitive_cols:
            df[f'{col}_zscore'] = np.around(np.abs(zscore(df[col])), decimals=1)
            outlier = df[df[f'{col}_zscore'] > zscore_threshold]
            outlier_count += outlier.shape[0]
            df.drop(outlier.index, inplace=True)
            if verbose:
                print(f'--> {outlier.shape[0]} outlier detected and removed from {col} column using zscore')
            continue
        hbos = HBOS(alpha=alpha, contamination=contamination, n_bins=n_bins, tol=tol)
        hbos.fit(df[[col]])
        df[f'{col}_anamoly_score'] = hbos.predict(df[[col]])
        outlier = df[df[f'{col}_anamoly_score'] == 1]
        outlier_count += outlier.shape[0]
        df.drop(outlier.index, inplace=True)
        if verbose:
            print(f'--> {outlier.shape[0]} outlier detected and removed from {col} column using HBOS algorithm')

    outlier_score_cols_mask = (df.columns.str.contains('anamoly_score')) | (df.columns.str.contains('zscore'))
    df = df.loc[:, ~outlier_score_cols_mask]

    print(f'Outlier detection completed. Number of removed outlier: {outlier_count}')

    return df.reset_index(drop=True)


def __remove_date_outliers(df, month, verbose=False):
    """

    ----------------------------------------------

    :param
        df(pd.DataFrame):
        month(integer):
        verbose(boolean):
    :returns:
        pd.DataFrame:
    """
    day = __get_days_per_month(month=month)
    early_outliers = df[df['pickup_datetime'] < datetime.datetime(2020, month, 1)]
    late_outliers = df[df['pickup_datetime'] > datetime.datetime(2020, month, day, 23, 59, 59)]
    if verbose:
        print(f'--> {early_outliers.shape[0]} earlier date entries found in pickup')
        print(f'--> {late_outliers.shape[0]} later date entries found in pickup')
    df.drop(early_outliers.index, inplace=True)
    df.drop(late_outliers.index, inplace=True)
    date_outliers = early_outliers.shape[0] + late_outliers.shape[0]
    print(f'{date_outliers} invalid date entries have been successfully dropped!')

    return df.reset_index(drop=True)


def __get_days_per_month(month):
    """

    ----------------------------------------------

    :param
        month(integer):
    :returns:
        integer:
    """
    if month == 2:
        return 29
    m = [1, 3, 5, 7, 8, 10, 12]
    if month in m:
        return 31
    return 30


def clean_dataset(df, month, verbose=False):
    """

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'tpep_dropoff_datetime': 'dropoff_datetime'},
              inplace=True)
    __set_column_types(df)
    df = __remove_invalid_numeric_data(df, verbose=verbose)
    df = __remove_date_outliers(df, month=month, verbose=verbose)
    __get_duration(df)
    df = __replace_ids(df)
    df = __remove_outliers(df,
                           density_sensitive_cols=['passenger_count', 'total_amount', 'trip_duration_minutes',
                                                   'trip_distance', 'congestion_surcharge', 'tip_amount'],
                           n_bins='auto',
                           verbose=verbose)
    __get_date_components(df)
    __is_weekend(df)
    return df
