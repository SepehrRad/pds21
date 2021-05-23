import warnings
import numpy as np
import pandas as pd
import calendar
from pyod.models.hbos import HBOS
from scipy.stats import zscore
from yellowcab.io.input import read_geo_dataset
from yellowcab.io.utils import get_data_path
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from yellowcab.io.input import read_parquet
from yellowcab.io.output import write_parquet

warnings.filterwarnings("ignore")

column_description = {
    "cyclical_features": [
        "start_month",
        "start_day",
        "start_hour",
        "end_hour",
        "end_day",
        "end_month",
    ],
    "categorical_features": ["RatecodeID", "payment_type"],
    "temporal_features": ["pickup_datetime", "dropoff_datetime"],
    "spatial_features": ["PULocationID", "DOLocationID"],
}


def _get_date_components(df):
    """
    This function splits the pickup- and dropoff-time columns into 3 columns each, divided by month, day and hour &
    returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    """
    for col in column_description.get("temporal_features"):
        name = col[:-9]
        df[f"{name}_month"] = df[col].dt.month
        df[f"{name}_day"] = df[col].dt.day
        df[f"{name}_hour"] = df[col].dt.hour


def _is_weekend(df):
    """
    This function adds a 'weekend'-column to the given DataFrame. It indicates whether the the trip is on the
    weekend (TRUE) or not (FALSE).

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    """
    # get day of week from pickup
    df["weekend"] = df["pickup_datetime"].dt.dayofweek > 4


def _get_duration(df):
    """
    This function adds a 'trip_duration_minutes'-column to the given DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    """
    df["trip_duration_minutes"] = (
                                          df["dropoff_datetime"] - df["pickup_datetime"]
                                  ).dt.total_seconds() / 60


def _replace_ids(df):
    """
    This function replaces the IDs of 'rate_id_dict' and 'payment_type_dict' & returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    rate_id_dict = {
        1.0: "Standard rate",
        2.0: "JFK",
        3.0: "Newark",
        4.0: "Nassau/Westchester",
        5.0: "Negotiated fare",
        6.0: "Group ride",
    }
    payment_type_dict = {
        1.0: "Credit card",
        2.0: "Cash",
        3.0: "No charge",
        4.0: "Dispute",
        5.0: "Unknown",
        6.0: "Voided trip",
    }
    df["RatecodeID"].replace(rate_id_dict, inplace=True)
    df["payment_type"].replace(payment_type_dict, inplace=True)
    df = df[pd.to_numeric(df["RatecodeID"], errors="coerce").isna()]
    df = df[pd.to_numeric(df["payment_type"], errors="coerce").isna()]
    return df


def _set_column_types(df):
    """
    Sets the column types for the given DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    categorical_cols = column_description.get("categorical_features")
    spatial_cols = column_description.get("spatial_features")
    df[categorical_cols] = df[categorical_cols].astype("category")
    df[spatial_cols] = df[spatial_cols].astype("str")


def _remove_invalid_numeric_data(df, verbose=False):
    """
    This functions removes negative (faulty) numeric values from the given DataFrame & returns the processed DataFrame.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
        verbose(boolean): Set 'True' to get detailed logging information.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    df_numeric_view = df.select_dtypes(include="number")
    sum_negative_entries = 0
    for col in df_numeric_view.columns:
        if col in ["passenger_count", "trip_distance"]:
            negative_entries = df[df[col] <= 0]
        else:
            negative_entries = df[df[col] < 0]
        if verbose:
            print(f"--> {negative_entries.shape[0]} invalid entries found in {col}")
        sum_negative_entries += negative_entries.shape[0]
        df.drop(negative_entries.index, inplace=True)
    print(f"{sum_negative_entries} invalid entries have been successfully dropped!")
    return df.reset_index(drop=True)


def _remove_outliers(
    df,
    density_sensitive_cols,
    excluded_cols=None,
    n_bins=10,
    zscore_threshold=4.5,
    verbose=False,
    contamination=0.1,
    tol=0.5,
    alpha=0.1,
):
    """
    This functions removes outliers by applying two different algorithms on specific columns:

    - outliers in density sensitive columns get detected by 'zscore'-algorthm.
    - outliers in other columns get detected by 'HBOS'-algorithm.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
        density_sensitive_cols(list): Columns to run 'zscore'-algorithm on.
        excluded_cols(list): Columns without outlier detection.
        n_bins(int): Hyperparameter for 'HBOS'-algorithm.
        zscore_threshold(float): Hyperparameter for 'zscore'-algorithm.
        verbose(boolean): Set 'True' to get detailed logging information.
        contamination(float): Hyperparameter for 'HBOS'-algorithm.
        tol(float): Hyperparameter for 'HBOS'-algorithm.
        alpha(float): Hyperparameter for 'HBOS'-algorithm.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    if n_bins == "auto":
        n_bins = int(
            1 + 3.322 * np.log(df.shape[0])
        )  # Sturge’s Rule for detecting bin numbers automatically

    outlier_count = 0
    df_numeric_view = df.select_dtypes(include="number")

    for col in df_numeric_view.columns:
        if excluded_cols and col in excluded_cols:
            continue
        if col in density_sensitive_cols:
            df[f"{col}_zscore"] = np.around(np.abs(zscore(df[col])), decimals=1)
            outlier = df[df[f"{col}_zscore"] > zscore_threshold]
            outlier_count += outlier.shape[0]
            df.drop(outlier.index, inplace=True)
            if verbose:
                print(
                    f"--> {outlier.shape[0]} outlier detected and removed from {col} column using zscore"
                )
            continue
        hbos = HBOS(alpha=alpha, contamination=contamination, n_bins=n_bins, tol=tol)
        hbos.fit(df[[col]])
        df[f"{col}_anamoly_score"] = hbos.predict(df[[col]])
        outlier = df[df[f"{col}_anamoly_score"] == 1]
        outlier_count += outlier.shape[0]
        df.drop(outlier.index, inplace=True)
        if verbose:
            print(
                f"--> {outlier.shape[0]} outlier detected and removed from {col} column using HBOS algorithm"
            )

    outlier_score_cols_mask = (df.columns.str.contains("anamoly_score")) | (
        df.columns.str.contains("zscore")
    )
    df = df.loc[:, ~outlier_score_cols_mask]

    print(f"Outlier detection completed. Number of removed outlier: {outlier_count}")

    return df.reset_index(drop=True)


def _remove_date_outliers(df, month):
    """
    This function removes trips from the given DataFrame, which do not start in the expected month.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
        month(integer): Month of the year (1 = January, 2 = February...).
    :returns:
        pd.DataFrame: Processed DataFrame.
    """
    date_outliers = df.shape[0]
    df = df[
        ~(
            (df["pickup_datetime"].dt.year != 2020)
            | (df["pickup_datetime"].dt.month != month)
        )
    ]
    date_outliers -= df.shape[0]
    print(f"{date_outliers} invalid date entries have been successfully dropped!")

    return df.reset_index(drop=True)


def _merge_geodata(df):
    """
    This function merges the given Dataframe with the geodata.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
    :returns:
        pd.DataFrame: Merged DataFrame.
    """
    zones_gdf = read_geo_dataset('taxi_zones.geojson')
    zones_gdf['centers_long'] = zones_gdf['geometry'].centroid.x
    zones_gdf['centers_lat'] = zones_gdf['geometry'].centroid.y
    zones_gdf['LocationID'] = zones_gdf['LocationID'].astype('str')
    df_gdf = df.merge(
        zones_gdf[['LocationID', 'centers_lat', 'centers_long']],
        how="left", left_on='PULocationID', right_on='LocationID')
    df_gdf = df_gdf.merge(
        zones_gdf[['LocationID', 'centers_lat', 'centers_long']],
        how="left", left_on='DOLocationID', right_on='LocationID', suffixes=("_pickup", "_dropoff"))
    zone_outliers = df_gdf.shape[0]
    df_gdf.dropna(inplace=True)
    zone_outliers -= df_gdf.shape[0]
    print(f"{zone_outliers} invalid zone entries have been successfully dropped!")
    return df_gdf.reset_index(drop=True)


def clean_dataset(df, month, verbose=False):
    """
    This function combines all functions of this 'cleaning'-class to detect and delete outliers and faulty trips.
    Furthermore the geodata is getting merged.

    ----------------------------------------------

    :param
        df(pd.DataFrame): DataFrame to be processed.
        month(integer): Month of the year (1 = January, 2 = February...).
        verbose(boolean): Set 'True' to get detailed logging information.
    :returns
        pd.DataFrame: Processed DataFrame.
    """
    df.rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "tpep_dropoff_datetime": "dropoff_datetime",
        },
        inplace=True,
    )
    _set_column_types(df)
    df = _remove_invalid_numeric_data(df, verbose=verbose)
    df = _remove_date_outliers(df, month=month)
    df = _merge_geodata(df)
    _get_duration(df)
    df = _replace_ids(df)
    df = _remove_outliers(
        df,
        density_sensitive_cols=[
            "passenger_count",
            "total_amount",
            "trip_duration_minutes",
            "trip_distance",
            "congestion_surcharge",
            "tip_amount",
        ],
        excluded_cols=[
            "centers_long_pickup",
            "centers_lat_pickup",
            "centers_long_dropoff",
            "centers_lat_dropoff",
        ],
        n_bins="auto",
        verbose=verbose,
    )
    _get_date_components(df)
    _is_weekend(df)
    df = df.loc[~(df['payment_type'] == 'Unknown')]
    return df.drop(columns=['LocationID_pickup', 'LocationID_dropoff'])


def clean_all_datasets(base_path=get_data_path(), relative_path="input/trip_data", verbose=False):
    """
    Doc String!
    """
    data_path = join(base_path, relative_path)
    data_sets = [dataset for dataset in listdir(data_path) if isfile(join(data_path, dataset))]
    if not all('.parquet' in name for name in data_sets):
        raise ValueError("The given directory includes non parquet files")
    for parquet_file in tqdm(data_sets):
        # assumes 01.parquet,02.parquet,...
        month = int(parquet_file.split('.parquet')[0])
        print(f'Started cleaning the {calendar.month_name[month]} data set')
        cleaned_df = clean_dataset(read_parquet(parquet_file), month=month, verbose=verbose)
        write_parquet(cleaned_df, filename=f'{calendar.month_name[month]}_cleaned.parquet')
        print(f'Finished cleaning the {calendar.month_name[month]} data set')
