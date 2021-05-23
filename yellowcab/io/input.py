import os
import pickle

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from .utils import get_data_path


def read_parquet(
    file, base_path=get_data_path(), relative_path="input/trip_data", columns=None
):
    """
    This function reads a parquet file & returns it as a pd.DataFrame.
    ----------------------------------------------
    :param
        file(String): Name of file.
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with file in base_path. Defaults to input/trip_data.
        columns(list[String]): Only reads specified from file. Default is None(reads all columns).
        For more information use pyarrow.parquet.read_table documentation.
    :returns
        pd.DataFrame: DataFrame containing data from a single parquet.
    """
    path = os.path.join(base_path, relative_path, file)
    try:
        table = pq.read_table(path, columns=columns)
        df = table.to_pandas()
        df = df.reset_index(drop=True)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_parquet_dataset(
    base_path=get_data_path(), relative_path="input/trip_data", columns=None
):
    """
    This function reads a directory of parquet files & returns them as a single pd.DataFrame.
    ----------------------------------------------
    :param
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with parquet dataset in base_path. Defaults to input/trip_data.
        columns(list[String]): Only reads specified from file. Default is None(reads all columns).
        For more information use pyarrow.parquet.read_table documentation.
    :returns
        pd.DataFrame: DataFrame containing data from all parquet files in path.
    """
    path = os.path.join(base_path, relative_path)
    try:
        dataset = pq.ParquetDataset(path)
        table = dataset.read(columns=columns)
        df = table.to_pandas()
        df = df.reset_index(drop=True)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_parquet_sample(
    file,
    base_path=get_data_path(),
    relative_path="input/trip_data",
    columns=None,
    frac=0.1,
):
    """
    This function reads a parquet file & returns a random data sample as a pd.DataFrame.
    ----------------------------------------------
    :param
        file(String): Name of file.
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with file in base_path. Defaults to input/trip_data.
        columns(list[String]): Only reads specified from file. Default is None(reads all columns).
        For more information use pyarrow.parquet.read_table documentation.
        frac(float): Sample size in % (0.0 - 1.0). Default is 0.1.
    :returns
        pd.DataFrame: DataFrame containing a random data sample from a single parquet file.
    """
    df = read_parquet(
        file=file, base_path=base_path, relative_path=relative_path, columns=columns
    )
    df = df.sample(frac=frac).reset_index(drop=True)
    return df


def read_parquet_dataset_sample(
    base_path=get_data_path(), relative_path="input/trip_data", columns=None, frac=0.1
):
    """
    This function reads a directory of parquet files & returns a random data sample in a single pd.DataFrame.
    ----------------------------------------------
    :param
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with parquet dataset in base_path. Defaults to input/trip_data.
        columns(list[String]): Only reads specified from file. Default is None(reads all columns).
        For more information use pyarrow.parquet.read_table documentation.
        frac(float): Sample size in % (0.0 - 1.0). Default is 0.1.
    :returns
        pd.DataFrame: DataFrame containing a random data sample from all parquet files in path.
    """
    df = read_parquet_dataset(
        base_path=base_path, relative_path=relative_path, columns=columns
    )
    df = df.sample(frac=frac).reset_index(drop=True)
    return df


def read_geo_dataset(
    geojson_file, base_path=get_data_path(), relative_path="input/taxi_zones"
):
    """
    This function reads a geojson and an associated csv file & returns it as a pd.DataFrame.
    ----------------------------------------------
    :param
        csv_file(String): Name of csv file.
        geojson_file(String): Name of geojson file.
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with file in base_path. Defaults to input/trip_data.
    :returns
        pd.DataFrame: DataFrame containing data from both input files.
    """
    geojson_path = os.path.join(base_path, relative_path, geojson_file)
    try:
        nyc_zones = gpd.read_file(geojson_path)
        nyc_zones.crs = "epsg:2263"
        nyc_zones = nyc_zones.to_crs("EPSG:4326")
    except FileNotFoundError:
        print("Data file not found. Path was " + geojson_path)
    return nyc_zones


def read_model(name="model.pkl"):
    path = os.path.join(get_data_path(), "output", name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
