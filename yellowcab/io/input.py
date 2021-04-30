import os
import pickle
import pyarrow.parquet as pq
import pandas as pd

from .utils import get_data_path


def read_file(path=os.path.join(get_data_path(), "input/trip_data", "<My_data>.parquet")):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_parquet(file, base_path=get_data_path(), relative_path="input/trip_data"):
    """
    This function reads a parquet file & returns it as a pd.DataFrame.
    ----------------------------------------------
    :param
        file(String): Name of file
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with file in base_path. Defaults to input/trip_data.
    :returns
        pd.DataFrame: DataFrame containing data from a single parquet.
    """
    path = os.path.join(base_path, relative_path, file)
    try:
        table = pq.read_table(path)
        df = table.to_pandas()
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_parquet_dataset(base_path=get_data_path(), relative_path="input/trip_data"):
    """
    This function reads a directory of parquet files & returns them as a single pd.DataFrame.
    ----------------------------------------------
    :param
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with parquet dataset in base_path. Defaults to input/trip_data.
    :returns
        pd.DataFrame: DataFrame containing random sample data from all parquet files in path.
    """
    path = os.path.join(base_path, relative_path)
    try:
        dataset = pq.ParquetDataset(path)
        table = dataset.read()
        df = table.to_pandas()
        df = df.reset_index()
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_model(name="model.pkl"):
    path = os.path.join(get_data_path(), "output", name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
