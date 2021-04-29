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


def read_parquet(path=os.path.join(get_data_path(), "input/trip_data"), file):
    """
    This function reads a parquet file & returns it as a pd.DataFrame.
    ----------------------------------------------
    :param
        path(String): Path to directory. Defaults to wd/data/input/trip_data.
        file(String): Name of file
    :returns
        pd.DataFrame: DataFrame containing data from parquet.
    """
    try:
        table = pq.read_table(os.path.join(path, file))
        df = table.to_pandas()
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_parquet_dataset(path=os.path.join(get_data_path(), "input/trip_data")):
    """
    This function reads a directory of parquet files & returns them as a single pd.DataFrame.
    ----------------------------------------------
    :param
        path(String): Path to directory. Defaults to wd/data/input/trip_data.
    :returns
        pd.DataFrame: DataFrame containing data from all parquet files in path.
    """
    try:
        dataset = pq.ParquetDataset(path)
        table = dataset.read()
        df = table.to_pandas()
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_model(name="model.pkl"):
    path = os.path.join(get_data_path(), "output", name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
