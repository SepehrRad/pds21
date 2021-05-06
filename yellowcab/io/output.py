import os
import pickle

import pyarrow as pa
import pyarrow.parquet as pq

from .utils import get_data_path


def write_parquet(df, filename, base_path=get_data_path(), relative_path="output"):
    """
    This function reads a pd.Dataframe file & returns it as a parquet file.

    ----------------------------------------------

    :param
        df(pd.Dataframe): Given dataframe.
        file(String): Name of file.
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with file in base_path. Defaults to input/trip_data.
    :raises
        FileNotFoundError: Data file not found in given directory.
    """
    path = os.path.join(base_path, relative_path, filename)
    try:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), "output", "model.pkl"), 'wb'))
