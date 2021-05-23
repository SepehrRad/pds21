import os
import pickle
import geopandas
from .utils import get_data_path


def write_parquet(df, filename, base_path=get_data_path(), relative_path="output"):
    """
    This function reads a pd.Dataframe, casts it to a GeoDataFrame and saves it as a parquet.

    ----------------------------------------------

    :param
        df(pd.Dataframe): Given dataframe.
        file(String): Name of file.
        base_path(String): Path to data directory. Defaults to wd/data.
        relative_path(String): Path to directory with file in base_path. Defaults to input/trip_data.
    """
    path = os.path.join(base_path, relative_path, filename)
    try:
        df.to_parquet(path)
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), "output", "model.pkl"), "wb"))
