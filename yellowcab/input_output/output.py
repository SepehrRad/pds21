import os
import pickle

from .utils import get_data_path


def write_parquet(df, filename, base_path=get_data_path(), relative_path="output"):
    """
    This function reads a pd.Dataframe and saves it as a parquet.

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


def save_model(model, model_name):
    """
    This function saves a trained model in a pickle file in our output folder.
    :param model: The name of the model that we want to save.
    :param model_name: For specifying to what task the model belongs to.
    """
    file_name = "{}.pkl".format(model_name)
    pickle.dump(model, open(os.path.join(get_data_path(), "output", file_name), "wb"))

    
def write_parquet_file(df, filename, path=get_data_path()):
    """
    This function reads a pd.Dataframe saves it as a parquet.
    ----------------------------------------------
    :param
        df(pd.Dataframe): Given dataframe.
        path(String): Path to data directory. Defaults to wd/data.
    """
    path = os.path.join(path, filename)
    try:
        df.to_parquet(path)
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


