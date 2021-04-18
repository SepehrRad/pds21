from .utils import get_data_path
import pandas as pd
import os
import pickle


def read_file(path=os.path.join(get_data_path(), "input", "<My_data>.parquet")):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)


def read_model(name="model.pkl"):
    path = os.path.join(get_data_path(), "output", name)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
