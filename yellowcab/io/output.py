import os
import pickle

from .utils import get_data_path


def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), "output", "model.pkl"), "wb"))
