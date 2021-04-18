from .. import io
from sklearn.linear_model import LinearRegression


def train():
    lin = LinearRegression()
    print("Linear model created")
    print("Training...")

    io.save_model(lin)
