from sklearn.linear_model import LinearRegression

from .. import io


def train():
    lin = LinearRegression()
    print("Linear model created")
    print("Training...")

    io.save_model(lin)
