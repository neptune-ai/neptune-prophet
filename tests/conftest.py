import pandas as pd
import pytest

from prophet import Prophet


@pytest.fixture()
def dataset():
    return pd.read_csv("tests/dataset.csv")


@pytest.fixture()
def model(dataset):
    model = Prophet()
    model.fit(dataset)
    return model
