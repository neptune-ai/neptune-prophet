import pandas as pd
import pytest

from prophet import Prophet


@pytest.fixture()
def dataset():
    return pd.read_csv("tests/dataset.csv")


@pytest.fixture()
def model(dataset):  # pylint: disable=redefined-outer-name
    model = Prophet()  # pylint: disable=redefined-outer-name
    model.fit(dataset)
    return model
