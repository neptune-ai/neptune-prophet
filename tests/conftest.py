# pylint: disable=redefined-outer-name
import pandas as pd
import pytest
from prophet import Prophet


@pytest.fixture(scope="session")
def dataset():
    yield pd.read_csv("tests/dataset.csv")


@pytest.fixture(scope="session")
def model(dataset):
    prophet_model = Prophet()
    prophet_model.fit(dataset)
    yield prophet_model


@pytest.fixture(scope="session")
def forecast(model):
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    yield forecast


@pytest.fixture(scope="session")
def predicted(model, dataset):
    predicted = model.predict(dataset)
    yield predicted
