import pandas as pd
import pytest


@pytest.fixture()
def dataset():
    return pd.read_csv("tests/dataset.csv")
