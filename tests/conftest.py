import pytest
import pandas as pd


@pytest.fixture()
def dataset():
    return pd.read_csv("tests/example_wp_log_peyton_manning.csv")
