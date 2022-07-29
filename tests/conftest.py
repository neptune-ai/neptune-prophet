import pandas as pd
import pytest


@pytest.fixture()
def dataset():
    return pd.read_csv("tests/example_wp_log_peyton_manning.csv")
