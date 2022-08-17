import json
import tempfile
import time
from pathlib import Path

import pytest

from neptune_prophet.impl import (
    create_forecast_plots,
    create_residual_diagnostics_plots,
    create_summary,
    get_forecast_components,
    get_model_config,
    get_serialized_model,
)

try:
    # neptune-client=0.9.0+ package structure
    from neptune.new import init_run
except ImportError:
    # neptune-client>=1.0.0 package structure
    from neptune import init_run


WAIT_FOR_LEADERBOARD = 10


def _test_with_run_initialization(*, pre, post):
    with init_run() as run:
        pre(run)
        time.sleep(WAIT_FOR_LEADERBOARD)
        post(run)


def _test_structure_get_model_config(run, base_namespace="data"):
    assert run.exists(base_namespace)
    assert run.exists(f"{base_namespace}/history_dates")
    assert run[f"{base_namespace}/history_dates"].fetch_extension() == "html"


def test_get_model_config(model):
    def initialize(run):
        run["data"] = get_model_config(model)

    _test_with_run_initialization(pre=initialize, post=_test_structure_get_model_config)


def _test_structure_get_serialized_model(run, base_namespace="data"):
    assert run.exists(base_namespace)

    with tempfile.TemporaryDirectory() as tmp:
        run[base_namespace].download(destination=tmp)

        with open(f"{tmp}/{Path(base_namespace).name}.json", "r", encoding="utf-8") as handler:
            _ = json.load(handler)


def test_get_serialized_model(model):
    def initialize(run):
        run["data"] = get_serialized_model(model)

    _test_with_run_initialization(pre=initialize, post=_test_structure_get_serialized_model)


def _test_structure_get_forecast_components(run, base_namespace="data"):
    assert run.exists(base_namespace)
    for column_name in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
        assert run.exists(f"{base_namespace}/{column_name}")


def test_get_forecast_components(model, forecast):
    def initialize(run):
        run["data"] = get_forecast_components(model, forecast)

    _test_with_run_initialization(pre=initialize, post=_test_structure_get_forecast_components)


def _test_structure_create_residual_diagnostics_plots(run, base_namespace="data"):
    assert run.exists(base_namespace)
    for column_name in [
        "histogram",
        "acf",
        "qq_plot",
        "actual_vs_normalized_errors",
        "ds_vs_normalized_errors",
    ]:
        assert run.exists(f"{base_namespace}/{column_name}")


@pytest.mark.parametrize("log_interactive", [False, True])
def test_create_residual_diagnostics_plots(dataset, predicted, log_interactive):
    def initialize(run):
        run["data"] = create_residual_diagnostics_plots(
            predicted,
            dataset.y,
            log_interactive=log_interactive,
        )

    _test_with_run_initialization(pre=initialize, post=_test_structure_create_residual_diagnostics_plots)


def _test_structure_create_forecast_plots(run, base_namespace="data"):
    assert run.exists(base_namespace)
    for column_name in ["forecast", "forecast_components", "forecast_changepoints"]:
        assert run.exists(f"{base_namespace}/{column_name}")


# @pytest.mark.parametrize("log_interactive", [False, True])
def test_create_forecast_plots(model, forecast):
    def initialize(run):
        run["data"] = create_forecast_plots(
            model,
            forecast,
            # log_interactive=log_interactive,
        )

    _test_with_run_initialization(pre=initialize, post=_test_structure_create_forecast_plots)


# @pytest.mark.parametrize("log_interactive", [False, True])
@pytest.mark.parametrize("log_charts", [False, True])
def test_create_summary(model, dataset, predicted, log_charts):
    def initialize(run):
        run["data"] = create_summary(
            model,
            df=dataset,
            fcst=predicted,
            log_charts=log_charts,
            # log_interactive=log_interactive,
        )

    def assert_structure(run):
        assert run.exists("data")
        assert run.exists("data/dataframes")
        _test_structure_get_model_config(run, "data/model/model_config")
        _test_structure_get_serialized_model(run, "data/model")

        if log_charts:
            assert run.exists("data/diagnostics_charts")
            _test_structure_create_residual_diagnostics_plots(
                run, "data/diagnostics_charts/residuals_diagnostics_charts"
            )
            _test_structure_create_forecast_plots(run, "data/diagnostics_charts")
        else:
            assert not run.exists("data/diagnostics_charts")

    _test_with_run_initialization(pre=initialize, post=assert_structure)
