import json
import tempfile
from pathlib import Path

import pytest
from neptune import init_run

from neptune_prophet.impl import (
    create_forecast_plots,
    create_residual_diagnostics_plots,
    create_summary,
    get_forecast_components,
    get_model_config,
    get_serialized_model,
)


def test_get_model_config(model):
    with init_run() as run:
        run["data"] = get_model_config(model)
        run.wait()
        _test_structure_get_model_config(run)


def test_get_serialized_model(model):
    with init_run() as run:
        run["data"] = get_serialized_model(model)
        run.wait()
        _test_structure_get_serialized_model(run)


def test_get_forecast_components(model, forecast):
    with init_run() as run:
        run["data"] = get_forecast_components(model, forecast)
        run.wait()
        _test_structure_get_forecast_components(run)


@pytest.mark.parametrize("log_interactive", [False, True])
def test_create_forecast_plots(model, forecast, log_interactive):
    with init_run() as run:
        run["data"] = create_forecast_plots(
            model,
            forecast,
            log_interactive=log_interactive,
        )
        run.wait()
        _test_structure_create_forecast_plots(run, interactive=log_interactive)


@pytest.mark.parametrize("log_interactive", [False, True])
def test_create_residual_diagnostics_plots(dataset, predicted, log_interactive):
    with init_run() as run:
        run["data"] = create_residual_diagnostics_plots(predicted, dataset.y, log_interactive=log_interactive)
        run.wait()
        _test_structure_create_residual_diagnostics_plots(run, interactive=log_interactive)


@pytest.mark.parametrize(
    "log_interactive, with_forecast, log_charts",
    [
        (True, True, True),
        (False, True, True),
        (False, False, True),
        (False, False, False),
    ],
)
def test_create_summary(model, dataset, with_forecast, log_interactive, log_charts):
    with init_run() as run:
        if with_forecast:
            forecast = model.predict(dataset)
        else:
            forecast = None

        run["data"] = create_summary(
            model, df=dataset, fcst=forecast, log_interactive=log_interactive, log_charts=log_charts
        )
        run.wait()

        assert run.exists("data")
        assert run.exists("data/dataframes")

        _test_structure_get_model_config(run, "data/model/model_config")
        _test_structure_get_serialized_model(run, "data/model/serialized_model")

        assert run.exists("data/diagnostics_charts") == log_charts

        if log_charts:
            _test_structure_create_forecast_plots(
                run=run, base_namespace="data/diagnostics_charts", interactive=log_interactive
            )
            _test_structure_create_residual_diagnostics_plots(
                run=run,
                interactive=log_interactive,
                base_namespace="data/diagnostics_charts/residuals_diagnostics_charts",
            )


def _test_structure_get_model_config(run, base_namespace="data"):
    assert run.exists(base_namespace)
    assert run.exists(f"{base_namespace}/history_dates")
    assert run[f"{base_namespace}/history_dates"].fetch_extension() == "html"


def _test_structure_get_serialized_model(run, base_namespace="data"):
    assert run.exists(base_namespace)

    with tempfile.TemporaryDirectory() as tmp:
        run[base_namespace].download(destination=tmp)

        with open(f"{tmp}/{Path(base_namespace).name}.json", "r", encoding="utf-8") as handler:
            _ = json.load(handler)


def _test_structure_get_forecast_components(run, base_namespace="data"):
    assert run.exists(base_namespace)
    for column_name in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
        assert run.exists(f"{base_namespace}/{column_name}")


def _test_plot(run, namespace, interactive):
    assert run.exists(namespace)
    assert run[namespace].fetch_extension() == ("html" if interactive else "png")


def _test_structure_create_forecast_plots(run, interactive, base_namespace="data"):
    assert run.exists(base_namespace)
    _test_plot(run, f"{base_namespace}/forecast", interactive=interactive)
    _test_plot(run, f"{base_namespace}/forecast_components", interactive=interactive)
    _test_plot(run, f"{base_namespace}/forecast_changepoints", interactive=False)
    _test_structure_get_forecast_components(run, base_namespace=base_namespace)


def _test_structure_create_residual_diagnostics_plots(run, interactive, base_namespace="data"):
    assert run.exists(base_namespace)
    _test_plot(run, f"{base_namespace}/histogram", interactive=False)
    _test_plot(run, f"{base_namespace}/acf", interactive=False)
    _test_plot(run, f"{base_namespace}/qq_plot", interactive=interactive)
    _test_plot(run, f"{base_namespace}/actual_vs_normalized_errors", interactive=interactive)
    _test_plot(run, f"{base_namespace}/ds_vs_normalized_errors", interactive=interactive)
