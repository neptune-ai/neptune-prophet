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
    import neptune.new as neptune
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune


@pytest.mark.parametrize("log_interactive", [False, True])
def test_e2e(dataset, model, log_interactive):

    if log_interactive:
        try:
            import plotly  # pylint: disable=import-outside-toplevel, unused-import
        except ModuleNotFoundError:
            pytest.skip("plotly is needed for log_interactive to work")

    run = neptune.init(
        project="common/fbprophet-integration",
        api_token="ANONYMOUS",
    )

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    run["artifacts/model"] = get_model_config(model)
    assert run.exists("artifacts/model")

    run["artifacts/serialized_model"] = get_serialized_model(model)
    assert run.exists("artifacts/serialized_model")

    run["artifacts/forecast_components"] = get_forecast_components(model, forecast)
    assert run.exists("artifacts/forecast_components")
    for column_name in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
        assert run.exists(f"artifacts/forecast_components/{column_name}")

    run["artifacts/forecast_plots"] = create_forecast_plots(
        model,
        forecast,
        log_interactive=log_interactive,
    )
    assert run.exists("artifacts/forecast_plots")
    for column_name in ["forecast", "forecast_components", "forecast_changepoints"]:
        assert run.exists(f"artifacts/forecast_plots/{column_name}")

    predicted = model.predict(dataset)

    run["artifacts/residual_diagnostics"] = create_residual_diagnostics_plots(
        predicted,
        dataset.y,
        log_interactive=log_interactive,
    )
    assert run.exists("artifacts/residual_diagnostics")
    for column_name in [
        "histogram",
        "acf",
        "qq_plot",
        "actual_vs_normalized_errors",
        "ds_vs_normalized_errors",
    ]:
        assert run.exists(f"artifacts/residual_diagnostics/{column_name}")

    run["artifacts/summary"] = create_summary(
        model,
        df=dataset,
        fcst=predicted,
        log_charts=True,
        log_interactive=log_interactive,
    )
    assert run.exists("artifacts/summary")
    assert run.exists("artifacts/summary/model/model_config")
    assert run.exists("artifacts/summary/model/serialized_model")
    assert run.exists("artifacts/summary/dataframes")
    assert run.exists("artifacts/summary/diagnostics_charts")

    run.stop()
