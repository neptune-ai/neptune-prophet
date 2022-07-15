import pytest
from prophet import Prophet
from neptune_prophet.impl import *

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune


@pytest.mark.parametrize("log_interactive", [False, True])
def test_e2e(dataset, log_interactive):

    if log_interactive:
        try:
            import plotly
        except ModuleNotFoundError:
            pytest.skip("plotly is needed for log_interactive to work")

    run = neptune.init(
        project="common/fbprophet-integration",
        api_token="ANONYMOUS",
    )

    model = Prophet()
    model.fit(dataset)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    run["model"] = get_model_config(model)
    run["serialized_model"] = get_serialized_model(model)

    run["forecast_plots"] = create_forecast_plots(
        model,
        forecast,
        log_interactive=log_interactive,
    )

    predicted = model.predict(dataset)

    run["residual_diagnostics"] = create_residual_diagnostics_plots(
        predicted,
        dataset.y,
        log_interactive=log_interactive,
    )

    run["summary"] = create_summary(
        model,
        df=dataset,
        fcst=predicted,
        log_charts=True,
        log_interactive=log_interactive,
    )

    run.stop()
