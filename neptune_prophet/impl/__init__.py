#
# Copyright (c) 2022, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

__all__ = [
    "get_model_config",
    "get_serialized_model",
    "create_forecast_plots",
    "create_residual_diagnostics_plots",
    "create_summary",
]

import tempfile

from typing import Any, Dict, List, Optional

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
    from neptune.new.types import File, FloatSeries
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune
    from neptune.types import File, FloatSeries

from neptune_prophet import __version__

INTEGRATION_VERSION_KEY = "source_code/integrations/neptune-prophet"

from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from prophet.serialize import model_to_json
import json
import tempfile


def _get_figure(figsize=(20, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    return fig, ax


def get_model_config(model: Prophet) -> Dict[str, Any]:
    """Extracts the configuration from the Prophet model.

    Args:
        model: Fitted Prophet model object.

    Returns:
        Dictionary with all summary items.

    Examples:
        from prophet import Prophet
        import neptune.new as neptune
        neptune.init_run()
        model = Prophet()
        model.fit(dataset)
        run["model_config"] = get_model_config(model)
    """
    config = model.__dict__
    model.history_dates = pd.DataFrame(model.history_dates)

    model_config = dict()
    for key, value in config.items():
        if key == "trend":
            continue
        elif isinstance(value, pd.DataFrame):
            model_config[f"{key}"] = File.as_html(value)
        elif isinstance(value, np.ndarray):
            model_config[f"{key}"] = File.as_html(pd.DataFrame(value))
        elif isinstance(value, pd.Series):
            model_config[f"{key}"] = File.as_html(pd.DataFrame(value))
        else:
            model_config[f"{key}"] = value

    return model_config


def get_serialized_model(model: Prophet) -> File:
    """Serializes the Prophet model.

    Args:
        model: Fitted Prophet model object.

    Returns:
        File containing the model.

    Examples:
        from prophet import Prophet
        import neptune.new as neptune
        neptune.init_run()
        model = Prophet()
        model.fit(dataset)
        run["model"] = get_serialized_model(model)
    """

    # create a temporary file and return File field with serialized model
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    json.dump(model_to_json(model), tmp)
    return File(tmp.name)


def _get_residuals(fcst: pd.DataFrame, y: pd.Series):
    if len(fcst.yhat) != len(y):
        raise ValueError("The lenghts of the true series and predicted series do not match.")

    return stats.zscore(
        y - fcst.yhat,
        nan_policy="omit",
    )


def _is_fcst(obj: Any) -> bool:
    if not isinstance(obj, pd.DataFrame):
        return False
    return "yhat" in obj.columns


def _extract_forecast_components(model: Prophet, fcst: pd.DataFrame) -> List[str]:
    # it is the same code as in Prophet, but simplified:
    # https://github.com/facebook/prophet/blob/ba9a5a2c6e2400206017a5ddfd71f5042da9f65b/python/prophet/plot.py#L127-L140
    components = ["trend"]
    if model.train_holiday_names is not None and "holidays" in fcst:
        components.append("holidays")
    components.extend([name for name in sorted(model.seasonalities) if name in fcst])
    return components


def _get_dataframe(df: pd.DataFrame, nrows: int = 1000) -> File:
    return File.as_html(df.head(n=nrows))


def create_forecast_plots(
    model: Prophet,
    fcst: pd.DataFrame,
    log_interactive: bool = True,
) -> Dict[str, Any]:
    """Prepares the Prophet plots to be saved to Neptune.

    Args:
        model: Fitted Prophet model object.
        fcst: Forecast returned by Prophet, as pandas DataFrame.
        log_interactive: Save the plots as interactive HTML files.

    Returns:
        Dictionary with all the plots.

    Examples:
        from prophet import Prophet
        import neptune.new as neptune
        neptune.init_run()
        model = Prophet()
        model.fit(dataset)
        run["forecast_plots"] = create_forecast_plots(model)
    """

    if log_interactive:
        try:
            import plotly
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plotly is needed for log_interactive to work.")

    if not _is_fcst(fcst):
        raise ValueError("fcst is not valid a Prophet forecast.")

    forecast_plots = dict()

    for column in ["yhat", "yhat_lower", "yhat_upper"] + _extract_forecast_components(
        model, fcst
    ):
        values = fcst.loc[:, column].tolist()
        forecast_plots[column] = FloatSeries(values)

    if log_interactive:
        fig1 = plot_plotly(model, fcst)
        forecast_plots["forecast"] = File.as_html(fig1)

        if "trend" in fcst.columns:
            fig2 = plot_components_plotly(model, fcst, figsize=(1000, 400))
            forecast_plots["forecast_components"] = File.as_html(fig2)

            fig3 = model.plot(fcst)
            changepoint_fig = add_changepoints_to_plot(fig3.gca(), model, fcst)
            forecast_plots["forecast_changepoints"] = File.as_image(
                changepoint_fig[-1].figure
            )
            plt.close(fig3)
        return forecast_plots
    else:
        fig1 = model.plot(fcst)
        forecast_plots["forecast"] = File.as_image(fig1)
        plt.close(fig1)

        if "trend" in fcst.columns:
            fig2 = model.plot_components(fcst)
            forecast_plots["forecast_components"] = File.as_image(fig2)

            changepoint_fig = add_changepoints_to_plot(fig1.gca(), model, fcst)
            forecast_plots["forecast_changepoints"] = File.as_image(
                changepoint_fig[-1].figure
            )
            plt.close(fig2)

        return forecast_plots


def create_residual_diagnostics_plots(
    fcst: pd.DataFrame,
    y: pd.Series,
    log_interactive: bool = True,
    alpha: float = 0.7,
) -> Dict[str, Any]:
    """Prepares additional diagnostic plots to be saved to Neptune.

    Args:
        fcst: Forecast returned by Prophet.
        y: True values that were predicted.
        log_interactive: Save the plots as interactive HTML files.
        alpha: Transparency level of the plots.

    Returns:
        Dictionary with all the plots.

    Examples:
        from prophet import Prophet
        import neptune.new as neptune
        neptune.init_run()
        model = Prophet()
        model.fit(dataset)
        run["residual_diagnostics_plot"] = create_residual_diagnostics_plots(model)
    """

    if log_interactive:
        try:
            import plotly
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plotly is needed for log_interactive to work.")

    if not _is_fcst(fcst):
        raise ValueError("fcst is not valid a Prophet forecast.")

    residuals = _get_residuals(fcst, y)
    plots = dict()

    fig1, ax1 = _get_figure()
    sm.qqplot(residuals, line="45", ax=ax1)
    ax1.set_title("Q-Q plot of normalized errors")

    fig2, ax2 = _get_figure()
    ax2.hist(residuals, bins="auto")
    ax2.set_xlabel("Normalized errors")
    ax2.set_title("Histogram of normalized errors")

    fig3, ax3 = _get_figure()
    ax3.scatter(y, residuals, alpha=alpha)
    ax3.set_title("Actual vs Normalized errors")
    ax3.set_ylabel("Normalized errors")
    ax3.set_xlabel("y")

    fig4, ax4 = _get_figure()
    sm.graphics.tsa.plot_acf(
        residuals,
        auto_ylims=True,
        ax=ax4,
        title="ACF of normalized errors",
    )

    fig5, ax5 = _get_figure()
    ax5.scatter(fcst["ds"], residuals, alpha=alpha)
    ax5.set_ylabel("Normalized errors")
    ax5.set_xlabel("Dates")
    ax5.set_title("DS vs Normalized errors")

    plots["histogram"] = File.as_image(fig2)
    plots["acf"] = File.as_image(fig4)

    if log_interactive:
        plots["qq_plot"] = File.as_html(fig1)
        plots["actual_vs_normalized_errors"] = File.as_html(fig3)
        plots["ds_vs_normalized_errors"] = File.as_html(fig5)
    else:
        plots["qq_plot"] = File.as_image(fig1)
        plots["actual_vs_normalized_errors"] = File.as_image(fig3)
        plots["ds_vs_normalized_errors"] = File.as_image(fig5)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)

    return plots


def create_summary(
    model: Prophet,
    df: Optional[pd.DataFrame] = None,
    fcst: Optional[pd.DataFrame] = None,
    log_charts: bool = True,
    log_interactive: bool = True,
    nrows: int = 1000,
) -> Dict[str, Any]:
    """Prepares additional diagnostic plots to be saved to Neptune.

    Args:
        model: Fitted Prophet model object.
        df: The dataset that was used for making the forecast.
            If provided, additional plots will be recorded.
        fcst: Forecast returned by Prophet.
            If not provided, it'll be calculated using the df data.
        log_charts: Aditionally save the diagnostic plots.
        log_interactive: Save the plots as interactive HTML files.
        nrows: Number of rows the dataset should be downsampled to.

    Returns:
        Dictionary with all the plots.

    Examples:
        from prophet import Prophet
        import neptune.new as neptune
        neptune.init_run()
        model = Prophet()
        model.fit(dataset)
        run["summary"] = create_summary(model)
    """

    if log_interactive:
        try:
            import plotly
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Plotly is needed for log_interactive to wor.k")

    alpha = 0.7
    prophet_summary = dict()

    prophet_summary["model"] = {
        "model_config": get_model_config(model),
        "serialized_model": get_serialized_model(model),
    }

    prophet_summary["dataframes"] = {"forecast": _get_dataframe(fcst, nrows=nrows)}

    if df is not None:
        prophet_summary[f"dataframes"]["df"] = File.as_html(df)

        if fcst is None:
            fcst = model.predict(fcst)
        elif not _is_fcst(fcst):
            raise ValueError("fcst is not valid a Prophet forecast.")

        if len(fcst.yhat) != len(df.y):
            raise RuntimeError("The lenghts of the true series and forecast series do not match.")

        if log_charts:
            prophet_summary["diagnostics_charts"] = {
                "residuals_diagnostics_charts": create_residual_diagnostics_plots(
                    fcst,
                    df.y,
                    log_interactive=log_interactive,
                    alpha=alpha,
                ),
                **create_forecast_plots(model, fcst, log_interactive=log_interactive),
            }
    else:
        if log_charts:
            prophet_summary["diagnostics_charts"] = {
                **create_forecast_plots(model, fcst, log_interactive=log_interactive)
            }

    return prophet_summary
