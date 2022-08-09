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
    "create_forecast_plots",
    "create_residual_diagnostics_plots",
    "create_summary",
    "get_forecast_components",
    "get_model_config",
    "get_serialized_model",
]

import json
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
    from neptune.new.types import File, FloatSeries
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune
    from neptune.types import File, FloatSeries

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_components_plotly, plot_plotly
from prophet.serialize import model_to_json


def create_summary(
    model: Prophet,
    df: Optional[pd.DataFrame] = None,
    fcst: Optional[pd.DataFrame] = None,
    log_charts: bool = True,
    log_interactive: bool = True,
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

    Returns:
        Dictionary with all the plots.

    Examples:
        import pandas as pd
        from prophet import Prophet
        import neptune.new as neptune

        neptune.init_run()

        dataset = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

        model = Prophet()
        model.fit(dataset)

        run["summary"] = create_summary(model, dataset)
    """

    if log_interactive:
        _fail_if_plotly_is_unavailable()

    prophet_summary = dict()

    prophet_summary["model"] = {
        "model_config": get_model_config(model),
        "serialized_model": get_serialized_model(model),
    }

    prophet_summary["dataframes"] = {"forecast": File.as_html(fcst)}

    if df is not None:
        prophet_summary["dataframes"]["df"] = File.as_html(df)

        if fcst is None:
            fcst = model.predict(fcst)
        else:
            _fail_if_invalid_fcst(fcst)

        if len(fcst.yhat) != len(df.y):
            raise RuntimeError("The lenghts of the true series and forecast series do not match.")

        if log_charts:
            prophet_summary["diagnostics_charts"] = {
                "residuals_diagnostics_charts": create_residual_diagnostics_plots(
                    fcst,
                    df.y,
                    log_interactive=log_interactive,
                ),
                **create_forecast_plots(model, fcst, log_interactive=log_interactive),
            }
    elif log_charts:
        prophet_summary["diagnostics_charts"] = create_forecast_plots(model, fcst, log_interactive=log_interactive)

    return prophet_summary


def get_model_config(model: Prophet) -> Dict[str, Any]:
    """Extracts the configuration from the Prophet model.

    Args:
        model: Fitted Prophet model object.

    Returns:
        Dictionary with all summary items.

    Examples:
        import pandas as pd
        from prophet import Prophet
        import neptune.new as neptune

        neptune.init_run()

        dataset = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

        model = Prophet()
        model.fit(dataset)

        run["model_config"] = get_model_config(model)
    """

    model_config = dict()
    for key, value in model.__dict__.items():
        if key == "trend":
            continue
        elif isinstance(value, pd.DataFrame):
            model_config[str(key)] = File.as_html(value)
        elif isinstance(value, (np.ndarray, pd.Series)):
            model_config[str(key)] = File.as_html(pd.DataFrame(value))
        else:
            model_config[str(key)] = value

        model_config["history_dates"] = pd.DataFrame(model.history_dates)

    return model_config


def get_serialized_model(model: Prophet) -> File:
    """Serializes the Prophet model.

    Args:
        model: Fitted Prophet model object.

    Returns:
        File containing the model.

    Examples:
        import pandas as pd
        from prophet import Prophet
        import neptune.new as neptune

        neptune.init_run()

        dataset = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

        model = Prophet()
        model.fit(dataset)

        run["model"] = get_serialized_model(model)
    """

    # create a temporary file and return File field with serialized model
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    json.dump(model_to_json(model), tmp)
    return File(tmp.name)


def get_forecast_components(model: Prophet, fcst: pd.DataFrame) -> Dict[str, Any]:
    """Get the Prophet forecast components to be saved to Neptune.

    Args:
        model: Fitted Prophet model object.
        fcst: Forecast returned by Prophet, as pandas DataFrame.

    Returns:
        Dictionary with all the plots.

    Examples:
        import pandas as pd
        from prophet import Prophet
        import neptune.new as neptune

        neptune.init_run()

        dataset = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

        model = Prophet()
        model.fit(dataset)
        predicted = model.predict(dataset)

        run["forecast_components"] = get_forecast_components(model, predicted)
    """
    forecast_components = dict()

    for column_name in _forecast_component_names(model, fcst):
        values = fcst.loc[:, column_name].tolist()
        forecast_components[column_name] = FloatSeries(values)

    return forecast_components


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
        import pandas as pd
        from prophet import Prophet
        import neptune.new as neptune

        neptune.init_run()

        dataset = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

        model = Prophet()
        model.fit(dataset)
        predicted = model.predict(dataset)

        run["forecast_plots"] = create_forecast_plots(model, predicted)
    """

    if log_interactive:
        _fail_if_plotly_is_unavailable()

    _fail_if_invalid_fcst(fcst)

    forecast_plots = get_forecast_components(model, fcst)

    if log_interactive:
        fig1 = plot_plotly(model, fcst)
        forecast_plots["forecast"] = File.as_html(fig1)

        if "trend" in fcst.columns:
            fig2 = plot_components_plotly(model, fcst, figsize=(1000, 400))
            forecast_plots["forecast_components"] = File.as_html(fig2)

            fig3 = model.plot(fcst)
            changepoint_fig = add_changepoints_to_plot(fig3.gca(), model, fcst)
            forecast_plots["forecast_changepoints"] = File.as_image(changepoint_fig[-1].figure)
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
            forecast_plots["forecast_changepoints"] = File.as_image(changepoint_fig[-1].figure)
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
        import pandas as pd
        from prophet import Prophet
        import neptune.new as neptune

        neptune.init_run()

        dataset = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

        model = Prophet()
        model.fit(dataset)
        predicted = model.predict(dataset)

        run["residual_diagnostics_plot"] = create_residual_diagnostics_plots(predicted, dataset.y)
    """

    if log_interactive:
        _fail_if_plotly_is_unavailable()

    _fail_if_invalid_fcst(fcst)

    residuals = _get_residuals(fcst, y)

    fig1 = _qq_plot(residuals)
    fig2 = _errors_histogram(residuals)
    fig3 = _actual_vs_normalized_errors_plot(y, alpha, residuals)
    fig4 = _acf_plot(residuals)
    fig5 = _normalized_errors_plot(fcst, residuals, alpha)

    plots = dict()
    plots["histogram"] = File.as_image(fig2)
    plots["acf"] = File.as_image(fig4)

    plots["qq_plot"] = _get_plot(fig1, log_interactive)
    plots["actual_vs_normalized_errors"] = _get_plot(fig3, log_interactive)
    plots["ds_vs_normalized_errors"] = _get_plot(fig5, log_interactive)

    _close_figs(fig1, fig2, fig3, fig4, fig5)

    return plots


def _get_residuals(fcst: pd.DataFrame, y: pd.Series) -> pd.Series:
    if len(fcst.yhat) != len(y):
        raise ValueError("The lenghts of the true series and predicted series do not match.")

    return stats.zscore(
        y - fcst.yhat,
        nan_policy="omit",
    )


def _fail_if_invalid_fcst(obj: Any):
    if not isinstance(obj, pd.DataFrame) or "yhat" not in obj.columns:
        raise ValueError("fcst is not valid a Prophet forecast.")


def _forecast_component_names(model: Prophet, fcst: pd.DataFrame) -> List[str]:
    # it is the same code as in Prophet, but simplified:
    # https://github.com/facebook/prophet/blob/ba9a5a2c6e2400206017a5ddfd71f5042da9f65b/python/prophet/plot.py#L127-L140
    components = ["yhat", "yhat_lower", "yhat_upper", "trend"]
    if model.train_holiday_names is not None and "holidays" in fcst:
        components.append("holidays")
    components.extend([name for name in sorted(model.seasonalities) if name in fcst])
    return components


def _fail_if_plotly_is_unavailable():
    try:
        import plotly  # pylint: disable=import-outside-toplevel, unused-import
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Plotly is needed for log_interactive to work.") from exc


def _qq_plot(residuals):
    fig1, ax1 = _get_figure()
    sm.qqplot(residuals, line="45", ax=ax1)
    ax1.set_title("Q-Q plot of normalized errors")
    return fig1


def _errors_histogram(residuals):
    fig2, ax2 = _get_figure()
    ax2.hist(residuals, bins="auto")
    ax2.set_xlabel("Normalized errors")
    ax2.set_title("Histogram of normalized errors")
    return fig2


def _actual_vs_normalized_errors_plot(y, alpha, residuals):
    fig3, ax3 = _get_figure()
    ax3.scatter(y, residuals, alpha=alpha)
    ax3.set_title("Actual vs Normalized errors")
    ax3.set_ylabel("Normalized errors")
    ax3.set_xlabel("y")
    return fig3


def _acf_plot(residuals):
    fig, ax = _get_figure()
    sm.graphics.tsa.plot_acf(
        residuals,
        auto_ylims=True,
        ax=ax,
        title="ACF of normalized errors",
    )
    return fig


def _get_plot(fig, log_interactive) -> File:
    if log_interactive:
        return File.as_html(fig)
    return File.as_image(fig)


def _close_figs(*args):
    for fig in args:
        plt.close(fig)


def _get_figure(figsize=(20, 10)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, figsize=figsize)
    return fig, ax


def _normalized_errors_plot(fcst, residuals, alpha):
    fig, ax = _get_figure()
    ax.scatter(fcst["ds"], residuals, alpha=alpha)
    ax.set_ylabel("Normalized errors")
    ax.set_xlabel("Dates")
    ax.set_title("DS vs Normalized errors")
    return fig
