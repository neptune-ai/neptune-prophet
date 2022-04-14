#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
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
    # TODO: add importable public names here, `neptune-client` uses `import *`
    # https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
]

# TODO: use `warnings.warn` for user caused problems: https://stackoverflow.com/a/14762106/1565454
import tempfile
import warnings

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
    from neptune.new.types import File, FloatSeries
    from neptune.new.internal.utils import verify_type
    from neptune.new.internal.utils.compatibility import expect_not_an_experiment
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune
    from neptune.internal.utils import verify_type
    from neptune.internal.utils.compatibility import expect_not_an_experiment

from neptune_prophet import __version__  # TODO: change module name

INTEGRATION_VERSION_KEY = "source_code/integrations/integration-template"  # TODO: change path

# TODO: Implementation of neptune-integration here

from fbprophet import Prophet
import pandas as pd
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from fbprophet.plot import plot_plotly, plot_components_plotly
from fbprophet.plot import plot_cross_validation_metric, add_changepoints_to_plot
from fbprophet.serialize import model_to_json
import json


def create_forecast_plots(model, forecast, df:pd.DataFrame=None, log_interactive=True):
    forecast_plots = dict()

    yhat_values = forecast.yhat.tolist()
    forecast_plots["yhat"] = FloatSeries(yhat_values)

    if df is not None:
        y_values = df.y.tolist()
        forecast_plots["y"] = FloatSeries(y_values)
        if "VWAP" in df.columns:
            vwap_values = df.VWAP.tolist()
            forecast_plots["VWAP"] = FloatSeries(vwap_values)

    if log_interactive:
        fig1 = plot_plotly(model, forecast)
        forecast_plots["forecast"] = File.as_html(fig1)
        if "trend" in forecast.columns:
            fig2 = plot_components_plotly(model, forecast, figsize=(1000,400))
            forecast_plots["forecast_components"] = File.as_html(fig2)
            fig3 = model.plot(forecast)
            changepoint_fig = add_changepoints_to_plot(fig3.gca(), model, forecast)
            forecast_plots["forecast_changepoints"] = File.as_image(changepoint_fig[-1].figure)
        return forecast_plots
    else:
        fig1 = model.plot(forecast)
        forecast_plots["forecast"] = File.as_image(fig1)
        if "trend" in forecast.columns:
            fig2 = model.plot_components(forecast)
            forecast_plots["forecast_components"] = File.as_image(fig2)
            changepoint_fig = add_changepoints_to_plot(fig1.gca(), model, forecast)
            forecast_plots["forecast_changepoints"] = File.as_image(changepoint_fig[-1].figure)
        return forecast_plots

def get_cross_validation_results( all_params, metrics: list, metric_name="rmse"):

    cv_results = dict()

    tuning_results = pd.DataFrame(all_params)
    tuning_results[metric_name] = metrics
    cv_results["tuning_results"] = tuning_results.to_dict()

    best_params = all_params[np.argmin(metrics)]
    cv_results["best_params"] = best_params


def get_figure(figsize=(20, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    return fig, ax

def create_residual_diagnostics_plot(residuals_forecast, y:pd.Series, alpha=0.7, log_interactive=True):
    if "e_z" not in residuals_forecast.columns:
        residuals_forecast = get_residuals(residuals_forecast, y)
    if "anomaly" not in residuals_forecast.columns:
        residuals_forecast = detect_anomalies(residuals_forecast, y)

    colors = {0: "#0079b9", 1: "red", -1: "red"}
    c = residuals_forecast.anomaly.map(colors) if "anomaly" in residuals_forecast else None
    plots = dict()

    fig1, ax1 = get_figure()
    sm.qqplot(residuals_forecast["e_z"], line="45" ,ax=ax1)
    ax1.set_title("QQ Plot of normalized errors")

    fig2, ax2 = get_figure()
    ax2.hist(residuals_forecast["e_z"], bins="auto")
    ax2.set_xlabel("Normalized e = y - yhat")
    ax2.set_title("Histogram of normalized errors")

    fig3, ax3 = get_figure()
    ax3.scatter(y, residuals_forecast["e_z"], c=c, alpha=alpha)
    ax3.set_title("Actual vs Normalized errors")
    ax3.set_ylabel("Normalized e = y - yhat")
    ax3.set_xlabel("y")

    fig4, ax4 = get_figure()
    sm.graphics.tsa.plot_acf(residuals_forecast.set_index("ds")["e_z"], auto_ylims=True, ax=ax4,
                             title="ACF of normalized errors")

    fig5, ax5 = get_figure()
    ax5.scatter(residuals_forecast["ds"], residuals_forecast["e_z"], c=c, alpha=alpha)
    ax5.set_ylabel("Normalized e = y - yhat")
    ax5.set_xlabel("Dates")
    ax5.set_title("DS vs Normalized errors")

    plots["Histogram"] =  File.as_image(fig2)
    plots["ACF"] =  File.as_image(fig4)

    if log_interactive:
        plots["QQ_plot"] = File.as_html(fig1)
        plots["Actual_vs_Normalized_errors"] =  File.as_html(fig3)
        plots["DS_vs_Normalized_errors"] = File.as_html(fig5)
    else:
        plots["QQ_plot"] =  File.as_image(fig1)
        plots["Actual_vs_Normalized_errors"] =  File.as_image(fig3)
        plots["DS_vs_Normalized_errors"] = File.as_image(fig5)

    return plots

def get_model_config(model: Prophet):
    model_config = dict()
    module = "numpy"
    if module not in sys.modules:
        raise Exception(f"{module} is not imported")

    config = copy.deepcopy(model.__dict__)
    model.history_dates = pd.DataFrame(model.history_dates)

    config["params"].pop("trend")

    for key, value in config.items():
        if isinstance(value, pd.DataFrame):
            model_config[f"{key}"] = File.as_html(value)
        elif isinstance(value, np.ndarray):
            model_config[f"{key}"] = File.as_html(pd.DataFrame(value))
        elif isinstance(value, pd.Series):
            model_config[f"{key}"] = File.as_html(pd.DataFrame(value))
        else:
            model_config[f"{key}"] = value

    return model_config

def get_residuals(forecast, y):
    forecast["e"] = y - forecast.yhat
    forecast["e_z"] = stats.zscore(forecast["e"], nan_policy="omit") # Normalization mean=0, std=1
    return forecast

def detect_anomalies(forecast, y):

    forecast["anomaly"] = 0

    forecast.loc[y > forecast["yhat_upper"], "anomaly"] = 1
    forecast.loc[y < forecast["yhat_lower"], "anomaly"] = -1

    # Anomaly importances
    forecast["importance"] = 0
    forecast.loc[forecast["anomaly"] ==1, "importance"] = \
        (y - forecast["yhat_upper"])/y
    forecast.loc[forecast["anomaly"] ==-1, "importance"] = \
        (forecast["yhat_lower"] - y)/y

    return forecast

def create_serialized_model(model):
    # create a temporary file and return File field with serialized model
    tmp = tempfile.TemporaryFile()
    tmp.write(json.dump(model_to_json(model)))
    return File(tmp)

def get_dataframe(df, nrows=1000):
    return File.as_html(df.head(n=nrows))

def create_summary(
        model, forecast:pd.DataFrame,
        df:pd.DataFrame = None,
        cv_metrics_df:pd.DataFrame=None,
        log_charts=True,
        log_interactive=True, alpha=0.7
    ):

    prophet_summary = dict()

    prophet_summary["model"] = {
        "model_config": get_model_config(model),
        "serialized_model": create_serialized_model(model)
    }

    prophet_summary["dataframes"] = {"forecast": get_dataframe(forecast)}

    if cv_metrics_df is not None:
        prophet_summary["metrics"] = cv_metrics_df.to_dict(orient="records")[0]

    if df is not None:
        prophet_summary[f"dataframes"]["df"] = File.as_html(df)
        if len(forecast) > len(df.y):
            forecast = forecast.truncate(after = len(df.y) - 1)

        residuals_forecast = get_residuals(forecast, y=df.y)
        residuals_forecast = detect_anomalies(residuals_forecast, y=df.y)
        residuals_forecast["y"] = df.y
        prophet_summary["dataframes"]["residuals_forecast"] = File.as_html(
            residuals_forecast[["y","e","e_z","anomaly","importance"]]
        )

        if log_charts:
            prophet_summary["diagnostics_charts"] =  {
                "residuals_diagnostics_charts":  create_residual_diagnostics_plot(residuals_forecast, df.y, alpha, log_interactive=log_interactive),
                **create_forecast_plots(model, forecast, df, log_interactive=log_interactive)
            }
    else:
        if log_charts:
            prophet_summary["diagnostics_charts"] =  {
                **create_forecast_plots(model, forecast, log_interactive=log_interactive)
            }

    plt.close("all")

    return prophet_summary