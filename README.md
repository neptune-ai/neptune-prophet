# Neptune + Prophet integration

Experiment tracking for Prophet-trained models.

## What will you get with this integration?

* Log, organize, visualize, and compare ML experiments in a single place
* Monitor model training live
* Version and query production-ready models and associated metadata (e.g., datasets)
* Collaborate with the team and across the organization

## What will be logged to Neptune?

* parameters,
* forecast data frames,
* residual diagnostic charts,
* [other metadata](https://docs.neptune.ai/logging/what_you_can_log)

![image](https://docs.neptune.ai/img/app/integrations/prophet.png)

## Resources

* [Documentation](https://docs.neptune.ai/integrations/prophet)
* [Code example on GitHub](https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/prophet/scripts)
* [Example project in the Neptune app](https://app.neptune.ai/o/common/org/fbprophet-integration/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=Diagnostic-charts-5855c208-c4b8-4171-b065-d0e8802b1b60&shortId=FBPROP-3211&type=run)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/prophet/notebooks/Neptune_prophet.ipynb)

## Example

### Before you start

- [Install and set up Neptune](https://docs.neptune.ai/setup/installation).
- Have Prophet installed.

### Installation

```
# On the command line
pip install neptune-prophet
```

### Logging example

```python
# In Python
import pandas as pd
from prophet import Prophet
import neptune
import neptune.integrations.prophet as npt_utils

# Start a run
run = neptune.init_run(project="common/fbprophet-integration", api_token=neptune.ANONYMOUS_API_TOKEN)

# Load dataset and fit model
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
)
model = Prophet()
model.fit(dataset)

# Log summary metadata (including model, dataset, forecast and charts)
run["prophet_summary"] = npt_utils.create_summary(model=model, df=df, fcst=forecast)

# Stop the run
run.stop()
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs.neptune.ai/getting_help).
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! In the Neptune app, click the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP).
* You can just shoot us an email at [support@neptune.ai](mailto:support@neptune.ai).
