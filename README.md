# Neptune + Prophet integration

Experiment tracking, model registry, data versioning, and live model monitoring for Prophet trained models.

## What will you get with this integration?

* Log, display, organize, and compare ML experiments in a single place
* Version, store, manage, and query trained models and model-building metadata
* Record and monitor model training, evaluation, or production runs live

## What will be logged to Neptune?

* parameters,
* forecast data frames,
* residual diagnostic charts,
* [other metadata](https://docs.neptune.ai/you-should-know/what-can-you-log-and-display)

![image](https://user-images.githubusercontent.com/97611089/188817349-973a49b2-e0d3-44dd-b51d-7dec670158f9.png)
*Example dashboard in the Neptune app showing diagnostic charts*

## Resources

* [Documentation](https://docs.neptune.ai/integrations-and-supported-tools/model-training/prophet)
* [Code example on GitHub](https://github.com/neptune-ai/examples/tree/main/integrations-and-supported-tools/prophet/scripts)
* [Example project in the Neptune app](https://app.neptune.ai/o/common/org/fbprophet-integration/experiments?split=tbl&dash=charts&viewId=standard-view)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/main/integrations-and-supported-tools/prophet/notebooks/Neptune_prophet.ipynb)

## Example

### Before you start

- [Install and set up Neptune](https://docs.neptune.ai/getting-started/installation).
- Have Prophet installed.

### Installation

```python
pip install neptune-prophet
```

### Logging example

```python
import pandas as pd
from prophet import Prophet
import neptune.new as neptune
import neptune.new.integrations.prophet as npt_utils


# Start a run
run = neptune.init_run(project="common/fbprophet-integration", api_token=neptune.ANONYMOUS_API_TOKEN)


# Load dataset and fit model
dataset = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'
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

* Check our [FAQ page](https://docs.neptune.ai/getting-started/getting-help#frequently-asked-questions).
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! In the Neptune app, click the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP).
* You can just shoot us an email at [support@neptune.ai](mailto:support@neptune.ai).
