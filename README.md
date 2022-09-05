# Neptune-Prophet integration

## Before you start

- [Install and set up Neptune](https://docs.neptune.ai/getting-started/installation).
- Have Prophet installed.

## Installation

```python
pip install neptune-prophet
```

## Logging example

```python
import pandas as pd
from prophet import Prophet
import neptune.new as neptune
import neptune.new.integrations.prophet as npt_utils

# Create a Neptune run
run = neptune.init_run()

# Train model
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
)
model = Prophet()
model.fit(dataset)

# Log results to Neptune run
run["prophet_summary"] = npt_utils.create_summary(model, dataset)
```

For more detailed instructions, see the [Neptune docs](https://docs.neptune.ai/integrations-and-supported-tools/model-training/prophet).
