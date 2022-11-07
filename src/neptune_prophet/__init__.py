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
    "__version__",
    "create_forecast_plots",
    "create_residual_diagnostics_plots",
    "create_summary",
    "get_forecast_components",
    "get_model_config",
    "get_serialized_model",
]

import sys

if sys.version_info >= (3, 8):
    from importlib.metadata import (
        PackageNotFoundError,
        version,
    )
else:
    from importlib_metadata import (
        PackageNotFoundError,
        version,
    )

from neptune_prophet.impl import (
    create_forecast_plots,
    create_residual_diagnostics_plots,
    create_summary,
    get_forecast_components,
    get_model_config,
    get_serialized_model,
)

try:
    __version__ = version("neptune-prophet")
except PackageNotFoundError:
    # package is not installed
    pass
