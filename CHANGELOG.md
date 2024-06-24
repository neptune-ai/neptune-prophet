## neptune-prophet 1.0.2

### Changes
- Constrained `numpy` to `<2.0.0` ([#35](https://github.com/neptune-ai/neptune-prophet/pull/35))

## neptune-prophet 1.0.1

### Changes
- Bumped `plotly` and unpinned `matplotlib` ([#33](https://github.com/neptune-ai/neptune-prophet/pull/33))

## neptune-prophet 1.0.0

### Changes
- Removed `neptune` and `neptune-client` from base requirements  ([#30](https://github.com/neptune-ai/neptune-prophet/pull/30))

## neptune-prophet 0.5.0
- Updated integration for compatibility with `neptune` v1 ([#26](https://github.com/neptune-ai/neptune-prophet/pull/26))

## neptune-prophet 0.4.1

### Changes
- Moved `neptune_prophet` package to `src` directory ([#15](https://github.com/neptune-ai/neptune-prophet/pull/15))
- Drop usage of deprecated File.extension init attribute ([#21](https://github.com/neptune-ai/neptune-prophet/pull/21))
- Poetry as a package builder ([#22](https://github.com/neptune-ai/neptune-prophet/pull/22))

### Fixes
- Fixed import from `impl` package ([#18](https://github.com/neptune-ai/neptune-prophet/pull/18))

## neptune-prophet 0.4.0

### Fixes

- Fixed required version of statsmodels that was causing problems in Colab ([#12](https://github.com/neptune-ai/neptune-prophet/pull/12))

## neptune-prophet 0.3.0

### Fixes
- Changed the default to `log_interactive = False` ([#10](https://github.com/neptune-ai/neptune-prophet/pull/10))

## neptune-prophet 0.2.0

### Fixes
- Fixed typo in length error ([#8](https://github.com/neptune-ai/neptune-prophet/pull/8))
- Added missing file extension for serialized model ([#8](https://github.com/neptune-ai/neptune-prophet/pull/8))

## neptune-prophet 0.1.0

### Features
- Logging model metadata, serialized models, Prophet's plots, and additional diagnostic plots ([#3](https://github.com/neptune-ai/neptune-prophet/pull/3))
