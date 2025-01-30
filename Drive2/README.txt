Running BatchTransform will run a transformation operation per pair provided in the input list. This splits the data into 4 week rolling windows and then profiles them under the DC framework.

The Dataset.py file facilitates the rolling window splitting, sampling of the data and application of indicators.
The Indicators.py file defines the indicators that can be applied to the data.
The Utils.py file defines a number of helper functions that are used across all experiment files.
The Run.py file controls the high level application of this code.
