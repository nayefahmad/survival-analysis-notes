# # Using lifelines to fit KM curves

# References:
#   - https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html  # noqa


from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.datasets import load_waltons
import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd

from IPython.core.interactiveshell import InteractiveShell  # noqa

InteractiveShell.ast_node_interactivity = "all"

# ## Example 1
waltons = load_waltons()
waltons.columns = ["time", "event", "group"]
waltons.describe(include="all").T

kmf = KaplanMeierFitter(label="waltons_data")
kmf.fit(waltons["time"], waltons["event"])
kmf.plot_survival_function()
plt.show()

# ## Smoothing the survival curve by using smoothed hazard function

# Reference: [github link](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-02-09_smoothing-the-km-estimate.md) # noqa

# This is the `gastricXelox` data in R package `asaur`.

time_months = [
    0.9256198,
    1.8512397,
    1.8512397,
    1.8512397,
    2.0826446,
    2.5454545,
    2.7768595,
    3.0082645,
    3.7024793,
    3.7024793,
    3.9338843,
    3.9338843,
    4.3966942,
    4.8595041,
    5.5537190,
    5.5537190,
    5.7851240,
    6.4793388,
    6.4793388,
    6.9421488,
    8.5619835,
    8.5619835,
    9.7190083,
    9.9504132,
    9.9504132,
    10.6446281,
    11.1074380,
    11.5702479,
    11.8016529,
    12.2644628,
    12.4958678,
    13.1900826,
    13.6528926,
    13.6528926,
    13.8842975,
    14.8099174,
    15.2727273,
    17.5867769,
    18.0495868,
    21.0578512,
    27.5371901,
    27.7685950,
    32.1652893,
    40.7272727,
    43.2727273,
    46.9752066,
    50.2148760,
    58.5454545,
]

delta = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

# KM curve:

km2 = KaplanMeierFitter(label="gastricXelox data")
km2.fit(time_months, delta)
fig, ax = plt.subplots()
km2.plot_survival_function(ax=ax)
ax.set_title("KM estimate of survival function")
fig.show()


# ## Nelson-Aalen-based cumulative hazard function and smoothed hazard fn:

na1 = NelsonAalenFitter()
na1.fit(time_months, delta)
fig, ax = plt.subplots()
na1.plot_cumulative_hazard(ax=ax)
ax.set_title("Estimated cumulative hazard function")
fig.show()

bandwidth = 3
fig, ax = plt.subplots()
na1.plot_hazard(bandwidth=bandwidth, ax=ax)
ax.set_title("Estimated hazard function")
fig.show()

# Recover the data underlying the plot using the `smoothed_hazard_()` method

df_smoothed_hazard = (
    na1.smoothed_hazard_(bandwidth=bandwidth)
    .reset_index()
    .rename(
        columns={"index": "time", "differenced-NA_estimate": "hazard_estimate"}
    )  # noqa
)
df_smoothed_hazard.head()
df_smoothed_hazard.tail()

n_haz = len(df_smoothed_hazard)
times_diff = df_smoothed_hazard["time"].diff()
times_diff = times_diff[1:].reset_index(drop=True)

hazards = df_smoothed_hazard["hazard_estimate"][0:-1]
assert len(hazards) == len(times_diff)

surv_smoothed = np.exp(-np.cumsum(hazards * times_diff))
# todo: fix x-values for plotting

# todo: show km estimate on same graph
# df_surv_smoothed = pd.DataFrame({'time': time_months,
#                                  'surv_from_smoothed_hazard': surv_smoothed})

fig, ax = plt.subplots()
surv_smoothed.plot(ax=ax)
ax.set_ylim((0, 1))
ax.set_title(
    f"Estimated survival curve based on smoothed estimate of hazard function \nBandwidth={bandwidth}"  # noqa
)
fig.show()

km2.survival_function_
