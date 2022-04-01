# # Using lifelines to fit survival curves

# This example uses KM curves, and Nelson-Aalen curves. It shows how we can
# use a smoothed estimate of the hazard function to get a smoothed survival
# function - similar to [this example in R](https://github.com/nayefahmad/survival-analysis-notes/blob/main/src/2022-02-09_smoothing-the-km-estimate.md)  # noqa

# References:
#   - https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html  # noqa


from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.datasets import load_waltons
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.core.interactiveshell import InteractiveShell  # noqa

InteractiveShell.ast_node_interactivity = "all"

# ## Example 1
waltons = load_waltons()
waltons.columns = ["time", "event", "group"]
waltons.describe(include="all").T

kmf = KaplanMeierFitter(label="waltons_data")

kmf.fit(waltons["time"], waltons["event"])
fig, ax = plt.subplots()
kmf.plot_survival_function(ax=ax)
ax.set_title("KM curve - Waltons data")
fig.show()

# Here is the estimated survival function as a dataframe:

kmf.survival_function_.head()

# Here's how predictions work:

times_to_predict = [2, 4, 6, 7, 8, 10]
kmf.predict(times_to_predict)


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
n_km_estimate = len(km2.survival_function_)
assert n_haz == n_km_estimate

times_diff_with_initial_na = df_smoothed_hazard["time"].diff()
times_diff = times_diff_with_initial_na[1:].reset_index(drop=True)

hazards = df_smoothed_hazard["hazard_estimate"][0:-1]
assert len(hazards) == len(times_diff)

surv_smoothed = np.exp(-np.cumsum(hazards * times_diff))

df_surv_smoothed = pd.DataFrame(
    {
        "time": np.cumsum(times_diff),
        "surv_smoothed": surv_smoothed,
    }
)

df_surv_smoothed

fig, ax = plt.subplots()
ax.plot(
    df_surv_smoothed["time"],
    df_surv_smoothed["surv_smoothed"],
    label="smoothed survival curve",
)
ax.plot(
    km2.survival_function_.reset_index()["timeline"],
    km2.survival_function_.reset_index()["gastricXelox data"],
    label="km estimate",
)
ax.set_ylim((0, 1))
ax.set_title(
    f"Estimated survival curve based on smoothed estimate of hazard function \nBandwidth={bandwidth}"  # noqa
)
ax.legend()
fig.show()

# ### Trying several different bandwidths in a loop:


def surv_smoothed(times: pd.Series, hazards: pd.Series) -> pd.DataFrame:
    times_diff = diff_and_drop_initial_na(times)
    hazards = hazards[0:-1]
    surv_smoothed = surv_from_hazard(hazards, times_diff)
    df_surv_smoothed = pd.DataFrame(
        {
            "time": np.cumsum(times_diff),
            "surv_smoothed": surv_smoothed,
        }
    )
    return df_surv_smoothed


def surv_from_hazard(hazards, times_diff):
    return np.exp(-np.cumsum(hazards * times_diff))


def test_surv_from_hazard():
    # todo: finish this
    pass


def diff_and_drop_initial_na(times: pd.Series):
    t_diff = times.diff()
    t_diff = t_diff[1:].reset_index(drop=True)
    return t_diff


def test_drop_na_after_diff():
    times = pd.Series([i for i in range(1, 11)])
    t_diff = diff_and_drop_initial_na(times)
    assert len(t_diff) == len(times) - 1
    assert all(t_diff.notnull())


fig = plt.figure(figsize=(20, 15))
for n, bandwidth in enumerate([0.1, 1, 2, 3, 5, 10]):
    df_smoothed_hazard = (
        na1.smoothed_hazard_(bandwidth=bandwidth)
        .reset_index()
        .rename(
            columns={
                "index": "time",
                "differenced-NA_estimate": "hazard_estimate",
            }  # noqa
        )  # noqa
    )

    df_surv_smoothed = surv_smoothed(
        df_smoothed_hazard["time"], df_smoothed_hazard["hazard_estimate"]
    )

    ax = plt.subplot(3, 2, n + 1)
    ax.plot(
        df_surv_smoothed["time"],
        df_surv_smoothed["surv_smoothed"],
        label="smoothed survival curve",
    )
    ax.plot(
        km2.survival_function_.reset_index()["timeline"],
        km2.survival_function_.reset_index()["gastricXelox data"],
        label="km estimate",
    )
    ax.set_ylim((0, 1))
    ax.set_title(
        f"Estimated survival curve based on smoothed estimate of hazard function \nBandwidth={bandwidth}"  # noqa
    )
    ax.legend()
fig.tight_layout()
fig.show()
