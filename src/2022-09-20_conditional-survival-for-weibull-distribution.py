# # Conditional survival curves for Weibull distributions

#

# References:
# []()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from lifelines import KaplanMeierFitter

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


shape = c = 1.5
x = weibull_min.rvs(c, size=1000)
df = pd.DataFrame({"values": x, "is_uncensored": np.ones_like(x)})
df.describe()

ticks = np.linspace(weibull_min.ppf(0.01, c), weibull_min.ppf(0.99, c), 100)
fig, ax = plt.subplots()
ax.plot(ticks, weibull_min.pdf(ticks, c))
ax.set_title(f"Weibull pdf; c = {c}")
fig.show()

kmf = KaplanMeierFitter()
kmf.fit(df["values"], df["is_uncensored"])

fig, ax = plt.subplots()
kmf.plot_survival_function(ax=ax)
ax.set_title("Weibull KM curve - no censoring")
fig.show()

df_km = kmf.survival_function_.copy()

# todo: wrap this in a function
# pick a specific value for s, the time that we have already survived.
s = np.quantile(df_km.index, 0.80)
s_round = round(s, 3)
unconditional_surv_s = df_km.loc[s].values[0]
df_km[f"CS(t|s={s_round})"] = df_km["KM_estimate"] / unconditional_surv_s
# correct for cases where KM value is > 1.0
df_km[f"CS(t|s={s_round})"] = np.where(
    df_km[f"CS(t|s={s_round})"] <= 1.0, df_km[f"CS(t|s={s_round})"], 1.0
)

# pull the medians:
median_surv_time_conditional = (
    df_km[f"CS(t|s={s_round})"][df_km[f"CS(t|s={s_round})"] > 0.50]
).index.values[-1]
median_surv_time_unconditional = (
    df_km["KM_estimate"][df_km["KM_estimate"] > 0.50]
).index.values[-1]


# Plot non-parametric condition and unconditional survival curves
fig, ax1 = plt.subplots()
ax1.plot(df_km.index.values, df_km["KM_estimate"], label="S(t)")
ax1.plot(df_km.index.values, df_km[f"CS(t|s={s_round})"], label=f"CS(t|s={s_round})")
ax1.set_title(
    "Unconditional and conditional Weibull survival functions  \n\n- S(t) represents unconditional survival probability at time t  \n- CS(t|s) represents conditional survival over the next t time-units, \n    given that you have already survived up to s",  # noqa
    loc="left",
)
ax1.axvline(x=s, lw=1, ls="dotted", color="grey")
ax1.axhline(y=0.50, lw=1, ls="dotted", color="grey")
ax1.annotate(
    f"s = {s_round}",
    xy=(s, 0),
    xytext=(s + 0.05, -0.01),
)
ax1.annotate(
    f"({round(median_surv_time_unconditional, 3)}, 0.5)",
    xy=(median_surv_time_unconditional, 0.5),
    xytext=(median_surv_time_unconditional, 0.5 + 0.01),
    size=8,
)
ax1.annotate(
    f"({round(median_surv_time_conditional, 3)}, 0.5)",
    xy=(median_surv_time_conditional, 0.5),
    xytext=(median_surv_time_conditional, 0.5 + 0.01),
    size=8,
)
ax1.legend()
ax1.set_xlabel("Time from start")
ax1.set_ylabel("Survival probability")
fig.tight_layout(pad=2)
fig.show()


# ## Parametric approach


def generate_weibull_km(c):
    x_values = set_up_data(c)
    km_values = weibull_km(x_values, c)
    return x_values, km_values


def generate_conditional_weibull_km(c, s):
    x_values = set_up_data(c)
    km_values = weibull_km_conditional(x_values, c, s)
    return x_values, km_values


def set_up_data(c):
    x_values = np.linspace(weibull_min.ppf(0.01, c), weibull_min.ppf(0.99, c), 100)
    return x_values


def weibull_km(x_values: np.array, c: float) -> np.array:
    km_values = np.exp(-1 * (x_values**c))
    return km_values


def weibull_km_conditional(x_values: np.array, c: float, s: float) -> np.array:
    """
    Generate conditional survival estimates
    :param x_values: points to evaluate the conditional km curve on
    :param c: shape param
    :param s: time value that we condition on: i.e. we assume survival up to this time
    :return:
    """
    km_value_at_s = np.exp(-1 * (s**c))
    all_km_values_unclipped = np.exp(-1 * (x_values**c)) / km_value_at_s
    all_km_values = np.where(all_km_values_unclipped <= 1, all_km_values_unclipped, 1)
    return all_km_values


def get_median_survival_time(x_values: np.array, km_values: np.array):
    """
    Picks the last time before the 51st percentile of survival times
    :param x_values:
    :param km_values:
    :return:
    """
    df = pd.DataFrame({"KM_estimate": km_values}, index=x_values)
    df.index.name = "timeline"
    median_surv = (df["KM_estimate"][df["KM_estimate"] > 0.50]).index.values[-1]
    return median_surv


def main():
    x_values, km_values = generate_weibull_km(c)
    median_surv_time_unconditional_parametric = get_median_survival_time(
        x_values, km_values
    )

    # fig, ax = plt.subplots()
    # ax.plot(x_values, km_values, label='Parametric Weibull survival function')
    # kmf.plot_survival_function(ax=ax)
    # ax.set_title('Comparing parametric Weibull survival function with KM estimate',
    #              loc='left')
    # ax.legend()
    # fig.show()

    x_values, km_values_conditional = generate_conditional_weibull_km(c, s)
    median_surv_time_conditional_parametric = get_median_survival_time(
        x_values, km_values_conditional
    )

    # Plot parametric condition and unconditional survival curves
    fig, ax2 = plt.subplots()
    ax2.plot(x_values, km_values, label="Parametric S(t)")
    ax2.plot(x_values, km_values_conditional, label=f"Parametric CS(t|s={s_round})")
    ax2.set_title(
        "Parametric conditional and unconditional Weibull survival functions  \n\n- S(t) represents unconditional survival probability at time t  \n- CS(t|s) represents conditional survival over the next t time-units, \n    given that you have already survived up to s",  # noqa
        loc="left",
    )
    ax2.axvline(x=s, lw=1, ls="dotted", color="grey")
    ax2.axhline(y=0.50, lw=1, ls="dotted", color="grey")
    ax2.annotate(
        f"s = {s_round}",
        xy=(s, 0),
        xytext=(s + 0.05, -0.01),
    )
    ax2.annotate(
        f"({round(median_surv_time_unconditional_parametric, 3)}, 0.5)",
        xy=(median_surv_time_unconditional_parametric, 0.5),
        xytext=(median_surv_time_unconditional_parametric, 0.5 + 0.01),
        size=8,
    )
    ax2.annotate(
        f"({round(median_surv_time_conditional_parametric, 3)}, 0.5)",
        xy=(median_surv_time_conditional_parametric, 0.5),
        xytext=(median_surv_time_conditional_parametric, 0.5 + 0.01),
        size=8,
    )
    ax2.legend()
    ax2.set_xlim((0, 3.5))
    ax2.set_xlabel("Time from start")
    ax2.set_ylabel("Survival probability")
    fig.tight_layout(pad=2)
    fig.show()


if __name__ == "__main__":
    main()
