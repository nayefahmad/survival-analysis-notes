"""
# Sampling from an arbitrary hazard function

This notebook shows how we can start with any arbitrary hazard function, and sample
failure times (both censored and uncensored). We show that the presence of censoring
causes histograms and ECDFs to be systematically biased if we only work with the
uncensored data points.

To sample from the hazard function, we do the following:
1. Integrate the hazard to get cumulative hazard, H(t)
2. Use a standard transform from cumulative hazard to survival function, S(t)
3. Use 1-S(t) to get the CDF
4. Use inversion to get inverse CDF, and then draw random samples from the CDF.

Once we set up a framework for sampling from hazard functions, we can use this to
generate data that is known to follow the assumptions of the CoxPH model, and we can
set the parameters of the true data generating function. Then we can test how well the
CoxPHFitter can recover the known true parameter values.

Reference: https://gist.github.com/jcrudy/10481743

"""

from typing import Callable, Tuple

from functools import partial

import lifelines
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.integrate
from statsmodels.distributions import ECDF


class HazardSampler:
    def __init__(self, hazard, start=0.0, step=None):
        self.hazard = hazard
        if step is None:
            # heuristic for setting step size to be used in numerical inverse CDF
            # implementation
            # todo: are there better heuristics?
            h0 = hazard(0.0)
            if h0 > 0:
                step = 2.0 / hazard(0.0)
            else:
                # Reasonable default. Not efficient in some cases.
                step = 200.0 / scipy.integrate.quad(hazard, 0.0, 100.0)
        self.cumulative_hazard = CumulativeHazard(hazard)
        self.survival_function = SurvivalFunction(self.cumulative_hazard)
        self.cdf = Cdf(self.survival_function)
        self.inverse_cdf = InverseCdf(self.cdf, start=start, step=step, lower=0.0)
        self.sampler = InversionTransformSampler(self.inverse_cdf)

    def draw(self):
        return self.sampler.draw()


class InversionTransformSampler:
    def __init__(self, inverse_cdf):
        self.inverse_cdf = inverse_cdf

    def draw(self):
        u = np.random.uniform(0, 1)
        return self.inverse_cdf(u)


class CumulativeHazard:
    def __init__(self, hazard):
        self.hazard = hazard

    def __call__(self, t):
        return scipy.integrate.quad(self.hazard, 0.0, t)[0]


class SurvivalFunction:
    def __init__(self, cumulative_hazard):
        self.cumulative_hazard = cumulative_hazard

    def __call__(self, t):
        return np.exp(-self.cumulative_hazard(t))


class Cdf:
    def __init__(self, survival_function):
        self.survival_function = survival_function

    def __call__(self, t):
        return 1.0 - self.survival_function(t)


class InverseCdf:
    def __init__(
        self, cdf, start, step, precision=1e-2, lower=float("-inf"), upper=float("inf")
    ):
        self.cdf = cdf
        self.precision = precision
        self.start = start
        self.step = step
        self.lower = lower
        self.upper = upper

    def __call__(self, target_p):
        """
        Takes a proportion p (y-axis value of the cdf) and returns the corresponding
        x value from the cdf (to within a certain precision level).
        """
        last_diff = None
        step = self.step
        current_x = self.start
        while True:
            current_cdf_value = self.cdf(current_x)
            diff = current_cdf_value - target_p
            if abs(diff) < self.precision:
                break
            elif diff < 0:
                # current_x is too far to the left, so we increase it
                current_x = min(current_x + step, self.upper)
                if last_diff is not None and last_diff > 0:
                    # if diff and last_diff are opposite signs, take smaller
                    # optimization steps
                    step *= 0.5
                last_diff = diff
            else:
                # current_x is too far to the right, so we decrease it
                current_x = max(current_x - step, self.lower)
                if last_diff is not None and last_diff < 0:
                    # if diff and last_diff are opposite signs, take smaller
                    # optimization steps
                    step *= 0.5
                last_diff = diff
        return current_x


def create_plots(idx: int, hazard: Callable) -> None:
    """
    Plot the hazard, samples from the hazard, and survival fn/ECDF comparison
    """

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(16, 8))
    xmax = 30

    txt = "Hazard function that we will sample from"
    x = np.linspace(0, xmax, 1000)
    y = hazard(x)
    ax[0].plot(x, y)
    ax[0].set_ylabel("hazard")
    ax[0].set_title(txt)
    ax[0].set_ylim(0, 1.5)
    ax[0].set_xlim(0, xmax + 10)

    # Sample failure times from the hazard function
    sampler = HazardSampler(hazard)
    print(f"Hazard {idx}, drawing samples for true failure times ... ")
    failure_times = np.array([sampler.draw() for _ in range(m)])
    print(f"Hazard {idx}, max failure time: {np.max(failure_times):.5f}")

    # Apply some non-informative right censoring, just to demonstrate how it's done
    # We use the same distribution to sample censoring times so that censoring is
    # independent from time interval. See: https://statwonk.com/weibull.html
    print(f"Hazard {idx}, drawing samples for censoring times ... ")
    censor_times = np.array([sampler.draw() for _ in range(m)])
    y = np.minimum(failure_times, censor_times)
    # c = 1.0 * (censor_times > failure_times)

    # Make some plots of the simulated data
    # Plot a histogram of failure times from this hazard function
    ax[1].hist(failure_times, bins=50)
    ax[1].set_title("True Failure Times")
    ax[1].set_xlim(0, xmax + 10)
    ax[1].set_ylabel("count")

    # Plot a histogram of censored failure times from this hazard function
    ax[2].hist(y, bins=50)
    ax[2].set_title("Observed Failure Times (Including Censored)")
    ax[2].set_xlim(0, xmax + 10)
    ax[2].set_ylabel("count")

    # Plot the empirical survival function (based on the censored sample) against
    # the actual survival function
    t = np.arange(0, xmax, 0.1)
    S = np.array([sampler.survival_function(t[i]) for i in range(len(t))])
    S_hat = 1.0 - ECDF(y)(t)

    ax[3].set_title("Survival Function Comparison")
    ax[3].plot(t, S, "r", lw=3, label="True survival function")
    ax[3].plot(t, S_hat, "b--", lw=3, label="1 - ECDF (biased due to censoring)")
    ax[3].legend()
    ax[3].set_xlabel("Time")
    ax[3].set_ylabel("Proportion Still Alive")
    ax[3].set_ylim(0, 1)
    ax[3].set_xlim(0, xmax + 10)

    plt.tight_layout()
    fig.show()


def set_params() -> np.array:
    """
    Set the coefficients of each variable that will be used in the simuatation of the
    Cox PH model
    """
    return np.array([-1, 1, 1, 0, 0.002, 0])


def individual_hazard_function(
    t: float, baseline_hazard: Callable, X: np.array
) -> float:
    """
    Use the baseline hazard to calculate baseline hazard, then adjust it using the
    relative hazard.

    - t: time at which to calculate individual hazard
    - X: np.array of time-invarying features for this individual
    """
    params = set_params()
    base = baseline_hazard(t)
    relative_hazard = np.exp(X @ params.T)
    return base * relative_hazard


def generate_dataset() -> pd.DataFrame:
    """
    Simulate features for all individuals, then for each individual, draw a failure
    time from their individual hazard function.
    """
    params = set_params()
    baseline_hazard = hazard_sine
    n_samples = 100
    n_features = params.shape[0]

    mean = np.zeros(n_features)
    covariance = np.eye(n_features)
    X = np.random.multivariate_normal(mean, covariance, size=n_samples)

    result = []
    for idx, row in enumerate(X):
        observed, event_indicator = sample_failure_time(row, baseline_hazard)
        df_temp = pd.DataFrame(
            {
                "idx": [idx],
                "observed_time": [observed],
                "is_uncensored": [event_indicator],
            }
        )
        result.append(df_temp)

    df_out = pd.concat(result).reset_index(drop=True)

    feature_cols = pd.DataFrame(X)
    feature_cols.columns = [f"x_{col}" for col in feature_cols.columns]
    df_out = pd.concat([feature_cols, df_out], axis=1)

    other_cols = [col for col in df_out.columns if col != "idx"]
    df_out = df_out[["idx"] + other_cols]

    return df_out


def sample_failure_time(X: np.ndarray, baseline_hazard: Callable) -> Tuple:
    """
    Use the features for an individual to create their individual hazard function,
    then sample from that hazard function.

    Returns failure time and censoring indicator
    """
    hazard = partial(
        individual_hazard_function, baseline_hazard=baseline_hazard, X=np.array(X)
    )

    sampler = HazardSampler(hazard)
    failure_time = sampler.draw()
    censor_time = sampler.draw()
    observed = np.min([failure_time, censor_time])
    event_indicator = 1.0 if failure_time < censor_time else 0.0

    return observed, event_indicator


def extract_params(cph: lifelines.CoxPHFitter) -> pd.DataFrame:
    """
    Extract fitted params from CPH model, and return along with the known true values.
    """
    params = set_params()
    df = pd.DataFrame({"actual_coef": params, "fitted": cph.summary["coef"]})
    df["abs_diff"] = np.abs(df["actual_coef"] - df["fitted"])
    return df


if __name__ == "__main__":
    # Set a random seed and sample size
    np.random.seed(11)
    m = 100

    # Define hazard functions to sample from
    def hazard_sine(t):
        return np.exp(np.sin(t) - 2.0)

    def hazard_piecewise(t):
        return np.where(t < 3, 0.3, 1.0)

    hazard_person_01 = partial(
        individual_hazard_function,
        baseline_hazard=hazard_sine,
        X=np.array([1, 0, 0, 0, 0, 0]),
    )

    # todo: add weibull hazard fn

    hazard_functions = [hazard_sine, hazard_piecewise, hazard_person_01]

    recreate_plots = False
    if recreate_plots:
        for idx, hazard in enumerate(hazard_functions):
            create_plots(idx, hazard)

    df = generate_dataset()

    df = df.drop(columns=["idx"])

    # Fit without regularization:
    cph = CoxPHFitter(penalizer=0)
    cph.fit(df, "observed_time", "is_uncensored")
    # cph.print_summary()
    df_params = extract_params(cph)
    print(df_params)

    # Fit with regularization:
    penalizer_grid = [0, 0.1, 0.2, 0.5, 1.0]

    for hyperparam in penalizer_grid:
        print(f"Penalizer: {hyperparam}".ljust(99, "-"))
        cph_02 = CoxPHFitter(penalizer=hyperparam)
        cph_02.fit(df, "observed_time", "is_uncensored")
        # cph_02.print_summary()
        df_params = extract_params(cph_02)
        print(df_params)
        print("\n")

    print("done")
