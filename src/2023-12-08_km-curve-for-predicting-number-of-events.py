"""
# Deriving expected number of events from the KM curve

## Overview
In the PMD we use the KM curve to derive P(>=1 event) in specified intervals. We are
also interested in E(num_events) estimates in those intervals, but currently we don't
have a good way of getting them, as the KM curve does not directly yield them. In fact,
the KM curve is designed for non-recurrent-event analysis, and would give very incorrect
E(num_events) values if used directly in the recurrent-event context.

In this file, the first thing to note is the kmf.event_table, which makes it clear that
the KM curve is not really set up for recurrent-event analysis.

However, we can generate a prediction of E(num_events) by using the event table, as
shown here. We are basically averaging over all entities to get the estimate, in a way
that's very similar to the MCF.

## Next steps:
This analysis is for an unconditional KM curve. That is, it answers the question:
what is E(num_event) for an entity that has just entered the analysis. It does not
account for cases where the new entity has already survived for X hours.

We need a way to generate conditional event tables (alongside conditional KM curves,
which we already do) in order to answer this question. Alternatively, maybe we can
just take the difference between two unconditional estimates? See
`n_events_between_10_and_20` below.
"""

import lifelines
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import exponweib, lognorm
from sklearn.model_selection import KFold
from typing import Tuple


def predict_events_for_new_person_using_event_table(
    prediction_time: float, event_table: pd.DataFrame, df_tbe: pd.DataFrame
) -> float:
    """
    This is a heuristic method for estimating the expected number of events that a
    new unit (starting at time 0) will experience by time=`prediction_time`.

    The logic is similar to how the MCF works: We average the total number of events
    observed up to time point `prediction_time` across the number of units that
    are at risk at the start.

    There is a critical difference between the way we define the number of units at
    risk here vs how it shows up in the KM event table. In the event table, each
    TBE interval is assumed to be associated with a separate unit, because it does
    not account for the recurrent nature of the data (which actually allows one unit to
    contribute multiple TBE intervals).

    Example:
    - prediction_time = 5.0
    - number of observed events = 6
    - number of contributing units = 8
    - expected number of events per unit within 5 time units = 6/8 = .75

    Breaking this down to the level of each id:
    - id=1 had 1 event on or before t=5
    - id=2 had 0 events on or before t=5
    - id=3 had 0 events on or before t=5
    - id=4 had 1 event on or before t=5
    - id=5 had 1 event on or before t=5
    - id=6 had 1 event on or before t=5
    - id=7 had 1 event on or before t=5
    - id=8 had 1 event on or before t=5
    """
    num_unique_units = df_tbe["id"].nunique()

    prediction = (
        event_table.query("event_at <= @prediction_time")["observed"].sum()
        / num_unique_units
    )
    return prediction


def predict_prob_one_or_more_events_by_specified_time(
    kmf: lifelines.fitters.kaplan_meier_fitter.KaplanMeierFitter, prediction_time: int
) -> float:
    """
    This is the current methodology used in the MHLH project to get predicted
    survival probabilities. Since we are in the recurrent events context, it is more
    correct to interpret these results as "prob of one *or more* events in the given
    timeframe", rather than "prob of an event in the given timeframe".
    """
    return kmf.predict(prediction_time)


def generate_conditional_event_table():
    # See overview above. This may not be necessary if we just take differences across
    # unconditional estimates.
    pass


def generate_data(
    distribution: scipy.stats._continuous_distns,
    horizon: int,
    params: dict,
    num_units: int = 50,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate simulated TBE data for a set of units. Each unit could be e.g. one tail,
    or one specific physical part.
    """
    dfs = []
    for unit in range(num_units):
        values, is_uncensored = generate_data_single_unit(
            distribution, horizon, params, seed
        )
        id_col = [unit] * len(values)
        df_temp = pd.DataFrame(
            {"id": id_col, "tbe_value": values, "is_uncensored": is_uncensored}
        )
        dfs.append(df_temp)
    df_out = pd.concat(dfs, axis=0)
    return df_out


def generate_data_single_unit(
    distribution: scipy.stats._continuous_distns,
    horizon: int,
    params: dict,
    seed: int = None,
) -> Tuple[np.array, np.array]:
    """
    Generates a set of TBE values from the given distribution, as well as the
    is_uncensored indicator values.
    """
    try:
        dist = distribution(**params)
    except TypeError as e:
        print(f"ERROR: {e}")
        raise TypeError("SUGGESTION: Check that parameters match given distribution")

    samples_sum = 0
    n = int(np.ceil(horizon / 50))  # heuristic
    while samples_sum < horizon:
        samples = dist.rvs(n, random_state=seed)
        samples_sum = samples.sum()
        n *= 2

    sample_values, is_uncensored = assign_censoring(samples, horizon)
    assert sample_values.shape[0] == is_uncensored.shape[0]
    return sample_values, is_uncensored


def assign_censoring(
    samples: np.array, horizon: int, tolerance: float = 1e-3
) -> Tuple[np.array, np.array]:
    """
    Checks for the point at which the sampled TBE intervals add up to a value greater
    than the simulation horizon. When this happens, we cut off and censor the last TBE
    interval.
    """
    samples_cumsum = np.cumsum(samples)
    check_greater_than_horizon = samples_cumsum > horizon
    censored_sample_index = check_greater_than_horizon.tolist().index(True)

    sample_values = samples[:censored_sample_index]
    last_sample = horizon - samples_cumsum[censored_sample_index - 1]
    sample_values = np.append(sample_values, [last_sample])
    assert np.abs(sample_values.sum() - horizon) < tolerance

    is_uncensored = np.append(np.ones(censored_sample_index), np.zeros(1))
    return sample_values, is_uncensored


SEED = 2024
generate_data_single_unit(exponweib, 999, {"a": 1, "c": 0.9, "scale": 100}, seed=SEED)
generate_data_single_unit(lognorm, 999, {"s": 0.8, "scale": 100}, seed=SEED)


def cross_validate(folds: int = 5):
    """
    todo: add `df_tbe_data: pd.DataFrame` as an arg. This is especially useful when it's
      real data (not simulated)
    """
    kf = KFold(n_splits=folds, shuffle=True)
    # ids = [id for id in range(df_simulated_data['id'].nunique())]
    ids = [id for id in range(50)]

    cv_scores = []
    for idx, (train, test) in enumerate(kf.split(ids)):
        print(f"fold_num: {idx}")
        print(f"train indices: {train}")
        print(f"test indices: {test}")
        print("calculate metric on test")
        print("fit KM and calculate metric on train")
        print("record error for this split: cv_scores[idx] = current_score")
        print("end".ljust(90, "-"))
        print("\n")

    return cv_scores


cross_validate()


# Main analysis:
df_tbe = pd.DataFrame(
    {
        "id": [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        "value": [10, 5, 10, 20, 1, 10, 3, 8, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1],
        "is_uncensored": [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    }
)
print(df_tbe)

kmf = KaplanMeierFitter().fit(df_tbe["value"], df_tbe["is_uncensored"])
kmf.plot_survival_function()
plt.show()
kmf.event_table

"""
Note that if we simply count the "dips" in the KM curve, we will grossly overestimate
the number of events a new id will have in time<=10. This is because we are ignoring
the recurrent nature of the data.
"""


predicted_events = predict_events_for_new_person_using_event_table(
    10, kmf.event_table, df_tbe
)

thresh = 1e-2
expected_result = 1.25  # makes sense given the number of events each person has had
assert abs(predicted_events - expected_result) <= thresh

predict_prob_one_or_more_events_by_specified_time(kmf, 10)


# Some more examples
# Note that a longer timeframe should always have the same number or more expected
# events than a shorter timeframe.
pred_times = [2, 5, 10, 12, 15, 20, 50, 100]
for time in pred_times:
    pred = predict_events_for_new_person_using_event_table(
        time, kmf.event_table, df_tbe
    )
    print(f"time: {time}, pred={pred}")


# We can find E(num_events) within a specific timeframe by taking the difference
# between two unconditional estimates:
n_events_at_10 = predict_events_for_new_person_using_event_table(
    10, kmf.event_table, df_tbe
)
n_events_at_20 = predict_events_for_new_person_using_event_table(
    20, kmf.event_table, df_tbe
)
n_events_between_10_and_20 = n_events_at_20 - n_events_at_10
assert n_events_between_10_and_20 == 0.125


# Cumulative hazard function using Nelson-Aalen:
naf = lifelines.NelsonAalenFitter().fit(df_tbe["value"], df_tbe["is_uncensored"])
naf.plot()
plt.show()

"""
Similar to the naive application of the KM curve for recurrent data, this will give an
incorrect estimate of number events.
"""


# Trying with CoxPH model to show that it's equivalent to KM when no covariates
# todo: seems like in lifelines you can't fit a CoxPH without covariates
# notes:
# - see CASL book for explanation of why coxph equivalent to KM when no covariates
cph = lifelines.CoxPHFitter().fit(
    df_tbe, duration_col="value", event_col="is_uncensored"
)
cph.print_summary()
prediction_time = 10
try:
    predicted_cumulative_hazard = cph.predict_cumulative_hazard(
        pd.DataFrame(), times=prediction_time
    )
    print(predicted_cumulative_hazard.values[0][0])
except KeyError:
    "doesn't work"


# todo: try coxph with 1 predictor - `tbe_sequence_number`. This let's us analyze
#  whether there is a "recurrence" problem
