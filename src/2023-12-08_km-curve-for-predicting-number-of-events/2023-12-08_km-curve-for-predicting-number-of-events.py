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
import pandas as pd
from prediction_and_evaluation import (
    predict_events_for_new_person_using_event_table,
    predict_prob_one_or_more_events_by_specified_time,
)

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
