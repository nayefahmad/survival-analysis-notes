import lifelines
import pandas as pd
from sklearn.model_selection import KFold


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


if __name__ == "__main__":
    cross_validate()
