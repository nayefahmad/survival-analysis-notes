import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import exponweib, lognorm
from typing import Tuple


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


if __name__ == "__main__":
    SEED = 2024
    generate_data_single_unit(
        exponweib, 999, {"a": 1, "c": 0.9, "scale": 100}, seed=SEED
    )
    generate_data_single_unit(lognorm, 999, {"s": 0.8, "scale": 100}, seed=SEED)
