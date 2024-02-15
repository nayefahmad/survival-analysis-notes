from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from typing import Tuple, Dict

from simulation import generate_data
from prediction_and_evaluation import cross_validate
from scipy.stats import exponweib

pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 500)


def default_params():
    return {"a": 1, "c": 0.9, "scale": 100}


@dataclass
class SimulationParams:
    seed: int = 2024
    cv_folds: int = 5

    dist: scipy.stats._continuous_distns = exponweib
    param_c_range: Tuple[float] = (0.7, 3.0)
    param_scale_range: Tuple[float] = (20, 500)
    params: Dict = field(default_factory=default_params)

    sim_horizon: int = 999
    num_iterations: int = 100
    forecast_horizon: int = 100
    num_tails: int = 15


# set baseline scenario:
params_baseline = SimulationParams()
np.random.seed(params_baseline.seed)

# create new scenarios by instantiating SimulationParams instances:
params_02 = SimulationParams(num_tails=100)
params_03 = SimulationParams()
params_03.params["c"] = 1.9
params_04 = SimulationParams(forecast_horizon=50)
params_05 = SimulationParams(forecast_horizon=200)
params_06 = SimulationParams(forecast_horizon=1000)
params_07 = SimulationParams(sim_horizon=300, num_iterations=300)
params_08 = SimulationParams(sim_horizon=200, num_iterations=300)

PARAMS = params_08  # update this as necessary

randomise_params = False
if randomise_params:
    c = np.random.uniform(low=PARAMS.param_c_range[0], high=PARAMS.param_c_range[1])
    scale = np.random.uniform(
        low=PARAMS.param_scale_range[0], high=PARAMS.param_scale_range[1]
    )
    PARAMS.params = {"a": 1, "c": c, "scale": scale}


print(f"INFO: params: {PARAMS}")
iter_results = []
iter_y_actuals = []
# In this simulation study, we run several iterations. For actual data, we can run over
# several parts/FFF groups
for idx in range(PARAMS.num_iterations):
    print(f"iter {idx}".ljust(88, "-"))
    df_tbe = None
    attempt = 0
    max_attempts = 10
    while df_tbe is None and attempt < max_attempts:
        try:
            df_tbe = generate_data(
                PARAMS.dist,
                PARAMS.sim_horizon,
                PARAMS.params,
                seed=PARAMS.seed + idx + attempt,
                num_units=PARAMS.num_tails,
            )

            cv_scores, y_actuals, y_preds = cross_validate(
                df_tbe, PARAMS.forecast_horizon, PARAMS.cv_folds
            )
            iter_results.append(cv_scores)
            iter_y_actuals.append(y_actuals)

        except AssertionError as e:
            print(f"ERROR: {e}")
            txt = "TODO: handle cases where the sim horizon is too short, where "
            txt += "the very first sample goes over the horizon"
            print(txt)
            # to prevent infinite while loop, we need to change the seed
            attempt += 1

        if attempt >= max_attempts:
            print("Maximum attempts reached, moving to next iteration.")
            break


cols = [f"error_fold_0{x + 1}" for x in range(PARAMS.cv_folds)]
results = pd.DataFrame(iter_results, columns=cols)
df_y_actuals = pd.DataFrame(iter_y_actuals)
mean_y_actual_all_sims = np.round(np.mean(df_y_actuals.values.flatten()), decimals=2)

mae_metric = results.abs().mean(axis=1)

df_results = pd.concat([results, mae_metric], axis=1)
df_results = df_results.rename(columns={0: "mean_abs_error"})
print(df_results)

txt = f"Distribution of cross-validated MAE across {len(df_results)} iterations"
txt += f"\nParams: {PARAMS.params}"
txt += f"\nNum tails: {PARAMS.num_tails}"
txt += f"\nForecast horizon:{PARAMS.forecast_horizon}; sim horizon:{PARAMS.sim_horizon}"
txt += f"\nNote: mean(y_actual)={mean_y_actual_all_sims}"
fig, ax = plt.subplots()
ax.hist(df_results["mean_abs_error"])
ax.set_title(txt)
plt.tight_layout()
plt.show()


print("done")
