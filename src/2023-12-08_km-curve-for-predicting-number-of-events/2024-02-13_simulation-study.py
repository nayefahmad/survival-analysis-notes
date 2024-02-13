import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation import generate_data
from prediction_and_evaluation import cross_validate
from scipy.stats import exponweib

pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 500)

SEED = 2024
np.random.seed(SEED)

NUM_ITERATIONS = 100
CV_FOLDS = 5

SIM_HORIZON = 999
FORECAST_HORIZON = 50
NUM_TAILS = 15

DIST = exponweib
PARAM_C_RANGE = [0.7, 3.0]
PARAM_SCALE_RANGE = [20, 500]
params = {"a": 1, "c": 0.9, "scale": 100}

randomise_params = False
if randomise_params:
    c = np.random.uniform(low=PARAM_C_RANGE[0], high=PARAM_C_RANGE[1])
    scale = np.random.uniform(low=PARAM_SCALE_RANGE[0], high=PARAM_SCALE_RANGE[1])
    params = {"a": 1, "c": c, "scale": scale}


print(f"INFO: params: {params}")
iter_results = []
# In this simulation study, we run several iterations. For actual data, we can run over
# several parts/FFF groups
for idx in range(NUM_ITERATIONS):
    print(f"iter {idx}".ljust(88, "-"))
    df_tbe = None
    attempt = 0
    max_attempts = 5
    while df_tbe is None and attempt < max_attempts:
        try:
            df_tbe = generate_data(
                DIST,
                SIM_HORIZON,
                params,
                seed=SEED + idx + attempt,
                num_units=NUM_TAILS,
            )

            cv_scores, y_actuals, y_preds = cross_validate(
                df_tbe, FORECAST_HORIZON, CV_FOLDS
            )
            iter_results.append(cv_scores)

        except AssertionError as e:
            print(f"ERROR: {e}")
            txt = "SUGGESTION: handle cases where the sim horizon is too short, where"
            txt += "the very first sample goes over the horizon"
            print(txt)
            # to prevent infinite while loop, we need to change the seed
            attempt += 1

        if attempt >= max_attempts:
            print("Maximum attempts reached, moving to next iteration.")
            break


cols = [f"error_fold_0{x + 1}" for x in range(CV_FOLDS)]
results = pd.DataFrame(iter_results, columns=cols)

mae_metric = results.abs().mean(axis=1)

df_results = pd.concat([results, mae_metric], axis=1)
df_results = df_results.rename(columns={0: "mean_abs_error"})
print(df_results)

txt = f"Distribution of cross-validated MAE across {len(df_results)} iterations"
txt += f"\nParams: {params}"
txt += f"\nNum tails: {NUM_TAILS}"
txt += f"\nForecast horizon: {FORECAST_HORIZON}"
fig, ax = plt.subplots()
ax.hist(df_results["mean_abs_error"])
ax.set_title(txt)
plt.tight_layout()
plt.show()


print("done")
