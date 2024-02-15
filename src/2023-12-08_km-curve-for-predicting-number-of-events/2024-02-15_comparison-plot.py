import pandas as pd
import matplotlib.pyplot as plt

# Filenames
files = ["01.csv", "02_uniform.csv", "03_always-one.csv"]

# Initialize list to store mean_abs_error data from each file
mean_abs_errors = []

# Read each file and append the mean_abs_error column to the list
for file in files:
    df = pd.read_csv(file)
    mean_abs_errors.append(df["mean_abs_error"])

# Determine the global min and max across all datasets to set uniform x-axis range
min_val = min([data.min() for data in mean_abs_errors])
max_val = max([data.max() for data in mean_abs_errors])

# Plot histograms
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

titles = ["event table method", "dummy: uniform(0,1)", "dummy: always 1.0"]
for i, data in enumerate(mean_abs_errors):
    axs[i].hist(data, bins=20, alpha=0.7, range=(min_val, max_val))
    axs[i].set_title(f"{titles[i]}")
    axs[i].set_xlabel("Mean Absolute Error")
    axs[i].set_ylabel("Frequency")

# Ensure the x-axis is the same for all plots
for ax in axs:
    ax.set_xlim([min_val, max_val])

txt = "Distribution of cross-validated MAE across 100 iterations "
txt += "\nparams_id: baseline case"
plt.suptitle(txt)
plt.tight_layout()
plt.show()


print("done")
