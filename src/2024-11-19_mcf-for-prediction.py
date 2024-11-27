import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Simulate data
np.random.seed(42)

# Define parts and aircraft
parts = range(1, 11)  # 10 parts
aircrafts = range(1, 51)  # 50 aircraft

# Simulate total cumulative flight hours for each aircraft
aircraft_total_hours = {
    aircraft: np.random.uniform(500, 1000) for aircraft in aircrafts
}

# Create a DataFrame for aircraft and their total_hours
aircraft_hours_df = pd.DataFrame(
    {
        "aircraft": list(aircraft_total_hours.keys()),
        "total_hours": list(aircraft_total_hours.values()),
    }
)

# Create DataFrame with all part-aircraft combinations
part_aircraft_df = pd.DataFrame(
    [(part, aircraft) for part in parts for aircraft in aircrafts],
    columns=["part", "aircraft"],
)

# Merge to get total_hours for each part-aircraft combination
part_aircraft_hours_df = part_aircraft_df.merge(
    aircraft_hours_df, on="aircraft", how="left"
)

# Simulate event data
data_list = []

for part in parts:
    for aircraft in aircrafts:
        total_hours = aircraft_total_hours[aircraft]
        # Simulate the number of events (e.g., removals) based on total hours
        num_events = np.random.poisson(lam=total_hours / 2000)
        # Generate event times uniformly distributed over total_hours
        if num_events > 0:
            event_times = np.sort(np.random.uniform(0, total_hours, size=num_events))
            for event_time in event_times:
                data_list.append(
                    {"part": part, "aircraft": aircraft, "event_time": event_time}
                )

# Create DataFrame of events
data = pd.DataFrame(data_list)


# Function to compute the MCF for a given part
def compute_mcf(part_data, part_aircraft_hours_df):
    # Get unique event times
    event_times = np.sort(part_data["event_time"].unique())
    if len(event_times) == 0:
        return np.array([]), np.array([])
    # Count events at each time
    event_counts = (
        part_data.groupby("event_time").size().reindex(event_times, fill_value=0)
    )
    # Get total_hours for all aircraft for the part
    aircraft_hours = part_aircraft_hours_df[part_aircraft_hours_df["part"] == part][
        ["aircraft", "total_hours"]
    ]
    n_risk = []
    for t in event_times:
        n_at_risk = (aircraft_hours["total_hours"] >= t).sum()
        n_risk.append(n_at_risk)
    n_risk = np.array(n_risk)
    # Calculate MCF increments and cumulative MCF
    delta_mcf = event_counts.values / n_risk
    mcf = np.cumsum(delta_mcf)
    return event_times, mcf


# Compute and store MCF for each part
mcf_dict = {}

for part in parts:
    part_data = data[data["part"] == part]
    event_times, mcf = compute_mcf(part_data, part_aircraft_hours_df)
    mcf_dict[part] = {"event_times": event_times, "mcf": mcf}
    # Plot MCF curve for the part if there are events
    if len(event_times) > 0:
        plt.figure()
        plt.step(event_times, mcf, where="post", label=f"Part {part}")
        plt.xlabel("Cumulative Flight Hours")
        plt.ylabel("Mean Cumulative Number of Events")
        plt.title(f"MCF Curve for Part {part}")
        plt.legend()
        plt.show()

# Fit a polynomial regression model with Lasso regularization to the MCF curve of each
# part
degree = 3  # Degree of the polynomial
alpha = 0.01  # Regularization strength  # todo: use lassoCV to find optimum alpha

models = {}

for part in parts:
    mcf_data = mcf_dict[part]
    X = np.array(mcf_data["event_times"]).reshape(-1, 1)
    y = np.array(mcf_data["mcf"])
    if len(X) > 0:
        # Create a pipeline that includes polynomial features, scaling, and Lasso
        # regression
        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree)),
                ("scaler", StandardScaler()),
                ("lasso", Lasso(alpha=alpha, max_iter=10000)),
            ]
        )
        # Fit the model
        model.fit(X, y)
        models[part] = model
        # Plot MCF curve and fitted polynomial
        X_plot = np.linspace(0, max(X), 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        plt.figure()
        plt.step(X.flatten(), y, where="post", label="MCF")
        plt.plot(X_plot, y_plot, label="Fitted Polynomial", color="red")
        plt.xlabel("Cumulative Flight Hours")
        plt.ylabel("Mean Cumulative Number of Events")
        plt.title(f"MCF Curve and Fitted Polynomial for Part {part}")
        plt.legend()
        plt.show()
    else:
        # If there are no events, use a model that predicts zero
        models[part] = None


# Function to extrapolate MCF to any cumulative flight hours value
def predict_mcf(part, cumulative_flight_hours):
    model = models[part]
    if model is not None:
        cumulative_flight_hours = np.array(cumulative_flight_hours).reshape(-1, 1)
        mcf_pred = model.predict(cumulative_flight_hours)
        # Ensure non-negative MCF predictions
        mcf_pred = np.maximum(mcf_pred, 0)
    else:
        mcf_pred = np.zeros_like(cumulative_flight_hours)
    return mcf_pred


# Predict the number of events in the next 100 flight hours for each part-aircraft
# combination
predictions = []

for part in parts:
    part_aircrafts = part_aircraft_hours_df[part_aircraft_hours_df["part"] == part]
    for idx, row in part_aircrafts.iterrows():
        aircraft = row["aircraft"]
        total_hours = row["total_hours"]
        # Predict MCF at current and future cumulative flight hours
        mcf_current = predict_mcf(part, [total_hours])[0]
        mcf_future = predict_mcf(part, [total_hours + 100])[0]
        # Expected number of events in the next 100 flight hours
        expected_events = mcf_future - mcf_current
        # Ensure non-negative expected events
        expected_events = max(expected_events, 0)
        predictions.append(
            {"part": part, "aircraft": aircraft, "expected_events": expected_events}
        )

# Create a DataFrame of the predictions
predictions_df = pd.DataFrame(predictions)

# Display the predictions
print(predictions_df)
