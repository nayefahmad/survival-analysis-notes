import pandas as pd
import matplotlib.pyplot as plt
from reliability.Repairable_systems import MCF_parametric
from reliability.Fitters import Fit_Exponential_2P


# Assuming your data is stored in a DataFrame named 'data'
# with columns 'part', 'aircraft', 'event_time', and 'total_hours'

# Prepare the data for the 'reliability' package
# Create a list of dictionaries for each system (aircraft)

data = pd.DataFrame()
parts = []
aircrafts = []

for part in parts:
    part_data = data[data["part"] == part]
    mcf_data = []

    for aircraft in aircrafts:
        # Get event times for this part-aircraft combination
        aircraft_events = part_data[part_data["aircraft"] == aircraft][
            "event_time"
        ].tolist()

        # If there are no events, add a censored time at total_hours
        if not aircraft_events:
            total_hours = data.loc[
                (data["part"] == part) & (data["aircraft"] == aircraft), "total_hours"
            ].values[0]
            mcf_data.append(
                {
                    "System ID": aircraft,
                    "Event Times": [total_hours],
                    "Censored": [True],
                }
            )
        else:
            mcf_data.append(
                {
                    "System ID": aircraft,
                    "Event Times": aircraft_events,
                    "Censored": [False] * len(aircraft_events),
                }
            )

    # Flatten the list for the MCF_parametric input
    mcf_input = []
    for entry in mcf_data:
        for idx, time in enumerate(entry["Event Times"]):
            mcf_input.append(
                {
                    "System ID": entry["System ID"],
                    "Event Time": time,
                    "Censored": entry["Censored"][idx],
                }
            )

    mcf_df = pd.DataFrame(mcf_input)

    # Compute the MCF using the 'reliability' package
    mcf = MCF_parametric(
        data=mcf_df,
        time_column="Event Time",
        system_id_column="System ID",
        censored_column="Censored",
    )

    # Plot the MCF
    mcf.plot(title=f"MCF Curve for Part {part}")
    plt.show()

    # Store the MCF for future predictions if needed
    # You can access mcf.times and mcf.MCF for the event times and MCF values


# Fit a parametric model to the MCF for extrapolation
# Here, we use the Exponential distribution as an example

for part in parts:
    # Assuming you have computed the MCF as shown above
    mcf = MCF_parametric(
        data=mcf_df,
        time_column="Event Time",
        system_id_column="System ID",
        censored_column="Censored",
    )

    # Fit an exponential model to the MCF
    fit = Fit_Exponential_2P(
        failures=mcf.times, right_censored=None, show_probability_plot=False
    )

    # Predict the expected number of events at future times
    total_hours_list = data["total_hours"].unique()
    predictions = []
    for total_hours in total_hours_list:
        mcf_current = fit.cdf(total_hours) * len(aircrafts)
        mcf_future = fit.cdf(total_hours + 100) * len(aircrafts)
        expected_events = mcf_future - mcf_current
        predictions.append(
            {
                "part": part,
                "total_hours": total_hours,
                "expected_events": expected_events,
            }
        )

    predictions_df = pd.DataFrame(predictions)
    print(predictions_df)
