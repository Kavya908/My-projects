import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("/content/RS_Session_246_AS11.csv","/content/RS_Session_246_AU_98_1.1.csv"))
# os.listdir() accepts only one argument which should be the path to the directory
# To print the files, list the paths individually
print(os.listdir("/content/")) # Assuming files are in /content/
# Or list the files by name
print("RS_Session_246_AS11.csv", "RS_Session_246_AU_98_1.1.csv")
df = pd.read_csv("/content/RS_Session_246_AU_98_1.1.csv")
df.tail()
df.drop(df.index[[36]], inplace=True)
df.isnull().sum() # There are no missing values
plt.figure(figsize=(15, 15));
df.groupby(["State/UT", "2021- Cases registered", "2022 - Cases registered", "2023- Cases registered"]).sum().plot(kind="bar",width=0.7, figsize=(15, 10), title="State Vs Total Case Registered Against Women");
plt.plot();
df.head()
plt.figure(figsize=(15, 15));
df.groupby(["State/UT", "2021 - Total rape Cases", "2022 - Total rape Cases", "2023 - Total rape Cases"]).sum().plot(kind="bar",width=0.7, figsize=(15, 10), title="State Vs Total Rape Case Registered Against Women");
plt.plot();
# Check the actual column names in your DataFrame
print(df.columns)

# Assuming there's a typo and the actual column name is "2021 - Total rape Cases"
# (as seen in the code of input-43)
print(df[["State/UT", "2021- Cases registered", "2021 - Total rape Cases"]])
print(df[["State/UT", "2022 - Cases registered", "2022 - Total rape Cases"]])
print(df[["State/UT", "2023- Cases registered", "2023 - Total rape Cases"]])
#rape percentage
df["Rape_2021_perc"] = (df["2021 - Total rape Cases"]/df["2021- Cases registered"])*100
df["Rape_2021_perc"] = df["Rape_2021_perc"].map(lambda x: round(x, 2))
df["Rape_2022_perc"] = (df["2022 - Total rape Cases"]/df["2022 - Cases registered"])*100
df["Rape_2022_perc"] = df["Rape_2022_perc"].map(lambda x: round(x, 2))
df["Rape_2023_perc"] = (df["2023 - Total rape Cases"]/df["2023- Cases registered"])*100
df["Rape_2023_perc"] = df["Rape_2023_perc"].map(lambda x: round(x, 2))
df.head()
# Display rape percentage for all states
print(df[["State/UT", "Rape_2021_perc", "Rape_2022_perc", "Rape_2023_perc"]])
ax = df.groupby('State/UT')[["Rape_2021_perc", "Rape_2022_perc", "Rape_2023_perc"]].sum().plot.bar(stacked=True, figsize=(15, 15), title="State Vs Rape Percentage")
ax.set_xlabel("States")
ax.set_ylabel("Percentage(%)")
import pandas as pd

# Read the CSV file into a DataFrame
df2 = pd.read_csv("/content/RS_Session_246_AS11.csv")

# Display all rows of the DataFrame
df2
# Remove total column
df2.drop(df2.index[[37, 38, 29]], inplace=True)
df_only_states_ut = df2.iloc[:37, :]
df_only_states_ut.drop(df_only_states_ut.index[29], inplace=True)
df_only_states_ut.groupby("States/UTs").sum().plot.barh(figsize=(15,25), width=0.7);
import pandas as pd

# Convert columns with numeric values to a numeric type before performing the sum
for column in df2.columns:
    if column != "States/UTs":  # Assuming "States/UTs" is the only non-numeric column
        try:
            # Try converting to numeric, errors will be coerced to NaN
            df2[column] = pd.to_numeric(df2[column], errors='coerce')
        except ValueError:
            pass  # Handle the error, e.g., print a message or skip the column

# Create a 'total_crime' column by summing across the crime type columns for each state:
df2["total_crime"] = df2.drop(columns=["States/UTs"]).sum(axis=1) # Sum across columns

df2  # Display all rows of df2 (including all states and total crime across 3 years)
df2.groupby(["States/UTs"])["total_crime"].sum().plot.bar(figsize=(15, 10))
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assuming df2 is already loaded and preprocessed

# Create a new dataframe with state, year, and total crime
df_for_prediction = df2[["States/UTs", "total_crime"]].copy()
df_for_prediction["year"] = [2021, 2022, 2023] * (len(df2) // 3)  # Assuming data from 2021-2023

# Create and train a linear regression model for each state
models = {}  # Store models for each state
for state in df_for_prediction["States/UTs"].unique():
    state_data = df_for_prediction[df_for_prediction["States/UTs"] == state]
    X = state_data[["year"]].values
    y = state_data["total_crime"].values
    model = LinearRegression()
    model.fit(X, y)
    models[state] = model  # Store the model for the state


# Predict and display for each year separately
for year in [2024, 2025]:
    print(f"\nPredictions for {year}:")
    for state in df_for_prediction["States/UTs"].unique():
        prediction = models[state].predict([[year]])[0]  # Get prediction for the year
        print(f"{state}: {prediction:.2f}")
        import pandas as pd
from sklearn.linear_model import LinearRegression

# ... (Data preparation and model training as before) ...

# Function to get predictions and percentages
def get_crime_info(state, future_years=[2024, 2025]):
    historical_total = df_for_prediction[df_for_prediction["States/UTs"] == state]["total_crime"].sum()

    # Get predictions for each future year separately
    future_predictions = [models[state].predict([[year]])[0] for year in future_years]

    # Separate predictions for 2024 and 2025
    prediction_2024 = future_predictions[0]  # Access the prediction for 2024 directly
    prediction_2025 = future_predictions[1]  # Access the prediction for 2025 directly

    future_total = sum(future_predictions)  # Calculate total future crimes
    total_crime = historical_total + future_total

    historical_percentage = (historical_total / total_crime) * 100
    future_percentage = (future_total / total_crime) * 100

    return {
        "historical_total_crime": historical_total,
        "future_total_crime": future_total,
        "total_crime": total_crime,
        "historical_percentage": historical_percentage,
        "future_percentage": future_percentage,
        "prediction_2024": prediction_2024,
        "prediction_2025": prediction_2025
    }


# Get user input for the state
state_input = input("Enter the state name: ")

# Display crime information and caution
try:
    crime_info = get_crime_info(state_input)
    print(f"\nCrime Information for {state_input}:")
    print(f"Total historical crimes (2021-2023): {crime_info['historical_total_crime']:.2f}")
    print(f"Predicted total future crimes (2024-2025): {crime_info['future_total_crime']:.2f}")
    print(f"Predicted crimes for 2024: {crime_info['prediction_2024']:.2f}")
    print(f"Predicted crimes for 2025: {crime_info['prediction_2025']:.2f}")
    print(f"Total crimes (historical + future): {crime_info['total_crime']:.2f}")
    print(f"Historical crime percentage: {crime_info['historical_percentage']:.2f}%")
    print(f"Future crime percentage: {crime_info['future_percentage']:.2f}%")
    print(f"\nCaution: If you are staying in {state_input}, beware of potential crimes.")
except KeyError:
    print(f"State '{state_input}' not found in the data.")
    import pandas as pd
from sklearn.linear_model import LinearRegression

# ... (Data preparation and model training as before) ...

# Function to get predictions and percentages
def get_crime_info(state, future_years=[2024, 2025]):
    historical_total = df_for_prediction[df_for_prediction["States/UTs"] == state]["total_crime"].sum()

    # Get predictions for each future year separately
    future_predictions = [models[state].predict([[year]])[0] for year in future_years]

    # Separate predictions for 2024 and 2025
    prediction_2024 = future_predictions[0]  # Access the prediction for 2024 directly
    prediction_2025 = future_predictions[1]  # Access the prediction for 2025 directly

    future_total = sum(future_predictions)  # Calculate total future crimes
    total_crime = historical_total + future_total

    historical_percentage = (historical_total / total_crime) * 100
    future_percentage = (future_total / total_crime) * 100

    return {
        "historical_total_crime": historical_total,
        "future_total_crime": future_total,
        "total_crime": total_crime,
        "historical_percentage": historical_percentage,
        "future_percentage": future_percentage,
        "prediction_2024": prediction_2024,
        "prediction_2025": prediction_2025
    }


# Get user input for the state
state_input = input("Enter the state name: ")

# Display crime information and caution
try:
    crime_info = get_crime_info(state_input)
    print(f"\nCrime Information for {state_input}:")
    print(f"Total historical crimes (2021-2023): {crime_info['historical_total_crime']:.2f}")
    print(f"Predicted total future crimes (2024-2025): {crime_info['future_total_crime']:.2f}")
    print(f"Predicted crimes for 2024: {crime_info['prediction_2024']:.2f}")
    print(f"Predicted crimes for 2025: {crime_info['prediction_2025']:.2f}")
    print(f"Total crimes (historical + future): {crime_info['total_crime']:.2f}")
    print(f"Historical crime percentage: {crime_info['historical_percentage']:.2f}%")
    print(f"Future crime percentage: {crime_info['future_percentage']:.2f}%")
    print(f"\nCaution: If you are staying in {state_input}, beware of potential crimes.")
except KeyError:
    print(f"State '{state_input}' not found in the data.")
    import pandas as pd
from sklearn.linear_model import LinearRegression

# ... (Data preparation and model training as before) ...

# Function to get predictions and percentages
def get_crime_info(state, future_years=[2024, 2025]):
    historical_total = df_for_prediction[df_for_prediction["States/UTs"] == state]["total_crime"].sum()

    # Get predictions for each future year separately
    future_predictions = [models[state].predict([[year]])[0] for year in future_years]

    # Separate predictions for 2024 and 2025
    prediction_2024 = future_predictions[0]  # Access the prediction for 2024 directly
    prediction_2025 = future_predictions[1]  # Access the prediction for 2025 directly

    future_total = sum(future_predictions)  # Calculate total future crimes
    total_crime = historical_total + future_total

    historical_percentage = (historical_total / total_crime) * 100
    future_percentage = (future_total / total_crime) * 100

    return {
        "historical_total_crime": historical_total,
        "future_total_crime": future_total,
        "total_crime": total_crime,
        "historical_percentage": historical_percentage,
        "future_percentage": future_percentage,
        "prediction_2024": prediction_2024,
        "prediction_2025": prediction_2025
    }


# Get user input for the state
state_input = input("Enter the state name: ")

# Display crime information and caution
try:
    crime_info = get_crime_info(state_input)
    print(f"\nCrime Information for {state_input}:")
    print(f"Total historical crimes (2021-2023): {crime_info['historical_total_crime']:.2f}")
    print(f"Predicted total future crimes (2024-2025): {crime_info['future_total_crime']:.2f}")
    print(f"Predicted crimes for 2024: {crime_info['prediction_2024']:.2f}")
    print(f"Predicted crimes for 2025: {crime_info['prediction_2025']:.2f}")
    print(f"Total crimes (historical + future): {crime_info['total_crime']:.2f}")
    print(f"Historical crime percentage: {crime_info['historical_percentage']:.2f}%")
    print(f"Future crime percentage: {crime_info['future_percentage']:.2f}%")
    print(f"\nCaution: If you are staying in {state_input}, beware of potential crimes.")
except KeyError:
    print(f"State '{state_input}' not found in the data.")
    import pandas as pd
from sklearn.linear_model import LinearRegression

# Function to get predictions and percentages
def get_crime_info(state, future_years=[2024, 2025]):
    # Get the sum of historical crimes for the state
    historical_total = df_for_prediction[df_for_prediction["States/UTs"] == state]["total_crime"].sum()

    # Get predictions for each future year separately using the trained model
    future_predictions = [models[state].predict([[year]])[0] for year in future_years]

    # Separate predictions for 2024 and 2025
    prediction_2024 = future_predictions[0]  # Prediction for 2024
    prediction_2025 = future_predictions[1]  # Prediction for 2025

    # Calculate total predicted future crimes
    future_total = sum(future_predictions)

    # Calculate overall crime (historical + future)
    total_crime = historical_total + future_total

    # Correct percentage calculations
    historical_percentage = (historical_total / total_crime) * 100 if total_crime != 0 else 0
    future_percentage = (future_total / total_crime) * 100 if total_crime != 0 else 0

    return {
        "historical_total_crime": historical_total,
        "future_total_crime": future_total,
        "total_crime": total_crime,
        "historical_percentage": historical_percentage,
        "future_percentage": future_percentage,
        "prediction_2024": prediction_2024,
        "prediction_2025": prediction_2025
    }

# Get user input for the state
state_input = input("Enter the state name: ")

# Display crime information and caution
try:
    crime_info = get_crime_info(state_input)
    print(f"\nCrime Information for {state_input}:")
    print(f"Total historical crimes (2021-2023): {crime_info['historical_total_crime']:.2f}")
    print(f"Predicted total future crimes (2024-2025): {crime_info['future_total_crime']:.2f}")
    print(f"Predicted crimes for 2024: {crime_info['prediction_2024']:.2f}")
    print(f"Predicted crimes for 2025: {crime_info['prediction_2025']:.2f}")
    print(f"Total crimes (historical + future): {crime_info['total_crime']:.2f}")
    print(f"Historical crime percentage: {crime_info['historical_percentage']:.2f}%")
    print(f"Future crime percentage: {crime_info['future_percentage']:.2f}%")
    print(f"\nCaution: If you are staying in {state_input}, beware of potential crimes.")
except KeyError:
    print(f"State '{state_input}' not found in the data.")
import matplotlib.pyplot as plt
import numpy as np

# Function to get predictions for all states
def get_all_states_predictions(future_years=[2024, 2025]):
    predictions = {}
    historical_totals = {}

    # Iterate over all states in the models
    for state in models.keys():
        # Predict future crimes for the given years
        future_predictions = [models[state].predict([[year]])[0] for year in future_years]

        # Sum of historical crimes for each state
        historical_total = df_for_prediction[df_for_prediction["States/UTs"] == state]["total_crime"].sum()

        # Store future predictions and historical totals
        predictions[state] = sum(future_predictions)
        historical_totals[state] = historical_total

    return predictions, historical_totals

# Get the predictions for all states and their historical totals
future_predictions, historical_totals = get_all_states_predictions()

# Calculate total future and historical crime for all states
total_future_crime = sum(future_predictions.values())
total_historical_crime = sum(historical_totals.values())

# Calculate the percentage increase
increase_percentage = ((total_future_crime - total_historical_crime) / total_historical_crime) * 100 if total_historical_crime != 0 else 0

# Convert predictions to percentages
predicted_crimes_percentages = [(value / total_future_crime) * 100 for value in future_predictions.values()]

# Prepare data for plotting
states = list(future_predictions.keys())

# Plotting the bar graph with adjusted bar width and spacing
plt.figure(figsize=(12, 6))
bar_width = 0.6  # Adjust bar width for more space between bars
bars = plt.bar(states, predicted_crimes_percentages, color='skyblue', width=bar_width)

# Rotate state names for better readability
plt.xticks(rotation=90)

# Add titles and labels
plt.title(f'Predicted Crimes for 2024 and 2025 by State\n'
          f'Total Historical Crimes (2021-2023): {total_historical_crime:.0f} | '
          f'Total Future Crimes (2024-2025): {total_future_crime:.0f}\n'
          f'Overall Crime Increase: {increase_percentage:.2f}%', fontsize=14)

plt.xlabel('States', fontsize=12)
plt.ylabel('Predicted Future Crimes (%)', fontsize=12)

# Add percentage labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}%', ha='center', va='bottom')

# Increase spacing between bars by adjusting x-ticks position
plt.gca().set_xticks(np.arange(len(states)))
plt.gca().set_xticklabels(states)
plt.gca().margins(x=0.01)  # Adding more space between bars

# Display the plot
plt.tight_layout()  # Adjust layout to make room for x-axis labels
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Function to get predictions for all states
def get_all_states_predictions(future_years=[2024, 2025]):
    predictions = {}
    historical_totals = {}

    # Iterate over all states in the models
    for state in models.keys():
        # Predict future crimes for the given years
        future_predictions = [models[state].predict([[year]])[0] for year in future_years]

        # Sum of historical crimes for each state
        historical_total = df_for_prediction[df_for_prediction["States/UTs"] == state]["total_crime"].sum()

        # Store future predictions and historical totals
        predictions[state] = sum(future_predictions)
        historical_totals[state] = historical_total

    return predictions, historical_totals

# Get the predictions for all states and their historical totals
future_predictions, historical_totals = get_all_states_predictions()

# Calculate total future and historical crime for all states
total_future_crime = sum(future_predictions.values())
total_historical_crime = sum(historical_totals.values())

# Calculate the percentage increase
increase_percentage = ((total_future_crime - total_historical_crime) / total_historical_crime) * 100 if total_historical_crime != 0 else 0

# Convert predictions to percentages
predicted_crimes_percentages = [(value / total_future_crime) * 100 for value in future_predictions.values()]

# Prepare data for plotting
states = list(future_predictions.keys())

# Plotting the bar graph with adjusted bar width and spacing
plt.figure(figsize=(12, 6))
bar_width = 0.6  # Adjust bar width for more space between bars
bars = plt.bar(states, predicted_crimes_percentages, color='skyblue', width=bar_width)

# Rotate state names for better readability
plt.xticks(rotation=90)

# Add titles and labels
plt.title(f'Predicted Crimes for 2024 and 2025 by State\n'
          f'Total Historical Crimes (2021-2023): {total_historical_crime:.0f} | '
          f'Total Future Crimes (2024-2025): {total_future_crime:.0f}\n'
          f'Overall Crime Increase: {increase_percentage:.2f}%', fontsize=14)

plt.xlabel('States', fontsize=12)
plt.ylabel('Predicted Future Crimes (%)', fontsize=12)

# Add percentage labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}%', ha='center', va='bottom')

# Increase spacing between bars by adjusting x-ticks position
plt.gca().set_xticks(np.arange(len(states)))
plt.gca().set_xticklabels(states)
plt.gca().margins(x=0.01)  # Adding more space between bars

# Display the plot
plt.tight_layout()  # Adjust layout to make room for x-axis labels
plt.show()
import pandas as pd
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox

# Sample data for historical crimes (replace with actual DataFrame)
df_for_prediction = pd.DataFrame({
    "States/UTs": ['State A', 'State B', 'State C', 'State D'],
    "total_crime": [1000, 1200, 800, 1100]
})

# Sample trained models (replace with actual trained models for each state)
models = {
    "State A": LinearRegression().fit([[2021], [2022], [2023]], [1000, 1100, 1050]),
    "State B": LinearRegression().fit([[2021], [2022], [2023]], [1200, 1150, 1250]),
    "State C": LinearRegression().fit([[2021], [2022], [2023]], [800, 850, 900]),
    "State D": LinearRegression().fit([[2021], [2022], [2023]], [1100, 1150, 1200])
}

# Function to get predictions and percentages
def get_crime_info(state, future_years=[2024, 2025]):
    if state not in df_for_prediction["States/UTs"].values:
        raise KeyError(f"State '{state}' not found")

    # Get the sum of historical crimes for the state
    historical_total = df_for_prediction[df_for_prediction["States/UTs"] == state]["total_crime"].sum()

    # Get predictions for each future year separately using the trained model
    future_predictions = [models[state].predict([[year]])[0] for year in future_years]

    # Separate predictions for 2024 and 2025
    prediction_2024 = future_predictions[0]
    prediction_2025 = future_predictions[1]

    # Calculate total predicted future crimes
    future_total = sum(future_predictions)

    # Calculate overall crime (historical + future)
    total_crime = historical_total + future_total

    # Correct percentage calculations
    historical_percentage = (historical_total / total_crime) * 100 if total_crime != 0 else 0
    future_percentage = (future_total / total_crime) * 100 if total_crime != 0 else 0

    return {
        "historical_total_crime": historical_total,
        "future_total_crime": future_total,
        "total_crime": total_crime,
        "historical_percentage": historical_percentage,
        "future_percentage": future_percentage,
        "prediction_2024": prediction_2024,
        "prediction_2025": prediction_2025
    }

# Function to display the crime info in a popup window
def show_crime_info():
    state_input = state_entry.get()  # Get the user input from the entry widget
    try:
        crime_info = get_crime_info(state_input)
        # Create a popup window to show the information
        messagebox.showinfo(f"Crime Information for {state_input}",
            f"Total historical crimes (2021-2023): {crime_info['historical_total_crime']:.2f}\n"
            f"Predicted total future crimes (2024-2025): {crime_info['future_total_crime']:.2f}\n"
            f"Predicted crimes for 2024: {crime_info['prediction_2024']:.2f}\n"
            f"Predicted crimes for 2025: {crime_info['prediction_2025']:.2f}\n"
            f"Total crimes (historical + future): {crime_info['total_crime']:.2f}\n"
            f"Historical crime percentage: {crime_info['historical_percentage']:.2f}%\n"
            f"Future crime percentage: {crime_info['future_percentage']:.2f}%\n"
            f"\nCaution: If you are staying in {state_input}, beware of potential crimes.")
    except KeyError:
        messagebox.showerror("Error", f"State '{state_input}' not found in the data.")

# Create the main window
root = tk.Tk()
root.title("Crime Prediction System")

# Create a label and entry for state name input
state_label = tk.Label(root, text="Enter State/UT Name:")
state_label.pack(pady=10)
state_entry = tk.Entry(root, width=30)
state_entry.pack(pady=5)

# Create a button to trigger the prediction
submit_button = tk.Button(root, text="Get Crime Prediction", command=show_crime_info)
submit_button.pack(pady=20)

# Run the application
root.mainloop()
