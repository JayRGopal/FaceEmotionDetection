import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to the xlsx file
MOOD_TRACKING_SHEET_PATH = 'path/to/your/mood_tracking_sheet.xlsx'

# Read the Excel file
xls = pd.ExcelFile(MOOD_TRACKING_SHEET_PATH)

# Get all sheet names that start with "S_"
sheet_names = [sheet for sheet in xls.sheet_names if sheet.startswith("S_")]

# Set up the plot grid
num_sheets = len(sheet_names)
num_cols = 3  # Number of columns in the grid
num_rows = int(np.ceil(num_sheets / num_cols))

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 6))
fig.suptitle('Mood Tracking Over Time', fontsize=36)

# Iterate through the sheets and plot
for idx, sheet_name in enumerate(sheet_names):
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Filter rows where 'Mood' is not empty and not NaN
    df = df.dropna(subset=['Mood'])
    df = df[df['Mood'].astype(bool)]

    # Convert the first column to datetime
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')

    # Remove rows with invalid datetime
    df = df.dropna(subset=[df.columns[0]])

    # Extract the datetime and mood
    time_data = df[df.columns[0]]
    mood_data = df['Mood']

    # Determine subplot position
    ax = axs[idx // num_cols, idx % num_cols]
    
    # Plot the data
    ax.plot(time_data, mood_data, marker='o')
    ax.set_title(f'Sheet: {sheet_name}', fontsize=20)
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel('Mood', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

# Hide any empty subplots
for idx in range(num_sheets, num_rows * num_cols):
    fig.delaxes(axs.flatten()[idx])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
