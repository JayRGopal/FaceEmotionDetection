import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

# Path to the xlsx file
MOOD_TRACKING_SHEET_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx'
MISC_FIGURE_PATH = f'/home/jgopal/NAS/Analysis/Misc_Figures/'

# Define patients with enough datapoints to be included in our analysis!
sheets_to_use = ['S_199', 'S_199b', 'S_201', 'S_207', 'S_10b', 'S_211', 'S_212', 'S_214', 'S_174', 
                 'S_217', 'S_218', 'S_210', 'S_219', 'S_221', 'S_224', 'S_226', 'S_227', 'S_230', 'S_231'] 

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

self_reports_dict_list_mood = []

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

    # Sort for chronological order
    df = df.sort_values(by=df.columns[0])

    # Extract the datetime and mood
    time_data = df[df.columns[0]]
    mood_data = df['Mood']

    # Calculate additional title information
    num_datapoints = len(mood_data)
    percent_variation = mood_data.std() / mood_data.mean() * 100 if mood_data.mean() != 0 else 0
    days_span = (time_data.max() - time_data.min()).days

    # Determine subplot position
    ax = axs[idx // num_cols, idx % num_cols]
    
    # Plot the data
    color = 'blue' if sheet_name in sheets_to_use else 'red'
    ax.plot(time_data, mood_data, marker='o', color=color)
    
    # Set the title with additional information
    title_color = 'black' if sheet_name in sheets_to_use else 'red'
    ax.set_title(f'{sheet_name}: N={num_datapoints}, Var={percent_variation:.2f}%, Days={days_span}', fontsize=20, color=title_color)
    
    ax.set_xlabel('Datetime', fontsize=18)
    ax.set_ylabel('Mood', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='x', rotation=45)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))

    self_reports_dict_list_mood.append({
        'Sheet Name': sheet_name,
        'Num Mood Datapoints': num_datapoints, 
        'Mood Pct Variation': percent_variation, 
        'Mood Days Spanned': days_span
    })

# Hide any empty subplots
for idx in range(num_sheets, num_rows * num_cols):
    fig.delaxes(axs.flatten()[idx])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(MISC_FIGURE_PATH + 'Mood_Over_Time.png', dpi=300)

plt.show()