import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import os
from openpyxl import load_workbook
from openpyxl.styles import Font
import itertools  # for cycling through colors

# Path to the xlsx file
MOOD_TRACKING_SHEET_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx'
MISC_FIGURE_PATH = '/home/jgopal/NAS/Analysis/Misc_Figures/'

# 1) Read the Excel, find all sheets that start with "S_"
xls = pd.ExcelFile(MOOD_TRACKING_SHEET_PATH)
sheet_names = [sheet for sheet in xls.sheet_names if sheet.startswith("S_")]

###############################################################################
# 2) Dynamically discover all self-report metrics/columns (except datetime, 'Notes')
###############################################################################
all_metrics = set()
for sheet_name in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    if df.empty:
        continue
    
    # Exclude the first column (assumed datetime) and anything named "Notes"
    for col in df.columns[1:]:
        if col.lower() != 'notes':
            all_metrics.add(col)

# Convert set to a sorted list
all_metrics = sorted(all_metrics)

###############################################################################
# 3) Prepare color cycling so each metric has a distinct color in the same plot
###############################################################################
color_cycle = itertools.cycle(plt.cm.tab10(np.linspace(0, 1, 10)))  # or any colormap you like
metric_color_map = {}
for metric in all_metrics:
    metric_color_map[metric] = next(color_cycle)

###############################################################################
# 4) Analyze each sheet (patient), collect stats & produce a single figure overlay
###############################################################################
patient_data = []

for sheet_name in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Prepare a dict to hold stats for this patient
    patient_info = {'Sheet Name': sheet_name}

    # Convert the first column to datetime where possible
    if not df.empty:
        if not pd.api.types.is_datetime64_any_dtype(df[df.columns[0]]):
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')
        df = df.dropna(subset=[df.columns[0]])  # remove rows where datetime is NaT
        df = df.sort_values(by=df.columns[0])
    
    # We'll plot all metrics on one figure, each in a different color
    plt.figure(figsize=(10, 6))
    
    # For each metric, apply the inclusion criteria & store stats
    for metric in all_metrics:
        # If the metric doesn't exist in this sheet, skip plotting & stats
        if metric not in df.columns:
            patient_info[f'Num Datapoints - {metric}'] = 0
            patient_info[f'Num Unique - {metric}'] = 0
            patient_info[f'Range - {metric}'] = 0
            patient_info[f'Is Included - {metric}'] = 'No'
            continue
        
        # Filter out rows with NaN or zero (False) in this metric
        metric_df = df.dropna(subset=[metric])
        metric_df = metric_df[metric_df[metric].astype(bool)]
        if metric_df.empty:
            patient_info[f'Num Datapoints - {metric}'] = 0
            patient_info[f'Num Unique - {metric}'] = 0
            patient_info[f'Range - {metric}'] = 0
            patient_info[f'Is Included - {metric}'] = 'No'
            continue
        
        # Sort again in case filtering changed row order
        metric_df = metric_df.sort_values(by=df.columns[0])
        
        # Basic stats
        values = metric_df[metric]
        num_datapoints = len(values)
        num_unique = values.nunique()
        val_range = values.max() - values.min()
        
        # Store in patient_info
        patient_info[f'Num Datapoints - {metric}'] = num_datapoints
        patient_info[f'Num Unique - {metric}'] = num_unique
        patient_info[f'Range - {metric}'] = val_range
        
        # Check if meets inclusion criteria:
        #   (1) >= 5 datapoints, (2) >= 3 unique, (3) range > 5
        if (num_datapoints >= 5) and (num_unique >= 3) and (val_range > 5):
            patient_info[f'Is Included - {metric}'] = 'Yes'
        else:
            patient_info[f'Is Included - {metric}'] = 'No'
        
        # Plot
        color = metric_color_map[metric]
        time_data = metric_df[df.columns[0]]
        plt.plot(time_data, values, marker='o', color=color, label=metric)
    
    # Customize the figure
    plt.title(f'{sheet_name}: All Metrics Over Time', fontsize=16)
    plt.xlabel('Datetime', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save figure for this patient
    out_path = os.path.join(MISC_FIGURE_PATH, f'{sheet_name}_Metrics_Over_Time.png')
    plt.savefig(out_path, dpi=300)
    plt.clf()  # Clear figure for next patient
    
    patient_data.append(patient_info)

###############################################################################
# 5) Save summary stats to Excel
###############################################################################
patient_df = pd.DataFrame(patient_data)

new_file_path = os.path.join(
    os.path.dirname(MOOD_TRACKING_SHEET_PATH),
    'Mood_Tracking_Overview.xlsx'
)

with pd.ExcelWriter(new_file_path, engine='openpyxl') as writer:
    patient_df.to_excel(writer, sheet_name='Data_Overview', index=False)

# Optionally bold rows if "any" metric is included
wb = load_workbook(new_file_path)
ws = wb['Data_Overview']

# Identify all "Is Included - metric" columns
include_cols = [col for col in patient_df.columns if col.startswith('Is Included - ')]

# For each row, check if there's a "Yes" in *any* inclusion column
for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
    # row is a tuple of cells; find "Is Included - {metric}" cells
    # We can map them by column headers
    row_values = {ws.cell(row=1, column=cell.column).value: cell for cell in row}
    
    # If any 'Is Included - metric' cell is 'Yes', bold entire row
    if any(row_values[col].value == 'Yes' for col in include_cols if col in row_values):
        for cell in row:
            cell.font = Font(bold=True)

wb.save(new_file_path)
