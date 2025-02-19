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
# 2) Dynamically discover all self-report metrics (exclude first col, "Notes", "Datetime")
###############################################################################
all_metrics = set()
for sheet_name in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    if df.empty:
        continue
    
    # Build a list of columns to examine, skipping the first column
    candidate_cols = df.columns[1:]
    
    for col in candidate_cols:
        col_lower = col.lower()
        # Exclude if the col is named "notes" or "datetime" (in any case)
        if col_lower not in ['notes', 'datetime']:
            all_metrics.add(col)

# Convert set to a sorted list
all_metrics = sorted(all_metrics)

###############################################################################
# 3) Prepare color cycling so each metric has a distinct color
###############################################################################
color_cycle = itertools.cycle(plt.cm.tab10(np.linspace(0, 1, 10)))  
metric_color_map = {}
for metric in all_metrics:
    metric_color_map[metric] = next(color_cycle)

###############################################################################
# 4) Inclusion criteria and plotting
###############################################################################
patient_data = []

for sheet_name in sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    patient_info = {'Sheet Name': sheet_name}
    
    # Convert the first column to datetime
    if not df.empty:
        if not pd.api.types.is_datetime64_any_dtype(df[df.columns[0]]):
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')
        df = df.dropna(subset=[df.columns[0]])
        df = df.sort_values(by=df.columns[0])

    # Start a plot for this patient (overlay all metrics)
    plt.figure(figsize=(10, 6))
    
    for metric in all_metrics:
        if metric not in df.columns:
            # If metric doesn't exist, record zero stats
            patient_info[f'Num Datapoints - {metric}'] = 0
            patient_info[f'Num Unique - {metric}'] = 0
            patient_info[f'Range - {metric}'] = 0
            patient_info[f'Is Included - {metric}'] = 'No'
            continue
        
        # Drop rows with NaN or zero in this metric
        metric_df = df.dropna(subset=[metric])
        metric_df = metric_df[metric_df[metric].astype(bool)]
        if metric_df.empty:
            patient_info[f'Num Datapoints - {metric}'] = 0
            patient_info[f'Num Unique - {metric}'] = 0
            patient_info[f'Range - {metric}'] = 0
            patient_info[f'Is Included - {metric}'] = 'No'
            continue
        
        metric_df = metric_df.sort_values(by=df.columns[0])
        
        # Calculate stats
        values = metric_df[metric]
        time_data = metric_df[df.columns[0]]
        
        num_datapoints = len(values)
        num_unique = values.nunique()
        val_range = values.max() - values.min()
        
        patient_info[f'Num Datapoints - {metric}'] = num_datapoints
        patient_info[f'Num Unique - {metric}'] = num_unique
        patient_info[f'Range - {metric}'] = val_range
        
        # Inclusion criteria: >=5 datapoints, >=3 unique, range>5
        if (num_datapoints >= 5) and (num_unique >= 3) and (val_range > 5):
            patient_info[f'Is Included - {metric}'] = 'Yes'
        else:
            patient_info[f'Is Included - {metric}'] = 'No'
        
        # Plot
        plt.plot(time_data, values,
                 marker='o',
                 color=metric_color_map[metric],
                 label=metric)
    
    plt.title(f'{sheet_name}: All Metrics Over Time', fontsize=16)
    plt.xlabel('Datetime', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    out_path = os.path.join(MISC_FIGURE_PATH, f'{sheet_name}_Metrics_Over_Time.png')
    plt.savefig(out_path, dpi=300)
    plt.clf()
    
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

# Optionally bold entire rows if any metric is 'Yes'
wb = load_workbook(new_file_path)
ws = wb['Data_Overview']

include_cols = [c for c in patient_df.columns if c.startswith('Is Included - ')]
for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
    # See if any "Is Included - {metric}" cell in this row is 'Yes'
    if any(cell.value == 'Yes' for cell in row if ws.cell(row=1, column=cell.column).value in include_cols):
        for cell in row:
            cell.font = Font(bold=True)

wb.save(new_file_path)
