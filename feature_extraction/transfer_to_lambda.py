import os
import pandas as pd

# Parameters
PAT_NOW = "S23_199"
CURRENT_CSV_PATH = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/'), f'combined_events_{PAT_NOW}.csv')
OLD_CSV_PATH = os.path.join(os.path.abspath(f'/home/jgopal/NAS/Analysis/outputs_EventAnalysis/old_combined_events/'), f'combined_events_{PAT_NOW}.csv')

# Load both CSVs
current_df = pd.read_csv(CURRENT_CSV_PATH)
old_df = pd.read_csv(OLD_CSV_PATH)

# Focus on unique combinations of Filename and Start Time
current_events = current_df[['Filename', 'Start Time']].drop_duplicates()
old_events = old_df[['Filename', 'Start Time']].drop_duplicates()

# Find missing events in SCRIPT 1
missing_events = pd.merge(old_events, current_events, on=['Filename', 'Start Time'], how='left', indicator=True)
missing_events = missing_events[missing_events['_merge'] == 'left_only']

# Output the missing events for inspection
missing_events_list = missing_events[['Filename', 'Start Time']]
print(f"Missing events in SCRIPT 1:\n{missing_events_list}")

# Save missing events to CSV for further analysis
missing_events_list.to_csv('/home/jgopal/Desktop/FaceEmotionDetection/feature_extraction/missing_events.csv', index=False)
