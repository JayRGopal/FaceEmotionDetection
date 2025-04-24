import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# Define paths and configurations
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']
INTERNAL_STATES = ['Mood']
TIME_WINDOWS = list(range(15, 241, 30))

# Create a mapping for readable method names
PREFIX_DISPLAY_MAP = {
    'OF_L_': 'OpenFace',
    'OGAU_L_': 'FaceDx AU',
    'OGAUHSE_L_': 'FaceDx Complete',
    'HSE_L_': 'FaceDx Emo'
}

# Function to parse filename to extract metadata
def parse_filename(filename):
    internal_state = filename.split('_features_')[0]
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    prefix_match = re.search(r'minutes_(.*)\.csv', filename)
    time_window = int(time_match.group(1)) if time_match else None
    prefix = prefix_match.group(1) if prefix_match else None
    return internal_state, time_window, prefix

# Get all patient folders
patient_folders = [folder for folder in os.listdir(FEATURE_SAVE_FOLDER) 
                  if os.path.isdir(os.path.join(FEATURE_SAVE_FOLDER, folder)) and folder.startswith('S')]
print(f"Found {len(patient_folders)} patient folders")

# Dictionary to store NaN counts for each method, internal state, time window, and column
nan_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

# Track the number of rows processed for each combination
row_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

print("\nAppending rows across all patients for each time window...")
for internal_state in INTERNAL_STATES:
    print(f"\nProcessing internal state: {internal_state}")
    
    for time_window in TIME_WINDOWS:
        print(f"  Processing time window: {time_window} minutes")
        
        for prefix in RESULTS_PREFIX_LIST:
            display_name = PREFIX_DISPLAY_MAP.get(prefix, prefix)
            print(f"    Processing method: {display_name}")
            
            # Collect data from all patients for this combination
            all_dfs = []
            
            for patient_id in patient_folders:
                patient_folder = os.path.join(FEATURE_SAVE_FOLDER, patient_id)
                csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]
                
                # Find the matching CSV file for this patient
                matching_file = None
                for file in csv_files:
                    file_state, file_time, file_prefix = parse_filename(file)
                    if file_state == internal_state and file_time == time_window and file_prefix == prefix:
                        matching_file = file
                        break
                
                if matching_file:
                    # Load data for this patient
                    df = pd.read_csv(os.path.join(patient_folder, matching_file))
                    all_dfs.append(df)
            
            if not all_dfs:
                print(f"      No data found for this combination")
                continue
                
            # Combine all dataframes
            combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
            
            # Count total rows
            total_rows = combined_df.shape[0]
            row_counts[prefix][internal_state][time_window] = total_rows
            
            # Check for NaN values in each column
            nan_column_counts = combined_df.isna().sum()
            
            # Store NaN counts for columns with at least one NaN
            for column, count in nan_column_counts.items():
                if count > 0:
                    nan_counts[prefix][internal_state][time_window][column] = count

# Print summary of NaN values for each method
print("\n\n======= NaN VALUES SUMMARY =======")

for prefix in RESULTS_PREFIX_LIST:
    display_name = PREFIX_DISPLAY_MAP.get(prefix, prefix)
    print(f"\n{display_name}:")
    
    # Track columns with NaNs in any time window
    all_nan_columns = set()
    
    # For each internal state and time window
    for internal_state in INTERNAL_STATES:
        for time_window in TIME_WINDOWS:
            # Check if we have data for this combination
            if row_counts[prefix][internal_state][time_window] > 0:
                # Get columns with NaNs
                nan_columns = nan_counts[prefix][internal_state][time_window]
                
                if nan_columns:
                    total_rows = row_counts[prefix][internal_state][time_window]
                    print(f"  {internal_state}, Time Window {time_window} min ({total_rows} rows):")
                    
                    # Sort columns by NaN count (descending)
                    sorted_columns = sorted(nan_columns.items(), key=lambda x: x[1], reverse=True)
                    
                    for column, count in sorted_columns:
                        percent = (count / total_rows) * 100
                        print(f"    {column}: {count} NaNs ({percent:.2f}%)")
                        all_nan_columns.add(column)
                else:
                    print(f"  {internal_state}, Time Window {time_window} min: No NaN values")
    
    # Print summary of all columns with NaNs for this method
    print(f"\n  Summary for {display_name}:")
    if all_nan_columns:
        print(f"  Columns with NaNs across all time windows:")
        for column in sorted(all_nan_columns):
            print(f"    {column}")
    else:
        print("  No columns with NaN values found")

print("\n======= ANALYSIS COMPLETE =======")