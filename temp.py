import os
import re
import pandas as pd
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

# Dictionary to store all columns for each method, internal state, and time window
all_columns = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

# Dictionary to store all columns found for each method
method_all_columns = defaultdict(set)

# Dictionary to store file counts for each combination
file_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# First pass: Collect all column names and check for NaN values
print("\nChecking for NaN values and collecting column names...")
for patient_id in patient_folders:
    patient_folder = os.path.join(FEATURE_SAVE_FOLDER, patient_id)
    csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]
    
    for file in csv_files:
        file_state, file_time, file_prefix = parse_filename(file)
        
        # Skip if not in our list of interest
        if file_state not in INTERNAL_STATES or file_prefix not in RESULTS_PREFIX_LIST or file_time not in TIME_WINDOWS:
            continue
        
        # Load data
        df = pd.read_csv(os.path.join(patient_folder, file))
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN values in {patient_id}/{file}")
            
            # Show columns with NaN values
            nan_cols = df.columns[df.isna().any()].tolist()
            print(f"  Columns with NaNs: {nan_cols}")
        
        # Store column names (excluding the last column, which is the target)
        feature_cols = df.columns[:-1].tolist()
        all_columns[file_prefix][file_state][file_time].update(feature_cols)
        method_all_columns[file_prefix].update(feature_cols)
        
        # Count this file
        file_counts[file_prefix][file_state][file_time] += 1

# Second pass: Compare column consistency across patients for each method
print("\nChecking column consistency across patients and time windows...")

for prefix in RESULTS_PREFIX_LIST:
    display_name = PREFIX_DISPLAY_MAP.get(prefix, prefix)
    print(f"\n{display_name}:")
    
    # Find union of all columns for this method
    all_method_columns = method_all_columns[prefix]
    
    # For each internal state and time window, find missing columns
    inconsistent_columns = set()
    
    for state in INTERNAL_STATES:
        for time_window in TIME_WINDOWS:
            # Skip if no files for this combination
            if file_counts[prefix][state][time_window] == 0:
                continue
                
            # Get columns for this combination
            combination_columns = all_columns[prefix][state][time_window]
            
            # Find columns that are in the method but not in this combination
            missing_columns = all_method_columns - combination_columns
            
            if missing_columns:
                print(f"  Missing columns in {state}, time window {time_window}:")
                print(f"    {sorted(missing_columns)}")
                inconsistent_columns.update(missing_columns)
    
    # Print summary of inconsistent columns for this method
    print(f"\n{display_name} - Inconsistent columns summary:")
    if inconsistent_columns:
        print(f"{sorted(inconsistent_columns)}")
    else:
        print("  No inconsistent columns found")

# Final summary
print("\nSummary of column consistency checks:")
for prefix in RESULTS_PREFIX_LIST:
    display_name = PREFIX_DISPLAY_MAP.get(prefix, prefix)
    all_method_columns = method_all_columns[prefix]
    print(f"{display_name}: {len(all_method_columns)} total unique columns")