import os
import re
import pandas as pd
from collections import defaultdict

# Define paths and configurations
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']
INTERNAL_STATES = ['Mood']
TIME_WINDOW = 30  # Just check one time window

# Function to parse filename
def parse_filename(filename):
    internal_state = filename.split('_features_')[0]
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    prefix_match = re.search(r'minutes_(.*)\.csv', filename)
    time_window = int(time_match.group(1)) if time_match else None
    prefix = prefix_match.group(1) if prefix_match else None
    return internal_state, time_window, prefix

# Get patient folders
patient_folders = [folder for folder in os.listdir(FEATURE_SAVE_FOLDER) 
                  if os.path.isdir(os.path.join(FEATURE_SAVE_FOLDER, folder)) and folder.startswith('S')]

# Store column types for each patient/method
type_data = defaultdict(lambda: defaultdict(dict))
inconsistent_types = defaultdict(set)

print("Checking data types across patients...")
for prefix in RESULTS_PREFIX_LIST:
    for state in INTERNAL_STATES:
        # Get data for first patient to establish baseline
        first_patient = None
        baseline_types = {}
        
        for patient_id in patient_folders:
            patient_folder = os.path.join(FEATURE_SAVE_FOLDER, patient_id)
            csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]
            
            # Find matching file
            matching_file = None
            for file in csv_files:
                file_state, file_time, file_prefix = parse_filename(file)
                if file_state == state and file_time == TIME_WINDOW and file_prefix == prefix:
                    matching_file = file
                    break
            
            if matching_file:
                # Load data
                file_path = os.path.join(patient_folder, matching_file)
                df = pd.read_csv(file_path)
                
                # Store column types for this patient
                type_data[prefix][state][patient_id] = {col: str(df[col].dtype) for col in df.columns}
                
                # Set baseline types from first patient
                if first_patient is None:
                    first_patient = patient_id
                    baseline_types = type_data[prefix][state][patient_id]
                else:
                    # Check for type inconsistencies
                    for col in baseline_types:
                        if col in type_data[prefix][state][patient_id]:
                            if baseline_types[col] != type_data[prefix][state][patient_id][col]:
                                inconsistent_types[prefix].add(col)

# Print only inconsistent columns by method
for prefix in RESULTS_PREFIX_LIST:
    if inconsistent_types[prefix]:
        print(f"\n{prefix} - Columns with inconsistent types:")
        for col in sorted(inconsistent_types[prefix]):
            # Show examples of the different types
            types_seen = set()
            for state in INTERNAL_STATES:
                for patient_id in type_data[prefix][state]:
                    if col in type_data[prefix][state][patient_id]:
                        types_seen.add(type_data[prefix][state][patient_id][col])
            print(f"  {col}: {', '.join(sorted(types_seen))}")
    else:
        print(f"\n{prefix}: All columns have consistent types across patients")

if not any(inconsistent_types.values()):
    print("\nAll methods have consistent column types across all patients!")