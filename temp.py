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
patient_folders.sort()  # Sort patient folders for consistent output
print(f"Found {len(patient_folders)} patient folders")

# Dictionary to store NaN percentages for each patient, method, internal state, and time window
patient_nan_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

print("\nAnalyzing NaN percentages for each patient individually...")
for patient_id in patient_folders:
    print(f"\nExamining patient: {patient_id}")
    patient_folder = os.path.join(FEATURE_SAVE_FOLDER, patient_id)
    csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]
    
    # Track if this patient has any files with NaNs
    has_nans = False
    
    for file in csv_files:
        file_state, file_time, file_prefix = parse_filename(file)
        
        # Skip if not in our list of interest
        if (file_state not in INTERNAL_STATES or 
            file_prefix not in RESULTS_PREFIX_LIST or 
            file_time not in TIME_WINDOWS):
            continue
        
        # Load data as-is without any type conversion
        df = pd.read_csv(os.path.join(patient_folder, file), dtype=str)
        
        # Convert back to appropriate types for NaN checking
        # This avoids type conversion issues
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass  # Keep as string if not numeric
        
        # Calculate NaN statistics
        total_cells = df.size
        nan_cells = df.isna().sum().sum()
        nan_percent = (nan_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Store statistics
        display_name = PREFIX_DISPLAY_MAP.get(file_prefix, file_prefix)
        patient_nan_stats[patient_id][file_prefix][file_state][file_time] = {
            'total_cells': total_cells,
            'nan_cells': nan_cells,
            'nan_percent': nan_percent
        }
        
        # Print NaN percentage for this file
        if nan_cells > 0:
            has_nans = True
            print(f"  {display_name}, {file_state}, Time Window {file_time} min:")
            print(f"    {nan_cells} NaNs out of {total_cells} cells ({nan_percent:.4f}%)")
            
            # Count NaNs by column 
            nan_by_col = df.isna().sum()
            nan_cols = nan_by_col[nan_by_col > 0]
            if not nan_cols.empty:
                print("    NaNs by column:")
                for col, count in nan_cols.items():
                    col_percent = (count / len(df)) * 100
                    print(f"      {col}: {count} NaNs ({col_percent:.2f}%)")
    
    if not has_nans:
        print("  No NaN values found in any files")

# Print summary of NaN percentages across all patients
print("\n\n======= SUMMARY OF NaN PERCENTAGES BY PATIENT =======")

# Calculate overall NaN percentage for each patient
patient_overall_stats = {}
for patient_id in patient_folders:
    total_cells_all = 0
    nan_cells_all = 0
    
    for prefix in RESULTS_PREFIX_LIST:
        for state in INTERNAL_STATES:
            for time_window in TIME_WINDOWS:
                if time_window in patient_nan_stats[patient_id][prefix][state]:
                    stats = patient_nan_stats[patient_id][prefix][state][time_window]
                    total_cells_all += stats['total_cells']
                    nan_cells_all += stats['nan_cells']
    
    overall_percent = (nan_cells_all / total_cells_all) * 100 if total_cells_all > 0 else 0
    patient_overall_stats[patient_id] = {
        'total_cells': total_cells_all,
        'nan_cells': nan_cells_all,
        'nan_percent': overall_percent
    }

# Print patients in order of NaN percentage (highest first)
sorted_patients = sorted(patient_overall_stats.items(), 
                        key=lambda x: x[1]['nan_percent'], 
                        reverse=True)

print("\nPatients ranked by overall NaN percentage:")
for patient_id, stats in sorted_patients:
    if stats['nan_cells'] > 0:
        print(f"{patient_id}: {stats['nan_cells']} NaNs out of {stats['total_cells']} cells ({stats['nan_percent']:.4f}%)")
    else:
        print(f"{patient_id}: No NaNs")

print("\n======= ANALYSIS COMPLETE =======")