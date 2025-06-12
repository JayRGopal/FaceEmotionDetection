import os
import re
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION --- #
FEATURE_SAVE_FOLDER = './temp_outputs/'
TIME_WINDOWS = list(range(30, 241, 30))
METHODS = ['OGAUHSE_L_', 'OF_L_']
INTERNAL_STATE = 'Mood'
LIMITED_FEATURES_SUBSTRINGS = ["AU10", "AU12", "AU25", "AU27", "AU6", "AU7"]

def parse_filename(filename):
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    time_window = int(time_match.group(1)) if time_match else None
    return time_window

def remove_duplicate_features(df):
    feature_cols = df.columns[:-1]
    unique_cols = []
    seen = set()
    for col in feature_cols:
        if col not in seen:
            seen.add(col)
            unique_cols.append(col)
    unique_cols.append(df.columns[-1])
    return df[unique_cols]

def filter_limited_features(df):
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]
    matching_cols = [col for col in feature_cols if any(substr in col for substr in LIMITED_FEATURES_SUBSTRINGS)]
    matching_cols.append(target_col)
    return df[matching_cols]

def main():
    print("Counting features for each patient, method, and time window:\n")
    for method in METHODS:
        print(f"Method: {method}")
        for limited in [False, True]:
            limited_str = " (limited features)" if limited else ""
            print(f"  {'Limited' if limited else 'All'} features:")
            feature_counts = {}  # (patient, time_window) -> n_features
            patient_folders = [pf for pf in os.listdir(FEATURE_SAVE_FOLDER) if os.path.isdir(os.path.join(FEATURE_SAVE_FOLDER, pf))]
            for patient_folder in tqdm(patient_folders, desc=f"    Patients for {method}{limited_str}", leave=False):
                patient_folder_path = os.path.join(FEATURE_SAVE_FOLDER, patient_folder)
                for filename in os.listdir(patient_folder_path):
                    if filename.endswith('.csv') and INTERNAL_STATE in filename and method in filename:
                        time_window = parse_filename(filename)
                        if time_window not in TIME_WINDOWS:
                            continue
                        file_path = os.path.join(patient_folder_path, filename)
                        df = pd.read_csv(file_path)
                        df = remove_duplicate_features(df)
                        if limited:
                            df = filter_limited_features(df)
                        n_features = df.shape[1] - 1  # Exclude target column
                        feature_counts[(patient_folder, time_window)] = n_features
            # Print per patient, per time window
            for patient in sorted(patient_folders):
                for tw in TIME_WINDOWS:
                    key = (patient, tw)
                    if key in feature_counts:
                        print(f"    Patient: {patient}, Time window: {tw} min, Features: {feature_counts[key]}")
                    else:
                        print(f"    Patient: {patient}, Time window: {tw} min, Features: MISSING")
            # Check for discrepancies
            # For each time window, check if all patients have the same feature count
            for tw in TIME_WINDOWS:
                counts = [feature_counts[(patient, tw)] for patient in patient_folders if (patient, tw) in feature_counts]
                if counts and len(set(counts)) > 1:
                    print(f"  WARNING: Discrepancy detected in feature counts for time window {tw} min: counts = {counts}")
            # For each patient, check if all time windows have the same feature count
            for patient in patient_folders:
                counts = [feature_counts[(patient, tw)] for tw in TIME_WINDOWS if (patient, tw) in feature_counts]
                if counts and len(set(counts)) > 1:
                    print(f"  WARNING: Discrepancy detected in feature counts for patient {patient}: counts = {counts}")
        print()

if __name__ == "__main__":
    main() 