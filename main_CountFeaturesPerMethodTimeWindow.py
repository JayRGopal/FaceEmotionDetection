import os
import re
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION --- #
FEATURE_SAVE_FOLDER = './temp_outputs/'
TIME_WINDOWS = list(range(30, 241, 30))
METHODS = ['OGAUHSE_L_']  # TEMPORARILY limit to just OGAUHSE_L_
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
    print("Comparing features for two patients (one with fewer, one with more features):\n")
    method = 'OGAUHSE_L_'
    limited = False  # Only standard features for now
    feature_counts = {}  # (patient, time_window) -> n_features
    feature_names = {}   # (patient, time_window) -> set of feature names
    patient_folders = [pf for pf in os.listdir(FEATURE_SAVE_FOLDER) if os.path.isdir(os.path.join(FEATURE_SAVE_FOLDER, pf))]
    for patient_folder in tqdm(patient_folders, desc=f"    Patients for {method}", leave=False):
        patient_folder_path = os.path.join(FEATURE_SAVE_FOLDER, patient_folder)
        for filename in os.listdir(patient_folder_path):
            if filename.endswith('.csv') and INTERNAL_STATE in filename and method in filename:
                time_window = parse_filename(filename)
                if time_window not in TIME_WINDOWS:
                    continue
                file_path = os.path.join(patient_folder_path, filename)
                df = pd.read_csv(file_path)
                df = remove_duplicate_features(df)
                n_features = df.shape[1] - 1  # Exclude target column
                feature_counts[(patient_folder, time_window)] = n_features
                feature_names[(patient_folder, time_window)] = set(df.columns[:-1])
    # Pick a time window that exists for at least two patients
    for tw in TIME_WINDOWS:
        patients_with_tw = [p for p in patient_folders if (p, tw) in feature_counts]
        if len(patients_with_tw) >= 2:
            # Sort by feature count
            sorted_patients = sorted(patients_with_tw, key=lambda p: feature_counts[(p, tw)])
            patient_few = sorted_patients[0]
            patient_more = sorted_patients[-1]
            print(f"Time window: {tw} min")
            print(f"  Patient with fewer features: {patient_few} ({feature_counts[(patient_few, tw)]} features)")
            print(f"  Patient with more features: {patient_more} ({feature_counts[(patient_more, tw)]} features)")
            delta = feature_names[(patient_more, tw)] - feature_names[(patient_few, tw)]
            print(f"  Features in patient with more features but NOT in patient with fewer features (delta):")
            print(delta)
            break
    else:
        print("No time window found with at least two patients having data.")

if __name__ == "__main__":
    main() 