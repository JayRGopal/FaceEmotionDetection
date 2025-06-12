import os
import re
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION --- #
FEATURE_SAVE_FOLDER = './temp_outputs/'
TIME_WINDOWS = list(range(30, 241, 30))
METHODS = ['OGAUHSE_L_']  # You can add more methods if needed
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
    print("\nFeature counts for every patient, every time window, every method:\n")
    all_data = {}  # method -> time_window -> patient -> n_features

    for method in METHODS:
        method_data = {}
        patient_folders = [pf for pf in os.listdir(FEATURE_SAVE_FOLDER) if os.path.isdir(os.path.join(FEATURE_SAVE_FOLDER, pf))]
        for tw in TIME_WINDOWS:
            tw_data = {}
            for patient_folder in patient_folders:
                patient_folder_path = os.path.join(FEATURE_SAVE_FOLDER, patient_folder)
                found = False
                for filename in os.listdir(patient_folder_path):
                    if filename.endswith('.csv') and INTERNAL_STATE in filename and method in filename:
                        time_window = parse_filename(filename)
                        if time_window == tw:
                            file_path = os.path.join(patient_folder_path, filename)
                            df = pd.read_csv(file_path)
                            df = remove_duplicate_features(df)
                            n_features = df.shape[1] - 1  # Exclude target column
                            tw_data[patient_folder] = n_features
                            found = True
                            break
                if not found:
                    tw_data[patient_folder] = None  # Mark as missing
            method_data[tw] = tw_data
        all_data[method] = method_data

    # Print in a nice table format
    for method in METHODS:
        print(f"\n=== Method: {method} ===")
        # Gather all patients for this method
        all_patients = set()
        for tw in TIME_WINDOWS:
            all_patients.update(all_data[method][tw].keys())
        all_patients = sorted(list(all_patients))
        # Print header
        header = ["Time Window (min)"] + all_patients
        print("{:<18}".format(header[0]), end="")
        for p in all_patients:
            print("{:>15}".format(p), end="")
        print()
        print("-" * (18 + 15 * len(all_patients)))
        # Print each row
        for tw in TIME_WINDOWS:
            print("{:<18}".format(str(tw)), end="")
            for p in all_patients:
                val = all_data[method][tw].get(p, None)
                if val is None:
                    print("{:>15}".format("-"), end="")
                else:
                    print("{:>15}".format(val), end="")
            print()
    print("\nLegend: Each cell shows the number of features for that patient and time window (or '-' if missing).")

if __name__ == "__main__":
    main()