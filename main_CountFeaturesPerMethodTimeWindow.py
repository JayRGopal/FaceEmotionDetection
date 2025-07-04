import os
import re
import pandas as pd

# --- CONFIGURATION --- #
FEATURE_SAVE_FOLDER = './temp_outputs/'
TIME_WINDOWS = list(range(30, 241, 30))
METHODS = ['OGAUHSE_L_', 'OF_L_']  # You can add more methods if needed
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

def print_narrow_table(header, rows, max_cols=5):
    """
    Print a table with a limited number of columns at a time.
    header: list of column names (first is row label)
    rows: list of lists (each row)
    max_cols: max number of patient columns to show at once
    """
    n_patients = len(header) - 1
    for start in range(0, n_patients, max_cols):
        end = min(start + max_cols, n_patients)
        sub_header = [header[0]] + header[1+start:1+end]
        print("{:<18}".format(sub_header[0]), end="")
        for h in sub_header[1:]:
            print("{:>15}".format(h), end="")
        print()
        print("-" * (18 + 15 * (end - start)))
        for row in rows:
            print("{:<18}".format(row[0]), end="")
            for val in row[1+start:1+end]:
                print("{:>15}".format(val), end="")
            print()
        print()  # Blank line between blocks

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

    # Print in a narrow table format (chunked by max_cols)
    max_cols = 5  # Number of patient columns per block
    for method in METHODS:
        print(f"\n=== Method: {method} ===")
        # Gather all patients for this method
        all_patients = set()
        for tw in TIME_WINDOWS:
            all_patients.update(all_data[method][tw].keys())
        all_patients = sorted(list(all_patients))
        # Prepare header and rows
        header = ["Time Window (min)"] + all_patients
        rows = []
        for tw in TIME_WINDOWS:
            row = [str(tw)]
            for p in all_patients:
                val = all_data[method][tw].get(p, None)
                row.append("-" if val is None else str(val))
            rows.append(row)
        print_narrow_table(header, rows, max_cols=max_cols)
    print("\nLegend: Each cell shows the number of features for that patient and time window (or '-' if missing).")

if __name__ == "__main__":
    main()