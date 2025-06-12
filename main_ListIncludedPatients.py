import pandas as pd

# ------------------- CONFIGURATION ------------------- #
OVERVIEW_EXCEL_PATH = "/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking_Overview.xlsx"

# ------------------- LOAD DATA ------------------- #
try:
    df_overview = pd.read_excel(OVERVIEW_EXCEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to read overview Excel file: {e}")

# ------------------- EXTRACT METRICS ------------------- #
# Find all moods by looking for columns that start with "Num_Self_Reports_"
mood_prefix = "Num_Self_Reports_"
metrics = sorted([
    col[len(mood_prefix):]
    for col in df_overview.columns
    if col.startswith(mood_prefix)
])
if not metrics:
    raise ValueError("No metrics found in overview file.")

# ------------------- INCLUSION CRITERIA ------------------- #
def inclusion_criteria(num_datapoints, score_range, num_unique):
    """
    Returns True if the patient meets the inclusion criteria:
    - At least 5 datapoints
    - Score range at least 5
    - At least 3 unique scores
    """
    if pd.isna(num_datapoints) or pd.isna(score_range) or pd.isna(num_unique):
        return False
    if num_datapoints < 5:
        return False
    if score_range < 5:
        return False
    if num_unique < 3:
        return False
    return True

# ------------------- MAIN LOGIC ------------------- #
# For each metric, collect included patients
all_included = dict()
for metric in metrics:
    col_num_datapoints = f"Num_Self_Reports_{metric}"
    col_num_unique = f"Num_Distinct_Scores_{metric}"
    col_range = f"Range_Self_Reports_{metric}"
    # Try both possible column names for patient/sheet
    if "Sheet Name" in df_overview.columns:
        col_sheet_name = "Sheet Name"
    elif "Patient" in df_overview.columns:
        col_sheet_name = "Patient"
    else:
        raise ValueError("Neither 'Sheet Name' nor 'Patient' column found in overview file.")

    required_cols = [col_num_datapoints, col_range, col_num_unique, col_sheet_name]
    missing_cols = [col for col in required_cols if col not in df_overview.columns]
    if missing_cols:
        continue  # Skip this metric if any required column is missing

    included_patients = []
    for _, row in df_overview.iterrows():
        num_datapoints = row[col_num_datapoints]
        score_range = row[col_range]
        num_unique = row[col_num_unique]
        sheet_name = row[col_sheet_name]
        if inclusion_criteria(num_datapoints, score_range, num_unique):
            included_patients.append(str(sheet_name))
    all_included[metric] = included_patients

# ------------------- OUTPUT ------------------- #
# Print the included patients for each metric (no file output)
for metric in metrics:
    included = all_included.get(metric, [])
    print(f"=== Metric: {metric} ===")
    if not included:
        print("No patients included.\n")
    else:
        for patient in included:
            print(patient)
        print()
