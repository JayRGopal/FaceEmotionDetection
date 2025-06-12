import os
import re
import pandas as pd

###############################################################################
# 0) Paths
###############################################################################
OVERVIEW_EXCEL_PATH = "/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking_Overview.xlsx"

###############################################################################
# 1) Read the overview data
###############################################################################
df_overview = pd.read_excel(OVERVIEW_EXCEL_PATH)

###############################################################################
# 2) Extract the list of metrics from column names
###############################################################################
pattern_metric = re.compile(r"^Num Datapoints - (.+)$")
metrics = []
for col in df_overview.columns:
    match = pattern_metric.match(col)
    if match:
        metrics.append(match.group(1))

metrics = sorted(set(metrics))  # unique metric names
print("Metrics found:", metrics)

###############################################################################
# 3) For each metric, print the list of patient sheet names that meet inclusion criteria
###############################################################################

def inclusion_criteria(num_datapoints, min_score, max_score, num_unique):
    """
    Returns True if the patient meets the inclusion criteria:
    - At least 5 datapoints
    - Score range at least 5
    - At least 3 unique scores
    """
    if pd.isna(num_datapoints) or pd.isna(min_score) or pd.isna(max_score) or pd.isna(num_unique):
        return False
    if num_datapoints < 5:
        return False
    score_range = max_score - min_score
    if score_range < 5:
        return False
    if num_unique < 3:
        return False
    return True

# DEBUG: Print all columns to check for typos or unexpected names
print("All columns in overview file:")
for col in df_overview.columns:
    print(f"  - {col}")

for metric in metrics:
    col_num_datapoints = f"Num Datapoints - {metric}"
    col_min_score = f"Min Score - {metric}"
    col_max_score = f"Max Score - {metric}"
    col_num_unique = f"Num Unique Scores - {metric}"
    # Try both possible column names for patient/sheet
    if "Sheet Name" in df_overview.columns:
        col_sheet_name = "Sheet Name"
    elif "Patient" in df_overview.columns:
        col_sheet_name = "Patient"
    else:
        print(f"  ERROR: Neither 'Sheet Name' nor 'Patient' column found for metric {metric}")
        continue

    # DEBUG: Print which columns are being checked for this metric
    print(f"\nChecking columns for metric '{metric}':")
    print(f"  - {col_num_datapoints}")
    print(f"  - {col_min_score}")
    print(f"  - {col_max_score}")
    print(f"  - {col_num_unique}")
    print(f"  - {col_sheet_name}")

    missing_cols = [col for col in [col_num_datapoints, col_min_score, col_max_score, col_num_unique, col_sheet_name] if col not in df_overview.columns]
    if missing_cols:
        print(f"  SKIPPING metric '{metric}' due to missing columns: {missing_cols}")
        continue  # Skip if any required column is missing

    included_patients = []
    for idx, row in df_overview.iterrows():
        num_datapoints = row[col_num_datapoints]
        min_score = row[col_min_score]
        max_score = row[col_max_score]
        num_unique = row[col_num_unique]
        # DEBUG: Print row info if something looks off
        if any(pd.isna(x) for x in [num_datapoints, min_score, max_score, num_unique]):
            print(f"  Row {idx} missing data: {row[col_sheet_name]} ({num_datapoints}, {min_score}, {max_score}, {num_unique})")
        if inclusion_criteria(num_datapoints, min_score, max_score, num_unique):
            included_patients.append(row[col_sheet_name])
    
    print(f"\n=== Metric: {metric} ===")
    if not included_patients:
        print("  No patients included.")
    else:
        for sheet_name in included_patients:
            print(f"  {sheet_name}")
    print()
