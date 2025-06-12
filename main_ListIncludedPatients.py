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
print(metrics)

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

for metric in metrics:
    col_num_datapoints = f"Num Datapoints - {metric}"
    col_min_score = f"Min Score - {metric}"
    col_max_score = f"Max Score - {metric}"
    col_num_unique = f"Num Unique Scores - {metric}"
    col_sheet_name = "Sheet Name" if "Sheet Name" in df_overview.columns else "Patient"

    if not all(col in df_overview.columns for col in [col_num_datapoints, col_min_score, col_max_score, col_num_unique, col_sheet_name]):
        continue  # Skip if any required column is missing

    included_patients = []
    for idx, row in df_overview.iterrows():
        num_datapoints = row[col_num_datapoints]
        min_score = row[col_min_score]
        max_score = row[col_max_score]
        num_unique = row[col_num_unique]
        if inclusion_criteria(num_datapoints, min_score, max_score, num_unique):
            included_patients.append(row[col_sheet_name])

    print(included_patients)
    print(f"=== Metric: {metric} ===")
    if not included_patients:
        print("  No patients included.")
    else:
        for sheet_name in included_patients:
            print(f"  {sheet_name}")
    print()
