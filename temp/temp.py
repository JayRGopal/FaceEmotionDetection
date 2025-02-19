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

###############################################################################
# 3) For each metric, print the list of patient sheet names with "Yes"
###############################################################################
for metric in metrics:
    col_included = f"Is Included - {metric}"
    if col_included not in df_overview.columns:
        continue  # Skip if somehow missing
    
    included_patients = df_overview.loc[df_overview[col_included] == "Yes", "Sheet Name"]
    
    print(f"=== Metric: {metric} ===")
    if included_patients.empty:
        print("  No patients included.")
    else:
        for sheet_name in included_patients:
            print(f"  {sheet_name}")
    print()
