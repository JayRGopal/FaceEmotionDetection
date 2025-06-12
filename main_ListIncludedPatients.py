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


Metrics found: ['Anxiety', 'Depression', 'Hunger', 'Mood', 'Pain', 'Sleepiness']
All columns in overview file:
  - Sheet Name
  - Num Datapoints - Anxiety
  - Num Unique - Anxiety
  - Range - Anxiety
  - Is Included - Anxiety
  - Num Datapoints - Depression
  - Num Unique - Depression
  - Range - Depression
  - Is Included - Depression
  - Num Datapoints - Hunger
  - Num Unique - Hunger
  - Range - Hunger
  - Is Included - Hunger
  - Num Datapoints - Mood
  - Num Unique - Mood
  - Range - Mood
  - Is Included - Mood
  - Num Datapoints - Pain
  - Num Unique - Pain
  - Range - Pain
  - Is Included - Pain
  - Num Datapoints - Sleepiness
  - Num Unique - Sleepiness
  - Range - Sleepiness
  - Is Included - Sleepiness

Checking columns for metric 'Anxiety':
  - Num Datapoints - Anxiety
  - Min Score - Anxiety
  - Max Score - Anxiety
  - Num Unique Scores - Anxiety
  - Sheet Name
  SKIPPING metric 'Anxiety' due to missing columns: ['Min Score - Anxiety', 'Max Score - Anxiety', 'Num Unique Scores - Anxiety']

Checking columns for metric 'Depression':
  - Num Datapoints - Depression
  - Min Score - Depression
  - Max Score - Depression
  - Num Unique Scores - Depression
  - Sheet Name
  SKIPPING metric 'Depression' due to missing columns: ['Min Score - Depression', 'Max Score - Depression', 'Num Unique Scores - Depression']

Checking columns for metric 'Hunger':
  - Num Datapoints - Hunger
  - Min Score - Hunger
  - Max Score - Hunger
  - Num Unique Scores - Hunger
  - Sheet Name
  SKIPPING metric 'Hunger' due to missing columns: ['Min Score - Hunger', 'Max Score - Hunger', 'Num Unique Scores - Hunger']

Checking columns for metric 'Mood':
  - Num Datapoints - Mood
  - Min Score - Mood
  - Max Score - Mood
  - Num Unique Scores - Mood
  - Sheet Name
  SKIPPING metric 'Mood' due to missing columns: ['Min Score - Mood', 'Max Score - Mood', 'Num Unique Scores - Mood']

Checking columns for metric 'Pain':
  - Num Datapoints - Pain
  - Min Score - Pain
  - Max Score - Pain
  - Num Unique Scores - Pain
  - Sheet Name
  SKIPPING metric 'Pain' due to missing columns: ['Min Score - Pain', 'Max Score - Pain', 'Num Unique Scores - Pain']

Checking columns for metric 'Sleepiness':
  - Num Datapoints - Sleepiness
  - Min Score - Sleepiness
  - Max Score - Sleepiness
  - Num Unique Scores - Sleepiness
  - Sheet Name
  SKIPPING metric 'Sleepiness' due to missing columns: ['Min Score - Sleepiness', 'Max Score - Sleepiness', 'Num Unique Scores - Sleepiness']
