import pandas as pd

# Updated paths
MOOD_TRACKING_SHEET_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx'
OUTPUT_SPREADSHEET_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking_Overview.xlsx'

# Expanded list of patients
patients_to_include = [
    'S_199', 'S_201', 'S_202', 'S_203', 'S_205', 'S_206', 'S_207', 'S_208', 'S_209', 'S_210', 'S_10b',
    'S_211', 'S_212', 'S_214', 'S_199b', 'S_174', 'S_217', 'S_218', 'S_219', 'S_221', 'S_224',
    'S_226', 'S_227', 'S_230', 'S_231', 'S_233', 'S_234', 'S_236', 'S_237', 'S_238', 'S_239',
    'S_240', 'S_242', 'S_244'
]

# List of moods to check
moods = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']

# Read the Excel file
xls = pd.ExcelFile(MOOD_TRACKING_SHEET_PATH)

# Prepare a list to store the overview data for each patient
overview_data = []

# Process each patient
for sheet_name in patients_to_include:
    patient_data = {'Patient': sheet_name}
    if sheet_name in xls.sheet_names:
        # Read data for the patient
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # For each mood, process stats
        for mood in moods:
            # Find the column (case-insensitive match)
            mood_col = None
            for col in df.columns:
                if col.strip().lower() == mood.lower():
                    mood_col = col
                    break
            if mood_col is not None:
                valid_mood_data = df[mood_col].dropna()
                # Remove empty strings and whitespace
                valid_mood_data = valid_mood_data.astype(str).str.strip()
                valid_mood_data = valid_mood_data[valid_mood_data != '']
                # Try to convert to numeric for range calculation
                valid_mood_data_numeric = pd.to_numeric(valid_mood_data, errors='coerce').dropna()
                num_self_reports = len(valid_mood_data)
                num_distinct_scores = valid_mood_data.nunique()
                if not valid_mood_data_numeric.empty:
                    value_range = valid_mood_data_numeric.max() - valid_mood_data_numeric.min()
                else:
                    value_range = 0
            else:
                num_self_reports = 0
                num_distinct_scores = 0
                value_range = 0
            patient_data[f"Num_Self_Reports_{mood}"] = num_self_reports
            patient_data[f"Num_Distinct_Scores_{mood}"] = num_distinct_scores
            patient_data[f"Range_Self_Reports_{mood}"] = value_range
    else:
        # No data for this patient, fill zeros
        for mood in moods:
            patient_data[f"Num_Self_Reports_{mood}"] = 0
            patient_data[f"Num_Distinct_Scores_{mood}"] = 0
            patient_data[f"Range_Self_Reports_{mood}"] = 0
    overview_data.append(patient_data)

# Create a DataFrame from the overview data
overview_df = pd.DataFrame(overview_data)

# Save the DataFrame to an Excel file
overview_df.to_excel(OUTPUT_SPREADSHEET_PATH, index=False)

# Print overview numbers with updated statistics
print("Overview Numbers:")
print(f"Total Patients Processed: {len(patients_to_include)}")
print(f"Patients with Data: {overview_df.filter(like='Num_Self_Reports').sum(axis=1).gt(0).sum()}")

for mood in moods:
    num_distinct_cols = overview_df.filter(like=f"Num_Distinct_Scores_{mood}").columns
    num_self_reports_cols = overview_df.filter(like=f"Num_Self_Reports_{mood}").columns
    range_cols = overview_df.filter(like=f"Range_Self_Reports_{mood}").columns
    if not num_distinct_cols.empty:
        print(f"\n{mood} Statistics:")
        print(f"Mean Num Distinct Scores: {overview_df[num_distinct_cols].mean().mean():.2f}")
        print(f"Median Num Distinct Scores: {overview_df[num_distinct_cols].median().median():.2f}")
        print(f"Mean Num Total Scores: {overview_df[num_self_reports_cols].mean().mean():.2f}")
        print(f"Median Num Total Scores: {overview_df[num_self_reports_cols].median().median():.2f}")
        print(f"Mean Range: {overview_df[range_cols].mean().mean():.2f}")
        print(f"Median Range: {overview_df[range_cols].median().median():.2f}")
