import pandas as pd

# Updated paths
MOOD_TRACKING_SHEET_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx'
OUTPUT_SPREADSHEET_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking_Overview.xlsx'

# Expanded list of patients
patients_to_include = [
    'S_199', 'S_201', 'S_202', 'S_203', 'S_205', 'S_206', 'S_207', 'S_208', 'S_209', 'S_10b',
    'S_211', 'S_212', 'S_214', 'S_199b', 'S_174', 'S_217', 'S_218', 'S_219', 'S_221', 'S_224',
    'S_226', 'S_227', 'S_230', 'S_231', 'S_233', 'S_234', 'S_236', 'S_237', 'S_238', 'S_239'
]

# Read the Excel file
xls = pd.ExcelFile(MOOD_TRACKING_SHEET_PATH)

# Prepare a list to store the overview data for each patient
overview_data = []

# Process each patient
for sheet_name in patients_to_include:
    if sheet_name in xls.sheet_names:
        # Read data for the patient
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Get columns related to moods (e.g., Mood, Depression, Anxiety)
        mood_columns = [col for col in df.columns if col.lower() in ['mood', 'depression', 'anxiety']]
        
        # Skip if no mood columns are found
        if not mood_columns:
            overview_data.append({
                'Patient': sheet_name,
                **{f"Num_Self_Reports_{mood}": 0 for mood in mood_columns},
                **{f"Num_Distinct_Scores_{mood}": 0 for mood in mood_columns}
            })
            continue
        
        # Initialize a dictionary to store the patient's data
        patient_data = {'Patient': sheet_name}
        
        # Process each mood column
        for mood in mood_columns:
            # Drop NA and empty values for the mood
            valid_mood_data = df[mood].dropna().astype(str).str.strip()
            valid_mood_data = valid_mood_data[valid_mood_data != '']
            
            # Calculate self-reports and distinct scores
            num_self_reports = len(valid_mood_data)
            num_distinct_scores = valid_mood_data.nunique()
            
            # Store the results
            patient_data[f"Num_Self_Reports_{mood}"] = num_self_reports
            patient_data[f"Num_Distinct_Scores_{mood}"] = num_distinct_scores
        
        # Append the patient's data to the overview list
        overview_data.append(patient_data)
    else:
        # Append empty row for patients with no data
        overview_data.append({
            'Patient': sheet_name,
            **{f"Num_Self_Reports_{mood}": 0 for mood in ['Mood', 'Depression', 'Anxiety']},
            **{f"Num_Distinct_Scores_{mood}": 0 for mood in ['Mood', 'Depression', 'Anxiety']}
        })

# Create a DataFrame from the overview data
overview_df = pd.DataFrame(overview_data)

# Save the DataFrame to an Excel file
overview_df.to_excel(OUTPUT_SPREADSHEET_PATH, index=False)

# Print overview numbers with updated statistics
print("Overview Numbers:")
print(f"Total Patients Processed: {len(patients_to_include)}")
print(f"Patients with Data: {overview_df.filter(like='Num_Self_Reports').sum(axis=1).gt(0).sum()}")

for mood in ['Mood', 'Depression', 'Anxiety']:
    mood_columns = overview_df.filter(like=f"Num_Distinct_Scores_{mood}").columns
    if not mood_columns.empty:
        print(f"\n{mood} Statistics:")
        print(f"Mean Num Distinct Scores: {overview_df[mood_columns].mean().mean():.2f}")
        print(f"Median Num Distinct Scores: {overview_df[mood_columns].median().median():.2f}")
        total_score_columns = overview_df.filter(like=f"Num_Self_Reports_{mood}").columns
        print(f"Mean Num Total Scores: {overview_df[total_score_columns].mean().mean():.2f}")
        print(f"Median Num Total Scores: {overview_df[total_score_columns].median().median():.2f}")
