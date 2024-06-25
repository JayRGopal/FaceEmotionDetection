


import pandas as pd

# Path to the Excel file
file_path = 'path_to_your_file.xlsx'

# Load the Excel file
excel_file = pd.ExcelFile(file_path)

# Initialize report dictionary
report = {}

# Loop through each sheet except those ending in _MONITOR
for sheet_name in excel_file.sheet_names:
    if sheet_name.endswith('_MONITOR'):
        continue
    
    # Load the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Check if 'Behavior' column exists
    if 'Behavior' not in df.columns:
        continue
    
    # Get the Behavior column (case insensitive match for 'smile', 'sad', 'discomfort', 'yawn')
    behavior_column = df['Behavior'].dropna().astype(str)
    
    # Count occurrences
    smile_count = behavior_column.str.contains('smile', case=False).sum()
    sad_count = behavior_column.str.contains('sad', case=False).sum()
    discomfort_count = behavior_column.str.contains('discomfort', case=False).sum()
    yawn_count = behavior_column.str.contains('yawn', case=False).sum()
    sleep_count = behavior_column.str.contains('sleep', case=False).sum()
    non_empty_count = behavior_column.shape[0]
    
    # Update report
    report[sheet_name] = {
        'smile_count': smile_count,
        'sad_count': sad_count,
        'discomfort_count': discomfort_count,
        'yawn_count': yawn_count,
        'sleep_count': sleep_count,
        'non_empty_count': non_empty_count
    }

# Print the report
for patient, counts in report.items():
    print(f"Patient: {patient}")
    print(f"  Number of 'smile': {counts['smile_count']}")
    print(f"  Number of 'sad': {counts['sad_count']}")
    print(f"  Number of 'discomfort': {counts['discomfort_count']}")
    print(f"  Number of 'yawn': {counts['yawn_count']}")
    print(f"  Number of 'sleep': {counts['sleep_count']}")
    print(f"  Total non-empty behavior rows: {counts['non_empty_count']}")
    print()


