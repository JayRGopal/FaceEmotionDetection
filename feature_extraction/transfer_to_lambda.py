import pandas as pd

# Path to the Excel file
file_path = 'path_to_your_file.xlsx'

# Load the Excel file
excel_file = pd.ExcelFile(file_path)

# Initialize report dictionary
report = {
    'Patient': [],
    'smile_count': [],
    'sad_count': [],
    'discomfort_count': [],
    'yawn_count': [],
    'sleep_count': [],
    'non_empty_count': []
}

# Loop through each sheet except those ending in _MONITOR
for sheet_name in excel_file.sheet_names:
    if sheet_name.endswith('_MONITOR'):
        continue
    
    # Load the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Check if 'Behavior' column exists
    if 'Behavior' not in df.columns:
        continue
    
    # Get the Behavior column (case insensitive match for 'smile', 'sad', 'discomfort', 'yawn', 'sleep')
    behavior_column = df['Behavior'].dropna().astype(str)
    
    # Count occurrences
    smile_count = behavior_column.str.contains('smile', case=False).sum()
    sad_count = behavior_column.str.contains('sad', case=False).sum()
    discomfort_count = behavior_column.str.contains('discomfort', case=False).sum()
    yawn_count = behavior_column.str.contains('yawn', case=False).sum()
    sleep_count = behavior_column.str.contains('sleep', case=False).sum()
    non_empty_count = behavior_column.shape[0]
    
    # Update report
    report['Patient'].append(sheet_name)
    report['smile_count'].append(smile_count)
    report['sad_count'].append(sad_count)
    report['discomfort_count'].append(discomfort_count)
    report['yawn_count'].append(yawn_count)
    report['sleep_count'].append(sleep_count)
    report['non_empty_count'].append(non_empty_count)

# Create a DataFrame from the report
report_df = pd.DataFrame(report)

# Display the DataFrame
import ace_tools as tools; tools.display_dataframe_to_user(name="Patient Behavior Report", dataframe=report_df)
