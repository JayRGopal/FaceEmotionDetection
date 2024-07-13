import pandas as pd

def save_to_excel(self_reports_dict_list, MOOD_TRACKING_SHEET_PATH):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(self_reports_dict_list)

    # Load the existing Excel file
    with pd.ExcelWriter(MOOD_TRACKING_SHEET_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='Data_Overview', index=False)

# Example usage:
self_reports_dict_list = [
    {"date": "2024-07-01", "mood": "happy", "hours_slept": 7},
    {"date": "2024-07-02", "mood": "sad", "hours_slept": 5},
    # Add more dictionaries as needed
]

MOOD_TRACKING_SHEET_PATH = 'path_to_your_file.xlsx'
save_to_excel(self_reports_dict_list, MOOD_TRACKING_SHEET_PATH)
