def save_to_new_excel(self_reports_dict_list, MOOD_TRACKING_SHEET_PATH):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(self_reports_dict_list)
    
    # Get the directory of the provided sheet path
    directory = os.path.dirname(MOOD_TRACKING_SHEET_PATH)
    
    # Define the new file path
    new_file_path = os.path.join(directory, 'Mood_Tracking_Overview.xlsx')
    
    # Save the DataFrame to the new Excel file
    with pd.ExcelWriter(new_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data_Overview', index=False)
