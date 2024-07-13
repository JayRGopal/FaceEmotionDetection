import pandas as pd
from openpyxl import load_workbook

def save_to_excel(self_reports_dict_list, MOOD_TRACKING_SHEET_PATH):
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(self_reports_dict_list)
    
    # Load the existing Excel file
    book = load_workbook(MOOD_TRACKING_SHEET_PATH)
    
    # Create a Pandas Excel writer using openpyxl as the engine
    with pd.ExcelWriter(MOOD_TRACKING_SHEET_PATH, engine='openpyxl') as writer:
        writer.book = book
        # Remove the sheet if it already exists
        if 'Data_Overview' in writer.book.sheetnames:
            del writer.book['Data_Overview']
        
        # Write DataFrame to a new sheet
        df.to_excel(writer, sheet_name='Data_Overview', index=False)
        
        # Save the workbook
        writer.save()
