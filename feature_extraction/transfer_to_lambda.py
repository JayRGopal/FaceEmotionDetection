Danny_Labels.replace({col: r'(\d{2}):(\d):(\d{2})$' for col in Danny_Labels.columns if 'Time' in col}, {'Time': r'\1:0\2:\3'}, regex=True, inplace=True)
