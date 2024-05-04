def get_dict_openface(output_dir):
    # Create an empty dictionary to hold the DataFrames
    dfs_openface = {}

    # Get a list of all the CSV files in the directory
    csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.csv')])

    # list of columns to keep, assuming they may have variable spaces
    columns_to_keep = ['frame', 'timestamp', 'success', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
                       'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 
                       'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 
                       'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU45_c']

    # Loop through the CSV files
    for csv_file in csv_files:
        # Load data into a pandas DataFrame
        csv_file_path = os.path.join(output_dir, csv_file)
        df_temp = pd.read_csv(csv_file_path)

        # Fix column names to not have leading or trailing spaces
        df_temp.columns = df_temp.columns.str.strip()

        # Keep every 6th row such that it's 5 fps!
        X = 6
        df_temp = df_temp[df_temp.index % X == 0]

        # Filter DataFrame to keep only columns in list, now that names are stripped
        df_temp = df_temp.loc[:, [col for col in columns_to_keep if col in df_temp.columns]]

        # Store the DataFrame in the dictionary with the csv file name as the key
        # remove the '.csv' by doing csv_file[:-4]
        dfs_openface[csv_file[:-4]] = df_temp
        del df_temp

    return dfs_openface
