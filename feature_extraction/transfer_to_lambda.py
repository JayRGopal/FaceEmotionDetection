def filter_by_timestamp_optimized(facedx_df, openface_df):
    # Reset index of the DataFrames at the beginning
    facedx_df.reset_index(drop=True, inplace=True)
    openface_df.reset_index(drop=True, inplace=True)

    # Convert timestamp columns to float
    facedx_df['timestamp'] = pd.to_numeric(facedx_df['timestamp'], errors='coerce')
    openface_df['timestamp'] = pd.to_numeric(openface_df['timestamp'], errors='coerce')

    # Initialize an empty list to store indices of rows in openface_df to keep
    indices_to_keep = []

    # Use broadcasting to find the absolute differences between each openface timestamp and all facedx timestamps
    for timestamp in openface_df['timestamp']:
        # Calculate the absolute difference between the current openface timestamp and all facedx timestamps
        abs_diff = np.abs(facedx_df['timestamp'] - timestamp)
        
        # Check if the minimum difference is within 0.2
        if (abs_diff.min() <= 0.2):
            indices_to_keep.append(True)
        else:
            indices_to_keep.append(False)

    # Filter openface_df using the indices_to_keep
    filtered_openface_df = openface_df[indices_to_keep].reset_index(drop=True)
    
    return filtered_openface_df