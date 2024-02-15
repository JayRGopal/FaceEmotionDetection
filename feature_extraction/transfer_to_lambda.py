

def filter_by_timestamp_optimized(facedx_df, openface_df):
    # Ensure the timestamp columns are of type float
    facedx_df['timestamp'] = pd.to_numeric(facedx_df['timestamp'], errors='coerce')
    openface_df['timestamp'] = pd.to_numeric(openface_df['timestamp'], errors='coerce')

    # Initialize an empty DataFrame to store filtered rows
    filtered_openface_df = pd.DataFrame()

    # Ensure no error from empty DataFrames
    if not facedx_df.empty and not openface_df.empty:
        # Expand the timestamp range in facedx by 0.2 in both directions and create a range series
        timestamp_ranges = pd.concat([facedx_df['timestamp'] - 0.2, facedx_df['timestamp'] + 0.2], axis=1)

        # For each row in openface, check if the timestamp falls within any range in facedx
        mask = openface_df['timestamp'].apply(lambda x: any((timestamp_ranges[0] <= x) & (x <= timestamp_ranges[1])))
        
        # Filter openface using the mask
        filtered_openface_df = openface_df[mask].copy()

    # Reset index of the resulting DataFrame
    filtered_openface_df.reset_index(drop=True, inplace=True)
    
    return filtered_openface_df


def filter_dictionaries(facedx_dict, openface_dict):
    filtered_openface_dict = {}
    for key in facedx_dict.keys():
        # Assuming both dictionaries have the same keys
        facedx_df = facedx_dict[key]
        openface_df = openface_dict[key]
        # Apply the optimized filtering function to each pair of DataFrames
        filtered_openface_df = filter_by_timestamp_optimized(facedx_df, openface_df)
        # Store the filtered DataFrame in the new dictionary with the same key
        filtered_openface_dict[key] = filtered_openface_df
    return filtered_openface_dict

