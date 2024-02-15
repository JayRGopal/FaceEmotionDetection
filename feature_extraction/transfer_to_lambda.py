

def filter_by_timestamp_optimized(facedx_df, openface_df):
    # This is the optimized filtering function we defined earlier
    timestamp_ranges = pd.concat([facedx_df['timestamp'] - 0.2, facedx_df['timestamp'] + 0.2], axis=1)
    mask = openface_df['timestamp'].apply(lambda x: any((timestamp_ranges[0] <= x) & (x <= timestamp_ranges[1])))
    filtered_openface_df = openface_df[mask].copy()
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

