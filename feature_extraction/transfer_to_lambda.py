def fill_empty_dfs_lists(dictionary):
    """
    Fill empty DataFrames in a nested dictionary structure with a DataFrame of zeros.
    
    Args:
        dictionary (dict): The dictionary containing nested dictionaries with lists of DataFrames.
    
    Returns:
        dict: A modified copy of the dictionary with empty DataFrames filled with zeros.
    """
    # Find the first non-empty DataFrame to use as a template for filling empty DataFrames
    non_empty_df = None
    for split_time, outer_dict in dictionary.items():
        for outer_key, inner_dict in outer_dict.items():
            for timestamp, df_list in inner_dict.items():
                for df in df_list:
                    if not df.empty:
                        non_empty_df = df
                        break
                if non_empty_df is not None:
                    break
            if non_empty_df is not None:
                break
        if non_empty_df is not None:
            break
    
    # If no non-empty DataFrame is found, return the original dictionary
    if non_empty_df is None:
        return dictionary

    # Create the modified dictionary
    modified_dictionary = {}
    for split_time, outer_dict in dictionary.items():
        modified_dictionary[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            modified_dictionary[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                modified_df_list = []
                for df in df_list:
                    if df.empty:
                        modified_df = pd.DataFrame(0, index=non_empty_df.index, columns=non_empty_df.columns)
                        # Preserve string columns from the non-empty DataFrame
                        for column in non_empty_df.columns:
                            if non_empty_df[column].dtype == object:
                                modified_df[column] = non_empty_df[column]
                    else:
                        modified_df = df.copy()
                    modified_df_list.append(modified_df)
                modified_dictionary[split_time][outer_key][timestamp] = modified_df_list

    return modified_dictionary
