def force_convert_to_float(dictionary):
    """
    Forcefully convert all DataFrames in a nested dictionary structure to have their columns as floats.

    Args:
        dictionary (dict): The dictionary containing nested dictionaries with lists of DataFrames.

    Returns:
        dict: A modified copy of the dictionary with all DataFrames converted to float.
    """
    new_dict = {}
    for split_time, outer_dict in dictionary.items():
        new_dict[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            new_dict[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                new_df_list = [df.astype(float) for df in df_list]
                new_dict[split_time][outer_key][timestamp] = new_df_list
    return new_dict