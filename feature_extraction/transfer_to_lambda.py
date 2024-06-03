def average_inner_dfs(dictionary):
    """
    Replace each list of DataFrames in a nested dictionary with the average of the DataFrames in each list.

    Args:
        dictionary (dict): The dictionary containing lists of DataFrames.

    Returns:
        dict: A modified copy of the dictionary with the average of each list of DataFrames.
    """
    new_dict = {}
    for split_time, outer_dict in dictionary.items():
        new_dict[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            new_dict[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                # Calculate the average of the DataFrames in the list
                avg_df = pd.concat(df_list).groupby(level=0).mean()
                new_dict[split_time][outer_key][timestamp] = avg_df
    return new_dict