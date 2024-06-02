def apply_function_to_dict_list(dictionary, func, **kwargs):
    """
    Apply a function to each DataFrame in a dictionary where values are LISTS of dfs and return a modified copy of the dictionary.

    Args:
        dictionary (dict): The dictionary containing DataFrames.
        func (function): The function to apply to each DataFrame.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        dict: A modified copy of the dictionary with the function applied to each DataFrame.
    """
    new_dict = {}
    for split_time, outer_dict in dictionary.items():
        new_dict[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            new_dict[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                new_dict[split_time][outer_key][timestamp] = [func(df, **kwargs) for df in df_list]
    return new_dict

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