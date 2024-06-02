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
    return {key: [func(df, **kwargs) for df in df_list] for key, df_list in dictionary.items()}

    