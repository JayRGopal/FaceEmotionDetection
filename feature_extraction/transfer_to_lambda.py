def time_splitter(input_dict, splitter_times):
    # Initialize the output dictionary
    output_dict = {}
    
    # Frame rate: 5 frames per second
    frame_rate = 5
    
    # Iterate over each split time
    for split_time in splitter_times:
        # Initialize the dictionary for the current split time
        output_dict[split_time] = {}
        
        # Calculate the number of rows per split
        rows_per_split = split_time * 60 * frame_rate
        
        # Iterate over the outer dictionary
        for outer_key, inner_dict in input_dict.items():
            # Initialize the dictionary for the current outer key
            output_dict[split_time][outer_key] = {}
            
            # Iterate over the inner dictionary
            for timestamp, df in inner_dict.items():
                # Split the DataFrame into chunks of the specified size
                split_dfs = [df.iloc[i:i+rows_per_split] for i in range(0, len(df), rows_per_split)]
                
                # Assign the list of split DataFrames to the appropriate location in the output dictionary
                output_dict[split_time][outer_key][timestamp] = split_dfs
    
    return output_dict

# Functions to apply feature processing to the inpatient dictionary structure

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
    For columns with strings, convert to numbers if possible and take the string from the first DataFrame otherwise.

    Args:
        dictionary (dict): The dictionary containing lists of DataFrames.

    Returns:
        dict: A modified copy of the dictionary with the average of each list of DataFrames.
    """
    def process_columns(df_list):
        """
        Process columns to calculate averages for numeric columns and keep strings from the first DataFrame.
        """
        combined_df = pd.concat(df_list, ignore_index=True)
        avg_df = pd.DataFrame()

        for column in combined_df.columns:
            # Try to convert the column to numeric
            numeric_series = pd.to_numeric(combined_df[column], errors='coerce')
            
            if numeric_series.notna().all():
                # If all values can be converted to numeric, calculate the mean
                avg_df[column] = [numeric_series.mean()]
            else:
                avg_df[column] = [df_list[0][column]]
        
        return avg_df
    
    new_dict = {}
    for split_time, outer_dict in dictionary.items():
        new_dict[split_time] = {}
        for outer_key, inner_dict in outer_dict.items():
            new_dict[split_time][outer_key] = {}
            for timestamp, df_list in inner_dict.items():
                avg_df = process_columns(df_list)
                new_dict[split_time][outer_key][timestamp] = avg_df
    return new_dict