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