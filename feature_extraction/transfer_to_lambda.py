def average_dfs(dfs):
    if not dfs:
        raise ValueError("The list of DataFrames is empty")

    # Ensure all DataFrames have the same shape and columns
    shape = dfs[0].shape
    columns = dfs[0].columns
    for df in dfs:
        if df.shape != shape or not df.columns.equals(columns):
            raise ValueError("All DataFrames must have the same shape and columns")
    
    # Initialize a DataFrame to store the result
    avg_df = pd.DataFrame(index=dfs[0].index, columns=columns)
    
    # Iterate over each column to handle strings and numbers separately
    for column in columns:
        if pd.api.types.is_string_dtype(dfs[0][column]):
            # If column is of string type, use the column from the first DataFrame
            avg_df[column] = dfs[0][column]
        else:
            # Calculate the average for numeric columns
            column_data = np.mean([df[column] for df in dfs], axis=0)
            avg_df[column] = column_data

    return avg_df