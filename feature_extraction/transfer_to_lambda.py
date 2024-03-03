def calculate_fac_tremor(df, window_size=5):
    # Pad the DataFrame at the beginning and end to handle edge cases
    df_padded = pd.concat([df.iloc[:window_size-1].copy(), df, df.iloc[-window_size+1:].copy()]).reset_index(drop=True)
    
    # Initialize a DataFrame to hold the median tremor values for each landmark
    tremor_medians = pd.DataFrame()
    
    for i in range(68):  # For each landmark
        # Prepare column names
        x_col = f'X_{i}'
        y_col = f'Y_{i}'
        z_col = f'Z_{i}'
        
        # Calculate rolling mean positions
        rolling_means = df_padded[[x_col, y_col, z_col]].rolling(window=window_size, center=True).mean()
        
        # Calculate Euclidean distance from each frame's position to the rolling mean position
        distances = np.sqrt((df_padded[x_col] - rolling_means[x_col])**2 + 
                            (df_padded[y_col] - rolling_means[y_col])**2 + 
                            (df_padded[z_col] - rolling_means[z_col])**2)
        
        # Calculate median of distances for each window
        tremor_median = distances.rolling(window=window_size, center=True).median()
        
        # Append the median tremor value for the landmark to the tremor_medians DataFrame
        tremor_medians[f'fac_tremor_median_{i+1:02d}'] = tremor_median
        
    # Calculate the mean of median tremors across all frames for each landmark
    output_df = tremor_medians.mean().rename(lambda x: f'{x}_mean').to_frame().transpose()
    
    # Adjust the DataFrame to start from the original index
    output_df.index = [0]
    
    return output_df