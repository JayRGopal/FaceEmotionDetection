def generate_random_timestamps(df_videoTimestamps, num_samples=500):
    # Determine the overall start and end times
    overall_start = df_videoTimestamps['VideoStart'].min()
    overall_end = df_videoTimestamps['VideoEnd'].max()
    
    # Generate random timestamps within the range
    random_timestamps = pd.to_datetime(np.random.uniform(
        overall_start.value,
        overall_end.value,
        num_samples
    ).astype('datetime64[ns]'))
    
    # Create a DataFrame with the random timestamps
    random_timestamps_df = pd.DataFrame(random_timestamps, columns=['Datetime'])
    
    return random_timestamps_df


random_timestamps_df = generate_random_timestamps(df_videoTimestamps)