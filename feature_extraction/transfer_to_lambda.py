import pandas as pd

def buffer_neither(smiles_df, sleep_df):
    # Convert time columns to datetime if not already
    smiles_df['Time Start'] = pd.to_datetime(smiles_df['Time Start'], errors='coerce')
    smiles_df['Time End'] = pd.to_datetime(smiles_df['Time End'], errors='coerce')
    sleep_df['Time Start'] = pd.to_datetime(sleep_df['Time Start'], errors='coerce')
    sleep_df['Time End'] = pd.to_datetime(sleep_df['Time End'], errors='coerce')

    # Drop rows with NaT values in both dataframes
    smiles_df = smiles_df.dropna(subset=['Time Start', 'Time End'])
    sleep_df = sleep_df.dropna(subset=['Time Start', 'Time End'])

    # Check if dataframes have valid data
    if smiles_df.empty and sleep_df.empty:
        return pd.DataFrame(columns=['Time'])  # Return empty dataframe if no data available

    # Use available data to define the range
    start_times = pd.Series(smiles_df['Time Start'].tolist() + sleep_df['Time Start'].tolist()).dropna()
    end_times = pd.Series(smiles_df['Time End'].tolist() + sleep_df['Time End'].tolist()).dropna()
    
    if start_times.empty or end_times.empty:
        return pd.DataFrame(columns=['Time'])  # Return empty dataframe if no valid times are available

    # Find the earliest and latest times
    start_time = start_times.min()
    end_time = end_times.max()

    # Create a DataFrame with fixed frequency for the time range
    time_range = pd.date_range(start=start_time, end=end_time, freq='10S')
    tracking_df = pd.DataFrame({'Time': time_range, 'BufferSafe': False})

    # Determine the BufferSafe column values
    for i in range(len(tracking_df)):
        time = tracking_df.loc[i, 'Time']
        buffer_before = time - pd.Timedelta(minutes=1)
        buffer_after = time + pd.Timedelta(minutes=1)

        has_smile_within_buffer = smiles_df[((smiles_df['Time Start'] <= buffer_after) & (smiles_df['Time End'] >= buffer_before))].shape[0] > 0
        has_sleep_within_buffer = sleep_df[((sleep_df['Time Start'] <= buffer_after) & (sleep_df['Time End'] >= buffer_before))].shape[0] > 0
        tracking_df.loc[i, 'BufferSafe'] = not (has_smile_within_buffer or has_sleep_within_buffer)

    # Filter the BufferSafe time intervals
    non_smile_non_sleep_buffer_times = tracking_df[tracking_df['BufferSafe']]['Time'].reset_index(drop=True)

    return non_smile_non_sleep_buffer_times
