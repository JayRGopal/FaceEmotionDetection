import pandas as pd

def buffer_neither(smiles_df, sleep_df):
    # Ensure Time Start and Time End columns are in datetime format
    smiles_df['Time Start'] = pd.to_datetime(smiles_df['Time Start'])
    smiles_df['Time End'] = pd.to_datetime(smiles_df['Time End'])
    sleep_df['Time Start'] = pd.to_datetime(sleep_df['Time Start'], errors='coerce')
    sleep_df['Time End'] = pd.to_datetime(sleep_df['Time End'], errors='coerce')

    # Drop rows with NaT values in sleep_df
    sleep_df = sleep_df.dropna(subset=['Time Start', 'Time End'])

    # Find the earliest and latest times from smiles_df
    smiles_earliest = smiles_df['Time Start'].min()
    smiles_latest = smiles_df['Time End'].max()

    # Find the earliest and latest times from sleep_df if not empty
    if not sleep_df.empty:
        sleep_earliest = sleep_df['Time Start'].min()
        sleep_latest = sleep_df['Time End'].max()
        # Determine the start and end times
        start_time = min(smiles_earliest, sleep_earliest) if pd.notna(sleep_earliest) else smiles_earliest
        end_time = max(smiles_latest, sleep_latest) if pd.notna(sleep_latest) else smiles_latest
    else:
        # Use smile times if sleep_df is empty or all NaN
        start_time = smiles_earliest
        end_time = smiles_latest

    # Create a DataFrame with fixed frequency for the time range
    time_range = pd.date_range(start=start_time, end=end_time, freq='10S')
    tracking_df = pd.DataFrame({'Time': time_range, 'BufferSafe': False})

    # Determine the BufferSafe column values
    for i in range(len(tracking_df)):
        time = tracking_df.loc[i, 'Time']
        buffer_before = time - pd.Timedelta(minutes=1)
        buffer_after = time + pd.Timedelta(minutes=1)

        has_smile_within_buffer = smiles_df[((smiles_df['Time Start'] <= buffer_after) & (smiles_df['Time End'] >= buffer_before))].shape[0] > 0
        has_sleep_within_buffer = sleep_df[((sleep_df['Time Start'] <= buffer_after) & (sleep_df['Time End'] >= buffer_before))].shape[0] > 0 if not sleep_df.empty else False
        tracking_df.loc[i, 'BufferSafe'] = not (has_smile_within_buffer or has_sleep_within_buffer)

    # Filter the BufferSafe time intervals
    non_smile_non_sleep_buffer_times = tracking_df[tracking_df['BufferSafe']]['Time'].reset_index(drop=True)

    return non_smile_non_sleep_buffer_times
