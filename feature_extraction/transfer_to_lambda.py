def buffer_neither(smiles_df, sleep_df):
    # returns a single-column pandas df with times when:
    # the surrounding 2 minutes (buffer) have neither a smile event nor a sleep event
    # note: you can replace smiles_df with any df that has events (e.g. yawns).
    # For random sampling, it's looking at a discrete list of datetimes separated by 10 seconds

    # Find the earliest and latest times from smiles_df
    smiles_earliest = smiles_df['Time Start'].min()
    smiles_latest = smiles_df['Time End'].max()

    # Find the earliest and latest times from sleep_df
    sleep_earliest = sleep_df['Time Start'].min()
    sleep_latest = sleep_df['Time End'].max()

    # Determine the start and end times
    start_time = min(smiles_earliest, sleep_earliest)
    end_time = max(smiles_latest, sleep_latest)

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
