def filter_df_by_behavior(df, desired_string):
    # Create a copy of the DataFrame
    filtered_df = df.copy()

    # Filter the DataFrame based on the desired string within 'Behavior' column
    filtered_df = filtered_df[filtered_df['Behavior'].str.contains(desired_string)]

    # Reset the index of the filtered DataFrame
    filtered_df = filtered_df.reset_index(drop=True)

    # Return the filtered DataFrame
    return filtered_df

import datetime


def add_time_strings(t1, t2):
    total_seconds = sum(x.total_seconds() for x in [pd.to_timedelta(t) for t in [t1, t2]])
    return str(pd.to_timedelta(total_seconds, unit='s'))

def convert_time(df1, df2):
    if df1.empty:
        return df1
    
    modified_df = df1.copy()

    filename_to_videostart = dict(zip(df2['Filename'], df2['VideoStart']))

    def handle_time_conversion(row, time_field):
        video_start = filename_to_videostart.get(row['Filename'], pd.NaT)
        if pd.isna(video_start):
            return pd.NaT

        try:
            video_start_timedelta = pd.to_timedelta(video_start.strftime('%H:%M:%S'))
        except AttributeError:
            video_start_timedelta = pd.to_timedelta(video_start)

        time_value = row[time_field]
        if pd.isna(time_value):
            return pd.NaT

        try:
            time_str = time_value.strftime('%H:%M:%S')
        except AttributeError:
            time_str = str(time_value)
        
        return add_time_strings(video_start_timedelta, pd.to_timedelta(time_str))

    modified_df['Time Start'] = modified_df.apply(lambda row: handle_time_conversion(row, 'Time Start'), axis=1)
    modified_df['Time End'] = modified_df.apply(lambda row: handle_time_conversion(row, 'Time End'), axis=1)

    return modified_df


import pandas as pd

def buffer_neither(smiles_df, sleep_df):
    # Convert time columns to datetime if not already
    smiles_df['Time Start'] = pd.to_datetime(smiles_df['Time Start'], errors='coerce')
    smiles_df['Time End'] = pd.to_datetime(smiles_df['Time End'], errors='coerce')

    if not sleep_df.empty:
        sleep_df['Time Start'] = pd.to_datetime(sleep_df['Time Start'], errors='coerce')
        sleep_df['Time End'] = pd.to_datetime(sleep_df['Time End'], errors='coerce')

    # Drop rows with NaT values in smiles dataframe
    smiles_df = smiles_df.dropna(subset=['Time Start', 'Time End'])

    # Check if sleep dataframe is not empty and drop rows with NaT values
    if not sleep_df.empty:
        sleep_df = sleep_df.dropna(subset=['Time Start', 'Time End'])

    # If both dataframes are empty
    if smiles_df.empty and (sleep_df.empty or sleep_df is None):
        return pd.DataFrame(columns=['Time'])  # Return empty dataframe if no data available

    # Define ranges using non-empty dataframe(s)
    start_times = pd.Series(smiles_df['Time Start'].tolist() + (sleep_df['Time Start'].tolist() if not sleep_df.empty else [])).dropna()
    end_times = pd.Series(smiles_df['Time End'].tolist() + (sleep_df['Time End'].tolist() if not sleep_df.empty else [])).dropna()

    if start_times.empty or end_times.empty:
        return pd.DataFrame(columns=['Time'])  # Return empty dataframe if no valid times are available

    # Find the earliest and latest times
    start_time = start_times.min()
    end_time = end_times.max()

    # Create a DataFrame with fixed frequency for the time range
    time_range = pd.date_range(start=start_time, end=end_time, freq='10S')
    tracking_df = pd.DataFrame({'Time': time_range, 'BufferSafe': True})

    # Set BufferSafe status based on proximity to smile and sleep events
    for i in range(len(tracking_df)):
        time = tracking_df.loc[i, 'Time']
        buffer_before = time - pd.Timedelta(minutes=1)
        buffer_after = time + pd.Timedelta(minutes=1)

        has_smile_within_buffer = smiles_df[((smiles_df['Time Start'] <= buffer_after) & (smiles_df['Time End'] >= buffer_before))].shape[0] > 0
        has_sleep_within_buffer = False if sleep_df.empty else (sleep_df[((sleep_df['Time Start'] <= buffer_after) & (sleep_df['Time End'] >= buffer_before))].shape[0] > 0)
        tracking_df.loc[i, 'BufferSafe'] = not (has_smile_within_buffer or has_sleep_within_buffer)

    return tracking_df[tracking_df['BufferSafe']]['Time'].reset_index(drop=True)


def create_event_detection_df(smiles_df, safe_series):
    # Create a new DataFrame for event detection
    event_detection_df = pd.DataFrame(columns=['Datetime', 'EventDetected'])
    # Iterate over each row in the smiles_df
    for index, row in smiles_df.iterrows():
        start_time = row['Time Start']
        end_time = row['Time End']

        # Generate a range of timestamps at a frequency of 1 second
        timestamps = pd.date_range(start=start_time, end=end_time, freq='S', inclusive='right')

        # Add each timestamp as a separate row to the event_detection_df
        for timestamp in timestamps:
            event_detection_df = pd.concat([event_detection_df, pd.DataFrame.from_records([{'Datetime': timestamp, 'EventDetected': 1}])], ignore_index=True)

    # Get the length of the smile event DataFrame
    num_smiles = len(event_detection_df)

    # Randomly sample from the buffer safe Series
    sampled_safe_series = safe_series.sample(n=num_smiles, replace=False)

    # Add nonsmile nonsleep events to the DataFrame
    nonsmile_nonsleep_times = sampled_safe_series.reset_index(drop=True)
    nonsmile_nonsleep_df = pd.DataFrame({'Datetime': nonsmile_nonsleep_times, 'EventDetected': 0})
    event_detection_df = pd.concat([event_detection_df, nonsmile_nonsleep_df], ignore_index=True)

    # Sort the DataFrame by DateTime in ascending order
    event_detection_df = event_detection_df.sort_values(by='Datetime').reset_index(drop=True)

    return event_detection_df

def get_labels(smile_string, sleep_string):
  # gets us our labels df (DateTime and EventDetected columns)
  # note: doesn't need to be smiles. Replace 'smile' with any other event as first arg.

  # smile string is what we want to detect
  # sleep string is what we label as neither smile nor non-smile
  # i.e. if a time period is labeled as sleep, exclude from dataset

  # Make sure Danny_Labels and df_videoTimestamps have been loaded in already!

  Smile_Labels = filter_df_by_behavior(Danny_Labels, smile_string)
  Sleep_Labels = filter_df_by_behavior(Danny_Labels, sleep_string)
  if not(Smile_Labels.empty):
      smiles_df = convert_time(Smile_Labels, df_videoTimestamps)
  else:
      smiles_df = Smile_Labels

  if not(Sleep_Labels.empty):
      sleep_df = convert_time(Sleep_Labels, df_videoTimestamps)
  else:
      sleep_df = Sleep_Labels

  non_smile_non_sleep_times = buffer_neither(smiles_df, sleep_df)


  return create_event_detection_df(smiles_df, non_smile_non_sleep_times)

Final_Smile_Labels = get_labels('smile', 'sleep')


