def convert_time(df1, df2):
    # Create a copy of the first DataFrame
    modified_df = df1.copy()

    # Create a dictionary mapping 'Filename' to 'VideoStart'
    filename_to_videostart = dict(zip(df2['Filename'], df2['VideoStart']))

    # Define a function to handle time conversion based on the type of VideoStart
    def handle_time_conversion(row, time_field):
        video_start = filename_to_videostart[row['Filename']]
        if isinstance(video_start, datetime.time):
            # If it's a datetime.time object, format it to a string and convert to timedelta
            video_start_timedelta = pd.to_timedelta(video_start.strftime('%H:%M:%S'))
        elif isinstance(video_start, pd.Timestamp):
            # If it's a Timestamp, convert to timedelta since midnight
            video_start_timedelta = pd.to_timedelta(video_start.time().strftime('%H:%M:%S'))
        else:
            # Otherwise, directly use it as a timedelta (assuming it's either a string or timedelta)
            video_start_timedelta = pd.to_timedelta(video_start)

        # Convert time fields based on specific format for different patients
        if PAT_SHORT_NAME == 'S_150':
            # For this patient, the manual labels are in format mm:ss.
            time_delta = pd.to_timedelta('00:' + row[time_field])
        else:
            # For all other patients, manual labels are in format mm:ss:00.
            time_delta = pd.to_timedelta('00:' + row[time_field][:-3])

        return video_start_timedelta + time_delta

    # Apply the conversion function to the 'Time Start' and 'Time End'
    modified_df['Time Start'] = modified_df.apply(lambda row: handle_time_conversion(row, 'Time Start'), axis=1)
    modified_df['Time End'] = modified_df.apply(lambda row: handle_time_conversion(row, 'Time End'), axis=1)

    # Return the modified DataFrame
    return modified_df
