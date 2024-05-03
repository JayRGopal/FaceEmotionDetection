import datetime

def convert_time(df1, df2):
    # Create a copy of the first DataFrame
    modified_df = df1.copy()

    # Create a dictionary mapping 'Filename' to 'VideoStart'
    filename_to_videostart = dict(zip(df2['Filename'], df2['VideoStart']))

    # Define a function to handle time conversion based on the type of VideoStart and time fields
    def handle_time_conversion(row, time_field):
        video_start = filename_to_videostart.get(row['Filename'], None)
        if video_start is None:
            return pd.NaT  # Handling missing video start times

        if isinstance(video_start, datetime.time):
            video_start_timedelta = pd.to_timedelta(video_start.strftime('%H:%M:%S'))
        elif isinstance(video_start, pd.Timestamp):
            video_start_timedelta = pd.to_timedelta(video_start.time().strftime('%H:%M:%S'))
        else:
            try:
                video_start_timedelta = pd.to_timedelta(video_start)
            except ValueError:
                return pd.NaT  # If conversion fails, return NaT

        # Extracting and converting the time field, handling potential NaN values
        time_value = row[time_field]
        if pd.isna(time_value):
            return pd.NaT  # Handle NaN values gracefully
        
        if isinstance(time_value, datetime.time):
            time_str = time_value.strftime('%H:%M:%S')
        else:
            time_str = str(time_value)  # Ensure conversion to string if not datetime.time

        try:
            if PAT_SHORT_NAME == 'S_150':
                time_delta = pd.to_timedelta('00:' + time_str)
            else:
                time_delta = pd.to_timedelta('00:' + time_str[:-3])
        except ValueError:
            return pd.NaT  # Handle errors in time conversion

        return video_start_timedelta + time_delta

    # Apply the conversion function to the 'Time Start' and 'Time End'
    modified_df['Time Start'] = modified_df.apply(lambda row: handle_time_conversion(row, 'Time Start'), axis=1)
    modified_df['Time End'] = modified_df.apply(lambda row: handle_time_conversion(row, 'Time End'), axis=1)

    # Return the modified DataFrame
    return modified_df