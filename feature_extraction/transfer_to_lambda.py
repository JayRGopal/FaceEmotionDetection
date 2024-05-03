def convert_time(df1, df2):
    # df1 has time start and time end for behavior
    # df2 has mapping from filename to video start

    # Create a copy of the first DataFrame
    modified_df = df1.copy()

    # Create a dictionary mapping 'Filename' to 'VideoStart'
    filename_to_videostart = dict(zip(df2['Filename'], df2['VideoStart']))

    # Convert 'Time Start' and 'Time End' columns to datetime based on the filename
    if PAT_SHORT_NAME == 'S_150':
      # For this patient, the manual labels are in format mm:ss.

      modified_df['Time Start'] = modified_df.apply(
          lambda row: pd.to_datetime(filename_to_videostart[row['Filename']]) +
                      pd.to_timedelta('00:' + row['Time Start'] + ' minutes'),
          axis=1
      )
      modified_df['Time End'] = modified_df.apply(
          lambda row: pd.to_datetime(filename_to_videostart[row['Filename']]) +
                      pd.to_timedelta('00:' + row['Time End'] + ' minutes'),
          axis=1
      )
    else:
      # For all other patients, manual labels are in format mm:ss:00.

      modified_df['Time Start'] = modified_df.apply(
          lambda row: pd.to_datetime(filename_to_videostart[row['Filename']]) +
                      pd.to_timedelta('00:' + row['Time Start'][:-3] + ' minutes'),
          axis=1
      )
      modified_df['Time End'] = modified_df.apply(
          lambda row: pd.to_datetime(filename_to_videostart[row['Filename']]) +
                      pd.to_timedelta('00:' + row['Time End'][:-3] + ' minutes'),
          axis=1
      )

    # Return the modified DataFrame
    return modified_df




TypeError: 'datetime.time' object is not subscriptable

