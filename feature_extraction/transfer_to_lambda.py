df_moodTracking['Datetime'] = df_moodTracking['Datetime'].apply(lambda x: x if len(x.split(':')) == 3 else x + ':00')
