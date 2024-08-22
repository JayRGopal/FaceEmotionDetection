import os


df_mood = pd.DataFrame(self_reports_dict_list_mood)
df_depression = pd.DataFrame(self_reports_dict_list_depression)
df_anxiety = pd.DataFrame(self_reports_dict_list_anxiety)

merged_df = df_mood.merge(df_depression, on='Sheet Name').merge(df_anxiety, on='Sheet Name')

# Fix Days Spanned
merged_df['Days Spanned'] = merged_df.filter(like='Days Spanned').max(axis=1)
merged_df = merged_df.drop(columns=merged_df.filter(like='Days Spanned').columns.difference(['Days Spanned']))

# Get the directory of the provided sheet path
directory = os.path.dirname(MOOD_TRACKING_SHEET_PATH)

# Define the new file path
new_file_path = os.path.join(directory, 'Mood_Tracking_Overview.xlsx')

# Save the DataFrame to the new Excel file
with pd.ExcelWriter(new_file_path, engine='openpyxl') as writer:
    merged_df.to_excel(writer, sheet_name='Data_Overview', index=False)
