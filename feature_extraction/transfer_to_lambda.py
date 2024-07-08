# Function to get average values of relevant columns for positive and negative examples
def get_average_values(data_dict, labels_df, columns):
    positive_values = {col: [] for col in columns}
    negative_values = {col: [] for col in columns}
    
    for timestamp, df in data_dict.items():
        if timestamp in labels_df['Datetime'].values:
            label = labels_df[labels_df['Datetime'] == timestamp]['EventDetected'].values[0]
            avg_features = df.mean()
            if label == 1:
                for col in columns:
                    positive_values[col].append(avg_features[col])
            else:
                for col in columns:
                    negative_values[col].append(avg_features[col])
    
    avg_positive = {col: np.mean(positive_values[col]) for col in columns}
    avg_negative = {col: np.mean(negative_values[col]) for col in columns}
    
    return avg_positive, avg_negative

# Define relevant columns for each data source
relevant_columns = {
    'openface_smile': ['AU06_r', 'AU12_r'],
    'opengraphau_smile': ['AU6', 'AU12'],
    'hsemotion_smile': ['Happiness']
}

# Dictionary to store average values
average_values = []

# Loop through each data source and calculate average values
for source, source_info in data_sources.items():
    data_dict = source_info['data']
    columns = relevant_columns[source]
    avg_positive, avg_negative = get_average_values(data_dict, Final_Smile_Labels, columns)
    
    for col in columns:
        average_values.append({
            'Source': source,
            'Column': col,
            'Positive Average': avg_positive[col],
            'Negative Average': avg_negative[col]
        })

# Create DataFrame for average values
average_values_df = pd.DataFrame(average_values)

# Display average values
print(average_values_df)
