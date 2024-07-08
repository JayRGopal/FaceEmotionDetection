# Function to get example timestamps of positive and negative predicted smiles
def get_example_predictions(rule_func, data_dict, threshold):
    positive_predictions = []
    negative_predictions = []
    
    for timestamp, df in data_dict.items():
        prediction = int(rule_func(df, threshold))
        if prediction == 1:
            positive_predictions.append(timestamp)
        else:
            negative_predictions.append(timestamp)
    
    return positive_predictions[:2], negative_predictions[:2]

# Dictionary to store example predictions
example_predictions = []

# Loop through each data source and rule, get example predictions
for source, source_info in data_sources.items():
    data_dict = source_info['data']
    for rule_name, rule_func in source_info['rules'].items():
        # Get best threshold for the current rule
        best_threshold = next(result['Best Threshold'] for result in results if result['Source'] == source and result['Rule'] == rule_name)
        
        positive_examples, negative_examples = get_example_predictions(rule_func, data_dict, best_threshold)
        
        example_predictions.append({
            'Source': source,
            'Rule': rule_name,
            'Positive Examples': positive_examples,
            'Negative Examples': negative_examples
        })

# Function to convert timestamps to minute:second marks
def get_minute_second_mark(timestamp, video_df):
    for _, row in video_df.iterrows():
        video_start = row['VideoStart']
        video_end = row['VideoEnd']
        if video_start <= timestamp <= video_end:
            time_diff = timestamp - video_start
            minutes, seconds = divmod(time_diff.total_seconds(), 60)
            return f"{row['Filename']} {int(minutes)}:{int(seconds):02d}"
    return None

# Dictionary to store example predictions with minute:second marks
example_predictions_with_marks = []

# Loop through each data source and rule, get example predictions with minute:second marks
for example in example_predictions:
    positive_examples = [get_minute_second_mark(ts, df_videoTimestamps) for ts in example['Positive Examples']]
    negative_examples = [get_minute_second_mark(ts, df_videoTimestamps) for ts in example['Negative Examples']]
    
    example_predictions_with_marks.append({
        'Source': example['Source'],
        'Rule': example['Rule'],
        'Positive Examples': positive_examples,
        'Negative Examples': negative_examples
    })

# Print example predictions with minute:second marks
for example in example_predictions_with_marks:
    print(f"Source: {example['Source']}, Rule: {example['Rule']}")
    print(f"Positive Examples: {example['Positive Examples']}")
    print(f"Negative Examples: {example['Negative Examples']}")
    print()
