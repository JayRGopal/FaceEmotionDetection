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

# Function to get example timestamps and their probabilities
def get_example_predictions_with_probs(rule_func, data_dict, threshold):
    positive_predictions = []
    negative_predictions = []
    
    for timestamp, df in data_dict.items():
        prediction = int(rule_func(df, threshold))
        avg_features = df.mean().to_dict()
        if prediction == 1:
            positive_predictions.append((timestamp, avg_features))
        else:
            negative_predictions.append((timestamp, avg_features))
    
    return positive_predictions[:2], negative_predictions[:2]

# Dictionary to store example predictions with minute:second marks and probabilities
example_predictions_with_marks_and_probs = []

# Loop through each data source and rule, get example predictions with minute:second marks and probabilities
for source, source_info in data_sources.items():
    data_dict = source_info['data']
    for rule_name, rule_func in source_info['rules'].items():
        # Get best threshold for the current rule
        best_threshold = next(result['Best Threshold'] for result in results if result['Source'] == source and result['Rule'] == rule_name)
        
        positive_examples, negative_examples = get_example_predictions_with_probs(rule_func, data_dict, best_threshold)
        
        positive_examples_with_marks = [(get_minute_second_mark(ts, df_videoTimestamps), probs) for ts, probs in positive_examples]
        negative_examples_with_marks = [(get_minute_second_mark(ts, df_videoTimestamps), probs) for ts, probs in negative_examples]
        
        example_predictions_with_marks_and_probs.append({
            'Source': source,
            'Rule': rule_name,
            'Positive Examples': positive_examples_with_marks,
            'Negative Examples': negative_examples_with_marks
        })

# Print example predictions with minute:second marks and probabilities
for example in example_predictions_with_marks_and_probs:
    print(f"Source: {example['Source']}, Rule: {example['Rule']}")
    print("Positive Examples:")
    for mark, probs in example['Positive Examples']:
        print(f"  {mark}, Probabilities: {probs}")
    print("Negative Examples:")
    for mark, probs in example['Negative Examples']:
        print(f"  {mark}, Probabilities: {probs}")
    print()
