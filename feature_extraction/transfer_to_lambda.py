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

# Print example predictions
for example in example_predictions:
    print(f"Source: {example['Source']}, Rule: {example['Rule']}")
    print(f"Positive Examples: {example['Positive Examples']}")
    print(f"Negative Examples: {example['Negative Examples']}")
    print()
