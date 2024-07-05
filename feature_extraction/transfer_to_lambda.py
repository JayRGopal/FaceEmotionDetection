# Function to sample data and calculate feature correlations
def sample_and_correlate(data_dict, labels_df, sample_fraction=0.8):
    sample_keys = np.random.choice(list(data_dict.keys()), size=int(len(data_dict) * sample_fraction), replace=False)
    sample_labels = labels_df[labels_df['Datetime'].isin(sample_keys)]
    
    combined_df = pd.DataFrame()
    for key in sample_keys:
        df = data_dict[key]
        df['Datetime'] = key
        combined_df = pd.concat([combined_df, df])
    
    combined_df = combined_df.merge(sample_labels, on='Datetime')
    
    correlations = combined_df.drop(columns=['Datetime', 'EventDetected']).apply(lambda x: x.corr(combined_df['EventDetected']))
    return correlations.abs().sort_values(ascending=False)

# Function to derive the optimal rule
def derive_optimal_rule(data_dict, labels_df):
    # Sample data and calculate feature correlations
    correlations = sample_and_correlate(data_dict, labels_df)
    
    # Select top correlated features
    top_features = correlations.index[:2]

    def optimal_rule(df, threshold):
        avg_features = df.mean()
        return (avg_features[top_features[0]] + avg_features[top_features[1]]) >= threshold
    
    # Tune threshold for the optimal rule
    thresholds = np.linspace(0, 2, 21)
    best_threshold, best_accuracy, best_auroc = tune_threshold(optimal_rule, data_dict, labels_df, thresholds)
    
    return optimal_rule, top_features, best_threshold, best_accuracy, best_auroc

best_rule, top_features, best_threshold, best_accuracy, best_auroc = derive_optimal_rule(opengraphau_smile, Final_Smile_Labels)

print(f"Best Rule: Using features {top_features} with Threshold: {best_threshold}")
print(f"Best Accuracy: {best_accuracy}")
print(f"Best AUROC: {best_auroc}")
