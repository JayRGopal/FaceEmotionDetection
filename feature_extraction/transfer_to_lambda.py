import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# Sample rule function with averaging
def smile_rule(df, threshold):
    avg_features = df.mean()
    return (avg_features['AU6'] + avg_features['AU12']) >= threshold

# Function to calculate accuracy and AUROC
def evaluate_rule(rule_func, data_dict, labels_df, threshold):
    y_true = []
    y_pred = []

    for timestamp, df in data_dict.items():
        if timestamp in labels_df['Datetime'].values:
            y_true.append(labels_df[labels_df['Datetime'] == timestamp]['EventDetected'].values[0])
            y_pred.append(int(rule_func(df, threshold)))

    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    
    return accuracy, auroc

# Function to tune threshold
def tune_threshold(rule_func, data_dict, labels_df, thresholds):
    results = []
    
    for threshold in thresholds:
        accuracy, auroc = evaluate_rule(rule_func, data_dict, labels_df, threshold)
        results.append({'Threshold': threshold, 'Accuracy': accuracy, 'AUROC': auroc})
    
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    # Assuming opengraphau_smile and Final_Smile_Labels are already defined
    # Example data
    opengraphau_smile = {
        '2024-07-05 10:00:00': pd.DataFrame({'AU6': [0.5, 0.3, 0.2, 0.4, 0.1], 'AU12': [0.7, 0.6, 0.8, 0.5, 0.9]}),
        '2024-07-05 10:05:00': pd.DataFrame({'AU6': [0.6, 0.4, 0.3, 0.5, 0.2], 'AU12': [0.6, 0.5, 0.7, 0.6, 0.8]}),
    }
    
    Final_Smile_Labels = pd.DataFrame({
        'Datetime': ['2024-07-05 10:00:00', '2024-07-05 10:05:00', '2024-07-05 10:10:00'],
        'EventDetected': [1, 0, 1]
    })

    # Define thresholds to test
    thresholds = np.linspace(0, 2, 21)

    # Tune threshold
    results_df = tune_threshold(smile_rule, opengraphau_smile, Final_Smile_Labels, thresholds)

    # Display results
    print(results_df)
