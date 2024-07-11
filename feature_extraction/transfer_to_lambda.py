from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd

# Rule functions for OpenFace
def smile_rule_openface_au6(df, threshold):
    avg_features = df.mean()
    return avg_features['AU06_r'] >= threshold

def smile_rule_openface_au12(df, threshold):
    avg_features = df.mean()
    return avg_features['AU12_r'] >= threshold

def smile_rule_openface_au6_au12(df, threshold):
    avg_features = df.mean()
    return (avg_features['AU06_r'] + avg_features['AU12_r']) >= threshold

# Rule functions for OpenGraph
def smile_rule_opengraph_au6(df, threshold):
    avg_features = df.mean()
    return avg_features['AU6'] >= threshold

def smile_rule_opengraph_au12(df, threshold):
    avg_features = df.mean()
    return avg_features['AU12'] >= threshold

def smile_rule_opengraph_au6_au12(df, threshold):
    avg_features = df.mean()
    return (avg_features['AU6'] + avg_features['AU12']) >= threshold

# Rule function for HSE
def smile_rule_hse(df, threshold):
    avg_features = df.mean()
    return avg_features['Happiness'] >= threshold

# Function to calculate metrics
def evaluate_rule(rule_func, data_dict, labels_df, threshold):
    y_true = []
    y_pred = []

    for timestamp, df in data_dict.items():
        if timestamp in labels_df['Datetime'].values:
            y_true.append(labels_df[labels_df['Datetime'] == timestamp]['EventDetected'].values[0])
            y_pred.append(int(rule_func(df, threshold)))

    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return accuracy, auroc, sensitivity, specificity

# Function to tune threshold
def tune_threshold(rule_func, data_dict, labels_df, thresholds):
    best_threshold = None
    best_accuracy = 0
    best_auroc = 0
    best_sensitivity = 0
    best_specificity = 0
    
    for threshold in thresholds:
        accuracy, auroc, sensitivity, specificity = evaluate_rule(rule_func, data_dict, labels_df, threshold)
        if accuracy > best_accuracy or (accuracy == best_accuracy and auroc > best_auroc):
            best_threshold = threshold
            best_accuracy = accuracy
            best_auroc = auroc
            best_sensitivity = sensitivity
            best_specificity = specificity
    
    return best_threshold, best_accuracy, best_auroc, best_sensitivity, best_specificity

# Define thresholds to test
thresholds = np.linspace(0.1, 2, 21)

# Dictionary of data sources and their corresponding rule functions
data_sources = {
    'openface_smile': {
        'data': openface_smile,
        'rules': {
            'AU6_r': smile_rule_openface_au6,
            'AU12_r': smile_rule_openface_au12,
            'AU6_r_AU12_r': smile_rule_openface_au6_au12
        }
    },
    'opengraphau_smile': {
        'data': opengraphau_smile,
        'rules': {
            'AU6': smile_rule_opengraph_au6,
            'AU12': smile_rule_opengraph_au12,
            'AU6_AU12': smile_rule_opengraph_au6_au12
        }
    },
    'hsemotion_smile': {
        'data': hsemotion_smile,
        'rules': {
            'Happiness': smile_rule_hse
        }
    }
}

# DataFrame to store results
results = []

# Loop through each data source and rule, tune thresholds and evaluate
for source, source_info in data_sources.items():
    data_dict = source_info['data']
    for rule_name, rule_func in source_info['rules'].items():
        best_threshold, best_accuracy, best_auroc, best_sensitivity, best_specificity = tune_threshold(rule_func, data_dict, Final_Smile_Labels, thresholds)
        results.append({
            'Source': source,
            'Rule': rule_name,
            'Best Threshold': best_threshold,
            'Accuracy': best_accuracy,
            'AUROC': best_auroc,
            'Sensitivity': best_sensitivity,
            'Specificity': best_specificity
        })

# Create DataFrame
results_df = pd.DataFrame(results)

# Display results
results_df
