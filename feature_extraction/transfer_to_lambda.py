from sklearn.metrics import accuracy_score, roc_auc_score

def smile_rule(df, threshold):
    avg_features = df.mean()
    return (avg_features['AU06_r'] + avg_features['AU12_r']) >= threshold

def smile_rule_hse(df, threshold):
    avg_features = df.mean()
    return (avg_features['Happiness']) >= threshold


def discomfort_rule(df, threshold):
    avg_features = df.mean()
    return (avg_features['AU04_r'] + avg_features['AU06_r'] + avg_features['AU09_r'] + avg_features['AU12_r']) >= threshold


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
    best_threshold = None
    best_accuracy = 0
    best_auroc = 0
    
    for threshold in thresholds:
        accuracy, auroc = evaluate_rule(rule_func, data_dict, labels_df, threshold)
        if accuracy > best_accuracy or (accuracy == best_accuracy and auroc > best_auroc):
            best_threshold = threshold
            best_accuracy = accuracy
            best_auroc = auroc
    
    return best_threshold, best_accuracy, best_auroc


# TESTING ONE RULE

# Define thresholds to test
thresholds = np.linspace(0, 1, 21)

# Tune threshold
best_threshold, best_accuracy, best_auroc = tune_threshold(smile_rule_hse, hsemotion_smile, Final_Smile_Labels, thresholds)
