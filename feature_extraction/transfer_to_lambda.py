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
