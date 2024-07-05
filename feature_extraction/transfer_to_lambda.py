# Function to evaluate an existing classifier
def evaluate_existing_model(clf, data_dict, labels_df):
    X, y = prepare_dataset(data_dict, labels_df)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    auroc = roc_auc_score(y, y_pred)
    
    return accuracy, auroc

# Example usage
accuracy, auroc = evaluate_existing_model(facedx_clf, facedx_smile, Final_Smile_Labels)
print(f"Accuracy: {accuracy}")
print(f"AUROC: {auroc}")