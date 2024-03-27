from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import binarize
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def find_optimal_alpha_and_binary_pred(vectors_return, y):
    # Step 1: Binarize y
    y_binarized = binarize(y.reshape(-1, 1), threshold=9.99, copy=True).reshape(-1)
    
    results = {}
    
    # Step 2: Loop through each key in vectors_return
    for key, X in vectors_return.items():
        # Step 4: Find the ideal alpha for LASSO logistic regression model
        # Note: Cs are the inverse of regularization strength; smaller values specify stronger regularization.
        # Using Leave-One-Out cross-validation
        clf = LogisticRegressionCV(
            Cs=10, cv=LeaveOneOut(), penalty='l1', solver='liblinear', scoring='accuracy', max_iter=1000
        ).fit(X, y_binarized)
        
        # Print the ideal alpha value (regularization strength) used
        optimal_alpha = 1 / clf.C_[0]
        print(f'Time Window {key} Minutes: Optimal alpha value is {optimal_alpha}')
        
        # Step 5: AUROC and accuracy
        proba = clf.predict_proba(X)[:, 1]
        predictions = clf.predict(X)

        accuracy = accuracy_score(y_binarized, predictions)

        auroc = roc_auc_score(y_binarized, proba)
        
        # Update results dictionary
        results[key] = (accuracy, auroc)

        print(f"Time Window {key} Minutes -- Accuracy: {accuracy}, AUROC: {auroc}")
    
    # Step 6: Output the dictionary mapping each key to (accuracy, auroc)
    return results

