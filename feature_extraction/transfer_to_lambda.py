from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import binarize
from scipy.stats import pearsonr
import numpy as np

def find_optimal_alpha_and_correlation(vectors_return, y):
    # Step 1: Binarize y
    y_binarized = binarize(y.reshape(-1, 1), threshold=8, copy=True).reshape(-1)
    
    results = {}
    
    # Step 2: Loop through each key in vectors_return
    for key, X in vectors_return.items():
        # Step 4: Find the ideal alpha for LASSO logistic regression model
        # Note: Cs are the inverse of regularization strength; smaller values specify stronger regularization.
        # Using Leave-One-Out cross-validation
        clf = LogisticRegressionCV(
            Cs=10, cv=len(X), penalty='l1', solver='liblinear', scoring='accuracy', max_iter=1000
        ).fit(X, y_binarized)
        
        # Print the ideal alpha value (regularization strength) used
        optimal_alpha = 1 / clf.C_[0]
        print(f'Key {key}: Optimal alpha value is {optimal_alpha}')
        
        # Step 5: Use that alpha value to train a logistic regression model with LASSO
        predictions = clf.predict(X)
        
        # Step 7: Get the Pearson's R correlation and p-value
        R_val, p_val = pearsonr(predictions, y_binarized)
        
        # Update results dictionary
        results[key] = (R_val, p_val)
    
    # Step 8: Output the dictionary mapping each key to (Pearson_R_val, p_val)
    return results
