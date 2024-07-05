import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Function to prepare the dataset
def prepare_dataset(data_dict, labels_df):
    combined_df = pd.DataFrame()
    
    for timestamp, df in data_dict.items():
        avg_features = df.mean().to_frame().T
        avg_features['Datetime'] = timestamp
        combined_df = pd.concat([combined_df, avg_features], ignore_index=True)
    
    combined_df = combined_df.merge(labels_df, on='Datetime')
    X = combined_df.drop(columns=['Datetime', 'EventDetected'])
    y = combined_df['EventDetected']
    
    return X, y

# Function to train decision tree and print decision process
def train_decision_tree(X, y):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    tree_rules = export_text(clf, feature_names=list(X.columns))
    
    return clf, tree_rules

# Function to calculate accuracy and AUROC
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)
    
    return accuracy, auroc

# Code to call the functions
X, y = prepare_dataset(opengraphau_smile, Final_Smile_Labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree and print decision process
clf, tree_rules = train_decision_tree(X_train, y_train)

print("Decision Tree Rules:\n")
print(tree_rules)

# Evaluate the model
accuracy, auroc = evaluate_model(clf, X_test, y_test)

print(f"Accuracy: {accuracy}")
print(f"AUROC: {auroc}")
