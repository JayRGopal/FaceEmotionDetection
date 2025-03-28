# Run this after main_FeatureExtractionInpatient_JustLinReg_extract_matrix.py



import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample
from scipy.stats import pearsonr
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Configuration
PAT_NOW = 'S23_199'
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
N_BOOTSTRAPS = 20

# Get patient folder
patient_folder = os.path.join(FEATURE_SAVE_FOLDER, PAT_NOW)
csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]

def parse_filename(filename):
    internal_state = filename.split('_features_')[0]
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    prefix_match = re.search(r'minutes_(.*)\.csv', filename)
    time_window = int(time_match.group(1)) if time_match else None
    prefix = prefix_match.group(1) if prefix_match else None
    return internal_state, time_window, prefix

for file in tqdm(csv_files, desc="Processing all CSVs"):
    internal_state, time_window, prefix = parse_filename(file)
    df = pd.read_csv(os.path.join(patient_folder, file))
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    print(f"\nLoaded: internal_state={internal_state} | time={time_window}min | prefix={prefix} | shape={df.shape}")

    # ---- Leave-One-Out Lasso ---- #
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    
    for train_idx, test_idx in loo.split(X):
        model = LassoCV(cv=5).fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = model.predict(X[test_idx])[0]
    
    loo_r, _ = pearsonr(y, preds)
    print(f"LOO Pearson R: {loo_r:.4f}")

    # ---- Bootstrap Lasso + Permutation ---- #
    coef_matrix = []
    bootstrap_r_values = []
    permutation_r_per_feature = defaultdict(list)

    for boot_iter in tqdm(range(N_BOOTSTRAPS), desc="Bootstrapping"):
        # Bootstrap sample
        X_boot, y_boot = resample(X, y, replace=True, n_samples=len(y), random_state=boot_iter)
        loo = LeaveOneOut()
        boot_preds = np.zeros_like(y_boot)
        boot_coefs = []
        test_indices = []

        for train_idx, test_idx in loo.split(X_boot):
            model = LassoCV(cv=5).fit(X_boot[train_idx], y_boot[train_idx])
            y_pred = model.predict(X_boot[test_idx])[0]
            boot_preds[test_idx[0]] = y_pred
            boot_coefs.append(model.coef_)
            test_indices.append(test_idx[0])

        r_boot, _ = pearsonr(y_boot, boot_preds)
        bootstrap_r_values.append(r_boot)
        coef_matrix.extend(boot_coefs)

        # ---- Permutation Test ---- #
        boot_coefs_arr = np.array(boot_coefs)
        X_test_matrix = X_boot[test_indices]
        for f_idx in range(X.shape[1]):
            X_test_permuted = X_test_matrix.copy()
            X_test_permuted[:, f_idx] = np.random.permutation(X_test_permuted[:, f_idx])

            perm_preds = np.sum(X_test_permuted * boot_coefs_arr, axis=1)
            r_perm, _ = pearsonr(y_boot[test_indices], perm_preds)
            permutation_r_per_feature[feature_names[f_idx]].append(r_perm)

    # ---- Results ---- #
    print(f"Bootstrap R values (20): {np.round(bootstrap_r_values, 4)}")

    coef_matrix = np.array(coef_matrix)  # shape: (N * 20, num_features)
    mean_importance = np.mean(np.abs(coef_matrix), axis=0)
    feature_importance = sorted(zip(feature_names, mean_importance), key=lambda x: -x[1])
    
    print("\nTop Features by Mean Absolute Coefficient (Bootstrap):")
    for fname, imp in feature_importance:
        print(f"{fname}: {imp:.4f}")

    print("\nFeature Permutation Impact (mean R drop):")
    for fname, perm_rs in sorted(permutation_r_per_feature.items(), key=lambda x: -np.mean(bootstrap_r_values) + np.mean(x[1])):
        r_drop = np.mean(bootstrap_r_values) - np.mean(perm_rs)
        print(f"{fname}: ΔR = {r_drop:.4f}")
