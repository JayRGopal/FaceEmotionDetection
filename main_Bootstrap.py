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

# ---------------- CONFIGURATION ---------------- #
PAT_NOW = 'S23_199'
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
N_BOOTSTRAPS = 20
ALPHAS = np.linspace(0.1, 2.0, 20)  # Custom alpha search range

# User-defined filters
TIME_WINDOWS = list(range(15, 241, 15))  # 15, 30, ..., 240
# INTERNAL_STATES = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']
INTERNAL_STATES = ['Mood']
RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']

# ---------------- FILE HANDLING ---------------- #
patient_folder = os.path.join(FEATURE_SAVE_FOLDER, PAT_NOW)
csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]

def parse_filename(filename):
    internal_state = filename.split('_features_')[0]
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    prefix_match = re.search(r'minutes_(.*)\.csv', filename)
    time_window = int(time_match.group(1)) if time_match else None
    prefix = prefix_match.group(1) if prefix_match else None
    return internal_state, time_window, prefix

# For saving final summary
summary_results = defaultdict(lambda: defaultdict(dict))  # internal_state -> prefix -> {time: mean_r}

# ---------------- MAIN LOOP ---------------- #
for file in tqdm(csv_files, desc="Processing all CSVs"):
    internal_state, time_window, prefix = parse_filename(file)

    # Apply user-defined filters
    if internal_state not in INTERNAL_STATES:
        continue
    if time_window not in TIME_WINDOWS:
        continue
    if prefix not in RESULTS_PREFIX_LIST:
        continue

    # Load data
    df = pd.read_csv(os.path.join(patient_folder, file))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    print(f"\nLoaded: internal_state={internal_state} | time={time_window}min | prefix={prefix} | shape={df.shape}")

    # ---- Leave-One-Out Lasso ---- #
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
        model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X[train_idx], y[train_idx])
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
        boot_preds = np.zeros_like(y_boot, dtype=float)
        boot_coefs = []
        test_indices = []

        for train_idx, test_idx in loo.split(X_boot):
            model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X_boot[train_idx], y_boot[train_idx])
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
    bootstrap_r_values = np.array(bootstrap_r_values)
    mean_r = np.mean(bootstrap_r_values)
    ci_lower = np.percentile(bootstrap_r_values, 2.5)
    ci_upper = np.percentile(bootstrap_r_values, 97.5)

    print(f"Bootstrap R values (20): {np.round(bootstrap_r_values, 4)}")
    print(f"Mean bootstrap R: {mean_r:.4f}")
    print(f"95% CI for LASSO LOO Pearson R: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Save to summary
    summary_results[internal_state][prefix][time_window] = mean_r

    # ---- Feature Importance ---- #
    coef_matrix = np.array(coef_matrix)  # shape: (N * 20, num_features)
    mean_importance = np.mean(np.abs(coef_matrix), axis=0)
    feature_importance = sorted(zip(feature_names, mean_importance), key=lambda x: -x[1])

    print("\nTop 5 Features by Mean Absolute Coefficient (Bootstrap):")
    top5_coef = feature_importance[:5]
    for fname, imp in top5_coef:
        print(f"{fname}: {imp:.4f}")

    print("\nTop 5 Features by Permutation Impact (ΔR):")
    permutation_impacts = [
        (fname, mean_r - np.mean(perm_rs))
        for fname, perm_rs in permutation_r_per_feature.items()
    ]
    top5_perm = sorted(permutation_impacts, key=lambda x: -x[1])[:5]
    for fname, r_drop in top5_perm:
        print(f"{fname}: ΔR = {r_drop:.4f}")

# ---------------- SUMMARY REPORT ---------------- #
print("\n====================== Summary of Mean Bootstrap R by Internal State + Prefix ======================\n")
for internal_state in summary_results:
    for prefix in summary_results[internal_state]:
        print(f"\n{internal_state} | {prefix}")
        time_r_dict = summary_results[internal_state][prefix]
        time_list = sorted(time_r_dict.keys())
        r_list = [round(time_r_dict[t], 4) for t in time_list]
        print(f"Time windows: {time_list}")
        print(f"Mean R values: {r_list}")