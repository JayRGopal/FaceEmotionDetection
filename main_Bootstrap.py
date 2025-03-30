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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURATION ---------------- #
PAT_NOW = 'S23_199'
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
BASE_RESULTS_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_Apr_2025/'
RESULTS_OUTPUT_PATH = os.path.join(BASE_RESULTS_PATH, PAT_NOW)
N_BOOTSTRAPS = 2
ALPHAS = np.linspace(0.1, 2.0, 20)
TIME_WINDOWS = list(range(15, 241, 15))
INTERNAL_STATES = ['Mood']
RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']

os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)

patient_folder = os.path.join(FEATURE_SAVE_FOLDER, PAT_NOW)
csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]

def parse_filename(filename):
    internal_state = filename.split('_features_')[0]
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    prefix_match = re.search(r'minutes_(.*)\.csv', filename)
    time_window = int(time_match.group(1)) if time_match else None
    prefix = prefix_match.group(1) if prefix_match else None
    return internal_state, time_window, prefix

summary_results = defaultdict(lambda: defaultdict(dict))
feature_heatmap_data = defaultdict(lambda: defaultdict(dict))
all_feature_names = set()

for file in tqdm(csv_files, desc="Processing all CSVs"):
    internal_state, time_window, prefix = parse_filename(file)
    if internal_state not in INTERNAL_STATES or time_window not in TIME_WINDOWS or prefix not in RESULTS_PREFIX_LIST:
        continue

    state_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state)
    prefix_folder = os.path.join(state_folder, prefix)
    overview_folder = os.path.join(state_folder, 'Overview')
    os.makedirs(prefix_folder, exist_ok=True)
    os.makedirs(overview_folder, exist_ok=True)

    df = pd.read_csv(os.path.join(patient_folder, file))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]
    all_feature_names.update(feature_names)

    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = model.predict(X[test_idx])[0]

    loo_r, _ = pearsonr(y, preds)

    coef_matrix = []
    bootstrap_r_values = []
    feature_selection_count = np.zeros(X.shape[1])

    for boot_iter in range(N_BOOTSTRAPS):
        X_boot, y_boot = resample(X, y, replace=True, n_samples=len(y), random_state=boot_iter)
        loo = LeaveOneOut()
        boot_preds = np.zeros_like(y_boot, dtype=float)
        boot_coefs = []
        for train_idx, test_idx in loo.split(X_boot):
            model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X_boot[train_idx], y_boot[train_idx])
            y_pred = model.predict(X_boot[test_idx])[0]
            boot_preds[test_idx[0]] = y_pred
            boot_coefs.append(model.coef_)
            feature_selection_count += (model.coef_ != 0).astype(int)

        r_boot, _ = pearsonr(y_boot, boot_preds)
        bootstrap_r_values.append(r_boot)
        coef_matrix.extend(boot_coefs)

    coef_matrix = np.array(coef_matrix)
    mean_r = np.mean(bootstrap_r_values)
    ci_lower = np.percentile(bootstrap_r_values, 2.5)
    ci_upper = np.percentile(bootstrap_r_values, 97.5)
    mean_importance = np.mean(np.abs(coef_matrix), axis=0)

    summary_results[internal_state][prefix][time_window] = {
        'r': mean_r,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

    for fname, imp in zip(feature_names, mean_importance):
        feature_heatmap_data[internal_state][fname][time_window] = imp

# --------- OVERVIEW AGGREGATE PLOTS ---------
for internal_state in summary_results:
    overview_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state, 'Overview')
    os.makedirs(overview_folder, exist_ok=True)

    for prefix in summary_results[internal_state]:
        time_r_dict = summary_results[internal_state][prefix]
        time_list = sorted(time_r_dict.keys())
        r_list = [time_r_dict[t]['r'] for t in time_list]
        lower_list = [time_r_dict[t]['ci_lower'] for t in time_list]
        upper_list = [time_r_dict[t]['ci_upper'] for t in time_list]

        plt.figure(figsize=(10, 6))
        plt.plot(time_list, r_list, marker='o', label='Pearson R')
        plt.fill_between(time_list, lower_list, upper_list, alpha=0.3, label='95% CI')
        plt.title(f"{internal_state} | {prefix} - Pearson R Across Time Windows", fontsize=14)
        plt.xlabel("Time Window (minutes)", fontsize=12)
        plt.ylabel("Mean Bootstrap Pearson R", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(overview_folder, f"summary_{prefix}_R_vs_time.png"))
        plt.close()

    # Feature Importance Heatmap
    feature_list = sorted(list(all_feature_names))
    time_list = sorted(TIME_WINDOWS)
    heatmap_matrix = np.zeros((len(feature_list), len(time_list)))

    for i, fname in enumerate(feature_list):
        for j, t in enumerate(time_list):
            heatmap_matrix[i, j] = feature_heatmap_data[internal_state].get(fname, {}).get(t, 0)

    plt.figure(figsize=(14, len(feature_list) * 0.4))
    sns.heatmap(heatmap_matrix, cmap='viridis', xticklabels=time_list, yticklabels=feature_list,
                cbar_kws={'label': 'Mean |Coefficient|'}, linewidths=0.5)
    plt.title(f"{internal_state} - Feature Importance Across Time Windows", fontsize=16)
    plt.xlabel("Time Window (minutes)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(overview_folder, f"{internal_state}_FeatureImportance_Heatmap.png"))
    plt.close()
