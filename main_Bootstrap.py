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
RESULTS_OUTPUT_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_Apr_2025/{PAT_NOW}/'
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
all_feature_stats = []

for file in tqdm(csv_files, desc="Processing all CSVs"):
    internal_state, time_window, prefix = parse_filename(file)
    if internal_state not in INTERNAL_STATES or time_window not in TIME_WINDOWS or prefix not in RESULTS_PREFIX_LIST:
        continue

    df = pd.read_csv(os.path.join(patient_folder, file))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    print(f"\nLoaded: internal_state={internal_state} | time={time_window}min | prefix={prefix} | shape={df.shape}")

    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = model.predict(X[test_idx])[0]

    loo_r, _ = pearsonr(y, preds)

    coef_matrix = []
    bootstrap_r_values = []
    permutation_r_per_feature = defaultdict(list)
    feature_selection_count = np.zeros(X.shape[1])

    for boot_iter in range(N_BOOTSTRAPS):
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
            feature_selection_count += (model.coef_ != 0).astype(int)
            test_indices.append(test_idx[0])

        r_boot, _ = pearsonr(y_boot, boot_preds)
        bootstrap_r_values.append(r_boot)
        coef_matrix.extend(boot_coefs)

        boot_coefs_arr = np.array(boot_coefs)
        X_test_matrix = X_boot[test_indices]
        for f_idx in range(X.shape[1]):
            X_test_permuted = X_test_matrix.copy()
            X_test_permuted[:, f_idx] = np.random.permutation(X_test_permuted[:, f_idx])
            perm_preds = np.sum(X_test_permuted * boot_coefs_arr, axis=1)
            r_perm, _ = pearsonr(y_boot[test_indices], perm_preds)
            permutation_r_per_feature[feature_names[f_idx]].append(r_perm)

    bootstrap_r_values = np.array(bootstrap_r_values)
    mean_r = np.mean(bootstrap_r_values)
    ci_lower = np.percentile(bootstrap_r_values, 2.5)
    ci_upper = np.percentile(bootstrap_r_values, 97.5)

    summary_results[internal_state][prefix][time_window] = mean_r

    coef_matrix = np.array(coef_matrix)
    mean_importance = np.mean(np.abs(coef_matrix), axis=0)
    feature_importance = sorted(zip(feature_names, mean_importance), key=lambda x: -x[1])

    top10 = feature_importance[:10]
    bottom10 = feature_importance[-10:]

    correlation_stats = []
    for fname, _ in top10 + bottom10:
        r, p = pearsonr(df[fname], y)
        correlation_stats.append({'feature': fname, 'pearson_r': r, 'p_value': p})

    feature_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_coef': mean_importance,
        'selection_freq': feature_selection_count / (N_BOOTSTRAPS * len(y)),
        'correlation_with_y': [pearsonr(df[f], y)[0] for f in feature_names]
    })

    base_fn = f"{internal_state}_time_{time_window}_prefix_{prefix}"
    feature_df.to_csv(os.path.join(RESULTS_OUTPUT_PATH, f"features_{base_fn}.csv"), index=False)
    pd.DataFrame(correlation_stats).to_csv(os.path.join(RESULTS_OUTPUT_PATH, f"correlations_{base_fn}.csv"), index=False)

    # --------- PLOTS ---------
    plt.figure(figsize=(10, 6))
    feature_bar_df = pd.DataFrame(top10 + bottom10, columns=['feature', 'mean_abs_coef'])
    sns.barplot(x='feature', y='mean_abs_coef', data=feature_bar_df, palette='coolwarm')
    plt.title(f'Top & Bottom 10 Features by Coefficient: {base_fn}', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_OUTPUT_PATH, f"barplot_features_{base_fn}.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sorted_feat_idx = np.argsort(mean_importance)[::-1]
    sorted_feats = np.array(feature_names)[sorted_feat_idx]
    sorted_freq = feature_selection_count[sorted_feat_idx] / (N_BOOTSTRAPS * len(y))
    sns.heatmap([sorted_freq], cmap='viridis', cbar_kws={'label': '% Bootstraps Selected'}, xticklabels=sorted_feats)
    plt.title(f"Feature Selection Heatmap: {base_fn}", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_OUTPUT_PATH, f"heatmap_selection_{base_fn}.png"))
    plt.close()

# --------- SUMMARY PLOT OF TIME WINDOWS ---------
for internal_state in summary_results:
    for prefix in summary_results[internal_state]:
        time_r_dict = summary_results[internal_state][prefix]
        time_list = sorted(time_r_dict.keys())
        r_list = [summary_results[internal_state][prefix][t] for t in time_list]

        plt.figure(figsize=(10, 6))
        plt.plot(time_list, r_list, marker='o')
        plt.fill_between(time_list, [ci_lower]*len(time_list), [ci_upper]*len(time_list), alpha=0.3)
        plt.title(f"{internal_state} | {prefix} - Pearson R Across Time Windows", fontsize=14)
        plt.xlabel("Time Window (minutes)", fontsize=12)
        plt.ylabel("Mean Bootstrap Pearson R", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_OUTPUT_PATH, f"summary_{internal_state}_{prefix}_R_vs_time.png"))
        plt.close()
        