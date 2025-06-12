"""
Simplified Mood Prediction Analysis Script for Paper-Ready Figures

Produces:
A) One panel per patient: time window (x) vs R (y), bar plot, OGAUHSE_L_
B) Group-level: average R across patients per time window, bar plot, OGAUHSE_L_
C) Group-level: average R across patients per time window, bar plot, OF_L_
D) Leave-one-patient-out decoding for OGAUHSE_L_
E) Binary decoding AUC per patient, per time window (bar plot, like R but for AUC)
F) Same as above, but with features limited to LIMITED_FEATURES_SUBSTRINGS

Permutation-based null distribution testing is included for all panels, with p-values reported for each real R/AUC value.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION --- #
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
RESULTS_OUTPUT_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_May_2025/MoodPrediction'
ALPHAS = np.linspace(0.1, 10.0, 10)
TIME_WINDOWS = list(range(30, 241, 30))
METHODS = ['OGAUHSE_L_', 'OF_L_']
INTERNAL_STATE = 'Mood'
LIMITED_FEATURES_SUBSTRINGS = ["AU10", "AU12", "AU25", "AU27", "AU6", "AU7"]
N_PERMUTATIONS = 50  # Number of permutations for null distribution

os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)

METHOD_DISPLAY_MAP = {
    'OF_L_': 'OpenFace',
    'OGAUHSE_L_': 'FaceDx Complete'
}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def parse_filename(filename):
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    time_window = int(time_match.group(1)) if time_match else None
    return time_window

def remove_duplicate_features(df):
    feature_cols = df.columns[:-1]
    unique_cols = []
    seen = set()
    for col in feature_cols:
        if col not in seen:
            seen.add(col)
            unique_cols.append(col)
    unique_cols.append(df.columns[-1])
    return df[unique_cols]

def filter_limited_features(df):
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]
    matching_cols = [col for col in feature_cols if any(substr in col for substr in LIMITED_FEATURES_SUBSTRINGS)]
    matching_cols.append(target_col)
    return df[matching_cols]

def binarize_mood(df):
    mood_col = df.columns[-1]
    unique_vals = df[mood_col].unique()
    if set(unique_vals).issubset({0, 1, -1}):
        return None
    try:
        df[mood_col] = pd.to_numeric(df[mood_col], errors='raise')
    except Exception:
        return None
    median_mood = df[mood_col].median()
    binary_values = (df[mood_col] > median_mood)
    if binary_values.nunique() == 1:
        return None
    new_df = df.copy()
    new_df[mood_col] = binary_values.astype(int)
    value_counts = new_df[mood_col].value_counts()
    if (value_counts < 2).any():
        return None
    return new_df

def inclusion_criteria(mood_scores):
    if len(mood_scores) < 5:
        return False
    score_range = mood_scores.max() - mood_scores.min()
    if score_range < 5:
        return False
    unique_perms = len(mood_scores.unique())
    if unique_perms < 3:
        return False
    return True

def load_patient_data(method, limited=False):
    all_patient_data = {}
    for patient_folder in os.listdir(FEATURE_SAVE_FOLDER):
        patient_folder_path = os.path.join(FEATURE_SAVE_FOLDER, patient_folder)
        if not os.path.isdir(patient_folder_path):
            continue
        patient_id = patient_folder
        patient_data_loaded = False
        patient_meets_criteria = False
        for filename in os.listdir(patient_folder_path):
            if filename.endswith('.csv') and INTERNAL_STATE in filename and method in filename:
                time_window = parse_filename(filename)
                file_path = os.path.join(patient_folder_path, filename)
                df = pd.read_csv(file_path)
                if not patient_data_loaded:
                    mood_scores = df.iloc[:, -1]
                    patient_meets_criteria = inclusion_criteria(mood_scores)
                    patient_data_loaded = True
                    if not patient_meets_criteria:
                        break
                if not patient_meets_criteria:
                    break
                df = remove_duplicate_features(df)
                if limited:
                    df = filter_limited_features(df)
                if patient_id not in all_patient_data:
                    all_patient_data[patient_id] = {}
                all_patient_data[patient_id][time_window] = df
    return all_patient_data

def permutation_test(X, y, model_type, alphas, binary, n_permutations=50, random_state=42):
    """
    Returns a null distribution of R or AUC by shuffling y n_permutations times.
    """
    rng = np.random.RandomState(random_state)
    null_scores = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        loo = LeaveOneOut()
        preds, actuals = [], []
        if binary:
            model = LogisticRegressionCV(Cs=1/np.array(alphas), cv=loo, penalty='l1', solver='liblinear', random_state=random_state)
        else:
            model = LassoCV(alphas=alphas, cv=loo, random_state=random_state)
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_perm[train_idx], y_perm[test_idx]
            try:
                model.fit(X_train, y_train)
                if binary:
                    y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
                preds.extend(y_pred)
                actuals.extend(y_test)
            except Exception:
                # Sometimes permutation can cause degenerate splits (e.g., all one class)
                preds.extend([np.nan])
                actuals.extend([np.nan])
        preds = np.array(preds)
        actuals = np.array(actuals)
        if binary:
            if len(np.unique(actuals[~np.isnan(actuals)])) > 1:
                try:
                    score = roc_auc_score(actuals[~np.isnan(actuals)], preds[~np.isnan(preds)])
                except Exception:
                    score = np.nan
            else:
                score = np.nan
        else:
            if len(actuals[~np.isnan(actuals)]) > 1:
                try:
                    score, _ = pearsonr(actuals[~np.isnan(actuals)], preds[~np.isnan(preds)])
                except Exception:
                    score = np.nan
            else:
                score = np.nan
        null_scores.append(score)
    return np.array(null_scores)

def permutation_test_lopo(X_train, y_train, X_test, y_test, model_type, alphas, binary, n_permutations=50, random_state=42):
    """
    Returns a null distribution of R or AUC for LOPO by shuffling y_train n_permutations times.
    """
    rng = np.random.RandomState(random_state)
    null_scores = []
    for i in range(n_permutations):
        y_train_perm = rng.permutation(y_train)
        if binary:
            model = LogisticRegressionCV(Cs=1/np.array(alphas), cv=5, penalty='l1', solver='liblinear', random_state=random_state)
        else:
            model = LassoCV(alphas=alphas, cv=5, random_state=random_state)
        try:
            model.fit(X_train, y_train_perm)
            if binary:
                preds = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                if len(np.unique(y_test)) > 1:
                    score = roc_auc_score(y_test, preds)
                else:
                    score = np.nan
            else:
                preds = model.predict(X_test)
                if len(y_test) > 1:
                    score, _ = pearsonr(y_test, preds)
                else:
                    score = np.nan
        except Exception:
            score = np.nan
        null_scores.append(score)
    return np.array(null_scores)

def compute_p_value(real_score, null_scores, tail='right'):
    """
    Compute permutation p-value for real_score given null_scores.
    """
    null_scores = null_scores[~np.isnan(null_scores)]
    if len(null_scores) == 0 or np.isnan(real_score):
        return np.nan
    if tail == 'right':
        p = (np.sum(null_scores >= real_score) + 1) / (len(null_scores) + 1)
    elif tail == 'left':
        p = (np.sum(null_scores <= real_score) + 1) / (len(null_scores) + 1)
    else:
        # two-sided
        p = (np.sum(np.abs(null_scores) >= np.abs(real_score)) + 1) / (len(null_scores) + 1)
    return p

def per_patient_r_barplots(all_patient_data, method, limited=False, binary=False, outdir=None, auc=False):
    # For each patient, plot time window (x) vs R (y) or AUC (y)
    for patient_id, patient_data in all_patient_data.items():
        results = []
        pvals = []
        for time_window in TIME_WINDOWS:
            if time_window not in patient_data:
                results.append(np.nan)
                pvals.append(np.nan)
                continue
            df = patient_data[time_window]
            if binary:
                df_bin = binarize_mood(df)
                if df_bin is None:
                    results.append(np.nan)
                    pvals.append(np.nan)
                    continue
                df = df_bin
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            loo = LeaveOneOut()
            preds, actuals = [], []
            if binary:
                model = LogisticRegressionCV(Cs=1/np.array(ALPHAS), cv=loo, penalty='l1', solver='liblinear', random_state=42)
            else:
                model = LassoCV(alphas=ALPHAS, cv=loo, random_state=42)
            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(X_train, y_train)
                if binary:
                    y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
                preds.extend(y_pred)
                actuals.extend(y_test)
            if binary:
                if len(np.unique(actuals)) > 1:
                    score = roc_auc_score(actuals, preds)
                else:
                    score = np.nan
            else:
                if len(actuals) > 1:
                    score, _ = pearsonr(actuals, preds)
                else:
                    score = np.nan
            results.append(score)
            # Permutation test for this patient/time_window
            if not np.isnan(score):
                null_scores = permutation_test(X, y, model_type='logistic' if binary else 'lasso', alphas=ALPHAS, binary=binary, n_permutations=N_PERMUTATIONS, random_state=42)
                pval = compute_p_value(score, null_scores, tail='right')
            else:
                pval = np.nan
            pvals.append(pval)
        # Plot
        plt.figure(figsize=(8, 5))
        plt.bar([str(tw) for tw in TIME_WINDOWS], results, color=COLORS[0])
        plt.xlabel("Time Window (min)")
        plt.ylabel("AUC" if binary else "Pearson r")
        plt.title(f"{METHOD_DISPLAY_MAP.get(method, method)} - {'Limited ' if limited else ''}{'Binary ' if binary else ''}Patient {patient_id}")
        plt.ylim(-0.1, 1.0 if binary else 1.0)
        # Annotate p-values above bars
        for i, (val, pval) in enumerate(zip(results, pvals)):
            if not np.isnan(val) and not np.isnan(pval):
                plt.text(i, val + 0.03, f"p={pval:.3f}", ha='center', va='bottom', fontsize=9, rotation=90)
        plt.tight_layout()
        suffix = ""
        if limited: suffix += "_limited"
        if binary: suffix += "_binary"
        if outdir is None:
            outdir = RESULTS_OUTPUT_PATH
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{method}{suffix}_patient_{patient_id}_barplot.png"), dpi=300)
        plt.close()
        # Save results to CSV
        pd.DataFrame({"time_window": TIME_WINDOWS, "score": results, "pval": pvals}).to_csv(
            os.path.join(outdir, f"{method}{suffix}_patient_{patient_id}_scores.csv"), index=False
        )

def group_level_barplot(all_patient_data, method, limited=False, binary=False, outdir=None):
    # For each time window, average R (or AUC) across patients
    all_scores = []
    all_pvals = []
    for patient_id, patient_data in all_patient_data.items():
        patient_scores = []
        patient_pvals = []
        for time_window in TIME_WINDOWS:
            if time_window not in patient_data:
                patient_scores.append(np.nan)
                patient_pvals.append(np.nan)
                continue
            df = patient_data[time_window]
            if binary:
                df_bin = binarize_mood(df)
                if df_bin is None:
                    patient_scores.append(np.nan)
                    patient_pvals.append(np.nan)
                    continue
                df = df_bin
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            loo = LeaveOneOut()
            preds, actuals = [], []
            if binary:
                model = LogisticRegressionCV(Cs=1/np.array(ALPHAS), cv=loo, penalty='l1', solver='liblinear', random_state=42)
            else:
                model = LassoCV(alphas=ALPHAS, cv=loo, random_state=42)
            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(X_train, y_train)
                if binary:
                    y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
                preds.extend(y_pred)
                actuals.extend(y_test)
            if binary:
                if len(np.unique(actuals)) > 1:
                    score = roc_auc_score(actuals, preds)
                else:
                    score = np.nan
            else:
                if len(actuals) > 1:
                    score, _ = pearsonr(actuals, preds)
                else:
                    score = np.nan
            patient_scores.append(score)
            # Permutation test for this patient/time_window
            if not np.isnan(score):
                null_scores = permutation_test(X, y, model_type='logistic' if binary else 'lasso', alphas=ALPHAS, binary=binary, n_permutations=N_PERMUTATIONS, random_state=42)
                pval = compute_p_value(score, null_scores, tail='right')
            else:
                pval = np.nan
            patient_pvals.append(pval)
        all_scores.append(patient_scores)
        all_pvals.append(patient_pvals)
    all_scores = np.array(all_scores)
    all_pvals = np.array(all_pvals)
    mean_scores = np.nanmean(all_scores, axis=0)
    sem_scores = np.nanstd(all_scores, axis=0) / np.sqrt(np.sum(~np.isnan(all_scores), axis=0))
    # For group-level p-value, use mean of patient p-values (or combine via Fisher's method if desired)
    group_pvals = np.nanmean(all_pvals, axis=0)
    plt.figure(figsize=(8, 5))
    plt.bar([str(tw) for tw in TIME_WINDOWS], mean_scores, yerr=sem_scores, color=COLORS[0], capsize=5)
    plt.xlabel("Time Window (min)")
    plt.ylabel("AUC" if binary else "Pearson r")
    plt.title(f"{METHOD_DISPLAY_MAP.get(method, method)} - {'Limited ' if limited else ''}{'Binary ' if binary else ''}Group Level")
    plt.ylim(-0.1, 1.0 if binary else 1.0)
    # Annotate group p-values above bars
    for i, (val, pval) in enumerate(zip(mean_scores, group_pvals)):
        if not np.isnan(val) and not np.isnan(pval):
            plt.text(i, val + 0.03, f"p={pval:.3f}", ha='center', va='bottom', fontsize=9, rotation=90)
    plt.tight_layout()
    suffix = ""
    if limited: suffix += "_limited"
    if binary: suffix += "_binary"
    if outdir is None:
        outdir = RESULTS_OUTPUT_PATH
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{method}{suffix}_group_barplot.png"), dpi=300)
    plt.close()
    # Save group results to CSV
    pd.DataFrame({"time_window": TIME_WINDOWS, "mean_score": mean_scores, "sem_score": sem_scores, "mean_pval": group_pvals}).to_csv(
        os.path.join(outdir, f"{method}{suffix}_group_scores.csv"), index=False
    )

def leave_one_patient_out_decoding(all_patient_data, method, limited=False, binary=False, outdir=None):
    # For each time window, train on all but one patient, test on left-out patient, collect R or AUC
    lopo_results = {tw: [] for tw in TIME_WINDOWS}
    lopo_pvals = {tw: [] for tw in TIME_WINDOWS}
    patient_ids = list(all_patient_data.keys())
    for test_patient in patient_ids:
        for time_window in TIME_WINDOWS:
            # Gather training data
            X_train, y_train, X_test, y_test = [], [], None, None
            for pid in patient_ids:
                if time_window not in all_patient_data[pid]:
                    continue
                df = all_patient_data[pid][time_window]
                if limited:
                    df = filter_limited_features(df)
                if binary:
                    df_bin = binarize_mood(df)
                    if df_bin is None:
                        continue
                    df = df_bin
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                if np.isnan(X).any():
                    X = np.nan_to_num(X, nan=0.0)
                if pid == test_patient:
                    X_test, y_test = X, y
                else:
                    X_train.append(X)
                    y_train.append(y)
            if X_test is None or len(X_train) == 0:
                lopo_results[time_window].append(np.nan)
                lopo_pvals[time_window].append(np.nan)
                continue
            X_train = np.vstack(X_train)
            y_train = np.concatenate(y_train)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if binary:
                model = LogisticRegressionCV(Cs=1/np.array(ALPHAS), cv=5, penalty='l1', solver='liblinear', random_state=42)
            else:
                model = LassoCV(alphas=ALPHAS, cv=5, random_state=42)
            model.fit(X_train, y_train)
            if binary:
                preds = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                if len(np.unique(y_test)) > 1:
                    score = roc_auc_score(y_test, preds)
                else:
                    score = np.nan
            else:
                preds = model.predict(X_test)
                if len(y_test) > 1:
                    score, _ = pearsonr(y_test, preds)
                else:
                    score = np.nan
            lopo_results[time_window].append(score)
            # Permutation test for this LOPO split
            if not np.isnan(score):
                null_scores = permutation_test_lopo(X_train, y_train, X_test, y_test, model_type='logistic' if binary else 'lasso', alphas=ALPHAS, binary=binary, n_permutations=N_PERMUTATIONS, random_state=42)
                pval = compute_p_value(score, null_scores, tail='right')
            else:
                pval = np.nan
            lopo_pvals[time_window].append(pval)
    # Plot: for each time window, bar of mean across patients
    mean_scores = [np.nanmean(lopo_results[tw]) for tw in TIME_WINDOWS]
    sem_scores = [np.nanstd(lopo_results[tw]) / np.sqrt(np.sum(~np.isnan(lopo_results[tw]))) for tw in TIME_WINDOWS]
    mean_pvals = [np.nanmean(lopo_pvals[tw]) for tw in TIME_WINDOWS]
    plt.figure(figsize=(8, 5))
    plt.bar([str(tw) for tw in TIME_WINDOWS], mean_scores, yerr=sem_scores, color=COLORS[1], capsize=5)
    plt.xlabel("Time Window (min)")
    plt.ylabel("AUC" if binary else "Pearson r")
    plt.title(f"{METHOD_DISPLAY_MAP.get(method, method)} - {'Limited ' if limited else ''}{'Binary ' if binary else ''}Leave-One-Patient-Out")
    plt.ylim(-0.1, 1.0 if binary else 1.0)
    # Annotate group p-values above bars
    for i, (val, pval) in enumerate(zip(mean_scores, mean_pvals)):
        if not np.isnan(val) and not np.isnan(pval):
            plt.text(i, val + 0.03, f"p={pval:.3f}", ha='center', va='bottom', fontsize=9, rotation=90)
    plt.tight_layout()
    suffix = ""
    if limited: suffix += "_limited"
    if binary: suffix += "_binary"
    if outdir is None:
        outdir = RESULTS_OUTPUT_PATH
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{method}{suffix}_lopo_barplot.png"), dpi=300)
    plt.close()
    # Save LOPO results to CSV
    pd.DataFrame({"time_window": TIME_WINDOWS, "mean_score": mean_scores, "sem_score": sem_scores, "mean_pval": mean_pvals}).to_csv(
        os.path.join(outdir, f"{method}{suffix}_lopo_scores.csv"), index=False
    )

def main():
    # --- A, B, D, E, F for OGAUHSE_L_ --- #
    # Standard features
    oga_data = load_patient_data('OGAUHSE_L_', limited=False)
    # A) Per-patient R barplots
    per_patient_r_barplots(oga_data, 'OGAUHSE_L_', limited=False, binary=False)
    # B) Group-level R barplot
    group_level_barplot(oga_data, 'OGAUHSE_L_', limited=False, binary=False)
    # D) Leave-one-patient-out decoding (R)
    leave_one_patient_out_decoding(oga_data, 'OGAUHSE_L_', limited=False, binary=False)
    # E) Per-patient AUC barplots (binary decoding)
    per_patient_r_barplots(oga_data, 'OGAUHSE_L_', limited=False, binary=True)
    # F) Limited features
    oga_data_limited = load_patient_data('OGAUHSE_L_', limited=True)
    per_patient_r_barplots(oga_data_limited, 'OGAUHSE_L_', limited=True, binary=False)
    group_level_barplot(oga_data_limited, 'OGAUHSE_L_', limited=True, binary=False)
    leave_one_patient_out_decoding(oga_data_limited, 'OGAUHSE_L_', limited=True, binary=False)
    per_patient_r_barplots(oga_data_limited, 'OGAUHSE_L_', limited=True, binary=True)
    # --- C for OF_L_ --- #
    of_data = load_patient_data('OF_L_', limited=False)
    group_level_barplot(of_data, 'OF_L_', limited=False, binary=False)
    # Optionally, add more for OF_L_ if needed (e.g., per-patient, binary, limited)
    print("All paper-ready figures generated.")

if __name__ == "__main__":
    main()