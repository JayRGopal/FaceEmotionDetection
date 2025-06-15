"""
Simplified Mood, Anxiety, and Depression Prediction Analysis Script for Paper-Ready Figures

Produces:
A) One panel per patient: time window (x) vs R (y), bar plot, OGAUHSE_L_
B) Group-level: average R across patients per time window, bar plot, OGAUHSE_L_
C) Group-level: average R across patients per time window, bar plot, OF_L_
D) Leave-one-patient-out decoding for OGAUHSE_L_
E) Binary decoding AUC per patient, per time window (bar plot, like R but for AUC)
F) Same as above, but with features limited to LIMITED_FEATURES_SUBSTRINGS

Permutation-based null distribution testing is included for all panels, with p-values reported for each real R/AUC value.

Additionally, prints a table of Mood, Anxiety, and Depression scores for each patient before analysis.
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

# Add tqdm for progress bars
from tqdm import tqdm

# --- CONFIGURATION --- #
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
RESULTS_OUTPUT_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_June_2025/MoodPrediction'
ALPHAS = np.logspace(-3, 1, 20)  # Log scale from 0.001 to 10
TIME_WINDOWS = list(range(30, 241, 30))
METHODS = ['OGAUHSE_L_', 'OF_L_']
INTERNAL_STATES = ['Depression', 'Mood', 'Anxiety']
LIMITED_FEATURES_SUBSTRINGS = ["AU10", "AU12", "AU25", "AU27", "AU6", "AU7"]
N_PERMUTATIONS = 50  # Number of permutations for null distribution

os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)

METHOD_DISPLAY_MAP = {
    'OF_L_': 'OpenFace',
    'OGAUHSE_L_': 'FaceDx Complete'
}

# --- PATIENT FILTERING --- #
INCLUDED_PATIENTS = [
    'S23_174', 'S23_199', 'S23_212', 'S23_214', 'S24_217', 'S24_219',
    'S24_224', 'S24_226', 'S24_227', 'S24_230', 'S24_231', 'S24_234'
]

# --- PATIENT NUMBER MAPPING --- #
# Full mapping for all 15 patients (in order)
PATIENT_NUMBER_MAP_FULL = {
    'S23_174': 1,
    'S23_199': 2,
    'S23_211': 3,
    'S23_212': 4,
    'S23_214': 5,
    'S24_217': 6,
    'S24_219': 7,
    'S24_222': 8,
    'S24_224': 9,
    'S24_226': 10,
    'S24_227': 11,
    'S24_230': 12,
    'S24_231': 13,
    'S24_233': 14,
    'S24_234': 15
}
# Only keep mapping for included patients
PATIENT_NUMBER_MAP = {pid: PATIENT_NUMBER_MAP_FULL[pid] for pid in INCLUDED_PATIENTS if pid in PATIENT_NUMBER_MAP_FULL}

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# --- NEW: Load Mood, Anxiety, Depression scores for each patient --- #
def load_behavioral_scores(xlsx_path, included_patients):
    """
    Loads Mood, Anxiety, and Depression scores for each patient from the Excel file.
    Returns a dict: patient_id -> DataFrame with columns ['Mood', 'Anxiety', 'Depression']
    """
    xl = pd.ExcelFile(xlsx_path)
    patient_scores = {}
    for pid in included_patients:
        # Convert S24_241 -> S_241
        if '_' in pid:
            s_code = 'S_' + pid.split('_')[1]
        else:
            s_code = pid
        if s_code not in xl.sheet_names:
            print(f"[WARN] Sheet {s_code} not found in {xlsx_path} for patient {pid}")
            continue
        df = xl.parse(s_code)
        # Only keep rows where Mood is not null
        if 'Mood' not in df.columns or 'Anxiety' not in df.columns or 'Depression' not in df.columns:
            print(f"[WARN] Missing columns in sheet {s_code} for patient {pid}")
            continue
        df = df.loc[df['Mood'].notnull(), ['Mood', 'Anxiety', 'Depression']]
        # Reset index for clean display
        df = df.reset_index(drop=True)
        patient_scores[pid] = df
    return patient_scores

def print_behavioral_scores_table(patient_scores):
    """
    Prints a table of Mood, Anxiety, and Depression scores for each patient.
    """
    print("\n=== Mood, Anxiety, and Depression Scores for Each Patient ===")
    # Find max number of entries for any patient
    max_len = max([len(df) for df in patient_scores.values()]) if patient_scores else 0
    # Build header
    header = ["Patient", "Entry#", "Mood", "Anxiety", "Depression"]
    print("{:<12} {:<7} {:<8} {:<8} {:<10}".format(*header))
    print("-" * 50)
    for pid in INCLUDED_PATIENTS:
        df = patient_scores.get(pid, None)
        if df is None or df.empty:
            print("{:<12} {:<7} {:<8} {:<8} {:<10}".format(pid, "", "N/A", "N/A", "N/A"))
            continue
        for idx, row in df.iterrows():
            print("{:<12} {:<7} {:<8} {:<8} {:<10}".format(
                pid if idx == 0 else "",
                idx+1,
                str(row['Mood']),
                str(row['Anxiety']),
                str(row['Depression'])
            ))
    print("-" * 50)
    print("")

# --- END NEW ---

def parse_filename(filename):
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    time_window = int(time_match.group(1)) if time_match else None
    return time_window

def remove_duplicate_features_take_second(df):
    """
    Remove duplicate columns, keeping the second occurrence (e.g., the .1 version).
    """
    feature_cols = list(df.columns[:-1])
    target_col = df.columns[-1]
    col_counts = {}
    keep_cols = []
    for col in feature_cols:
        base = col
        if base.endswith('.1'):
            base = base[:-2]
        col_counts.setdefault(base, []).append(col)
    # For each base, if duplicate, keep the last (second) occurrence
    for base, col_list in col_counts.items():
        if len(col_list) == 1:
            keep_cols.append(col_list[0])
        else:
            keep_cols.append(col_list[-1])
    keep_cols.append(target_col)
    return df[keep_cols]

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
    if (value_counts < 2).any() or len(value_counts) < 2:
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

def load_patient_data(method, internal_state, limited=False):
    all_patient_data = {}
    patient_folders = [pf for pf in os.listdir(FEATURE_SAVE_FOLDER) if os.path.isdir(os.path.join(FEATURE_SAVE_FOLDER, pf))]
    # Only keep patient folders that are in INCLUDED_PATIENTS
    patient_folders = [pf for pf in patient_folders if pf in INCLUDED_PATIENTS]
    # First, load all data and keep track of feature sets
    patient_time_features = {}
    patient_time_dfs = {}
    for patient_folder in tqdm(patient_folders, desc=f"Loading data for method {method}{' (limited)' if limited else ''} | {internal_state}"):
        patient_folder_path = os.path.join(FEATURE_SAVE_FOLDER, patient_folder)
        patient_id = patient_folder
        patient_data_loaded = False
        patient_meets_criteria = False
        for filename in os.listdir(patient_folder_path):
            if filename.endswith('.csv') and internal_state in filename and method in filename:
                time_window = parse_filename(filename)
                file_path = os.path.join(patient_folder_path, filename)
                df = pd.read_csv(file_path)
                # Remove duplicate features, keeping the second occurrence
                df = remove_duplicate_features_take_second(df)
                # --- NEW: Drop rows where the self-report (last column) is NaN --- #
                df = df[df.iloc[:, -1].notna()].reset_index(drop=True)
                # --- END NEW ---
                if limited:
                    df = filter_limited_features(df)
                if not patient_data_loaded:
                    mood_scores = df.iloc[:, -1]
                    patient_meets_criteria = inclusion_criteria(mood_scores)
                    patient_data_loaded = True
                    if not patient_meets_criteria:
                        break
                if not patient_meets_criteria:
                    break
                # Save feature set and df for this patient/time_window
                feature_set = set(df.columns[:-1])
                patient_time_features.setdefault(patient_id, {})[time_window] = feature_set
                patient_time_dfs.setdefault(patient_id, {})[time_window] = df
    # Now, for this method, only keep patients who have ALL time windows
    valid_patients = []
    for pid in patient_time_features:
        if all(tw in patient_time_features[pid] for tw in TIME_WINDOWS):
            valid_patients.append(pid)
    # Find intersection of features across all valid patients and all time windows
    all_feature_sets = []
    for pid in valid_patients:
        for tw in TIME_WINDOWS:
            all_feature_sets.append(patient_time_features[pid][tw])
    if len(all_feature_sets) == 0:
        print(f"WARNING: No valid patients for method {method} | {internal_state}")
        return {}
    common_features = set.intersection(*all_feature_sets)
    # For each patient/time_window, drop any extra columns not in common_features
    for pid in valid_patients:
        all_patient_data[pid] = {}
        for tw in TIME_WINDOWS:
            df = patient_time_dfs[pid][tw]
            # Only keep columns in common_features + target
            feature_cols = [col for col in df.columns[:-1] if col in common_features]
            feature_cols_sorted = sorted(feature_cols)  # sort for consistency
            cols_to_keep = feature_cols_sorted + [df.columns[-1]]
            df_clean = df[cols_to_keep]
            all_patient_data[pid][tw] = df_clean
    # Print concise confirmation
    n_features = len(common_features)
    print(f"Patients included for {method} | {internal_state}: {valid_patients}")
    if method.startswith('OGAUHSE'):
        print(f"all OGAUHSE has {n_features} features. Confirmed!")
    elif method.startswith('OF'):
        print(f"all OF has {n_features} features. Confirmed!")
    else:
        print(f"all {method} has {n_features} features. Confirmed!")
    return all_patient_data

def permutation_test(X, y, model_type, alphas, binary, n_permutations=50, random_state=42):
    """
    Returns a null distribution of R or AUC by shuffling y n_permutations times.
    """
    rng = np.random.RandomState(random_state)
    null_scores = []
    for i in tqdm(range(n_permutations), desc="Permutation test", leave=False):
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
    for i in tqdm(range(n_permutations), desc="Permutation test (LOPO)", leave=False):
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

def per_patient_r_barplots(all_patient_data, method, internal_state, limited=False, binary=False, outdir=None, auc=False):
    # Only consider included patients present in all_patient_data
    filtered_patient_ids = [pid for pid in INCLUDED_PATIENTS if pid in all_patient_data]
    for patient_id in tqdm(filtered_patient_ids, desc=f"Per-patient barplots ({method}{' limited' if limited else ''}{' binary' if binary else ''}) | {internal_state}"):
        patient_data = all_patient_data[patient_id]
        results = []
        pvals = []
        for time_window in tqdm(TIME_WINDOWS, desc=f"Patient {patient_id} time windows", leave=False):
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
            try:
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
            except Exception as e:
                print(f"[WARN] Exception in per-patient LOO barplot for patient {patient_id}, time_window {time_window}: {e}")
                results.append(np.nan)
            # Permutation test for this patient/time_window
            if not np.isnan(score):
                null_scores = permutation_test(X, y, model_type='logistic' if binary else 'lasso', alphas=ALPHAS, binary=binary, n_permutations=N_PERMUTATIONS, random_state=42)
                pval = compute_p_value(score, null_scores, tail='right')
            else:
                pval = np.nan
            pvals.append(pval)
        # Plot
        plt.figure(figsize=(8, 5))
        bars = plt.bar([str(tw) for tw in TIME_WINDOWS], results, color=COLORS[0])
        plt.xlabel("Time Window (min)")
        plt.ylabel("AUC" if binary else "Pearson r")
        # Use patient number for labeling
        patient_number = PATIENT_NUMBER_MAP.get(patient_id, None)
        patient_label = f"Patient #{patient_number}" if patient_number is not None else patient_id
        plt.title(f"{METHOD_DISPLAY_MAP.get(method, method)} - {'Limited ' if limited else ''}{'Binary ' if binary else ''}{patient_label} | {internal_state}")
        plt.ylim(-0.1, 1.0 if binary else 1.0)
        # Add patient number as an axis label (for clarity)
        plt.gca().text(1.02, 0.5, patient_label, transform=plt.gca().transAxes, rotation=270, va='center', ha='left', fontsize=13, color='gray')
        # Annotate p-values above bars, always above the top of the bar (even if bar is negative)
        for i, (val, pval) in enumerate(zip(results, pvals)):
            if not np.isnan(val) and not np.isnan(pval):
                bar = bars[i]
                height = bar.get_height()
                # If bar is positive, put text above the top; if negative, put text above the top (i.e., above zero)
                offset = 0.03 * (plt.ylim()[1] - plt.ylim()[0])
                if height >= 0:
                    y = height + offset
                    va = 'bottom'
                else:
                    y = height - offset
                    va = 'top'
                # Always place above the bar, not at the end
                plt.text(bar.get_x() + bar.get_width()/2, y, f"p={pval:.3f}", ha='center', va=va, fontsize=9, rotation=90)
        plt.tight_layout()
        suffix = ""
        if limited: suffix += "_limited"
        if binary: suffix += "_binary"
        if outdir is None:
            outdir = RESULTS_OUTPUT_PATH
        os.makedirs(outdir, exist_ok=True)
        # Save with patient number in filename for clarity
        plt.savefig(os.path.join(outdir, f"{method}{suffix}_patient_{patient_label.replace(' ', '_')}_barplot.png"), dpi=300)
        plt.close()
        # Save results to CSV (keep patient_id for traceability)
        pd.DataFrame({"time_window": TIME_WINDOWS, "score": results, "pval": pvals, "patient_number": patient_number, "patient_id": patient_id}).to_csv(
            os.path.join(outdir, f"{method}{suffix}_patient_{patient_label.replace(' ', '_')}_scores.csv"), index=False
        )

def group_level_barplot(all_patient_data, method, internal_state, limited=False, binary=False, outdir=None):
    # Only consider included patients present in all_patient_data
    filtered_patient_ids = [pid for pid in INCLUDED_PATIENTS if pid in all_patient_data]
    all_scores = []
    all_pvals = []
    patient_numbers = []
    for patient_id in tqdm(filtered_patient_ids, desc=f"Group-level ({method}{' limited' if limited else ''}{' binary' if binary else ''}) | {internal_state}"):
        patient_data = all_patient_data[patient_id]
        patient_scores = []
        patient_pvals = []
        for time_window in tqdm(TIME_WINDOWS, desc=f"Patient {patient_id} time windows", leave=False):
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
            # Only standardize X, not y
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            loo = LeaveOneOut()
            preds, actuals = [], []
            if binary:
                model = LogisticRegressionCV(Cs=1/np.array(ALPHAS), cv=loo, penalty='l1', solver='liblinear', random_state=42)
            else:
                model = LassoCV(alphas=ALPHAS, cv=loo, random_state=42)
            try:
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
            except Exception as e:
                print(f"[WARN] Exception in LOO-CV for patient {patient_id}, time_window {time_window}: {e}")
                patient_scores.append(np.nan)
            # Permutation test for this patient/time_window
            if not np.isnan(score):
                null_scores = permutation_test(X, y, model_type='logistic' if binary else 'lasso', alphas=ALPHAS, binary=binary, n_permutations=N_PERMUTATIONS, random_state=42)
                pval = compute_p_value(score, null_scores, tail='right')
            else:
                pval = np.nan
            patient_pvals.append(pval)
        all_scores.append(patient_scores)
        all_pvals.append(patient_pvals)
        patient_numbers.append(PATIENT_NUMBER_MAP.get(patient_id, None))
    all_scores = np.array(all_scores)
    all_pvals = np.array(all_pvals)
    mean_scores = np.nanmean(all_scores, axis=0)
    sem_scores = np.nanstd(all_scores, axis=0) / np.sqrt(np.sum(~np.isnan(all_scores), axis=0))
    # For group-level p-value, use mean of patient p-values (or combine via Fisher's method if desired)
    group_pvals = np.nanmean(all_pvals, axis=0)
    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(tw) for tw in TIME_WINDOWS], mean_scores, yerr=sem_scores, color=COLORS[0], capsize=5)
    plt.xlabel("Time Window (min)")
    plt.ylabel("AUC" if binary else "Pearson r")
    plt.title(f"{METHOD_DISPLAY_MAP.get(method, method)} - {'Limited ' if limited else ''}{'Binary ' if binary else ''}Group Level | {internal_state}")
    plt.ylim(-0.1, 1.0 if binary else 1.0)
    # Annotate group p-values above bars, always above the top of the bar (even if bar is negative)
    for i, (val, pval) in enumerate(zip(mean_scores, group_pvals)):
        if not np.isnan(val) and not np.isnan(pval):
            bar = bars[i]
            height = bar.get_height()
            # If bar is positive, put text above the top; if negative, put text above the top (i.e., above zero)
            offset = 0.03 * (plt.ylim()[1] - plt.ylim()[0])
            if height >= 0:
                y = height + offset
                va = 'bottom'
            else:
                y = height - offset
                va = 'top'
            # Always place above the bar, not at the end
            plt.text(bar.get_x() + bar.get_width()/2, y, f"p={pval:.3f}", ha='center', va=va, fontsize=9, rotation=90)
    plt.tight_layout()
    suffix = ""
    if limited: suffix += "_limited"
    if binary: suffix += "_binary"
    if outdir is None:
        outdir = RESULTS_OUTPUT_PATH
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{method}{suffix}_group_barplot.png"), dpi=300)
    plt.close()
    # Save group results to CSV, include patient numbers for traceability
    pd.DataFrame({"time_window": TIME_WINDOWS, "mean_score": mean_scores, "sem_score": sem_scores, "mean_pval": group_pvals}).to_csv(
        os.path.join(outdir, f"{method}{suffix}_group_scores.csv"), index=False
    )

def leave_one_patient_out_decoding(all_patient_data, method, internal_state, limited=False, binary=False, outdir=None):
    # Only consider included patients present in all_patient_data
    filtered_patient_ids = [pid for pid in INCLUDED_PATIENTS if pid in all_patient_data]
    
    lopo_results = {tw: [] for tw in TIME_WINDOWS}
    lopo_pvals = {tw: [] for tw in TIME_WINDOWS}
    patient_ids = filtered_patient_ids
    
    for test_patient in tqdm(patient_ids, desc=f"LOPO ({method}{' limited' if limited else ''}{' binary' if binary else ''}) | {internal_state}"):
        for time_window in tqdm(TIME_WINDOWS, desc=f"Test patient {test_patient} time windows", leave=False):
            # Gather training data
            X_train, y_train, X_test, y_test = [], [], None, None
            n_train_patients = 0
            for pid in patient_ids:
                if time_window not in all_patient_data[pid]:
                    print(f"  Time window {time_window} missing for patient {pid}")
                    continue
                df = all_patient_data[pid][time_window]
                if limited:
                    df = filter_limited_features(df)
                if binary:
                    df_bin = binarize_mood(df)
                    if df_bin is None:
                        print(f"  Binary conversion failed for patient {pid}, time window {time_window}")
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
                    n_train_patients += 1
            
            if X_test is None or len(X_train) == 0:
                print(f"  No valid data for time window {time_window}")
                lopo_results[time_window].append(np.nan)
                lopo_pvals[time_window].append(np.nan)
                continue
                
            X_train = np.vstack(X_train)
            y_train = np.concatenate(y_train)
            
            # Only standardize X, not y
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if binary:
                model = LogisticRegressionCV(Cs=1/np.array(ALPHAS), cv=5, penalty='l1', solver='liblinear', random_state=42)
            else:
                model = LassoCV(alphas=ALPHAS, cv=5, random_state=42)
            
            try:
                model.fit(X_train, y_train)
                fit_error = False
            except Exception as e:
                print(f"  Model fitting failed: {e}")
                fit_error = True
                
            if fit_error:
                score = np.nan
            else:
                if binary:
                    preds = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                    if len(np.unique(y_test)) > 1:
                        score = roc_auc_score(y_test, preds)
                    else:
                        print(f"  Not enough unique classes in y_test")
                        score = np.nan
                else:
                    preds = model.predict(X_test)
                    if len(y_test) > 1:
                        score, _ = pearsonr(y_test, preds)
                    else:
                        print(f"  Not enough samples in y_test")
                        score = np.nan
                        
            lopo_results[time_window].append(score)
            
            # Permutation test for this LOPO split
            if not np.isnan(score) and not fit_error:
                null_scores = permutation_test_lopo(X_train, y_train, X_test, y_test, model_type='logistic' if binary else 'lasso', alphas=ALPHAS, binary=binary, n_permutations=N_PERMUTATIONS, random_state=42)
                pval = compute_p_value(score, null_scores, tail='right')
            else:
                pval = np.nan
            lopo_pvals[time_window].append(pval)

    # Save LOPO results to CSV and plot
    if outdir is None:
        outdir = RESULTS_OUTPUT_PATH
    os.makedirs(outdir, exist_ok=True)
    # Prepare DataFrame: rows = patients, columns = time windows
    lopo_scores_df = pd.DataFrame(lopo_results, index=patient_ids)
    lopo_pvals_df = pd.DataFrame(lopo_pvals, index=patient_ids)
    lopo_scores_df.index.name = "patient_id"
    lopo_pvals_df.index.name = "patient_id"
    suffix = ""
    if limited: suffix += "_limited"
    if binary: suffix += "_binary"
    lopo_scores_df.to_csv(os.path.join(outdir, f"{method}{suffix}_lopo_scores.csv"))
    lopo_pvals_df.to_csv(os.path.join(outdir, f"{method}{suffix}_lopo_pvals.csv"))

    # Plot mean and SEM across patients for each time window
    mean_scores = lopo_scores_df.mean(axis=0, skipna=True).values
    sem_scores = lopo_scores_df.std(axis=0, skipna=True).values / np.sqrt(lopo_scores_df.notna().sum(axis=0).values)
    mean_pvals = lopo_pvals_df.mean(axis=0, skipna=True).values

    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(tw) for tw in TIME_WINDOWS], mean_scores, yerr=sem_scores, color=COLORS[1], capsize=5)
    plt.xlabel("Time Window (min)")
    plt.ylabel("AUC" if binary else "Pearson r")
    plt.title(f"{METHOD_DISPLAY_MAP.get(method, method)} - {'Limited ' if limited else ''}{'Binary ' if binary else ''}LOPO | {internal_state}")
    plt.ylim(-0.1, 1.0 if binary else 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{method}{suffix}_lopo_barplot.png"), dpi=300)
    plt.close()

def main():
    # --- NEW: Load and print Mood, Anxiety, Depression scores for each patient --- #
    BEHAVIORAL_XLSX_PATH = os.path.expanduser('~/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx')
    patient_scores = load_behavioral_scores(BEHAVIORAL_XLSX_PATH, INCLUDED_PATIENTS)
    #print_behavioral_scores_table(patient_scores)
    # --- END NEW ---

    for internal_state in INTERNAL_STATES:
        RESULTS_OUTPUT_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_June_2025/{internal_state}Prediction'
        os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)

        print(f"\n===== Running analysis for {internal_state} =====\n")
        # --- A, B, D, E, F for OGAUHSE_L_ --- #
        # Standard features
        print(f"Loading OGAUHSE_L_ data (standard features) for {internal_state}...")
        try:
            oga_data = load_patient_data('OGAUHSE_L_', internal_state, limited=False)
        except Exception as e:
            print(f"[ERROR] Failed to load OGAUHSE_L_ standard features for {internal_state}: {e}")
            oga_data = None

        # # A) Per-patient R barplots
        # try:
        #     per_patient_r_barplots(oga_data, 'OGAUHSE_L_', internal_state, limited=False, binary=False, outdir = RESULTS_OUTPUT_PATH)
        # except Exception as e:
        #     print(f"[ERROR] A) Per-patient R barplots failed for {internal_state}: {e}")

        # B) Group-level R barplot
        try:
            group_level_barplot(oga_data, 'OGAUHSE_L_', internal_state, limited=False, binary=False, outdir = RESULTS_OUTPUT_PATH)
        except Exception as e:
            print(f"[ERROR] B) Group-level R barplot failed for {internal_state}: {e}")

        # D) Leave-one-patient-out decoding (R)
        try:
            leave_one_patient_out_decoding(oga_data, 'OGAUHSE_L_', internal_state, limited=False, binary=False, outdir = RESULTS_OUTPUT_PATH)
        except Exception as e:
            print(f"[ERROR] D) Leave-one-patient-out decoding failed for {internal_state}: {e}")

        # # E) Per-patient AUC barplots (binary decoding)
        # try:
        #     per_patient_r_barplots(oga_data, 'OGAUHSE_L_', internal_state, limited=False, binary=True, outdir = RESULTS_OUTPUT_PATH)
        # except Exception as e:
        #     print(f"[ERROR] E) Per-patient AUC barplots (binary decoding) failed for {internal_state}: {e}")

        # E) Group-level AUC barplot
        try:
            group_level_barplot(oga_data, 'OGAUHSE_L_', internal_state, limited=False, binary=True, outdir = RESULTS_OUTPUT_PATH)
        except Exception as e:
            print(f"[ERROR] E) Group-level AUC barplot failed for {internal_state}: {e}")


        # F) Limited features
        print(f"Loading OGAUHSE_L_ data (limited features) for {internal_state}...")
        try:
            oga_data_limited = load_patient_data('OGAUHSE_L_', internal_state, limited=True)
        except Exception as e:
            print(f"[ERROR] Failed to load OGAUHSE_L_ limited features for {internal_state}: {e}")
            oga_data_limited = None

        # try:
        #     per_patient_r_barplots(oga_data_limited, 'OGAUHSE_L_', internal_state, limited=True, binary=False, outdir = RESULTS_OUTPUT_PATH)
        # except Exception as e:
        #     print(f"[ERROR] F1) Per-patient R barplots (limited features) failed for {internal_state}: {e}")

        try:
            group_level_barplot(oga_data_limited, 'OGAUHSE_L_', internal_state, limited=True, binary=False, outdir = RESULTS_OUTPUT_PATH)
        except Exception as e:
            print(f"[ERROR] F2) Group-level R barplot (limited features) failed for {internal_state}: {e}")

        # try:
        #     leave_one_patient_out_decoding(oga_data_limited, 'OGAUHSE_L_', internal_state, limited=True, binary=False, outdir = RESULTS_OUTPUT_PATH)
        # except Exception as e:
        #     print(f"[ERROR] F3) Leave-one-patient-out decoding (limited features) failed for {internal_state}: {e}")

        # try:
        #     per_patient_r_barplots(oga_data_limited, 'OGAUHSE_L_', internal_state, limited=True, binary=True, outdir = RESULTS_OUTPUT_PATH)
        # except Exception as e:
        #     print(f"[ERROR] F4) Per-patient AUC barplots (limited features, binary) failed for {internal_state}: {e}")

        try:
            group_level_barplot(oga_data_limited, 'OGAUHSE_L_', internal_state, limited=True, binary=True, outdir = RESULTS_OUTPUT_PATH)
        except Exception as e:
            print(f"[ERROR] F5) Group-level AUC barplot (limited features, binary) failed for {internal_state}: {e}")

        # --- C and G for OF_L_ --- #
        print(f"Loading OF_L_ data (standard features) for {internal_state}...")
        try:
            of_data = load_patient_data('OF_L_', internal_state, limited=False)
        except Exception as e:
            print(f"[ERROR] Failed to load OF_L_ standard features for {internal_state}: {e}")
            of_data = None
        # G) Per-patient R barplots
        # try:
        #     per_patient_r_barplots(of_data, 'OF_L_', internal_state, limited=False, binary=False, outdir = RESULTS_OUTPUT_PATH)
        # except Exception as e:
        #     print(f"[ERROR] G) Per-patient R barplots failed for {internal_state}: {e}")

        # C) OF plots
        try:
            group_level_barplot(of_data, 'OF_L_', internal_state, limited=False, binary=False, outdir = RESULTS_OUTPUT_PATH)
        except Exception as e:
            print(f"[ERROR] C) Group-level R barplot (OF_L_) failed for {internal_state}: {e}")
         

    print("All paper-ready figures generated for all internal states.")

def main2():
    """Quick debugging function for LOPO and group level analyses"""
    internal_state = 'Mood'
    method = 'OGAUHSE_L_'
    
    print(f"\n===== Loading data for {internal_state} =====\n")
    # Load data
    try:
        oga_data = load_patient_data('OGAUHSE_L_', internal_state, limited=False)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
        
    print("\n===== Running LOPO Analysis =====\n")
    # Run LOPO
    try:
        leave_one_patient_out_decoding(oga_data, method, internal_state, limited=False, binary=False)
    except Exception as e:
        print(f"[ERROR] LOPO failed: {e}")
    
    print("\n===== Running Group Level Analysis =====\n")
    # Run group level for first two time windows
    try:
        # Create a copy of data with only first two time windows
        debug_data = {}
        for pid in oga_data:
            debug_data[pid] = {tw: oga_data[pid][tw] for tw in TIME_WINDOWS[:2]}
        
        # Add debug prints for group level
        print("\nGroup Level Debug:")
        for pid in debug_data:
            print(f"\nPatient {pid}:")
            for tw in TIME_WINDOWS[:2]:
                df = debug_data[pid][tw]
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                print(f"  Time window {tw}:")
                print(f"    X shape: {X.shape}")
                print(f"    y shape: {y.shape}")
                print(f"    y values: {y}")
        
        group_level_barplot(debug_data, method, internal_state, limited=False, binary=False)
    except Exception as e:
        print(f"[ERROR] Group level failed: {e}")

if __name__ == "__main__":
    #main()
    main2()  # Temporarily use main2 for debugging

"""

===== Loading data for Mood =====

Loading data for method OGAUHSE_L_ | Mood: 100%|█| 11/11 [00:00<00:00, 34.42it/s
Patients included for OGAUHSE_L_ | Mood: ['S24_234', 'S23_174', 'S24_227', 'S23_212', 'S24_230', 'S24_231', 'S24_217', 'S24_219', 'S24_226', 'S23_214', 'S23_199']
all OGAUHSE has 56 features. Confirmed!

===== Running LOPO Analysis =====

LOPO (OGAUHSE_L_) | Mood: 100%|█████████████████| 11/11 [00:28<00:00,  2.63s/it]
                                                                                
===== Running Group Level Analysis =====                                        


Group Level Debug:

Patient S24_234:
  Time window 30:
    X shape: (16, 56)
    y shape: (16,)
    y values: [10.   7.5  5.   2.5  7.5 10.   5.   5.   5.   5.   5.   5.   0.   5.
  5.   5. ]
  Time window 60:
    X shape: (16, 56)
    y shape: (16,)
    y values: [10.   7.5  5.   2.5  7.5 10.   5.   5.   5.   5.   5.   5.   0.   5.
  5.   5. ]

Patient S23_174:
  Time window 30:
    X shape: (13, 56)
    y shape: (13,)
    y values: [ 5.         10.          0.          0.          1.66666667  0.
  3.33333333 10.          6.66666667  1.66666667  3.33333333  0.
  0.        ]
  Time window 60:
    X shape: (13, 56)
    y shape: (13,)
    y values: [ 5.         10.          0.          0.          1.66666667  0.
  3.33333333 10.          6.66666667  1.66666667  3.33333333  0.
  0.        ]

Patient S24_227:
  Time window 30:
    X shape: (18, 56)
    y shape: (18,)
    y values: [ 0.    0.    0.    0.    0.   10.   10.   10.   10.    6.25  8.75  8.75
  3.75  6.25  6.25  7.5   8.75  5.  ]
  Time window 60:
    X shape: (18, 56)
    y shape: (18,)
    y values: [ 0.    0.    0.    0.    0.   10.   10.   10.   10.    6.25  8.75  8.75
  3.75  6.25  6.25  7.5   8.75  5.  ]

Patient S23_212:
  Time window 30:
    X shape: (21, 56)
    y shape: (21,)
    y values: [ 7.5  2.5  2.5  5.   5.   5.   5.   5.   5.   0.   5.   7.5  7.5  7.5
  0.   5.  10.   5.   2.5  5.   5. ]
  Time window 60:
    X shape: (21, 56)
    y shape: (21,)
    y values: [ 7.5  2.5  2.5  5.   5.   5.   5.   5.   5.   0.   5.   7.5  7.5  7.5
  0.   5.  10.   5.   2.5  5.   5. ]

Patient S24_230:
  Time window 30:
    X shape: (13, 56)
    y shape: (13,)
    y values: [ 0.         10.          0.          5.71428571  8.57142857  8.57142857
  8.57142857  8.57142857  8.57142857  8.57142857  8.57142857  5.71428571
  8.57142857]
  Time window 60:
    X shape: (13, 56)
    y shape: (13,)
    y values: [ 0.         10.          0.          5.71428571  8.57142857  8.57142857
  8.57142857  8.57142857  8.57142857  8.57142857  8.57142857  5.71428571
  8.57142857]

Patient S24_231:
  Time window 30:
    X shape: (50, 56)
    y shape: (50,)
    y values: [10.   0.  10.   0.   2.5  0.   0.   0.   0.   0.   0.   2.5  0.   5.
  5.   0.   0.   2.5  0.   2.5  2.5  0.   0.   0.   0.   2.5  0.   0.
  0.  10.  10.  10.   7.5  5.  10.   7.5 10.   7.5 10.   5.   7.5  5.
 10.   7.5  5.   7.5  7.5 10.  10.  10. ]
  Time window 60:
    X shape: (50, 56)
    y shape: (50,)
    y values: [10.   0.  10.   0.   2.5  0.   0.   0.   0.   0.   0.   2.5  0.   5.
  5.   0.   0.   2.5  0.   2.5  2.5  0.   0.   0.   0.   2.5  0.   0.
  0.  10.  10.  10.   7.5  5.  10.   7.5 10.   7.5 10.   5.   7.5  5.
 10.   7.5  5.   7.5  7.5 10.  10.  10. ]

Patient S24_217:
  Time window 30:
    X shape: (15, 56)
    y shape: (15,)
    y values: [ 8.88888889  8.88888889  5.55555556  4.44444444  0.          8.88888889
  8.88888889  7.77777778  7.77777778  5.55555556  5.55555556  6.66666667
 10.         10.          8.88888889]
  Time window 60:
    X shape: (15, 56)
    y shape: (15,)
    y values: [ 8.88888889  8.88888889  5.55555556  4.44444444  0.          8.88888889
  8.88888889  7.77777778  7.77777778  5.55555556  5.55555556  6.66666667
 10.         10.          8.88888889]

Patient S24_219:
  Time window 30:
    X shape: (14, 56)
    y shape: (14,)
    y values: [ 0.          0.          0.          0.          0.          3.33333333
  0.          6.66666667  3.33333333  3.33333333  3.33333333  6.66666667
  3.33333333 10.        ]
  Time window 60:
    X shape: (14, 56)
    y shape: (14,)
    y values: [ 0.          0.          0.          0.          0.          3.33333333
  0.          6.66666667  3.33333333  3.33333333  3.33333333  6.66666667
  3.33333333 10.        ]

Patient S24_226:
  Time window 30:
    X shape: (19, 56)
    y shape: (19,)
    y values: [10. 10.  2.  0.  2.  2.  2.  0.  6.  6.  2.  6. 10.  6.  2.  8. 10. 10.
  8.]
  Time window 60:
    X shape: (19, 56)
    y shape: (19,)
    y values: [10. 10.  2.  0.  2.  2.  2.  0.  6.  6.  2.  6. 10.  6.  2.  8. 10. 10.
  8.]

Patient S23_214:
  Time window 30:
    X shape: (12, 56)
    y shape: (12,)
    y values: [ 6.25  7.5   7.5  10.   10.   10.    7.5   5.    0.   10.    7.5  10.  ]
  Time window 60:
    X shape: (12, 56)
    y shape: (12,)
    y values: [ 6.25  7.5   7.5  10.   10.   10.    7.5   5.    0.   10.    7.5  10.  ]

Patient S23_199:
  Time window 30:
    X shape: (18, 56)
    y shape: (18,)
    y values: [ 8.33333333  5.          6.66666667  5.          5.          6.66666667
  6.66666667  5.          3.33333333  3.33333333 10.          3.33333333
  3.33333333  5.          0.          0.          0.          0.        ]
  Time window 60:
    X shape: (18, 56)
    y shape: (18,)
    y values: [ 8.33333333  5.          6.66666667  5.          5.          6.66666667
  6.66666667  5.          3.33333333  3.33333333 10.          3.33333333
  3.33333333  5.          0.          0.          0.          0.        ]

"""