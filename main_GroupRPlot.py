import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr, ttest_1samp
from scipy.stats import spearmanr

# Paths and configurations
RUNTIME_VAR_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Runtime_Vars/'
RESULTS_PATH_BASE = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results/Group/'
PREFIX_1 = "OGAU_L_"
PREFIX_1_PAIN = "OGAUHSE_L_"
PREFIX_2 = "OF_L_"

LABEL_1 = "FaceDx"
LABEL_2 = "OpenFace"

METRICS = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']
SHOW_PREFIX_2 = True

pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')
os.makedirs(RESULTS_PATH_BASE, exist_ok=True)

# Load variable
def load_var(variable_name, RUNTIME_VAR_PATH):
    with open(os.path.join(RUNTIME_VAR_PATH, f'{variable_name}.pkl'), 'rb') as file:
        return pickle.load(file)

# Detect all patients
def detect_patients():
    files = os.listdir(RUNTIME_VAR_PATH)
    patient_names = set()
    for file in files:
        if file.startswith('predictions_S_'):
            parts = file.split('_')
            patient_name = '_'.join(parts[1:3])
            patient_names.add(patient_name)
    return list(patient_names)

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

def permutation_test_r2(y_true, y_pred, num_permutations=100):
    """
    Simple permutation test using R^2:
      1. Compute real_r2 = Pearson’s R^2 on (y_true, y_pred).
      2. Shuffle y_true multiple times. For each shuffle, compute r^2.
      3. p-value = fraction of shuffles that have r^2 >= real_r2.
    """

    # Compute actual R^2
    real_r, _ = pearsonr(y_true, y_pred)
    real_r2 = real_r ** 2

    count = 0
    for _ in range(num_permutations):
        y_shuffled = np.random.permutation(y_true)
        shuffle_r, _ = pearsonr(y_shuffled, y_pred)
        if (shuffle_r ** 2) >= real_r2:
            count += 1

    p_value = count / num_permutations
    return real_r2, p_value

# Preprocess mood tracking
def preprocess_mood_tracking(PAT_SHORT_NAME):
    MOOD_TRACKING_SHEET_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx'
    df = pd.read_excel(MOOD_TRACKING_SHEET_PATH, sheet_name=f'{PAT_SHORT_NAME}')
    df = df.replace('', np.nan).replace(' ', np.nan).fillna(value=np.nan)
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.strftime('%-m/%-d/%Y %H:%M:%S')
    df = df.drop(columns=['Notes'], errors='ignore')
    return df

# Function to check if a patient meets the inclusion criteria
def meets_inclusion_criteria(df, metric):
    if metric not in df.columns:
        return False

    series_clean = df[metric].dropna()

    # 1) Number of self-reports >= 5
    if len(series_clean) < 5:
        return False

    # 2) Median score > 0
    if series_clean.median() <= 0:
        return False

    # 3) Score range >= 5 (for 0–10 scale)
    if (series_clean.max() - series_clean.min()) < 5:
        return False

    # 4) Number of unique values >= 3
    if len(series_clean.unique()) < 3:
        return False

    return True




# Detect patients
patients = detect_patients()

for metric in METRICS:
    r2_values_prefix_1 = []
    r2_values_prefix_2 = []
    r2_values_prefix_1_included = []
    r2_values_prefix_2_included = []
    p_values_prefix_1 = []
    p_values_prefix_2 = []
    p_values_prefix_1_included = []
    p_values_prefix_2_included = []
    spearman_values_prefix_1_included = []
    spearman_values_prefix_2_included = []
    ccc_values_prefix_1_included = []
    ccc_values_prefix_2_included = []

    variance_list = []
    sample_size_list = []
    random_distributions = []

    print(f"\nResults for {metric.capitalize()}:")

    for patient in patients:
        try:
            # Preprocess mood tracking for the patient
            df_moodTracking = preprocess_mood_tracking(patient)

            # Load predictions
            predictions_prefix_1 = load_var(f'predictions_{patient}_{PREFIX_1_PAIN if metric == "Pain" else PREFIX_1}', RUNTIME_VAR_PATH)[metric]
            predictions_prefix_2 = load_var(f'predictions_{patient}_{PREFIX_2}', RUNTIME_VAR_PATH)[metric]

            y_true_1 = predictions_prefix_1['y_true']
            preds_1 = predictions_prefix_1['preds'][predictions_prefix_1['best_time_radius']]

            # Calculate R^2 + Perm test - prefix 1
            real_r2_1, p_value_1 = permutation_test_r2(y_true_1, preds_1, num_permutations=100)
            if np.isnan(real_r2_1):
                print(f"{patient} excluded due to NaN values.")
                continue

            print(f"[{patient} -- {metric} -- {PREFIX_1_PAIN if metric == 'Pain' else PREFIX_1}] Permutation Test R^2 = {real_r2_1:.3f}, p = {p_value_1:.3f}")

            y_true_2 = predictions_prefix_2['y_true']
            preds_2 = predictions_prefix_2['preds'][predictions_prefix_2['best_time_radius']]

            # Calculate R^2 + Perm test - prefix 2
            real_r2_2, p_value_2 = permutation_test_r2(y_true_2, preds_2, num_permutations=100)
            if np.isnan(real_r2_2):
                print(f"{patient} excluded due to NaN values.")
                continue
            print(f"[{patient} -- {metric} -- {PREFIX_2}] Permutation Test R^2 = {real_r2_2:.3f}, p = {p_value_2:.3f}")


            if meets_inclusion_criteria(df_moodTracking, metric):
                r2_values_prefix_1_included.append(real_r2_1)
                r2_values_prefix_2_included.append(real_r2_2)
                p_values_prefix_1_included.append(p_value_1)
                p_values_prefix_2_included.append(p_value_2)

                # Spearman Correlation
                spearman_1, _ = spearmanr(y_true_1, preds_1)
                spearman_2, _ = spearmanr(y_true_2, preds_2)
                spearman_values_prefix_1_included.append(spearman_1)
                spearman_values_prefix_2_included.append(spearman_2)

                # CCC
                ccc_1 = concordance_correlation_coefficient(y_true_1, preds_1)
                ccc_2 = concordance_correlation_coefficient(y_true_2, preds_2)
                ccc_values_prefix_1_included.append(ccc_1)
                ccc_values_prefix_2_included.append(ccc_2)

            # Variance and sample size
            variance = np.var(df_moodTracking[metric].dropna())
            sample_size = len(df_moodTracking[metric].dropna())

            variance_list.append(variance)
            sample_size_list.append(sample_size)

            r2_values_prefix_1.append(real_r2_1)
            p_values_prefix_1.append(p_value_1)

            r2_values_prefix_2.append(real_r2_2)
            p_values_prefix_2.append(p_value_2)


        except Exception as e:
            print(f"Error processing {patient}: {e}")
            continue

    # Scatterplot: x = sample size, y = variance, color = R^2
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(sample_size_list, variance_list, c=r2_values_prefix_1, cmap='viridis', s=100)
    plt.colorbar(scatter, label='$R^2$')
    plt.xlabel('Sample Size')
    plt.ylabel('Variance')
    plt.title(f'{metric.capitalize()} - Variance vs. Sample Size')
    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_scatter_variance_sampleSize.png'), bbox_inches='tight')
    plt.close()



    # Create and save the group R^2 boxplot
    data = [r2_values_prefix_1]
    labels = [LABEL_1]

    if SHOW_PREFIX_2:
        data.append(r2_values_prefix_2)
        labels.append(LABEL_2)

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, vert=False, labels=labels, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})

    # 1) Overlay points for prefix_1
    group_1_y = 1
    jitter_amount = 0.05
    for r2_val, p_val in zip(r2_values_prefix_1, p_values_prefix_1):
        color = 'red' if p_val < 0.05 else 'black'
        # Slight random vertical jitter so points don’t overlap exactly
        y_jittered = group_1_y + np.random.uniform(-jitter_amount, jitter_amount)
        plt.scatter(r2_val, y_jittered, color=color, s=60, alpha=0.7)

    # 2) Overlay points for prefix_2 (only if SHOW_PREFIX_2 is True)
    if SHOW_PREFIX_2:
        group_2_y = 2
        for r2_val, p_val in zip(r2_values_prefix_2, p_values_prefix_2):
            color = 'red' if p_val < 0.05 else 'black'
            y_jittered = group_2_y + np.random.uniform(-jitter_amount, jitter_amount)
            plt.scatter(r2_val, y_jittered, color=color, s=60, alpha=0.7)

    plt.title(f'Group $R^2$ for {metric.capitalize()}, N = {len(r2_values_prefix_1)}', fontsize=24)
    plt.xlabel("$R^2$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR2_ALL.png'), bbox_inches='tight')
    plt.close()




    # Create and save the violin plot
    plt.figure(figsize=(10, 6))
    plt.violinplot(data, vert=False, showmeans=True, showmedians=True)
    
    # Overlay points for prefix_1 at y=1
    group_1_y = 1
    for r2_val, p_val in zip(r2_values_prefix_1, p_values_prefix_1):
        color = 'red' if p_val < 0.05 else 'black'
        plt.scatter(r2_val, group_1_y + np.random.uniform(-0.05, 0.05),
                    color=color, s=60, alpha=0.7)

    # Overlay points for prefix_2 at y=2
    if SHOW_PREFIX_2:
        group_2_y = 2
        for r2_val, p_val in zip(r2_values_prefix_2, p_values_prefix_2):
            color = 'red' if p_val < 0.05 else 'black'
            plt.scatter(r2_val, group_2_y + np.random.uniform(-0.05, 0.05),
                        color=color, s=60, alpha=0.7)

    if SHOW_PREFIX_2:
        plt.yticks([1, 2], labels)
    else:
        plt.yticks([1], labels)
    plt.title(f'{metric.capitalize()} - Violin Plot of $R^2$', fontsize=24)
    plt.xlabel("$R^2$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR2_violin_ALL.png'), bbox_inches='tight')
    plt.close()


    # ONLY MEETING INCLUSION CRITERIA
    # Create and save the group R^2 boxplot
    data = [r2_values_prefix_1_included]
    labels = [LABEL_1]

    if SHOW_PREFIX_2:
        data.append(r2_values_prefix_2_included)
        labels.append(LABEL_2)

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, vert=False, labels=labels, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})
    
    # Overlay each included patient's R^2 with color by p < 0.05
    group_1_y = 1
    jitter_amount = 0.05

    # 1) Group 1 points
    for r2_val, p_val in zip(r2_values_prefix_1_included, p_values_prefix_1_included):
        color = 'red' if p_val < 0.05 else 'black'
        y_jittered = group_1_y + np.random.uniform(-jitter_amount, jitter_amount)
        plt.scatter(r2_val, y_jittered, color=color, s=60, alpha=0.7)

    # 2) Group 2 points (only if SHOW_PREFIX_2 is True)
    if SHOW_PREFIX_2:
        group_2_y = 2
        for r2_val, p_val in zip(r2_values_prefix_2_included, p_values_prefix_2_included):
            color = 'red' if p_val < 0.05 else 'black'
            y_jittered = group_2_y + np.random.uniform(-jitter_amount, jitter_amount)
            plt.scatter(r2_val, y_jittered, color=color, s=60, alpha=0.7)
    
    plt.title(f'Group $R^2$ for {metric.capitalize()}, N = {len(r2_values_prefix_1_included)}', fontsize=24)
    plt.xlabel("$R^2$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR2.png'), bbox_inches='tight')
    plt.close()

    # Create and save the violin plot
    plt.figure(figsize=(10, 6))
    plt.violinplot(data, vert=False, showmeans=True, showmedians=True)
    
    # Overlay each included patient's R^2 with color by p < 0.05
    group_1_y = 1
    jitter_amount = 0.05

    # 1) Group 1 points
    for r2_val, p_val in zip(r2_values_prefix_1_included, p_values_prefix_1_included):
        color = 'red' if p_val < 0.05 else 'black'
        y_jittered = group_1_y + np.random.uniform(-jitter_amount, jitter_amount)
        plt.scatter(r2_val, y_jittered, color=color, s=60, alpha=0.7)

    # 2) Group 2 points (only if SHOW_PREFIX_2 is True)
    if SHOW_PREFIX_2:
        group_2_y = 2
        for r2_val, p_val in zip(r2_values_prefix_2_included, p_values_prefix_2_included):
            color = 'red' if p_val < 0.05 else 'black'
            y_jittered = group_2_y + np.random.uniform(-jitter_amount, jitter_amount)
            plt.scatter(r2_val, y_jittered, color=color, s=60, alpha=0.7)
    
    if SHOW_PREFIX_2:
        plt.yticks([1, 2], labels)
    else:
        plt.yticks([1], labels)
    plt.title(f'{metric.capitalize()} - Violin Plot of $R^2$', fontsize=24)
    plt.xlabel("$R^2$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR2_violin.png'), bbox_inches='tight')
    plt.close()


    # -------------------------------------------------------
    # SPEARMAN’S RHO: BOX PLOT (INCLUDED PATIENTS ONLY)
    # -------------------------------------------------------
    data_spearman = [spearman_values_prefix_1_included]
    labels_spearman = [LABEL_1]

    if SHOW_PREFIX_2:
        data_spearman.append(spearman_values_prefix_2_included)
        labels_spearman.append(LABEL_2)

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_spearman, vert=False, labels=labels_spearman, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})

    plt.title(f'Spearman Correlation for {metric} (Included), N = {len(spearman_values_prefix_1_included)}', fontsize=18)
    plt.xlabel("Spearman's ρ", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_spearman_included_box.png'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # SPEARMAN’S RHO: VIOLIN PLOT (INCLUDED PATIENTS ONLY)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.violinplot(data_spearman, vert=False, showmeans=True, showmedians=True)

    if SHOW_PREFIX_2:
        plt.yticks([1, 2], labels_spearman)
    else:
        plt.yticks([1], labels_spearman)

    plt.title(f'{metric}: Spearman (Included)', fontsize=18)
    plt.xlabel("Spearman's ρ", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_spearman_included_violin.png'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # CCC: BOX PLOT (INCLUDED PATIENTS ONLY)
    # -------------------------------------------------------
    data_ccc = [ccc_values_prefix_1_included]
    labels_ccc = [LABEL_1]

    if SHOW_PREFIX_2:
        data_ccc.append(ccc_values_prefix_2_included)
        labels_ccc.append(LABEL_2)

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_ccc, vert=False, labels=labels_ccc, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})

    plt.title(f'CCC for {metric} (Included), N = {len(ccc_values_prefix_1_included)}', fontsize=18)
    plt.xlabel("Concordance Correlation Coefficient", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_ccc_included_box.png'), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    # CCC: VIOLIN PLOT (INCLUDED PATIENTS ONLY)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.violinplot(data_ccc, vert=False, showmeans=True, showmedians=True)

    if SHOW_PREFIX_2:
        plt.yticks([1, 2], labels_ccc)
    else:
        plt.yticks([1], labels_ccc)

    plt.title(f'{metric}: CCC (Included)', fontsize=18)
    plt.xlabel("Concordance Correlation Coefficient", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_ccc_included_violin.png'), bbox_inches='tight')
    plt.close()








