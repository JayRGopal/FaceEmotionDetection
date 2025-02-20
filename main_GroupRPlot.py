import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr, ttest_1samp

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
    values = df[metric].dropna().unique()
    return len(df[metric].dropna()) >= 5 and len(values) >= 4


# Generate permutation test distribution
def permutation_test(y_true, preds, num_permutations=10000):
    random_r2s = []
    for _ in range(num_permutations):
        y_true_shuffled = np.random.permutation(y_true)
        r, _ = pearsonr(preds, y_true_shuffled)
        random_r2s.append(r**2)
    return random_r2s

# Function to generate random chance performance distribution
def generate_random_chance_distribution(y_true, preds, num_shuffles=10000):
    random_r2s = []
    for _ in range(num_shuffles):
        y_true_shuffled = np.random.permutation(y_true)
        r, _ = pearsonr(preds, y_true_shuffled)
        random_r2s.append(r ** 2)  # Store R^2 instead of R
    return np.percentile(random_r2s, [25, 75])  # Return 25th and 75th percentiles for shading


# Detect patients
patients = detect_patients()

for metric in METRICS:
    r2_values_prefix_1 = []
    r2_values_prefix_2 = []
    r2_values_prefix_1_included = []
    r2_values_prefix_2_included = []
    variance_list = []
    sample_size_list = []
    patient_permutation_distributions_prefix_1 = []
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

            # Calculate R^2
            r2_1 = pearsonr(y_true_1, preds_1)[0] ** 2

            if np.isnan(r2_1):
                print(f"{patient} excluded due to NaN values.")
                continue

            y_true_2 = predictions_prefix_2['y_true']
            preds_2 = predictions_prefix_2['preds'][predictions_prefix_2['best_time_radius']]

            # Calculate R^2
            r2_2 = pearsonr(y_true_2, preds_2)[0] ** 2

            if np.isnan(r2_2):
                print(f"{patient} excluded due to NaN values.")
                continue
            
            if meets_inclusion_criteria(df_moodTracking, metric):
                r2_values_prefix_1_included.append(r2_1)
                r2_values_prefix_2_included.append(r2_2)

            # Variance and sample size
            variance = np.var(df_moodTracking[metric].dropna())
            sample_size = len(df_moodTracking[metric].dropna())

            variance_list.append(variance)
            sample_size_list.append(sample_size)

            r2_values_prefix_1.append(r2_1)

            r2_values_prefix_2.append(r2_2)

            # Permutation test
            perm_distribution_1 = permutation_test(y_true_1, preds_1)
            patient_permutation_distributions_prefix_1.append(perm_distribution_1)


            # Generate random chance performance distribution using the correct method
            random_chance_distribution = generate_random_chance_distribution(predictions_prefix_1['y_true'], predictions_prefix_1['preds'][predictions_prefix_1['best_time_radius']])
            random_distributions.append(random_chance_distribution)


        except Exception as e:
            print(f"Error processing {patient}: {e}")
            continue

    # Permutation test summary
    true_mean_r2_1 = np.mean(r2_values_prefix_1)
    perm_means_1 = [np.mean(perm) for perm in patient_permutation_distributions_prefix_1]
    t_stat_1, p_value_1 = ttest_1samp(perm_means_1, true_mean_r2_1)

    print(f"Permutation Test Mean R^2: {true_mean_r2_1:.3f}, p-value: {p_value_1:.3g}")

    # Scatterplot: x = sample size, y = variance, color = R^2
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(sample_size_list, variance_list, c=r2_values_prefix_1, cmap='viridis', s=100)
    plt.colorbar(scatter, label='$R^2$')
    plt.xlabel('Sample Size')
    plt.ylabel('Variance')
    plt.title(f'{metric.capitalize()} - Variance vs. Sample Size')
    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_scatter_variance_sampleSize.png'), bbox_inches='tight')
    plt.close()

    # Histogram of permutation test results
    plt.figure(figsize=(10, 6))
    for perm_dist in patient_permutation_distributions_prefix_1:
        plt.hist(perm_dist, bins=30, alpha=0.3, label=f"Patient")
    plt.axvline(true_mean_r2_1, color='red', linestyle='dotted', label='True Mean $R^2$')
    plt.legend()
    plt.title(f'Permutation Test Distribution for {metric.capitalize()}')
    plt.xlabel('$R^2$')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_permutation_histogram.png'), bbox_inches='tight')
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
    plt.title(f'Group $R^2$ for {metric.capitalize()}, N = {len(r2_values_prefix_1)}', fontsize=24)
    plt.xlabel("$R^2$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add random chance performance region
    random_chance_25th, random_chance_75th = np.mean(random_distributions, axis=0)
    plt.axvspan(random_chance_25th, random_chance_75th, color='gray', alpha=0.3)
    plt.axvline(random_chance_25th, color='gray', linestyle='dotted')
    plt.axvline(random_chance_75th, color='gray', linestyle='dotted')

    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR2_ALL.png'), bbox_inches='tight')
    plt.close()

    # Create and save the violin plot
    plt.figure(figsize=(10, 6))
    plt.violinplot(data, vert=False, showmeans=True, showmedians=True)
    if SHOW_PREFIX_2:
        plt.yticks([1, 2], labels)
    else:
        plt.yticks([1], labels)
    plt.title(f'{metric.capitalize()} - Violin Plot of $R^2$', fontsize=24)
    plt.xlabel("$R^2$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axvspan(random_chance_25th, random_chance_75th, color='gray', alpha=0.3)
    plt.axvline(random_chance_25th, color='gray', linestyle='dotted')
    plt.axvline(random_chance_75th, color='gray', linestyle='dotted')


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
    plt.title(f'Group $R^2$ for {metric.capitalize()}, N = {len(r2_values_prefix_1_included)}', fontsize=24)
    plt.xlabel("$R^2$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Add random chance performance region
    random_chance_25th, random_chance_75th = np.mean(random_distributions, axis=0)
    plt.axvspan(random_chance_25th, random_chance_75th, color='gray', alpha=0.3)
    plt.axvline(random_chance_25th, color='gray', linestyle='dotted')
    plt.axvline(random_chance_75th, color='gray', linestyle='dotted')

    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR2.png'), bbox_inches='tight')
    plt.close()

    # Create and save the violin plot
    plt.figure(figsize=(10, 6))
    plt.violinplot(data, vert=False, showmeans=True, showmedians=True)
    if SHOW_PREFIX_2:
        plt.yticks([1, 2], labels)
    else:
        plt.yticks([1], labels)
    plt.title(f'{metric.capitalize()} - Violin Plot of $R^2$', fontsize=24)
    plt.xlabel("$R^2$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axvspan(random_chance_25th, random_chance_75th, color='gray', alpha=0.3)
    plt.axvline(random_chance_25th, color='gray', linestyle='dotted')
    plt.axvline(random_chance_75th, color='gray', linestyle='dotted')


    plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR2_violin.png'), bbox_inches='tight')
    plt.close()




