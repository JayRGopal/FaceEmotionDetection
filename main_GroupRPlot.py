# import os
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import warnings
# from scipy.stats import pearsonr

# # Define the paths and prefixes
# RUNTIME_VAR_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Runtime_Vars/'
# RESULTS_PATH_BASE = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results/Group/'
# PREFIX_1 = "OGAU_L_"
# PREFIX_1_PAIN = "OGAUHSE_L_"  # Just for pain, we use AU + HSE
# PREFIX_2 = "OF_L_"

# LABEL_1 = "FaceDx"
# LABEL_2 = "OpenFace"

# METRICS = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']  # Add all metrics you're interested in

# SHOW_PREFIX_2 = True  # Boolean flag to control whether PREFIX_2 should be included in the plots

# # Ignore all warnings
# pd.options.mode.chained_assignment = None
# pd.set_option('mode.chained_assignment', None)
# warnings.filterwarnings('ignore')

# # Ensure the results directory exists
# os.makedirs(RESULTS_PATH_BASE, exist_ok=True)

# # Function to load variables
# def load_var(variable_name, RUNTIME_VAR_PATH):
#     with open(os.path.join(RUNTIME_VAR_PATH, f'{variable_name}.pkl'), 'rb') as file:
#         return pickle.load(file)

# # Function to detect all unique patient short names based on saved files
# def detect_patients():
#     files = os.listdir(RUNTIME_VAR_PATH)
#     patient_names = set()

#     for file in files:
#         if file.startswith('predictions_S_'):
#             # Extract the patient name assuming it starts with 'S_' and is followed by '_'
#             parts = file.split('_')
#             patient_name = '_'.join(parts[1:3])  # This ensures you capture the full patient name after 'predictions_'
#             patient_names.add(patient_name)
    
#     return list(patient_names)

# # Function to check if a patient meets the inclusion criteria
# def meets_inclusion_criteria(df, metric):
#     if metric not in df.columns:
#         return False
#     values = df[metric].dropna().unique()
#     return len(df[metric].dropna()) >= 5 and len(values) >= 4

# # Load and preprocess the mood tracking sheet
# def preprocess_mood_tracking(PAT_SHORT_NAME):
#     MOOD_TRACKING_SHEET_PATH = f'/home/jgopal/NAS/Analysis/AudioFacialEEG/Behavioral Labeling/Mood_Tracking.xlsx'
#     df = pd.read_excel(MOOD_TRACKING_SHEET_PATH, sheet_name=f'{PAT_SHORT_NAME}')

#     # Properly deal with the missing values
#     df = df.replace('', np.nan).replace(' ', np.nan).fillna(value=np.nan)

#     df['Datetime'] = pd.to_datetime(df['Datetime']).dt.strftime('%-m/%-d/%Y %H:%M:%S')

#     # Drop unnecessary columns
#     df = df.drop(columns=['Notes'], errors='ignore')

#     return df

# # Function to generate random chance performance distribution
# def generate_random_chance_distribution(y_true, preds, num_shuffles=100):
#     random_rs = []
#     for _ in range(num_shuffles):
#         y_true_shuffled = np.random.permutation(y_true)
#         r, _ = pearsonr(preds, y_true_shuffled)
#         random_rs.append(r)
#     return np.percentile(random_rs, [25, 75])

# # Calculate variance as the percentage of possible variation covered
# def calculate_variance_percentage(y_true, metric):
#     num_unique_values = len(np.unique(y_true))
#     if metric == 'Pain':
#         max_possible_values = 8  # Pain scale has 8 distinct values
#     else:
#         max_possible_values = 11  # Other metrics have 11 distinct values
#     variance_percentage = (num_unique_values / max_possible_values) * 100
#     return variance_percentage


# # Detect all patients
# patients = detect_patients()


# # Loop through each metric
# for metric in METRICS:
#     r_values_prefix_1 = []
#     r_values_prefix_2 = []
#     variance_list = []
#     sample_size_list = []
#     random_distributions = []

#     # Print header for the metric
#     print(f"\nResults for {metric.capitalize()}:")

#     # Loop through each patient
#     for patient in patients:
#         try:
#             # Preprocess mood tracking for the patient
#             df_moodTracking = preprocess_mood_tracking(patient)

#             # Check if patient meets the inclusion criteria
#             # if not meets_inclusion_criteria(df_moodTracking, metric):
#             #     print(f"{patient} excluded for {metric} due to not meeting inclusion criteria.")
#             #     continue

#             # Load the predictions for both prefixes
#             if metric == 'Pain':
#                 predictions_prefix_1 = load_var(f'predictions_{patient}_{PREFIX_1_PAIN}', RUNTIME_VAR_PATH)[f'{metric}']
#             else:
#                 predictions_prefix_1 = load_var(f'predictions_{patient}_{PREFIX_1}', RUNTIME_VAR_PATH)[f'{metric}']
#             predictions_prefix_2 = load_var(f'predictions_{patient}_{PREFIX_2}', RUNTIME_VAR_PATH)[f'{metric}'] if SHOW_PREFIX_2 else None

#             # Calculate Pearson's R for each
#             r_1 = np.corrcoef(predictions_prefix_1['y_true'], predictions_prefix_1['preds'][predictions_prefix_1['best_time_radius']])[0, 1]
#             r_2 = np.corrcoef(predictions_prefix_2['y_true'], predictions_prefix_2['preds'][predictions_prefix_2['best_time_radius']])[0, 1] if SHOW_PREFIX_2 else None

#             # Exclude patient if either score is NaN
#             if np.isnan(r_1) or (SHOW_PREFIX_2 and np.isnan(r_2)):
#                 print(f"{patient} excluded due to NaN values.")
#                 continue

#             # Calculate variance and sample size
#             variance = np.var(df_moodTracking[metric].dropna())
#             sample_size = len(df_moodTracking[metric].dropna())

#             # Generate random chance performance distribution using the correct method
#             random_chance_distribution = generate_random_chance_distribution(predictions_prefix_1['y_true'], predictions_prefix_1['preds'][predictions_prefix_1['best_time_radius']])
#             random_distributions.append(random_chance_distribution)

#             # Append to the respective lists
#             r_values_prefix_1.append(r_1)
#             variance_list.append(variance)
#             sample_size_list.append(sample_size)
#             if SHOW_PREFIX_2:
#                 r_values_prefix_2.append(r_2)

#             # Print the scores for this patient
#             if SHOW_PREFIX_2:
#                 print(f"{patient}: {LABEL_1} = {r_1:.2f}, {LABEL_2} = {r_2:.2f}")
#             else:
#                 print(f"{patient}: {LABEL_1} = {r_1:.2f}")

#         except Exception as e:
#             print(f"Error loading or processing data for patient {patient}, metric {metric}: {str(e)}")
#             continue

#     # Create the box and whisker plot with random chance performance
#     data = [r_values_prefix_1]
#     labels = [LABEL_1]

#     if SHOW_PREFIX_2:
#         data.append(r_values_prefix_2)
#         labels.append(LABEL_2)

#     plt.figure(figsize=(10, 6))
#     plt.boxplot(data, vert=False, labels=labels, showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10})

#     # Add random chance performance region
#     random_chance_25th, random_chance_75th = np.mean(random_distributions, axis=0)
#     plt.axvspan(random_chance_25th, random_chance_75th, color='gray', alpha=0.3)
#     plt.axvline(random_chance_25th, color='gray', linestyle='dotted')
#     plt.axvline(random_chance_75th, color='gray', linestyle='dotted')

#     # Capitalize the first letter of the metric
#     metric_label = metric.capitalize()

#     # Title with the metric and number of patients (updated to reflect the current count)
#     plt.title(f'Predicting {metric_label}, N = {len(r_values_prefix_1)}', fontsize=24)

#     # Set axis labels
#     plt.xlabel("Pearson's R", fontsize=18)
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)

#     # Save the plot
#     plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupR.png'), bbox_inches='tight')
#     plt.close()

#     # Ensure the lists for variance and sample size correspond to the included patients only
#     variance_list = []
#     sample_size_list = []

#     # Loop through each patient to calculate variance and sample size
#     for i, patient in enumerate(patients):
#         try:
#             predictions_prefix_1 = load_var(f'predictions_{patient}_{PREFIX_1 if metric != "Pain" else PREFIX_1_PAIN}', RUNTIME_VAR_PATH)[f'{metric}']
#             y_true = predictions_prefix_1['y_true']

#             # Calculate variance and sample size only for the included patients
#             variance_list.append(calculate_variance_percentage(y_true, metric))
#             sample_size_list.append(len(y_true))

#         except Exception as e:
#             # If any patient does not match, skip them
#             continue

#     # Create and save the dot plot colored by variance
#     plt.figure(figsize=(10, 6))
#     plt.scatter(r_values_prefix_1, np.full_like(r_values_prefix_1, 1), c=variance_list[:len(r_values_prefix_1)], cmap='viridis', s=100, label=LABEL_1)
#     if SHOW_PREFIX_2:
#         plt.scatter(r_values_prefix_2, np.full_like(r_values_prefix_2, 2), c=variance_list[:len(r_values_prefix_2)], cmap='viridis', s=100, label=LABEL_2)

#     # Add the random chance performance region
#     random_chance_25th, random_chance_75th = np.mean(random_distributions, axis=0)
#     plt.axvspan(random_chance_25th, random_chance_75th, color='gray', alpha=0.3)
#     plt.axvline(random_chance_25th, color='gray', linestyle='dotted')
#     plt.axvline(random_chance_75th, color='gray', linestyle='dotted')

#     plt.colorbar(label='Variance Percentage')
#     plt.title(f'{metric_label} - Correlation Coefficients Colored by Variance')
#     plt.yticks([1, 2], [LABEL_1, LABEL_2])
#     plt.xlabel("Pearson's R")
#     plt.ylabel("Prefix")
#     plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupRdots_variance.png'), bbox_inches='tight')
#     plt.close()

#     # Create and save the dot plot colored by sample size
#     plt.figure(figsize=(10, 6))
#     plt.scatter(r_values_prefix_1, np.full_like(r_values_prefix_1, 1), c=sample_size_list[:len(r_values_prefix_1)], cmap='plasma', s=100, label=LABEL_1)
#     if SHOW_PREFIX_2:
#         plt.scatter(r_values_prefix_2, np.full_like(r_values_prefix_2, 2), c=sample_size_list[:len(r_values_prefix_2)], cmap='plasma', s=100, label=LABEL_2)

#     # Add the random chance performance region
#     random_chance_25th, random_chance_75th = np.mean(random_distributions, axis=0)
#     plt.axvspan(random_chance_25th, random_chance_75th, color='gray', alpha=0.3)
#     plt.axvline(random_chance_25th, color='gray', linestyle='dotted')
#     plt.axvline(random_chance_75th, color='gray', linestyle='dotted')

#     plt.colorbar(label='Sample Size')
#     plt.title(f'{metric_label} - Correlation Coefficients Colored by Sample Size')
#     plt.yticks([1, 2], [LABEL_1, LABEL_2])
#     plt.xlabel("Pearson's R")
#     plt.ylabel("Prefix")
#     plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupRdots_samples.png'), bbox_inches='tight')
#     plt.close()

#     # Create and save the violin plot for patients meeting inclusion criteria
#     plt.figure(figsize=(10, 6))
#     data_inclusion = [r_values_prefix_1]
#     labels_inclusion = [LABEL_1]

#     if SHOW_PREFIX_2:
#         data_inclusion.append(r_values_prefix_2)
#         labels_inclusion.append(LABEL_2)

#     plt.violinplot(data_inclusion, vert=False, showmeans=True, showmedians=True)
#     plt.yticks([1, 2], labels_inclusion)
#     plt.title(f'{metric_label} - Violin Plot of Correlation Coefficients')
#     plt.xlabel("Pearson's R")
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)

#     # Add random chance performance region
#     random_chance_25th, random_chance_75th = np.mean(random_distributions, axis=0)
#     plt.axvspan(random_chance_25th, random_chance_75th, color='gray', alpha=0.3)
#     plt.axvline(random_chance_25th, color='gray', linestyle='dotted')
#     plt.axvline(random_chance_75th, color='gray', linestyle='dotted')

#     plt.savefig(os.path.join(RESULTS_PATH_BASE, f'{metric}_groupRviolin.png'), bbox_inches='tight')
#     plt.close()

# print("Plots generated and saved successfully.")


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




