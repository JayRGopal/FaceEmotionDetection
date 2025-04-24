# Leave One Patient Out Cross-Validation and Statistical Testing
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIGURATION ---------------- #
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
BASE_RESULTS_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_Apr_2025/'
RESULTS_OUTPUT_PATH = os.path.join(BASE_RESULTS_PATH, 'AcrossPatients')
ALPHAS = np.linspace(0.1, 10.0, 30)
TIME_WINDOWS = list(range(15, 241, 30))
INTERNAL_STATES = ['Mood']
RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']
NUM_NULL_PERMUTATIONS = 100  # Number of permutations for null distribution

# Create output directories
os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)

# Create a mapping to decode feature prefixes to more readable names for plots
PREFIX_DISPLAY_MAP = {
    'OF_L_': 'OpenFace',
    'OGAU_L_': 'FaceDx AU',
    'OGAUHSE_L_': 'FaceDx Complete',
    'HSE_L_': 'FaceDx Emo'
}

# Setup plot style for professional presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Define professional color palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Function to parse filename to extract metadata
def parse_filename(filename):
    internal_state = filename.split('_features_')[0]
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    prefix_match = re.search(r'minutes_(.*)\.csv', filename)
    time_window = int(time_match.group(1)) if time_match else None
    prefix = prefix_match.group(1) if prefix_match else None
    return internal_state, time_window, prefix

# Get all patient folders
patient_folders = [folder for folder in os.listdir(FEATURE_SAVE_FOLDER) 
                  if os.path.isdir(os.path.join(FEATURE_SAVE_FOLDER, folder)) and folder.startswith('S')]
print(f"Found {len(patient_folders)} patient folders: {patient_folders}")

# Initialize data structures for results
all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# For each combination of internal state, feature prefix, and time window
for internal_state in INTERNAL_STATES:
    print(f"\nProcessing internal state: {internal_state}")
    
    # Set up folders for this internal state
    state_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state)
    os.makedirs(state_folder, exist_ok=True)
    
    for prefix in RESULTS_PREFIX_LIST:
        print(f"  Processing feature set: {PREFIX_DISPLAY_MAP.get(prefix, prefix)}")
        
        prefix_folder = os.path.join(state_folder, prefix)
        os.makedirs(prefix_folder, exist_ok=True)
        
        for time_window in TIME_WINDOWS:
            print(f"    Processing time window: {time_window} minutes")
            
            # First, collect data from all patients for this combination
            all_patient_data = []
            patient_ids = []
            
            for patient_id in patient_folders:
                patient_folder = os.path.join(FEATURE_SAVE_FOLDER, patient_id)
                csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]
                
                # Find the matching CSV file for this patient
                matching_file = None
                for file in csv_files:
                    file_state, file_time, file_prefix = parse_filename(file)
                    if file_state == internal_state and file_time == time_window and file_prefix == prefix:
                        matching_file = file
                        break
                
                if matching_file:
                    # Load data for this patient
                    df = pd.read_csv(os.path.join(patient_folder, matching_file))
                    # Store data and patient ID
                    all_patient_data.append(df)
                    patient_ids.append(patient_id)
            
            if len(all_patient_data) < 2:
                print(f"      Skipping {internal_state}/{prefix}/{time_window} - not enough patients")
                continue
                
            print(f"      Found data for {len(all_patient_data)} patients")
            
            # Verify column consistency across patients
            feature_cols = all_patient_data[0].columns[:-1]  # All but the last column (target)
            
            # Now perform leave-one-patient-out cross-validation
            results = []
            predictions = []
            actuals = []
            patient_labels = []
            
            for i, test_patient_idx in enumerate(range(len(all_patient_data))):
                test_df = all_patient_data[test_patient_idx]
                test_patient = patient_ids[test_patient_idx]
                
                # Combine all other patients' data for training
                train_dfs = [all_patient_data[j] for j in range(len(all_patient_data)) if j != test_patient_idx]
                train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
                
                # Extract features and targets
                X_train = train_df[feature_cols].values
                y_train = train_df.iloc[:, -1].values
                X_test = test_df[feature_cols].values
                y_test = test_df.iloc[:, -1].values
                
                # Train a model using LassoCV
                model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut())
                if np.isnan(X_train).any(): 
                    print(f"WARNING: NaNs found in X_train for patient {test_patient}, time window {time_window}, method {prefix}. NaN columns: {feature_cols[np.isnan(X_train).any(axis=0)]}")
                model.fit(X_train, y_train)
                
                # Predict on test patient
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if len(y_test) > 1:
                    pearson_r, pearson_p = pearsonr(y_test, y_pred)
                    spearman_r, spearman_p = spearmanr(y_test, y_pred)
                else:
                    # Can't compute correlation with just one sample
                    pearson_r, pearson_p = np.nan, np.nan
                    spearman_r, spearman_p = np.nan, np.nan
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Store results
                results.append({
                    'patient': test_patient,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                })
                
                # Store predictions and actuals for later plotting
                predictions.extend(y_pred)
                actuals.extend(y_test)
                patient_labels.extend([test_patient] * len(y_test))
                
                # Run null distribution test (shuffle test patient's targets)
                null_distribution = {
                    'pearson_r': [],
                    'spearman_r': [],
                    'r2': [],
                    'rmse': [],
                    'mae': []
                }
                
                for perm in range(NUM_NULL_PERMUTATIONS):
                    y_shuffled = np.random.permutation(y_test)
                    
                    # Calculate null metrics
                    if len(y_test) > 1:
                        null_pearson_r, _ = pearsonr(y_shuffled, y_pred)
                        null_spearman_r, _ = spearmanr(y_shuffled, y_pred)
                    else:
                        null_pearson_r = np.nan
                        null_spearman_r = np.nan
                    
                    null_r2 = r2_score(y_shuffled, y_pred)
                    null_rmse = np.sqrt(mean_squared_error(y_shuffled, y_pred))
                    null_mae = mean_absolute_error(y_shuffled, y_pred)
                    
                    null_distribution['pearson_r'].append(null_pearson_r)
                    null_distribution['spearman_r'].append(null_spearman_r)
                    null_distribution['r2'].append(null_r2)
                    null_distribution['rmse'].append(null_rmse)
                    null_distribution['mae'].append(null_mae)
                
                # Calculate p-values from null distribution
                for metric in ['pearson_r', 'spearman_r', 'r2']:
                    actual_value = results[-1][metric]
                    if not np.isnan(actual_value):
                        # For metrics where higher is better
                        null_values = np.array(null_distribution[metric])
                        null_values = null_values[~np.isnan(null_values)]  # Remove NaNs
                        if len(null_values) > 0:
                            perm_p_value = (np.sum(null_values >= actual_value) + 1) / (len(null_values) + 1)
                            results[-1][f'{metric}_perm_p'] = perm_p_value
                
                for metric in ['rmse', 'mae']:
                    actual_value = results[-1][metric]
                    if not np.isnan(actual_value):
                        # For metrics where lower is better
                        null_values = np.array(null_distribution[metric])
                        null_values = null_values[~np.isnan(null_values)]  # Remove NaNs
                        if len(null_values) > 0:
                            perm_p_value = (np.sum(null_values <= actual_value) + 1) / (len(null_values) + 1)
                            results[-1][f'{metric}_perm_p'] = perm_p_value
                
                # Save null distribution plot for this patient
                if len(y_test) > 1:
                    plt.figure(figsize=(10, 6))
                    sns.kdeplot(null_distribution['pearson_r'], fill=True, color=COLORS[3], alpha=0.5, label="Null Distribution")
                    plt.axvline(pearson_r, color='red', linestyle='--', linewidth=2, label=f"Actual r={pearson_r:.3f}")
                    plt.title(f"Statistical Significance - Patient {test_patient}\n{internal_state} | {PREFIX_DISPLAY_MAP.get(prefix, prefix)} | {time_window} min", fontsize=16)
                    plt.xlabel("Pearson Correlation (r)", fontsize=14)
                    plt.ylabel("Density", fontsize=14)
                    plt.legend(fontsize=12)
                    
                    # Add p-value annotation
                    perm_p_value = results[-1].get('pearson_r_perm_p', np.nan)
                    if not np.isnan(perm_p_value):
                        p_value_text = f"p-value: {perm_p_value:.3f}"
                        significance_text = "Significant (p<0.05)" if perm_p_value < 0.05 else "Not significant (p≥0.05)"
                        plt.annotate(f"{p_value_text}\n{significance_text}", xy=(0.95, 0.95), xycoords='axes fraction',
                                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                                    ha='right', va='top', fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(prefix_folder, f"patient_{test_patient}_null_dist_time_{time_window}.png"), dpi=300)
                    plt.close()
            
            # Create combined results dataframe
            results_df = pd.DataFrame(results)
            
            # Calculate average performance across patients
            avg_results = {
                'mean_pearson_r': results_df['pearson_r'].mean(),
                'mean_spearman_r': results_df['spearman_r'].mean(),
                'mean_r2': results_df['r2'].mean(),
                'mean_rmse': results_df['rmse'].mean(),
                'mean_mae': results_df['mae'].mean(),
                'std_pearson_r': results_df['pearson_r'].std(),
                'std_spearman_r': results_df['spearman_r'].std(),
                'std_r2': results_df['r2'].std(),
                'std_rmse': results_df['rmse'].std(),
                'std_mae': results_df['mae'].std()
            }
            
            # Store results for this combination
            all_results[internal_state][prefix][time_window] = {
                'patient_results': results,
                'avg_results': avg_results,
                'predictions': predictions,
                'actuals': actuals,
                'patient_labels': patient_labels
            }
            
            # Save patient-level results to CSV
            results_df.to_csv(os.path.join(prefix_folder, f"patient_results_time_{time_window}.csv"), index=False)
            
            # Save aggregate results to CSV
            pd.DataFrame([avg_results]).to_csv(os.path.join(prefix_folder, f"avg_results_time_{time_window}.csv"), index=False)
            
            # Create scatter plot of all predictions vs actuals
            plt.figure(figsize=(10, 8))
            
            # Plot data points
            for i, patient in enumerate(set(patient_labels)):
                patient_mask = [p == patient for p in patient_labels]
                plt.scatter(
                    [actuals[j] for j, m in enumerate(patient_mask) if m],
                    [predictions[j] for j, m in enumerate(patient_mask) if m],
                    alpha=0.7, s=60, label=f"Patient {patient}"
                )
            
            # Add regression line for all data
            z = np.polyfit(actuals, predictions, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(actuals), max(actuals), 100)
            plt.plot(x_range, p(x_range), linestyle='--', color='black', linewidth=2)
            
            # Add identity line (perfect prediction)
            min_val, max_val = min(min(actuals), min(predictions)), max(max(actuals), max(predictions))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, linewidth=1)
            
            # Overall correlation
            overall_r, overall_p = pearsonr(actuals, predictions)
            
            plt.title(f"Predicted vs. Actual {internal_state} - Leave-One-Patient-Out\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)} | {time_window} min", fontsize=16)
            plt.xlabel(f"Actual {internal_state}", fontsize=14)
            plt.ylabel(f"Predicted {internal_state}", fontsize=14)
            
            # Add metrics info on plot
            metrics_text = (f"Overall Pearson r: {overall_r:.3f} (p={overall_p:.3f})\n"
                           f"Mean Pearson r: {avg_results['mean_pearson_r']:.3f} ± {avg_results['std_pearson_r']:.3f}\n"
                           f"Mean RMSE: {avg_results['mean_rmse']:.3f} ± {avg_results['std_rmse']:.3f}")
            plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                         ha='left', va='top', fontsize=12)
            
            plt.legend(fontsize=10, loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(prefix_folder, f"all_patients_predictions_time_{time_window}.png"), dpi=300)
            plt.close()
            
            # Create bar chart of per-patient performance
            plt.figure(figsize=(12, 8))
            patients = results_df['patient'].values
            pearson_values = results_df['pearson_r'].values
            
            # Sort patients by performance
            sorted_indices = np.argsort(pearson_values)
            sorted_patients = [patients[i] for i in sorted_indices]
            sorted_pearson = [pearson_values[i] for i in sorted_indices]
            
            # Create color coding based on significance
            bar_colors = []
            for idx in sorted_indices:
                if 'pearson_r_perm_p' in results_df.columns and results_df['pearson_r_perm_p'].iloc[idx] < 0.05:
                    bar_colors.append(COLORS[0])  # Significant
                else:
                    bar_colors.append(COLORS[1])  # Not significant
            
            # Create bar chart
            bars = plt.bar(sorted_patients, sorted_pearson, color=bar_colors)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=avg_results['mean_pearson_r'], color='black', linestyle='-', alpha=0.8,
                       label=f"Mean r = {avg_results['mean_pearson_r']:.3f}")
            
            plt.title(f"Per-Patient Prediction Performance - {internal_state}\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)} | {time_window} min", fontsize=16)
            plt.xlabel("Patient", fontsize=14)
            plt.ylabel("Pearson Correlation (r)", fontsize=14)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(prefix_folder, f"per_patient_performance_time_{time_window}.png"), dpi=300)
            plt.close()

# Create summary plots across time windows for each internal state and feature prefix
print("\nCreating summary plots across time windows...")

for internal_state in INTERNAL_STATES:
    state_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state)
    overview_folder = os.path.join(state_folder, 'Overview')
    os.makedirs(overview_folder, exist_ok=True)
    
    # Prepare data for time window comparison plot
    time_window_data = defaultdict(list)
    
    for prefix in RESULTS_PREFIX_LIST:
        time_windows = sorted([tw for tw in all_results[internal_state][prefix].keys()])
        
        if not time_windows:
            continue
            
        pearson_values = []
        pearson_std = []
        
        for tw in time_windows:
            avg_results = all_results[internal_state][prefix][tw]['avg_results']
            pearson_values.append(avg_results['mean_pearson_r'])
            pearson_std.append(avg_results['std_pearson_r'])
        
        time_window_data['prefix'].append(prefix)
        time_window_data['display_name'].append(PREFIX_DISPLAY_MAP.get(prefix, prefix))
        time_window_data['time_windows'].append(time_windows)
        time_window_data['pearson_values'].append(pearson_values)
        time_window_data['pearson_std'].append(pearson_std)
    
    # Create plot comparing performance across time windows
    plt.figure(figsize=(12, 8))
    
    for i in range(len(time_window_data['prefix'])):
        prefix = time_window_data['prefix'][i]
        display_name = time_window_data['display_name'][i]
        time_windows = time_window_data['time_windows'][i]
        pearson_values = time_window_data['pearson_values'][i]
        pearson_std = time_window_data['pearson_std'][i]
        
        plt.errorbar(time_windows, pearson_values, yerr=pearson_std, 
                    fmt='o-', color=COLORS[i], label=display_name, capsize=5)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Window (minutes)', fontsize=14)
    plt.ylabel('Mean Pearson Correlation (r)', fontsize=14)
    plt.title(f"{internal_state} - Leave-One-Patient-Out Performance Across Time Windows", fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(overview_folder, f"{internal_state}_cross_patient_performance.png"), dpi=300)
    plt.close()
    
    # Create summary dataframe and save to CSV
    summary_rows = []
    
    for prefix in RESULTS_PREFIX_LIST:
        for time_window in sorted(all_results[internal_state][prefix].keys()):
            avg_results = all_results[internal_state][prefix][time_window]['avg_results']
            
            # Count significant patients
            sig_patients = 0
            total_patients = len(all_results[internal_state][prefix][time_window]['patient_results'])
            
            for patient_result in all_results[internal_state][prefix][time_window]['patient_results']:
                if 'pearson_r_perm_p' in patient_result and patient_result['pearson_r_perm_p'] < 0.05:
                    sig_patients += 1
            
            summary_rows.append({
                'Internal_State': internal_state,
                'Feature_Set': PREFIX_DISPLAY_MAP.get(prefix, prefix),
                'Time_Window': time_window,
                'Mean_Pearson_R': avg_results['mean_pearson_r'],
                'Std_Pearson_R': avg_results['std_pearson_r'],
                'Mean_R2': avg_results['mean_r2'],
                'Std_R2': avg_results['std_r2'],
                'Mean_RMSE': avg_results['mean_rmse'],
                'Std_RMSE': avg_results['std_rmse'],
                'Total_Patients': total_patients,
                'Significant_Patients': sig_patients,
                'Percent_Significant': (sig_patients / total_patients) * 100 if total_patients > 0 else 0
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(state_folder, f"{internal_state}_summary.csv"), index=False)

print("Leave-one-patient-out analysis complete!")