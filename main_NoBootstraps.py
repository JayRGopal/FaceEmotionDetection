# Run this after main_FeatureExtractionInpatient_JustLinReg_extract_matrix.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
ALPHAS = np.linspace(0.3, 6.0, 20)
TIME_WINDOWS = list(range(15, 241, 15))
# INTERNAL_STATES = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']
INTERNAL_STATES = ['Mood']
RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']
# RESULTS_PREFIX_LIST = ['OGAU_L_']
NUM_TOP_FEATURES = 5  # Number of top features to analyze
NUM_BOTTOM_FEATURES = 5  # Number of bottom features to analyze
NUM_NULL_PERMUTATIONS = 2  # Number of permutations for null distribution
NUM_PERMUTATION_IMPORTANCE = 2 # Num iterations for perm importance

# Create output directories
os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)

# Create a mapping to decode feature prefixes to more readable names for plots
PREFIX_DISPLAY_MAP = {
    'OF_L_': 'OpenFace',
    'OGAU_L_': 'FaceDx AU',
    'OGAUHSE_L_': 'FaceDx Complete',
    'HSE_L_': 'FaceDx Emo'
}

# Function to parse filename to extract metadata
def parse_filename(filename):
    internal_state = filename.split('_features_')[0]
    time_match = re.search(r'time_(\d+)_minutes_', filename)
    prefix_match = re.search(r'minutes_(.*)\.csv', filename)
    time_window = int(time_match.group(1)) if time_match else None
    prefix = prefix_match.group(1) if prefix_match else None
    return internal_state, time_window, prefix

# Initialize data structures to store results
summary_results = defaultdict(lambda: defaultdict(dict))
feature_heatmap_data = defaultdict(lambda: defaultdict(dict))
permutation_impact_data = defaultdict(lambda: defaultdict(dict))
feature_selection_frequency = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
feature_correlation_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
null_distribution_data = defaultdict(lambda: defaultdict(dict))
all_feature_names = set()

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

# Process all CSV files
patient_folder = os.path.join(FEATURE_SAVE_FOLDER, PAT_NOW)
csv_files = [f for f in os.listdir(patient_folder) if f.endswith('.csv')]

for file in tqdm(csv_files, desc="Processing all CSVs"):
    internal_state, time_window, prefix = parse_filename(file)
    if internal_state not in INTERNAL_STATES or time_window not in TIME_WINDOWS or prefix not in RESULTS_PREFIX_LIST:
        continue

    # Create folder structure
    state_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state)
    prefix_folder = os.path.join(state_folder, prefix)
    overview_folder = os.path.join(state_folder, 'Overview')
    csv_folder = os.path.join(state_folder, 'CSV_Results')
    
    for folder in [prefix_folder, overview_folder, csv_folder]:
        os.makedirs(folder, exist_ok=True)

    # Load the data
    df = pd.read_csv(os.path.join(patient_folder, file))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]
    all_feature_names.update(feature_names)

    # Initialize result metrics
    all_metrics = {
        'pearson_r': [],
        'spearman_r': [],
        'r2': [],
        'rmse': [],
        'mae': []
    }
    
    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    alpha_store = []
    coef_matrix = []
    feature_selection_count = np.zeros(X.shape[1])
    for train_idx, test_idx in loo.split(X):
        model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = model.predict(X[test_idx])[0]
        alpha_store.append(model.alpha_)
        coef_matrix.append(model.coef_)
        
        # Track which features were selected (non-zero coefficients)
        non_zero_features = (model.coef_ != 0).astype(int)
        feature_selection_count += non_zero_features
        
        for f_idx, is_selected in enumerate(non_zero_features):
            if is_selected:
                feature_selection_frequency[internal_state][prefix][feature_names[f_idx]] += 1

    coef_matrix = np.array(coef_matrix)

    # Calculate initial metrics
    loo_r, loo_p = pearsonr(y, preds)
    loo_spearman, loo_spearman_p = spearmanr(y, preds)
    loo_r2 = r2_score(y, preds)
    loo_rmse = np.sqrt(mean_squared_error(y, preds))
    loo_mae = mean_absolute_error(y, preds)

    # Store feature correlations with self-reports
    feature_correlations = {}
    for i, feat_name in enumerate(feature_names):
        r_val, p_val = pearsonr(X[:, i], y)
        feature_correlations[feat_name] = {'r': r_val, 'p': p_val}

    # Generate null distribution by shuffling y values
    null_metrics = {
        'pearson_r': [],
        'spearman_r': [],
        'r2': [],
        'rmse': [],
        'mae': []
    }
    
    for perm_iter in range(NUM_NULL_PERMUTATIONS):
        y_shuffled = np.random.permutation(y)
        perm_preds = np.zeros_like(y_shuffled, dtype=float)
        
        for train_idx, test_idx in loo.split(X):
            model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X[train_idx], y_shuffled[train_idx])
            perm_preds[test_idx[0]] = model.predict(X[test_idx])[0]
        
        # Calculate null metrics
        null_r, _ = pearsonr(y_shuffled, perm_preds)
        null_spearman, _ = spearmanr(y_shuffled, perm_preds)
        null_r2 = r2_score(y_shuffled, perm_preds)
        null_rmse = np.sqrt(mean_squared_error(y_shuffled, perm_preds))
        null_mae = mean_absolute_error(y_shuffled, perm_preds)
        
        null_metrics['pearson_r'].append(null_r)
        null_metrics['spearman_r'].append(null_spearman)
        null_metrics['r2'].append(null_r2)
        null_metrics['rmse'].append(null_rmse)
        null_metrics['mae'].append(null_mae)

    # Calculate p-values based on null distribution
    p_values = {}
    for metric in null_metrics:
        if metric in ['pearson_r', 'spearman_r', 'r2']:
            # For metrics where higher is better
            actual_value = locals()[f'loo_{metric.replace("_r", "")}']
            p_values[metric] = (np.sum(np.array(null_metrics[metric]) >= actual_value) + 1) / (NUM_NULL_PERMUTATIONS + 1)
        else:
            # For metrics where lower is better
            actual_value = locals()[f'loo_{metric}']
            p_values[metric] = (np.sum(np.array(null_metrics[metric]) <= actual_value) + 1) / (NUM_NULL_PERMUTATIONS + 1)

    # Store null distribution data
    null_distribution_data[internal_state][prefix][time_window] = null_metrics
    # Store feature correlations with self-reports
    feature_correlations = {}
    for i, feat_name in enumerate(feature_names):
        r_val, p_val = pearsonr(X[:, i], y)
        feature_correlations[feat_name] = {'r': r_val, 'p': p_val}

    
    # Calculate feature importance based on coefficient magnitude
    mean_importance = np.mean(np.abs(coef_matrix), axis=0)
    
    # Calculate permutation importance
    permutation_impacts = []
    for f_idx, fname in enumerate(feature_names):
        impact_scores = []
        for perm_iter in range(NUM_PERMUTATION_IMPORTANCE):
            X_perm = X.copy()
            X_perm[:, f_idx] = np.random.permutation(X_perm[:, f_idx])
            perm_preds = np.zeros_like(y, dtype=float)
            
            for train_idx, test_idx in loo.split(X_perm):
                model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X_perm[train_idx], y[train_idx])
                perm_preds[test_idx[0]] = model.predict(X_perm[test_idx])[0]
            
            perm_r, _ = pearsonr(y, perm_preds)
            impact = loo_r - perm_r
            impact_scores.append(impact)
        
        mean_impact = np.mean(impact_scores)
        permutation_impacts.append(mean_impact)
        permutation_impact_data[internal_state][fname][time_window] = mean_impact
    
    # Get top and bottom features by importance
    feature_importance_data = list(zip(feature_names, mean_importance, permutation_impacts))
    feature_importance_data.sort(key=lambda x: x[2], reverse=True)  # Sort by permutation impact
    top_features = feature_importance_data[:NUM_TOP_FEATURES]
    bottom_features = feature_importance_data[-NUM_BOTTOM_FEATURES:]

    # Analyze correlation between top/bottom features and target variable
    top_bottom_correlations = {}
    for feature_set, label in [(top_features, 'top'), (bottom_features, 'bottom')]:
        for fname, _, _ in feature_set:
            idx = list(feature_names).index(fname)
            r_val, p_val = pearsonr(X[:, idx], y)
            top_bottom_correlations[f"{label}_{fname}"] = {'r': r_val, 'p': p_val}
            feature_correlation_data[internal_state][prefix][fname][time_window] = {'r': r_val, 'p': p_val}
    
    # Calculate confidence intervals from null distribution
    ci_results = {}
    for metric in null_metrics:
        actual_value = locals()[f'loo_{metric.replace("_r", "")}'] if metric in ['pearson_r', 'spearman_r', 'r2'] else locals()[f'loo_{metric}']
        ci_results[metric] = {
            'value': actual_value,
            'p_value': p_values[metric],
            'null_mean': np.mean(null_metrics[metric]),
            'null_ci_lower': np.percentile(null_metrics[metric], 2.5),
            'null_ci_upper': np.percentile(null_metrics[metric], 97.5)
        }

    # Store results in summary
    summary_results[internal_state][prefix][time_window] = ci_results
    
    # Store feature importance data
    for fname, imp in zip(feature_names, mean_importance):
        feature_heatmap_data[internal_state][fname][time_window] = imp

    # Calculate feature selection frequency
    total_models = len(list(loo.split(X)))
    feature_selection_percentages = {}
    for f_idx, f_name in enumerate(feature_names):
        percentage = (feature_selection_count[f_idx] / total_models) * 100
        feature_selection_percentages[f_name] = percentage
    
    # Add to summary results
    summary_results[internal_state][prefix][time_window]['feature_selection'] = feature_selection_percentages
    

    # Save CSV results
    result_dict = {
        'Metric': ['Pearson R', 'Spearman R', 'R²', 'RMSE', 'MAE'],
        'Value': [ci_results['pearson_r']['value'], ci_results['spearman_r']['value'], 
                 ci_results['r2']['value'], ci_results['rmse']['value'], ci_results['mae']['value']],
        'P_Value': [ci_results['pearson_r']['p_value'], ci_results['spearman_r']['p_value'],
                   ci_results['r2']['p_value'], ci_results['rmse']['p_value'], ci_results['mae']['p_value']],
        'Null_Mean': [ci_results['pearson_r']['null_mean'], ci_results['spearman_r']['null_mean'],
                     ci_results['r2']['null_mean'], ci_results['rmse']['null_mean'], ci_results['mae']['null_mean']],
        'Null_CI_Lower': [ci_results['pearson_r']['null_ci_lower'], ci_results['spearman_r']['null_ci_lower'],
                         ci_results['r2']['null_ci_lower'], ci_results['rmse']['null_ci_lower'], ci_results['mae']['null_ci_lower']],
        'Null_CI_Upper': [ci_results['pearson_r']['null_ci_upper'], ci_results['spearman_r']['null_ci_upper'],
                         ci_results['r2']['null_ci_upper'], ci_results['rmse']['null_ci_upper'], ci_results['mae']['null_ci_upper']]
    }
    metrics_df = pd.DataFrame(result_dict)
    metrics_df.to_csv(os.path.join(csv_folder, f"{internal_state}_{prefix}_time_{time_window}_metrics.csv"), index=False)







    # Save alpha search distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(alpha_store, bins=20, color='#1f77b4')
    plt.title(f"Alpha Regularization Parameter Distribution\n{internal_state} | {PREFIX_DISPLAY_MAP.get(prefix, prefix)} | {time_window} min", fontsize=16)
    plt.xlabel("Alpha Value", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(prefix_folder, f"alpha_distribution_time_{time_window}.png"), dpi=300)
    plt.close()

    # Save feature importance plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot top features
    top_names = [f[0] for f in top_features]
    top_importance = [f[2] for f in top_features]
    top_correlations = [feature_correlations[f]['r'] for f in top_names]

    y_pos = np.arange(len(top_names))
    axes[0].barh(y_pos, top_importance, color=COLORS[0])
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([f"{name} (r={corr:.2f})" for name, corr in zip(top_names, top_correlations)])
    axes[0].set_title(f"Top {NUM_TOP_FEATURES} Features by Permutation Impact", fontsize=16)
    axes[0].set_xlabel("Permutation Impact (ΔR)", fontsize=14)

    # Plot bottom features
    bottom_names = [f[0] for f in bottom_features]
    bottom_importance = [f[2] for f in bottom_features]
    bottom_correlations = [feature_correlations[f]['r'] for f in bottom_names]

    y_pos = np.arange(len(bottom_names))
    axes[1].barh(y_pos, bottom_importance, color=COLORS[1])
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{name} (r={corr:.2f})" for name, corr in zip(bottom_names, bottom_correlations)])
    axes[1].set_title(f"Bottom {NUM_BOTTOM_FEATURES} Features by Permutation Impact", fontsize=16)
    axes[1].set_xlabel("Permutation Impact (ΔR)", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(prefix_folder, f"feature_importance_time_{time_window}.png"), dpi=300)
    plt.close()

    # Save feature selection frequency plot
    plt.figure(figsize=(12, 8))
    sorted_indices = np.argsort(feature_selection_count)[::-1]
    top_indices = sorted_indices[:20]  # Show only top 20 most frequently selected features

    plt.bar(np.arange(len(top_indices)), 
            [feature_selection_percentages[feature_names[i]] for i in top_indices],
            color=COLORS[2])
    plt.xticks(np.arange(len(top_indices)), 
            [feature_names[i] for i in top_indices], 
            rotation=45, ha='right')
    plt.title(f"Feature Selection Frequency\n{internal_state} | {PREFIX_DISPLAY_MAP.get(prefix, prefix)} | {time_window} min", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Selection Frequency (%)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(prefix_folder, f"feature_selection_frequency_time_{time_window}.png"), dpi=300)
    plt.close()

    # Save actual vs predicted scatter plot with null distribution information
    plt.figure(figsize=(10, 8))
    plt.scatter(y, preds, alpha=0.7, s=80, color=COLORS[0])

    # Add regression line
    z = np.polyfit(y, preds, 1)
    p = np.poly1d(z)
    plt.plot(y, p(y), linestyle='--', color=COLORS[1], linewidth=2)

    # Add identity line (perfect prediction)
    min_val, max_val = min(min(y), min(preds)), max(max(y), max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, linewidth=1)

    plt.title(f"Predicted vs. Actual {internal_state}\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)} | {time_window} min", fontsize=16)
    plt.xlabel(f"Actual {internal_state}", fontsize=14)
    plt.ylabel(f"Predicted {internal_state}", fontsize=14)

    # Add metrics info on plot with p-values
    metrics_text = (f"Pearson r: {ci_results['pearson_r']['value']:.3f} (p={ci_results['pearson_r']['p_value']:.3f})\n"
                    f"RMSE: {ci_results['rmse']['value']:.3f} (p={ci_results['rmse']['p_value']:.3f})\n"
                    f"R²: {ci_results['r2']['value']:.3f} (p={ci_results['r2']['p_value']:.3f})")
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                ha='left', va='top', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(prefix_folder, f"actual_vs_predicted_time_{time_window}.png"), dpi=300)
    plt.close()

    # Save statistical significance visualization for Pearson R
    plt.figure(figsize=(10, 6))
    sns.kdeplot(null_metrics['pearson_r'], fill=True, color=COLORS[3], alpha=0.5, label="Null Distribution")
    plt.axvline(loo_r, color='red', linestyle='--', linewidth=2, label=f"Actual (r={loo_r:.3f})")
    plt.title(f"Statistical Significance - Pearson R\n{internal_state} | {PREFIX_DISPLAY_MAP.get(prefix, prefix)} | {time_window} min", fontsize=16)
    plt.xlabel("Pearson Correlation (r)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)

    # Add p-value annotation
    p_value_text = f"p-value: {p_values['pearson_r']:.3f}"
    significance_text = "Significant (p<0.05)" if p_values['pearson_r'] < 0.05 else "Not significant (p≥0.05)"
    plt.annotate(f"{p_value_text}\n{significance_text}", xy=(0.95, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                ha='right', va='top', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(prefix_folder, f"significance_pearson_r_time_{time_window}.png"), dpi=300)
    plt.close()

    # Save top/bottom feature correlations as CSV
    correlation_dict = {
        'Feature': [],
        'Type': [],
        'Correlation': [],
        'P_Value': [],
        'Permutation_Impact': []
    }

    for feat, imp, perm in top_features:
        correlation_dict['Feature'].append(feat)
        correlation_dict['Type'].append('Top')
        correlation_dict['Correlation'].append(feature_correlations[feat]['r'])
        correlation_dict['P_Value'].append(feature_correlations[feat]['p'])
        correlation_dict['Permutation_Impact'].append(perm)

    for feat, imp, perm in bottom_features:
        correlation_dict['Feature'].append(feat)
        correlation_dict['Type'].append('Bottom')
        correlation_dict['Correlation'].append(feature_correlations[feat]['r'])
        correlation_dict['P_Value'].append(feature_correlations[feat]['p'])
        correlation_dict['Permutation_Impact'].append(perm)

    correlation_df = pd.DataFrame(correlation_dict)
    correlation_df.to_csv(os.path.join(csv_folder, f"{internal_state}_{prefix}_time_{time_window}_feature_correlations.csv"), index=False)


# --------- OVERVIEW AGGREGATE PLOTS ---------
# Process summary results and create visualizations
print("Processing summary results and creating visualizations...")

for internal_state in tqdm(INTERNAL_STATES, desc="Processing internal states"):
    state_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state)
    overview_folder = os.path.join(state_folder, 'Overview')
    csv_folder = os.path.join(state_folder, 'CSV_Results')
    
    # Ensure all required folders exist
    for folder in [overview_folder, csv_folder]:
        os.makedirs(folder, exist_ok=True)
    
    # 1. Process performance metrics over time
    performance_over_time = defaultdict(lambda: defaultdict(list))
    
    for prefix in RESULTS_PREFIX_LIST:
        for time_window in TIME_WINDOWS:
            if time_window in summary_results[internal_state][prefix]:
                for metric in ['pearson_r', 'spearman_r', 'r2', 'rmse', 'mae']:
                    if metric in summary_results[internal_state][prefix][time_window]:
                        value = summary_results[internal_state][prefix][time_window][metric]['value']
                        p_value = summary_results[internal_state][prefix][time_window][metric]['p_value']
                        performance_over_time[prefix][f"{metric}_value"].append(value)
                        performance_over_time[prefix][f"{metric}_p"].append(p_value)
                        performance_over_time[prefix]["time_windows"].append(time_window)
    
    # Save performance data to CSV
    for prefix in RESULTS_PREFIX_LIST:
        if len(performance_over_time[prefix]["time_windows"]) > 0:
            perf_df = pd.DataFrame({
                'Time_Window': performance_over_time[prefix]["time_windows"],
                'Pearson_R': performance_over_time[prefix]["pearson_r_value"],
                'Pearson_R_p': performance_over_time[prefix]["pearson_r_p"],
                'Spearman_R': performance_over_time[prefix]["spearman_r_value"],
                'Spearman_R_p': performance_over_time[prefix]["spearman_r_p"],
                'R2': performance_over_time[prefix]["r2_value"],
                'R2_p': performance_over_time[prefix]["r2_p"],
                'RMSE': performance_over_time[prefix]["rmse_value"],
                'RMSE_p': performance_over_time[prefix]["rmse_p"],
                'MAE': performance_over_time[prefix]["mae_value"],
                'MAE_p': performance_over_time[prefix]["mae_p"]
            })
            perf_df.to_csv(os.path.join(csv_folder, f"{internal_state}_{prefix}_performance_over_time.csv"), index=False)
    
    # 2. Create Pearson correlation time series plot
    plt.figure(figsize=(12, 8))
    
    for idx, prefix in enumerate(RESULTS_PREFIX_LIST):
        if len(performance_over_time[prefix]["time_windows"]) > 0:
            time_windows = performance_over_time[prefix]["time_windows"]
            pearson_values = performance_over_time[prefix]["pearson_r_value"]
            p_values = performance_over_time[prefix]["pearson_r_p"]
            
            # Plot line
            plt.plot(time_windows, pearson_values, 'o-', color=COLORS[idx], 
                     label=f"{PREFIX_DISPLAY_MAP.get(prefix, prefix)}")
            
            # Mark significant points with a different marker
            significant_times = [t for t, p in zip(time_windows, p_values) if p < 0.05]
            significant_values = [v for v, p in zip(pearson_values, p_values) if p < 0.05]
            if significant_times:
                plt.plot(significant_times, significant_values, 'D', color=COLORS[idx], 
                         markersize=10, markeredgecolor='black', markeredgewidth=1.5,
                         label=f"{PREFIX_DISPLAY_MAP.get(prefix, prefix)} (p<0.05)")
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Window (minutes)', fontsize=14)
    plt.ylabel('Pearson Correlation (r)', fontsize=14)
    plt.title(f"{internal_state} Decoding Performance Over Time", fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(overview_folder, f"{internal_state}_pearson_over_time.png"), dpi=300)
    plt.close()
    
    # 3. Create multi-metric comparison plot (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    metrics = ['pearson_r', 'spearman_r', 'r2', 'rmse']
    metric_labels = ['Pearson Correlation (r)', 'Spearman Correlation (ρ)', 'R²', 'RMSE']
    
    for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[metric_idx]
        
        for idx, prefix in enumerate(RESULTS_PREFIX_LIST):
            if len(performance_over_time[prefix]["time_windows"]) > 0:
                time_windows = performance_over_time[prefix]["time_windows"]
                values = performance_over_time[prefix][f"{metric}_value"]
                p_values = performance_over_time[prefix][f"{metric}_p"]
                
                # Plot line
                ax.plot(time_windows, values, 'o-', color=COLORS[idx], 
                         label=f"{PREFIX_DISPLAY_MAP.get(prefix, prefix)}")
                
                # Mark significant points with a different marker
                significant_times = [t for t, p in zip(time_windows, p_values) if p < 0.05]
                significant_values = [v for v, p in zip(values, p_values) if p < 0.05]
                if significant_times:
                    ax.plot(significant_times, significant_values, 'D', color=COLORS[idx], 
                             markersize=8, markeredgecolor='black', markeredgewidth=1.0)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Window (minutes)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"{label} Over Time", fontsize=14)
        
        if metric_idx == 0:  # Only add legend to first plot
            ax.legend(loc='best', fontsize=10)
    
    plt.suptitle(f"{internal_state} Multi-Metric Comparison", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(overview_folder, f"{internal_state}_multi_metric_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Create feature selection frequency heatmap for each prefix
    for prefix in RESULTS_PREFIX_LIST:
        all_features = set()
        all_times = []
        
        # Collect all features and times
        for time_window in TIME_WINDOWS:
            if time_window in summary_results[internal_state][prefix]:
                if 'feature_selection' in summary_results[internal_state][prefix][time_window]:
                    all_features.update(summary_results[internal_state][prefix][time_window]['feature_selection'].keys())
                    all_times.append(time_window)
        
        if not all_features or not all_times:
            continue
            
        # Convert to lists for consistent ordering
        all_features = list(all_features)
        all_times.sort()
        
        # Create matrix for heatmap
        heatmap_data = np.zeros((len(all_features), len(all_times)))
        
        # Fill matrix with frequency data
        for i, feature in enumerate(all_features):
            for j, time_window in enumerate(all_times):
                if (time_window in summary_results[internal_state][prefix] and 
                    'feature_selection' in summary_results[internal_state][prefix][time_window] and
                    feature in summary_results[internal_state][prefix][time_window]['feature_selection']):
                    heatmap_data[i, j] = summary_results[internal_state][prefix][time_window]['feature_selection'][feature]
        
        # Sort features by average selection frequency
        feature_avg_freq = np.mean(heatmap_data, axis=1)
        sorted_indices = np.argsort(feature_avg_freq)[::-1]
        
        # Take top 30 features only for readability
        if len(sorted_indices) > 30:
            sorted_indices = sorted_indices[:30]
        
        sorted_features = [all_features[i] for i in sorted_indices]
        sorted_data = heatmap_data[sorted_indices, :]
        
        # Create heatmap
        plt.figure(figsize=(12, max(8, len(sorted_features) * 0.3)))
        sns.heatmap(sorted_data, cmap='viridis', annot=False, 
                    xticklabels=all_times, yticklabels=sorted_features)
        plt.xlabel('Time Window (minutes)', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title(f"{internal_state} - {PREFIX_DISPLAY_MAP.get(prefix, prefix)} Feature Selection Frequency", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_feature_selection_heatmap.png"), dpi=300)
        plt.close()
    
    # 5. Create coefficient importance heatmap
    # Create a combined heatmap for permutation impact across all prefixes
    for prefix in RESULTS_PREFIX_LIST:
        # Collect features with impact data
        features_with_data = set()
        for feature in permutation_impact_data[internal_state]:
            if any(time in permutation_impact_data[internal_state][feature] for time in TIME_WINDOWS):
                features_with_data.add(feature)
        
        if not features_with_data:
            continue
            
        # Create matrix for heatmap
        features_list = list(features_with_data)
        perm_heatmap = np.zeros((len(features_list), len(TIME_WINDOWS)))
        perm_heatmap.fill(np.nan)  # Fill with NaN initially
        
        # Fill matrix with impact data
        for i, feature in enumerate(features_list):
            for j, time_window in enumerate(TIME_WINDOWS):
                if time_window in permutation_impact_data[internal_state][feature]:
                    perm_heatmap[i, j] = permutation_impact_data[internal_state][feature][time_window]
        
        # Sort features by average impact
        avg_impact = np.nanmean(perm_heatmap, axis=1)
        sorted_indices = np.argsort(avg_impact)[::-1]
        
        # Take top 30 features only for readability
        if len(sorted_indices) > 30:
            sorted_indices = sorted_indices[:30]
            
        sorted_features = [features_list[i] for i in sorted_indices]
        sorted_heatmap = perm_heatmap[sorted_indices, :]
        
        # Create heatmap
        plt.figure(figsize=(14, max(8, len(sorted_features) * 0.3)))
        ax = sns.heatmap(sorted_heatmap, cmap='coolwarm', center=0, 
                         xticklabels=TIME_WINDOWS, yticklabels=sorted_features, 
                         mask=np.isnan(sorted_heatmap))
        plt.xlabel('Time Window (minutes)', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title(f"{internal_state} - {PREFIX_DISPLAY_MAP.get(prefix, prefix)} Feature Permutation Impact", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_permutation_impact_heatmap.png"), dpi=300)
        plt.close()
    
    # 6. Find best time window and model for each prefix
    best_models = {}
    for prefix in RESULTS_PREFIX_LIST:
        if len(performance_over_time[prefix]["pearson_r_value"]) > 0:
            best_idx = np.argmax(performance_over_time[prefix]["pearson_r_value"])
            best_time = performance_over_time[prefix]["time_windows"][best_idx]
            best_pearson = performance_over_time[prefix]["pearson_r_value"][best_idx]
            best_p_value = performance_over_time[prefix]["pearson_r_p"][best_idx]
            
            best_models[prefix] = {
                'time_window': best_time,
                'pearson_r': best_pearson,
                'p_value': best_p_value
            }
    
    # 7. Create feature-to-target correlation plots for top features (best time window)
    best_prefix = None
    best_time = None
    best_pearson = -1
    
    for prefix, data in best_models.items():
        if data['pearson_r'] > best_pearson:
            best_pearson = data['pearson_r']
            best_prefix = prefix
            best_time = data['time_window']
    
    if best_prefix and best_time in summary_results[internal_state][best_prefix]:
        # Find the CSV file for the best model to load feature data
        best_csv_file = None
        for file in os.listdir(patient_folder):
            internal_file_state, file_time, file_prefix = parse_filename(file)
            if (internal_file_state == internal_state and 
                file_time == best_time and 
                file_prefix == best_prefix):
                best_csv_file = file
                break
        
        if best_csv_file:
            # Load the data
            df = pd.read_csv(os.path.join(patient_folder, best_csv_file))
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = df.columns[:-1]
            
            # Find feature importance data
            feature_importance_data = []
            for feature in feature_names:
                if feature in permutation_impact_data[internal_state] and best_time in permutation_impact_data[internal_state][feature]:
                    feature_importance_data.append((feature, permutation_impact_data[internal_state][feature][best_time]))
            
            feature_importance_data.sort(key=lambda x: x[1], reverse=True)
            top_10_features = feature_importance_data[:10]
            
            # Create scatter plots for top features
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for i, (feature, importance) in enumerate(top_10_features):
                if i >= len(axes):
                    break
                    
                feature_idx = list(feature_names).index(feature)
                x_vals = X[:, feature_idx]
                
                axes[i].scatter(x_vals, y, alpha=0.7, s=60, color=COLORS[i % len(COLORS)])
                
                # Add regression line
                z = np.polyfit(x_vals, y, 1)
                p = np.poly1d(z)
                axes[i].plot(sorted(x_vals), p(sorted(x_vals)), linestyle='--', color='red', linewidth=1.5)
                
                # Calculate correlation
                r_val, p_val = pearsonr(x_vals, y)
                axes[i].set_title(f"{feature}\nr={r_val:.2f}, p={p_val:.3f}", fontsize=10)
                axes[i].set_xlabel(feature, fontsize=9)
                if i % 5 == 0:
                    axes[i].set_ylabel(internal_state, fontsize=10)
            
            plt.suptitle(f"Top Feature Correlations for {internal_state}\n{PREFIX_DISPLAY_MAP.get(best_prefix, best_prefix)} - {best_time} min", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            plt.savefig(os.path.join(overview_folder, f"{internal_state}_top_feature_correlations.png"), dpi=300)
            plt.close()
    
    # 8. Create plot of alpha regularization parameters across time windows
    for prefix in RESULTS_PREFIX_LIST:
        alpha_over_time = defaultdict(list)
        
        for time_window in TIME_WINDOWS:
            csv_path = os.path.join(csv_folder, f"{internal_state}_{prefix}_time_{time_window}_metrics.csv")
            if os.path.exists(csv_path):
                # Check if we have alphas data for this time window (needs to be extracted from files or saved earlier)
                prefix_folder = os.path.join(state_folder, prefix)
                if os.path.exists(os.path.join(prefix_folder, f"alpha_distribution_time_{time_window}.png")):
                    alpha_over_time['time_windows'].append(time_window)
        
        if alpha_over_time['time_windows']:
            plt.figure(figsize=(12, 6))
            plt.plot(alpha_over_time['time_windows'], alpha_over_time['alphas'], 'o-', color=COLORS[0])
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time Window (minutes)', fontsize=14)
            plt.ylabel('Optimal Alpha Value', fontsize=14)
            plt.title(f"{internal_state} - {PREFIX_DISPLAY_MAP.get(prefix, prefix)} Alpha Parameter Over Time", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_alpha_over_time.png"), dpi=300)
            plt.close()
    
    # 9. Create summary dashboard (2x2 grid) combining key visualizations
    fig = plt.figure(figsize=(20, 16))
    
    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Performance metrics across time windows
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, prefix in enumerate(RESULTS_PREFIX_LIST):
        if len(performance_over_time[prefix]["time_windows"]) > 0:
            time_windows = performance_over_time[prefix]["time_windows"]
            pearson_values = performance_over_time[prefix]["pearson_r_value"]
            p_values = performance_over_time[prefix]["pearson_r_p"]
            
            # Plot line
            ax1.plot(time_windows, pearson_values, 'o-', color=COLORS[idx], 
                     label=f"{PREFIX_DISPLAY_MAP.get(prefix, prefix)}")
            
            # Mark significant points with a different marker
            significant_times = [t for t, p in zip(time_windows, p_values) if p < 0.05]
            significant_values = [v for v, p in zip(pearson_values, p_values) if p < 0.05]
            if significant_times:
                ax1.plot(significant_times, significant_values, 'D', color=COLORS[idx], 
                         markersize=8, markeredgecolor='black', markeredgewidth=1.0,
                         label=f"{PREFIX_DISPLAY_MAP.get(prefix, prefix)} (p<0.05)")
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time Window (minutes)', fontsize=12)
    ax1.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax1.set_title(f"Decoding Performance Over Time", fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    
    # Plot 2: Top feature importance for best performing time window
    ax2 = fig.add_subplot(gs[0, 1])
    
    if best_prefix and best_time in summary_results[internal_state][best_prefix]:
        # Use previously collected top features data
        if 'top_10_features' in locals():
            x_pos = np.arange(len(top_10_features))
            ax2.barh(x_pos, [imp for _, imp in top_10_features], color=COLORS[0])
            ax2.set_yticks(x_pos)
            ax2.set_yticklabels([f[:20] + "..." if len(f) > 20 else f for f, _ in top_10_features])
            ax2.invert_yaxis()  # Highest values at the top
            ax2.set_xlabel('Permutation Impact', fontsize=12)
            ax2.set_title(f"Top Features - {PREFIX_DISPLAY_MAP.get(best_prefix, best_prefix)} at {best_time} min", fontsize=14)
    
    # Plot 3: Feature selection frequency heatmap for best prefix
    ax3 = fig.add_subplot(gs[1, 0])
    
    if best_prefix:
        all_features = set()
        all_times = []
        
        # Collect all features and times (reuse from earlier)
        for time_window in TIME_WINDOWS:
            if time_window in summary_results[internal_state][best_prefix]:
                if 'feature_selection' in summary_results[internal_state][best_prefix][time_window]:
                    all_features.update(summary_results[internal_state][best_prefix][time_window]['feature_selection'].keys())
                    all_times.append(time_window)
        
        if all_features and all_times:
            # Convert to lists for consistent ordering
            all_features = list(all_features)
            all_times.sort()
            
            # Create matrix for heatmap
            heatmap_data = np.zeros((len(all_features), len(all_times)))
            
            # Fill matrix with frequency data
            for i, feature in enumerate(all_features):
                for j, time_window in enumerate(all_times):
                    if (time_window in summary_results[internal_state][best_prefix] and 
                        'feature_selection' in summary_results[internal_state][best_prefix][time_window] and
                        feature in summary_results[internal_state][best_prefix][time_window]['feature_selection']):
                        heatmap_data[i, j] = summary_results[internal_state][best_prefix][time_window]['feature_selection'][feature]
            
            # Sort features by average selection frequency
            feature_avg_freq = np.mean(heatmap_data, axis=1)
            sorted_indices = np.argsort(feature_avg_freq)[::-1]
            
            # Take top 20 features only for readability in dashboard
            if len(sorted_indices) > 20:
                sorted_indices = sorted_indices[:20]
            
            sorted_features = [all_features[i][:15] + "..." if len(all_features[i]) > 15 else all_features[i] for i in sorted_indices]
            sorted_data = heatmap_data[sorted_indices, :]
            
            # Create heatmap in the subplot
            sns.heatmap(sorted_data, cmap='viridis', annot=False, 
                        xticklabels=all_times, yticklabels=sorted_features, ax=ax3)
            ax3.set_xlabel('Time Window (minutes)', fontsize=12)
            ax3.set_ylabel('Features', fontsize=12)
            ax3.set_title(f"Feature Selection Frequency - {PREFIX_DISPLAY_MAP.get(best_prefix, best_prefix)}", fontsize=14)
    
    # Plot 4: Multiple metrics comparison for best prefix
    ax4 = fig.add_subplot(gs[1, 1])
    
    if best_prefix and len(performance_over_time[best_prefix]["time_windows"]) > 0:
        time_windows = performance_over_time[best_prefix]["time_windows"]
        
        metrics_to_plot = [
            ("pearson_r_value", "Pearson R", COLORS[0]),
            ("spearman_r_value", "Spearman R", COLORS[1]),
            ("r2_value", "R²", COLORS[2])
        ]
        
        for metric_key, metric_label, color in metrics_to_plot:
            values = performance_over_time[best_prefix][metric_key]
            p_values = performance_over_time[best_prefix][metric_key.replace("_value", "_p")]
            
            # Plot line
            ax4.plot(time_windows, values, 'o-', color=color, label=metric_label)
            
            # Mark significant points with a different marker
            significant_times = [t for t, p in zip(time_windows, p_values) if p < 0.05]
            significant_values = [v for v, p in zip(values, p_values) if p < 0.05]
            if significant_times:
                ax4.plot(significant_times, significant_values, 'D', color=color, 
                         markersize=8, markeredgecolor='black', markeredgewidth=1.0)
        
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Time Window (minutes)', fontsize=12)
        ax4.set_ylabel('Metric Value', fontsize=12)
        ax4.set_title(f"Multiple Metrics - {PREFIX_DISPLAY_MAP.get(best_prefix, best_prefix)}", fontsize=14)
        ax4.legend(loc='best', fontsize=10)
    
    plt.suptitle(f"{internal_state} Summary Dashboard", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(overview_folder, f"{internal_state}_summary_dashboard.png"), dpi=300)
    plt.close()

# 10. Create final summary CSV with best performances
summary_rows = []

for internal_state in INTERNAL_STATES:
    for prefix in RESULTS_PREFIX_LIST:
        best_time = None
        best_pearson = -1
        best_p_value = 1.0
        
        for time_window in TIME_WINDOWS:
            if (time_window in summary_results[internal_state][prefix] and 
                'pearson_r' in summary_results[internal_state][prefix][time_window]):
                pearson_value = summary_results[internal_state][prefix][time_window]['pearson_r']['value']
                p_value = summary_results[internal_state][prefix][time_window]['pearson_r']['p_value']
                
                if p_value < 0.05 and pearson_value > best_pearson:
                    best_pearson = pearson_value
                    best_time = time_window
                    best_p_value = p_value
        
        if best_time is not None:
            # Get best model metrics
            best_metrics = summary_results[internal_state][prefix][best_time]
            
            # Get top features if available
            top_features = []
            for feature in permutation_impact_data[internal_state]:
                if best_time in permutation_impact_data[internal_state][feature]:
                    impact = permutation_impact_data[internal_state][feature][best_time]
                    top_features.append((feature, impact))
            
            top_features.sort(key=lambda x: x[1], reverse=True)
            top_features_str = "; ".join([f"{f} ({i:.3f})" for f, i in top_features[:5]])
            
            # Add to summary rows
            summary_rows.append({
                'Internal_State': internal_state,
                'Model_Type': PREFIX_DISPLAY_MAP.get(prefix, prefix),
                'Best_Time_Window': best_time,
                'Pearson_R': best_metrics['pearson_r']['value'],
                'Pearson_R_p': best_metrics['pearson_r']['p_value'],
                'Spearman_R': best_metrics['spearman_r']['value'],
                'Spearman_R_p': best_metrics['spearman_r']['p_value'],
                'R2': best_metrics['r2']['value'],
                'R2_p': best_metrics['r2']['p_value'],
                'RMSE': best_metrics['rmse']['value'],
                'RMSE_p': best_metrics['rmse']['p_value'],
                'Top_Features': top_features_str
            })

# Save final summary CSV
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_OUTPUT_PATH, "final_decoding_summary.csv"), index=False)
    print(f"Final summary saved to {os.path.join(RESULTS_OUTPUT_PATH, 'final_decoding_summary.csv')}")

print("Results processing complete!")