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
N_BOOTSTRAPS = 2
ALPHAS = np.linspace(0.1, 2.0, 20)
TIME_WINDOWS = list(range(15, 241, 15))
# INTERNAL_STATES = ['Mood', 'Depression', 'Anxiety', 'Hunger', 'Pain']
INTERNAL_STATES = ['Mood']
# RESULTS_PREFIX_LIST = ['OF_L_', 'OGAU_L_', 'OGAUHSE_L_', 'HSE_L_']
RESULTS_PREFIX_LIST = ['OGAU_L_']
NUM_TOP_FEATURES = 5  # Number of top features to analyze
NUM_BOTTOM_FEATURES = 5  # Number of bottom features to analyze

# Create output directories
os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)

# Create a mapping to decode feature prefixes to more readable names for plots
PREFIX_DISPLAY_MAP = {
    'OF_L_': 'Optical Flow',
    'OGAU_L_': 'OpenFace AU',
    'OGAUHSE_L_': 'OpenFace AU + HSE',
    'HSE_L_': 'Head-Shoulder-Eye'
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
all_feature_names = set()

# Setup plot style for professional presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
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
    
    for train_idx, test_idx in loo.split(X):
        model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = model.predict(X[test_idx])[0]
        alpha_store.append(model.alpha_)

    # Calculate initial metrics
    loo_r, _ = pearsonr(y, preds)
    loo_spearman, _ = spearmanr(y, preds)
    loo_r2 = r2_score(y, preds)
    loo_rmse = np.sqrt(mean_squared_error(y, preds))
    loo_mae = mean_absolute_error(y, preds)

    # Store feature correlations with target
    feature_correlations = {}
    for i, feat_name in enumerate(feature_names):
        r_val, p_val = pearsonr(X[:, i], y)
        feature_correlations[feat_name] = {'r': r_val, 'p': p_val}

    # Bootstrap analysis
    coef_matrix = []
    permutation_r_per_feature = defaultdict(list)
    bootstrap_metrics = defaultdict(list)
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
            
            # Track which features were selected (non-zero coefficients)
            non_zero_features = (model.coef_ != 0).astype(int)
            feature_selection_count += non_zero_features
            
            for f_idx, is_selected in enumerate(non_zero_features):
                if is_selected:
                    feature_selection_frequency[internal_state][prefix][feature_names[f_idx]] += 1
                    
            test_indices.append(test_idx[0])

        # Calculate bootstrap metrics
        r_boot, _ = pearsonr(y_boot, boot_preds)
        spearman_boot, _ = spearmanr(y_boot, boot_preds)
        r2_boot = r2_score(y_boot, boot_preds)
        rmse_boot = np.sqrt(mean_squared_error(y_boot, boot_preds))
        mae_boot = mean_absolute_error(y_boot, boot_preds)
        
        bootstrap_metrics['pearson_r'].append(r_boot)
        bootstrap_metrics['spearman_r'].append(spearman_boot)
        bootstrap_metrics['r2'].append(r2_boot)
        bootstrap_metrics['rmse'].append(rmse_boot)
        bootstrap_metrics['mae'].append(mae_boot)
        
        coef_matrix.extend(boot_coefs)

        # Permutation importance calculation
        boot_coefs_arr = np.array(boot_coefs)
        X_test_matrix = X_boot[test_indices]
        for f_idx in range(X.shape[1]):
            X_test_permuted = X_test_matrix.copy()
            X_test_permuted[:, f_idx] = np.random.permutation(X_test_permuted[:, f_idx])
            perm_preds = np.sum(X_test_permuted * boot_coefs_arr, axis=1)
            r_perm, _ = pearsonr(y_boot[test_indices], perm_preds)
            permutation_r_per_feature[feature_names[f_idx]].append(r_perm)

    # Process results
    coef_matrix = np.array(coef_matrix)
    mean_importance = np.mean(np.abs(coef_matrix), axis=0)
    mean_perm_impact = []
    mean_r = np.mean(bootstrap_metrics['pearson_r'])
    
    for f in feature_names:
        perm_impact = mean_r - np.mean(permutation_r_per_feature[f])
        mean_perm_impact.append(perm_impact)
        permutation_impact_data[internal_state][f][time_window] = perm_impact

    # Calculate confidence intervals for all metrics
    ci_results = {}
    for metric in bootstrap_metrics:
        values = bootstrap_metrics[metric]
        ci_results[metric] = {
            'mean': np.mean(values),
            'ci_lower': np.percentile(values, 2.5),
            'ci_upper': np.percentile(values, 97.5)
        }

    # Store results in summary
    summary_results[internal_state][prefix][time_window] = ci_results
    
    # Store feature importance data
    for fname, imp in zip(feature_names, mean_importance):
        feature_heatmap_data[internal_state][fname][time_window] = imp

    # Calculate feature selection frequency
    total_models = N_BOOTSTRAPS * len(list(loo.split(X)))
    feature_selection_percentage = (feature_selection_count / total_models) * 100
    
    # Get top and bottom features by importance
    feature_importance_data = list(zip(feature_names, mean_importance, mean_perm_impact))
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
    
    # Save CSV results
    result_dict = {
        'Metric': ['Pearson R', 'Spearman R', 'R²', 'RMSE', 'MAE'],
        'Value': [ci_results['pearson_r']['mean'], ci_results['spearman_r']['mean'], 
                 ci_results['r2']['mean'], ci_results['rmse']['mean'], ci_results['mae']['mean']],
        'CI_Lower': [ci_results['pearson_r']['ci_lower'], ci_results['spearman_r']['ci_lower'],
                    ci_results['r2']['ci_lower'], ci_results['rmse']['ci_lower'], ci_results['mae']['ci_lower']],
        'CI_Upper': [ci_results['pearson_r']['ci_upper'], ci_results['spearman_r']['ci_upper'],
                   ci_results['r2']['ci_upper'], ci_results['rmse']['ci_upper'], ci_results['mae']['ci_upper']]
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
    sorted_indices = np.argsort(feature_selection_percentage)[::-1]
    top_indices = sorted_indices[:20]  # Show only top 20 most frequently selected features
    
    plt.bar(np.arange(len(top_indices)), 
            [feature_selection_percentage[i] for i in top_indices],
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
    
    # Save actual vs predicted scatter plot
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
    
    # Add metrics info on plot
    metrics_text = (f"Pearson r: {ci_results['pearson_r']['mean']:.3f} [{ci_results['pearson_r']['ci_lower']:.3f}, {ci_results['pearson_r']['ci_upper']:.3f}]\n"
                    f"RMSE: {ci_results['rmse']['mean']:.3f}\n"
                    f"R²: {ci_results['r2']['mean']:.3f}")
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 ha='left', va='top', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(prefix_folder, f"actual_vs_predicted_time_{time_window}.png"), dpi=300)
    plt.close()

    # Save top/bottom feature correlations as CSV
    correlation_dict = {
        'Feature': [],
        'Type': [],
        'Correlation': [],
        'P_Value': []
    }
    
    for feat, imp, perm in top_features:
        correlation_dict['Feature'].append(feat)
        correlation_dict['Type'].append('Top')
        correlation_dict['Correlation'].append(feature_correlations[feat]['r'])
        correlation_dict['P_Value'].append(feature_correlations[feat]['p'])
    
    for feat, imp, perm in bottom_features:
        correlation_dict['Feature'].append(feat)
        correlation_dict['Type'].append('Bottom')
        correlation_dict['Correlation'].append(feature_correlations[feat]['r'])
        correlation_dict['P_Value'].append(feature_correlations[feat]['p'])
    
    correlation_df = pd.DataFrame(correlation_dict)
    correlation_df.to_csv(os.path.join(csv_folder, f"{internal_state}_{prefix}_time_{time_window}_feature_correlations.csv"), index=False)

# --------- OVERVIEW AGGREGATE PLOTS ---------
for internal_state in summary_results:
    overview_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state, 'Overview')
    csv_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state, 'CSV_Results')
    os.makedirs(overview_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    # Prepare data for time-series performance metrics plot
    all_metrics_data = defaultdict(lambda: defaultdict(list))
    
    for prefix in summary_results[internal_state]:
        # Create overview data structures
        time_list = sorted(summary_results[internal_state][prefix].keys())
        
        # Prepare data for CSV output
        overview_data = {
            'Time_Window': time_list
        }
        
        # Track data for all metrics
        for metric in ['pearson_r', 'spearman_r', 'r2', 'rmse', 'mae']:
            overview_data[f'{metric}_mean'] = []
            overview_data[f'{metric}_ci_lower'] = []
            overview_data[f'{metric}_ci_upper'] = []
            
            for t in time_list:
                if t in summary_results[internal_state][prefix]:
                    metric_data = summary_results[internal_state][prefix][t][metric]
                    overview_data[f'{metric}_mean'].append(metric_data['mean'])
                    overview_data[f'{metric}_ci_lower'].append(metric_data['ci_lower'])
                    overview_data[f'{metric}_ci_upper'].append(metric_data['ci_upper'])
                    
                    # Store for multi-metric plot
                    all_metrics_data[metric]['times'].append(t)
                    all_metrics_data[metric]['values'].append(metric_data['mean'])
                    all_metrics_data[metric]['prefix'].append(prefix)
        
        # Save overview data as CSV
        overview_df = pd.DataFrame(overview_data)
        overview_df.to_csv(os.path.join(csv_folder, f"{internal_state}_{prefix}_overview_metrics.csv"), index=False)
        
        # Create Pearson R time series plot with confidence intervals
        r_means = [summary_results[internal_state][prefix][t]['pearson_r']['mean'] for t in time_list]
        r_lower = [summary_results[internal_state][prefix][t]['pearson_r']['ci_lower'] for t in time_list]
        r_upper = [summary_results[internal_state][prefix][t]['pearson_r']['ci_upper'] for t in time_list]
        
        plt.figure(figsize=(12, 8))
        plt.plot(time_list, r_means, marker='o', markersize=8, linewidth=2, label=f'Pearson R', color=COLORS[0])
        plt.fill_between(time_list, r_lower, r_upper, alpha=0.3, color=COLORS[0], label='95% CI')
        
        plt.title(f"{internal_state} Detection Performance Over Time\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)}", fontsize=18)
        plt.xlabel("Time Window (minutes)", fontsize=16)
        plt.ylabel("Pearson Correlation Coefficient (r)", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_pearson_r_over_time.png"), dpi=300)
        plt.close()
        
        # Create multi-metric time series plot
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('pearson_r', 'Pearson R', COLORS[0]),
            ('spearman_r', 'Spearman R', COLORS[1]),
            ('r2', 'R²', COLORS[2]),
            ('rmse', 'RMSE (lower is better)', COLORS[3])
        ]
        
        for i, (metric, label, color) in enumerate(metrics_to_plot):
            means = [summary_results[internal_state][prefix][t][metric]['mean'] for t in time_list]
            lower = [summary_results[internal_state][prefix][t][metric]['ci_lower'] for t in time_list]
            upper = [summary_results[internal_state][prefix][t][metric]['ci_upper'] for t in time_list]
            
            axes[i].plot(time_list, means, marker='o', markersize=8, linewidth=2, color=color)
            axes[i].fill_between(time_list, lower, upper, alpha=0.3, color=color)
            
            axes[i].set_title(f"{label} Over Time", fontsize=16)
            axes[i].set_xlabel("Time Window (minutes)", fontsize=14)
            axes[i].set_ylabel(label, fontsize=14)
            axes[i].grid(True, alpha=0.3)
            
        plt.suptitle(f"{internal_state} Multiple Performance Metrics\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)}", fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_multiple_metrics_over_time.png"), dpi=300)
        plt.close()
        
    # Create feature selection frequency heatmap
    feature_list = sorted(list(all_feature_names))
    time_list = sorted(TIME_WINDOWS)
    
    for prefix in RESULTS_PREFIX_LIST:
        # Calculate normalized selection frequencies
        total_models = N_BOOTSTRAPS * len(list(LeaveOneOut().split(X))) * len(time_list)
        selection_frequency_matrix = np.zeros((len(feature_list), len(time_list)))
        
        for i, feat in enumerate(feature_list):
            for j, t in enumerate(time_list):
                freq = feature_selection_frequency[internal_state][prefix].get(feat, 0)
                selection_frequency_matrix[i, j] = (freq / total_models) * 100
        
        # Create a mask for features that are never selected
        mask = (selection_frequency_matrix.sum(axis=1) == 0).reshape(-1, 1)
        
        # Plot only features that were selected at least once
        non_zero_features = [f for i, f in enumerate(feature_list) if not mask[i].item()]
        non_zero_matrix = selection_frequency_matrix[~mask.flatten()]
        
        if len(non_zero_features) > 0:
            plt.figure(figsize=(14, max(8, len(non_zero_features) * 0.4)))
            sns.heatmap(non_zero_matrix, cmap='viridis', xticklabels=time_list, yticklabels=non_zero_features,
                        cbar_kws={'label': '% of Bootstrap Models'}, linewidths=0.5)
            plt.title(f"{internal_state} - Feature Selection Frequency\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)}", fontsize=18)
            plt.xlabel("Time Window (minutes)", fontsize=16)
            plt.ylabel("Feature", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_feature_selection_frequency.png"), dpi=300)
            plt.close()
    
    # Create coefficient importance and permutation impact heatmaps
    feature_list = sorted(list(all_feature_names))
    time_list = sorted(TIME_WINDOWS)
    
    for prefix in RESULTS_PREFIX_LIST:
        for matrix_type, data_source, title, fname_suffix in [
            ('Coefficient Importance', feature_heatmap_data, 'Mean |Coefficient|', 'Coef'),
            ('Permutation Impact', permutation_impact_data, 'Permutation Impact (ΔR)', 'Perm')
        ]:
            heatmap_matrix = np.zeros((len(feature_list), len(time_list)))
            
            for i, fname in enumerate(feature_list):
                for j, t in enumerate(time_list):
                    heatmap_matrix[i, j] = data_source[internal_state].get(fname, {}).get(t, 0)
            
            # Create a mask for features that are never important
            mask = (np.abs(heatmap_matrix).sum(axis=1) < 1e-6).reshape(-1, 1)
            
            # Plot only features that have some importance
            non_zero_features = [f for i, f in enumerate(feature_list) if not mask[i].item()]
            non_zero_matrix = heatmap_matrix[~mask.flatten()]
            
            if len(non_zero_features) > 0:
                plt.figure(figsize=(14, max(8, len(non_zero_features) * 0.4)))
                sns.heatmap(non_zero_matrix, cmap='viridis', xticklabels=time_list, yticklabels=non_zero_features,
                            cbar_kws={'label': title}, linewidths=0.5)
                plt.title(f"{internal_state} - {matrix_type} Across Time Windows\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)}", fontsize=18)
                plt.xlabel("Time Window (minutes)", fontsize=16)
                plt.ylabel("Feature", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_{fname_suffix}_Heatmap.png"), dpi=300)
                plt.close()
    
    # Create feature-to-target correlation plots
    for prefix in RESULTS_PREFIX_LIST:
        # Get feature correlation data
        feature_corr_data = {}
        for feat in feature_correlation_data[internal_state][prefix]:
            for time_window in feature_correlation_data[internal_state][prefix][feat]:
                if feat not in feature_corr_data:
                    feature_corr_data[feat] = []
                feature_corr_data[feat].append((time_window, feature_correlation_data[internal_state][prefix][feat][time_window]['r']))
        
        # Sort features by average correlation
        avg_correlations = {}
        for feat, values in feature_corr_data.items():
            avg_correlations[feat] = np.mean([v[1] for v in values])
        
        # Get top N features by absolute correlation
        sorted_features = sorted(avg_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:NUM_TOP_FEATURES]
        top_corr_features = [f[0] for f in sorted_features]
        
        # Plot correlation over time for top features
        plt.figure(figsize=(12, 8))
        
        for i, feat in enumerate(top_corr_features):
            if feat in feature_corr_data:
                data_points = feature_corr_data[feat]
                data_points.sort(key=lambda x: x[0])  # Sort by time window
                times = [p[0] for p in data_points]
                correlations = [p[1] for p in data_points]
                
                plt.plot(times, correlations, marker='o', linewidth=2, label=feat, color=COLORS[i % len(COLORS)])
        
        plt.title(f"{internal_state} - Top Features Correlation with Target\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)}", fontsize=18)
        plt.xlabel("Time Window (minutes)", fontsize=16)
        plt.ylabel("Pearson Correlation (r)", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_feature_correlations.png"), dpi=300)
        plt.close()
    
    # Create correlation matrix for all feature selection frequency
    for prefix in RESULTS_PREFIX_LIST:
        # Process feature selection data across time windows
        selection_data = {}
        for t in time_list:
            features_selected = []
            for feat in feature_list:
                if feature_selection_frequency[internal_state][prefix].get(feat, 0) > 0:
                    features_selected.append(feat)
            
            selection_data[t] = features_selected
        
        # Create CSV with feature selection data
        selection_df_data = {'Time_Window': time_list}
        for feat in feature_list:
            selection_df_data[feat] = [
                (feature_selection_frequency[internal_state][prefix].get(feat, 0) / total_models) * 100
                for t in time_list
            ]
        
        selection_df = pd.DataFrame(selection_df_data)
        selection_df.to_csv(os.path.join(csv_folder, f"{internal_state}_{prefix}_feature_selection_frequency.csv"), index=False)
        
        # Create alpha value distribution across time windows
        alpha_data = defaultdict(list)
        
        for file in csv_files:
            internal_state_file, time_window_file, prefix_file = parse_filename(file)
            if internal_state_file == internal_state and prefix_file == prefix and time_window_file in time_list:
                # Load the data
                df = pd.read_csv(os.path.join(patient_folder, file))
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                # Run LassoCV
                loo = LeaveOneOut()
                for train_idx, test_idx in loo.split(X):
                    model = LassoCV(alphas=ALPHAS, cv=LeaveOneOut()).fit(X[train_idx], y[train_idx])
                    alpha_data[time_window_file].append(model.alpha_)
        
        # Plot alpha distribution across time windows
        alpha_means = [np.mean(alpha_data[t]) for t in time_list if t in alpha_data and alpha_data[t]]
        alpha_stds = [np.std(alpha_data[t]) for t in time_list if t in alpha_data and alpha_data[t]]
        valid_times = [t for t in time_list if t in alpha_data and alpha_data[t]]
        
        if valid_times:
            plt.figure(figsize=(12, 6))
            plt.errorbar(valid_times, alpha_means, yerr=alpha_stds, marker='o', markersize=8, 
                        linewidth=2, elinewidth=1, capsize=5, color=COLORS[4])
            
            plt.title(f"{internal_state} - Alpha Regularization Parameter Across Time Windows\n{PREFIX_DISPLAY_MAP.get(prefix, prefix)}", fontsize=18)
            plt.xlabel("Time Window (minutes)", fontsize=16)
            plt.ylabel("Mean Alpha Value (± SD)", fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(overview_folder, f"{internal_state}_{prefix}_alpha_over_time.png"), dpi=300)
            plt.close()
            
            # Save alpha data as CSV
            alpha_df_data = {'Time_Window': valid_times, 'Mean_Alpha': alpha_means, 'Std_Alpha': alpha_stds}
            alpha_df = pd.DataFrame(alpha_df_data)
            alpha_df.to_csv(os.path.join(csv_folder, f"{internal_state}_{prefix}_alpha_values.csv"), index=False)

# Create a summary report with key findings
for internal_state in summary_results:
    overview_folder = os.path.join(RESULTS_OUTPUT_PATH, internal_state, 'Overview')
    
    # Create a summary figure with 4 key plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    
    # Plot 1: Performance metrics across time windows for all prefix types
    ax1 = axes[0, 0]
    for i, prefix in enumerate(RESULTS_PREFIX_LIST):
        time_list = sorted(summary_results[internal_state][prefix].keys())
        r_means = [summary_results[internal_state][prefix][t]['pearson_r']['mean'] for t in time_list]
        r_lower = [summary_results[internal_state][prefix][t]['pearson_r']['ci_lower'] for t in time_list]
        r_upper = [summary_results[internal_state][prefix][t]['pearson_r']['ci_upper'] for t in time_list]
        
        ax1.plot(time_list, r_means, marker='o', markersize=8, linewidth=2, 
                label=f"{PREFIX_DISPLAY_MAP.get(prefix, prefix)}", color=COLORS[i % len(COLORS)])
        ax1.fill_between(time_list, r_lower, r_upper, alpha=0.2, color=COLORS[i % len(COLORS)])
    
    ax1.set_title("Pearson R Performance Across Time Windows", fontsize=16)
    ax1.set_xlabel("Time Window (minutes)", fontsize=14)
    ax1.set_ylabel("Pearson Correlation (r)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=12)
    
    # Plot 2: Top feature importance for the best performing time window
    ax2 = axes[0, 1]
    
    # Find best time window and prefix
    best_r = -1
    best_prefix = None
    best_time = None
    
    for prefix in summary_results[internal_state]:
        for time_window in summary_results[internal_state][prefix]:
            r_val = summary_results[internal_state][prefix][time_window]['pearson_r']['mean']
            if r_val > best_r:
                best_r = r_val
                best_prefix = prefix
                best_time = time_window
    
    if best_prefix and best_time:
        # Create list of feature importances from permutation impact
        feature_imp = []
        for feat in feature_list:
            if feat in permutation_impact_data[internal_state] and best_time in permutation_impact_data[internal_state][feat]:
                feature_imp.append((feat, permutation_impact_data[internal_state][feat][best_time]))
        
        # Sort and get top N
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        top_10_features = feature_imp[:NUM_TOP_FEATURES]
        
        # Plot
        feature_names = [f[0] for f in top_10_features]
        importance_vals = [f[1] for f in top_10_features]
        
        y_pos = np.arange(len(feature_names))
        ax2.barh(y_pos, importance_vals, color=COLORS[0])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(feature_names)
        ax2.set_title(f"Top {NUM_TOP_FEATURES} Features (Best Performance: {best_time}min, {PREFIX_DISPLAY_MAP.get(best_prefix, best_prefix)})", fontsize=16)
        ax2.set_xlabel("Permutation Impact (ΔR)", fontsize=14)
    
    # Plot 3: Feature selection frequency heatmap for best prefix
    ax3 = axes[1, 0]
    
    if best_prefix:
        # Get features with highest selection frequency
        feature_freq = []
        for feat in feature_list:
            total_freq = sum(feature_selection_frequency[internal_state][best_prefix].get(feat, 0) for t in time_list)
            if total_freq > 0:
                feature_freq.append((feat, total_freq))
        
        # Sort and get top 15
        feature_freq.sort(key=lambda x: x[1], reverse=True)
        top_15_freq_features = [f[0] for f in feature_freq[:15]]
        
        # Create mini heatmap
        mini_matrix = np.zeros((len(top_15_freq_features), len(time_list)))
        for i, feat in enumerate(top_15_freq_features):
            for j, t in enumerate(time_list):
                freq = feature_selection_frequency[internal_state][best_prefix].get(feat, 0)
                mini_matrix[i, j] = (freq / total_models) * 100
        
        # Plot
        sns.heatmap(mini_matrix, cmap='viridis', xticklabels=time_list, yticklabels=top_15_freq_features,
                   cbar_kws={'label': '% of Bootstrap Models'}, ax=ax3, linewidths=0.5)
        ax3.set_title(f"Feature Selection Frequency ({PREFIX_DISPLAY_MAP.get(best_prefix, best_prefix)})", fontsize=16)
        ax3.set_xlabel("Time Window (minutes)", fontsize=14)
        ax3.set_ylabel("Feature", fontsize=14)
    
    # Plot 4: Multiple metrics for best prefix
    ax4 = axes[1, 1]
    
    if best_prefix:
        metrics_to_plot = [
            ('pearson_r', 'Pearson R', COLORS[0]),
            ('spearman_r', 'Spearman R', COLORS[1]),
            ('r2', 'R²', COLORS[2])
        ]
        
        for metric, label, color in metrics_to_plot:
            means = [summary_results[internal_state][best_prefix][t][metric]['mean'] for t in time_list]
            ax4.plot(time_list, means, marker='o', markersize=6, linewidth=2, label=label, color=color)
        
        ax4.set_title(f"Multiple Metrics ({PREFIX_DISPLAY_MAP.get(best_prefix, best_prefix)})", fontsize=16)
        ax4.set_xlabel("Time Window (minutes)", fontsize=14)
        ax4.set_ylabel("Metric Value", fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=12)
    
    # Overall title
    plt.suptitle(f"{internal_state} Analysis Summary\nPatient: {PAT_NOW}", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(overview_folder, f"{internal_state}_summary_dashboard.png"), dpi=300)
    plt.close()

# Create final summary CSV with best performances for each configuration
final_summary_data = {
    'Internal_State': [],
    'Prefix': [],
    'Best_Time_Window': [],
    'Pearson_R': [],
    'Spearman_R': [],
    'R2': [],
    'RMSE': [],
    'MAE': [],
    'Top_Feature_1': [],
    'Top_Feature_2': [],
    'Top_Feature_3': []
}

for internal_state in summary_results:
    for prefix in summary_results[internal_state]:
        # Find best time window by Pearson R
        best_r = -1
        best_time = None
        
        for time_window in summary_results[internal_state][prefix]:
            r_val = summary_results[internal_state][prefix][time_window]['pearson_r']['mean']
            if r_val > best_r:
                best_r = r_val
                best_time = time_window
        
        if best_time:
            # Get top features for this configuration
            feature_imp = []
            for feat in feature_list:
                if feat in permutation_impact_data[internal_state] and best_time in permutation_impact_data[internal_state][feat]:
                    feature_imp.append((feat, permutation_impact_data[internal_state][feat][best_time]))
            
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            top_3_features = feature_imp[:3] if len(feature_imp) >= 3 else feature_imp + [('N/A', 0)] * (3 - len(feature_imp))
            
            # Get metrics
            metrics = summary_results[internal_state][prefix][best_time]
            
            # Add to summary data
            final_summary_data['Internal_State'].append(internal_state)
            final_summary_data['Prefix'].append(prefix)
            final_summary_data['Best_Time_Window'].append(best_time)
            final_summary_data['Pearson_R'].append(metrics['pearson_r']['mean'])
            final_summary_data['Spearman_R'].append(metrics['spearman_r']['mean'])
            final_summary_data['R2'].append(metrics['r2']['mean'])
            final_summary_data['RMSE'].append(metrics['rmse']['mean'])
            final_summary_data['MAE'].append(metrics['mae']['mean'])
            final_summary_data['Top_Feature_1'].append(f"{top_3_features[0][0]} ({top_3_features[0][1]:.3f})")
            final_summary_data['Top_Feature_2'].append(f"{top_3_features[1][0]} ({top_3_features[1][1]:.3f})")
            final_summary_data['Top_Feature_3'].append(f"{top_3_features[2][0]} ({top_3_features[2][1]:.3f})")

# Save final summary CSV
final_summary_df = pd.DataFrame(final_summary_data)
final_summary_df.to_csv(os.path.join(RESULTS_OUTPUT_PATH, f"{PAT_NOW}_final_summary.csv"), index=False)

print(f"Analysis complete. Results saved to {RESULTS_OUTPUT_PATH}")