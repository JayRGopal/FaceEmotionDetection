"""
Comprehensive Mood Prediction Analysis Script
- Predicts mood using OpenFace and FaceDx Complete features
- Performs analyses at individual and group levels
- Tests with separate and concatenated time windows
- Runs statistical significance testing
- Creates binary classification models for low/high mood
"""
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut, train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

# ---------------- CONFIGURATION ---------------- #
FEATURE_SAVE_FOLDER = '/home/jgopal/Desktop/FaceEmotionDetection/temp_outputs/'
RESULTS_OUTPUT_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_May_2025/MoodPrediction'
ALPHAS = np.linspace(0.1, 10.0, 30)
TIME_WINDOWS = list(range(30, 241, 30))  # Starting at 30 and going up in intervals of 30
METHODS = ['OF_L_', 'OGAUHSE_L_']  # OpenFace and FaceDx Complete
NUM_NULL_PERMUTATIONS = 100  # Number of permutations for null distribution
INTERNAL_STATE = 'Mood'
LIMITED_FEATURES_SUBSTRINGS = ["AU10", "AU12", "AU25", "AU27", "AU6", "AU7"]

# Create output directories
os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)
for method in METHODS:
    os.makedirs(os.path.join(RESULTS_OUTPUT_PATH, method), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_OUTPUT_PATH, f"{method}_LIMITED"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_OUTPUT_PATH, f"{method}_BINARY"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_OUTPUT_PATH, f"{method}_LIMITED_BINARY"), exist_ok=True)

# Create a mapping to decode feature prefixes to more readable names for plots
METHOD_DISPLAY_MAP = {
    'OF_L_': 'OpenFace',
    'OGAUHSE_L_': 'FaceDx Complete'
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

# Function to remove duplicate columns
def remove_duplicate_features(df):
    feature_cols = df.columns[:-1]  # All but the last column (target)
    unique_cols = []
    seen = set()
    
    for col in feature_cols:
        if col not in seen:
            seen.add(col)
            unique_cols.append(col)
        else:
            print(f"  Dropping duplicate column: {col}")
    
    # Keep only unique columns plus the target column
    unique_cols.append(df.columns[-1])  # Add the target column
    return df[unique_cols]

# Function to filter dataframe to limited features
def filter_limited_features(df):
    # Get feature columns (all except the last one which is the target)
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]
    
    # Filter columns that contain any of the specified substrings
    matching_cols = [col for col in feature_cols if any(substr in col for substr in LIMITED_FEATURES_SUBSTRINGS)]
    
    # Add the target column back
    matching_cols.append(target_col)
    
    # Return filtered dataframe
    filtered_df = df[matching_cols]
    print(f"  Filtered from {len(feature_cols)} to {len(matching_cols)-1} features based on AU criteria")
    return filtered_df

# Function to binarize mood scores
def binarize_mood(df):
    """Binarize mood scores around median. Returns None if all scores would be same value.
    Args:
        df: DataFrame with mood scores in last column
    Returns:
        DataFrame with binarized scores, or None if all scores would be identical
    """
    mood_col = df.columns[-1]
    
    # Debug print original data
    print(f"  DEBUG - Pre-binarization check:")
    print(f"    Column name: {mood_col}")
    print(f"    Data type: {df[mood_col].dtype}")
    print(f"    Values: {df[mood_col].values}")
    
    # First check if we already have binary values
    unique_vals = df[mood_col].unique()
    if set(unique_vals).issubset({0, 1, -1}):
        print(f"  WARNING: Mood scores already appear to be binary/categorical: {unique_vals}")
        return None
    
    # Ensure we're working with numeric data
    try:
        df[mood_col] = pd.to_numeric(df[mood_col], errors='raise')
    except Exception as e:
        print(f"  ERROR: Could not convert mood scores to numeric: {e}")
        return None
    
    median_mood = df[mood_col].median()
    binary_values = (df[mood_col] > median_mood)
    
    # Check if all values would map to same binary value
    if binary_values.nunique() == 1:
        print(f"  All mood scores would map to same binary value (all {'above' if binary_values.iloc[0] else 'below'} median {median_mood}). Skipping binarization.")
        return None
    
    # Create new dataframe to avoid modifying original
    new_df = df.copy()
    new_df[mood_col] = binary_values.astype(int)
    
    # Final verification
    final_unique = new_df[mood_col].unique()
    if not set(final_unique).issubset({0, 1}):
        print(f"  ERROR: Unexpected values after binarization: {final_unique}")
        return None
    
    # Check minimum samples per class
    value_counts = new_df[mood_col].value_counts()
    if (value_counts < 2).any():
        print(f"  Insufficient samples per class (need at least 2): {value_counts}")
        return None
        
    print(f"  Successful binarization:")
    print(f"    Value counts: {value_counts}")
    
    return new_df

def inclusion_criteria(mood_scores):
    """Check if mood scores meet inclusion criteria.
    Args:
        mood_scores: Series/array of mood scores (0-10 scale)
    Returns:
        bool: True if criteria met, False otherwise
    """
    if len(mood_scores) < 5:  # Criterion 1: ≥5 reports
        return False
    
    score_range = mood_scores.max() - mood_scores.min()
    if score_range < 5:  # Criterion 2: range ≥5 (50% of 0-10 scale)
        return False
        
    unique_perms = len(mood_scores.unique())
    if unique_perms < 3:  # Criterion 3: ≥3 unique scores
        return False
        
    return True


# Function to calculate null distribution and p-value
def calculate_null_distribution(X, y_true, y_pred, metric_func, model_func, is_classification=False):
    null_values = []
    
    # For regression, we need both y_true and y_pred
    # For classification, we additionally need X to retrain the model
    
    for _ in range(NUM_NULL_PERMUTATIONS):
        if is_classification:
            # For classification, shuffle the true labels and refit the model
            y_shuffled = np.random.permutation(y_true)
            model = model_func()
            model.fit(X, y_shuffled)
            y_shuffled_pred = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
            null_value = metric_func(y_shuffled, y_shuffled_pred)
        else:
            # For regression, just shuffle the true values and compare with predictions
            y_shuffled = np.random.permutation(y_true)
            null_value = metric_func(y_shuffled, y_pred)
        
        null_values.append(null_value)
    
    actual_value = metric_func(y_true, y_pred)
    
    if is_classification or metric_func == roc_auc_score:
        # For metrics where higher is better (accuracy, AUC, correlation)
        perm_p_value = (np.sum(np.array(null_values) >= actual_value) + 1) / (len(null_values) + 1)
    else:
        # For metrics where lower is better (RMSE, etc.)
        perm_p_value = (np.sum(np.array(null_values) <= actual_value) + 1) / (len(null_values) + 1)
    
    return null_values, actual_value, perm_p_value

# Function to plot null distribution
def plot_null_distribution(null_values, actual_value, p_value, metric_name, title, output_path):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(null_values, fill=True, color=COLORS[3], alpha=0.5, label="Null Distribution")
    plt.axvline(actual_value, color='red', linestyle='--', linewidth=2, label=f"Actual {metric_name}={actual_value:.3f}")
    plt.title(title, fontsize=16)
    plt.xlabel(metric_name, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    
    # Add p-value annotation
    significance_text = "Significant (p<0.05)" if p_value < 0.05 else "Not significant (p≥0.05)"
    plt.annotate(f"p-value: {p_value:.3f}\n{significance_text}", xy=(0.95, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                ha='right', va='top', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# Function to run regression analysis for a single patient
def analyze_single_patient(patient_id, patient_data, time_windows, method, output_folder, 
                          is_limited=False, is_binary=False):
    print(f"\nAnalyzing patient {patient_id} with method {METHOD_DISPLAY_MAP.get(method, method)}")
    
    # Set up subfolder
    suffix = ""
    if is_limited:
        suffix += "_LIMITED"
    if is_binary:
        suffix += "_BINARY"
    patient_folder = os.path.join(output_folder, f"{method}{suffix}", patient_id)
    os.makedirs(patient_folder, exist_ok=True)
    
    # Initialize results storage
    results = []
    
    # Set up a multi-panel figure for this patient
    n_plots = len(time_windows)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # For each time window
    for tw_idx, time_window in enumerate(time_windows):
        if time_window not in patient_data:
            print(f"  No data for time window {time_window}")
            continue
        
        df = patient_data[time_window]
        
        if is_limited:
            df = filter_limited_features(df)
        
        if is_binary:
            binarized_df = binarize_mood(df)
            if binarized_df is None:
                print(f"  Skipping binary analysis for patient {patient_id} at time window {time_window} - invalid binarization")
                continue
            
            print("\nDEBUG - After binarization:")
            print(f"  Values in binarized_df: {binarized_df.iloc[:, -1].unique()}")
            df = binarized_df
        
        # Extract features and target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        print("\nDEBUG - After X,y split:")
        print(f"  y values: {np.unique(y)}")
        
        # Handle NaN values
        if np.isnan(X).any():
            print(f"  WARNING: NaNs found in features. Filling with 0s.")
            X = np.nan_to_num(X, nan=0.0)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("\nDEBUG - After standardization:")
        print(f"  y values: {np.unique(y)}")
        
        # Run leave-one-out cross-validation
        loo = LeaveOneOut()
        predictions = []
        actuals = []
        
        # Define model based on classification or regression
        if is_binary:
            model_func = lambda: LogisticRegressionCV(
                Cs=1/np.array(ALPHAS), 
                cv=loo,  # Use leave-one-out instead of fixed 5-fold CV
                penalty='l1', 
                solver='liblinear',
                random_state=42
            )
            model = model_func()
        else:
            model_func = lambda: LassoCV(
                alphas=ALPHAS, 
                cv=loo,  # Use leave-one-out instead of fixed 5-fold CV
                random_state=42
            )
            model = model_func()
        
        
        # Debug each LOO split
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"\nDEBUG - LOO split:")
            print(f"  Train set values: {np.unique(y_train)}")
            print(f"  Test set value: {y_test[0]}")
            
            if len(np.unique(y_train)) < 2:
                print(f"  DEBUG WARNING: Training set has only one class: {np.unique(y_train)}")
                continue
            
            print(f"  DEBUG - Before model fit:")
            print(f"    Train set shape: {X_train.shape}")
            print(f"    Train set classes: {np.unique(y_train)}")
            
            model.fit(X_train, y_train)
            
            if is_binary:
                y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            predictions.extend(y_pred)
            actuals.extend(y_test)
        
        # Calculate metrics
        if is_binary:
            metric_func = roc_auc_score
            metric_name = "AUC"
            score = metric_func(actuals, predictions) if len(np.unique(actuals)) > 1 else np.nan
        else:
            metric_func = pearsonr
            metric_name = "Pearson r"
            score, p_value = metric_func(actuals, predictions) if len(actuals) > 1 else (np.nan, np.nan)
            # Convert pearsonr function to a simple metric function for null distribution
            metric_func = lambda y_true, y_pred: pearsonr(y_true, y_pred)[0]
            
        # Null distribution
        if not np.isnan(score) and len(actuals) > 1:
            # Refit model on all data for null distribution testing
            model.fit(X, y)
            if is_binary:
                all_predictions = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
            else:
                all_predictions = model.predict(X)
                
            null_values, actual_value, perm_p_value = calculate_null_distribution(
                X, y, all_predictions, metric_func, model_func, is_binary
            )
            
            # Plot null distribution
            plot_title = (f"Statistical Significance - Patient {patient_id}\n"
                         f"{INTERNAL_STATE} | {METHOD_DISPLAY_MAP.get(method, method)} | {time_window} min")
            
            plot_null_distribution(
                null_values, actual_value, perm_p_value, metric_name,
                plot_title, os.path.join(patient_folder, f"null_dist_time_{time_window}.png")
            )
        else:
            perm_p_value = np.nan
            
        # Store results
        results.append({
            'time_window': time_window,
            'metric_value': score,
            'metric_name': metric_name,
            'p_value': perm_p_value,
            'significant': perm_p_value < 0.05 if not np.isnan(perm_p_value) else False
        })
        
        # Add subplot
        row = tw_idx // n_cols
        col = tw_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        if is_binary:
            # For binary classification, create a confusion matrix
            if len(np.unique(actuals)) > 1:
                binary_preds = np.array(predictions) > 0.5
                conf_matrix = confusion_matrix(actuals, binary_preds)
                
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f"Time: {time_window} min\nAUC: {score:.3f}" + 
                           (f"\np: {perm_p_value:.3f}*" if perm_p_value < 0.05 else f"\np: {perm_p_value:.3f}"))
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                ax.set_title(f"Time: {time_window} min\nNo data")
        else:
            # For regression, create a scatter plot
            ax.scatter(actuals, predictions)
            
            # Add regression line
            if len(actuals) > 1:
                z = np.polyfit(actuals, predictions, 1)
                p = np.poly1d(z)
                ax.plot(sorted(actuals), p(sorted(actuals)), 'r--')
                
                # Add identity line (perfect prediction)
                min_val = min(min(actuals), min(predictions))
                max_val = max(max(actuals), max(predictions))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4)
                
                ax.set_title(f"Time: {time_window} min\nr: {score:.3f}" + 
                           (f"\np: {perm_p_value:.3f}*" if perm_p_value < 0.05 else f"\np: {perm_p_value:.3f}"))
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                ax.set_title(f"Time: {time_window} min\nNo data")
    
    # Adjust layout and save the multi-panel figure
    plt.tight_layout()
    suffix_str = "_limited" if is_limited else ""
    suffix_str += "_binary" if is_binary else ""
    plt.savefig(os.path.join(output_folder, f"{method}{suffix}", f"{patient_id}_all_time_windows{suffix_str}.png"), dpi=300)
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, f"{method}{suffix}", f"{patient_id}_results{suffix_str}.csv"), index=False)
    
    return results

# Function to run leave-one-patient-out analysis
def analyze_leave_one_patient_out(all_patient_data, time_windows, method, output_folder, 
                                 is_limited=False, is_binary=False):
    print(f"\nRunning leave-one-patient-out analysis with method {METHOD_DISPLAY_MAP.get(method, method)}")
    
    # Set up subfolder
    suffix = ""
    if is_limited:
        suffix += "_LIMITED"
    if is_binary:
        suffix += "_BINARY"
    lopo_folder = os.path.join(output_folder, f"{method}{suffix}", "LeaveOneOut")
    os.makedirs(lopo_folder, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    
    # For each time window
    time_window_results = {}
    
    for time_window in time_windows:
        print(f"  Processing time window {time_window}")
        
        # Collect data for this time window across all patients
        patient_data_for_window = {}
        patient_ids = []
        
        for patient_id, patient_windows in all_patient_data.items():
            if time_window in patient_windows:
                df = patient_windows[time_window]
                
                if is_limited:
                    df = filter_limited_features(df)
                    
                if is_binary:
                    binarized_df = binarize_mood(df)
                    if binarized_df is None:
                        print(f"  Skipping binary analysis for patient {patient_id} at time window {time_window} - invalid binarization")
                        continue
                    
                    df = binarized_df
                
                patient_data_for_window[patient_id] = df
                patient_ids.append(patient_id)
        
        if len(patient_data_for_window) < 2:
            print(f"    Skipping time window {time_window} - not enough patients")
            continue
            
        print(f"    Found data for {len(patient_data_for_window)} patients")
        
        # Initialize results for this time window
        results = []
        
        # For each patient as test set
        for test_patient_id in patient_ids:
            test_df = patient_data_for_window[test_patient_id]
            
            # Combine all other patients' data for training
            train_dfs = [patient_data_for_window[pid] for pid in patient_ids if pid != test_patient_id]
            train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
            
            # Extract features and targets
            X_train = train_df.iloc[:, :-1].values
            y_train = train_df.iloc[:, -1].values
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values
            
            # Handle NaN values
            if np.isnan(X_train).any():
                print(f"    WARNING: NaNs found in training features for time window {time_window}. Filling with 0s.")
                X_train = np.nan_to_num(X_train, nan=0.0)
            if np.isnan(X_test).any():
                print(f"    WARNING: NaNs found in test features for time window {time_window}. Filling with 0s.")
                X_test = np.nan_to_num(X_test, nan=0.0)
                
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Define model based on classification or regression
            if is_binary:
                model = LogisticRegressionCV(
                    Cs=1/np.array(ALPHAS), 
                    cv=5, 
                    penalty='l1', 
                    solver='liblinear',
                    random_state=42
                )
                model_func = lambda: LogisticRegressionCV(
                    Cs=1/np.array(ALPHAS), 
                    cv=5, 
                    penalty='l1', 
                    solver='liblinear',
                    random_state=42
                )
            else:
                model = LassoCV(
                    alphas=ALPHAS, 
                    cv=5,
                    random_state=42
                )
                model_func = lambda: LassoCV(
                    alphas=ALPHAS, 
                    cv=5,
                    random_state=42
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict on test patient
            if is_binary:
                y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            if is_binary:
                metric_func = roc_auc_score
                metric_name = "AUC"
                if len(np.unique(y_test)) > 1:
                    score = metric_func(y_test, y_pred)
                else:
                    score = np.nan
            else:
                metric_func = pearsonr
                metric_name = "Pearson r"
                if len(y_test) > 1:
                    score, p_value = metric_func(y_test, y_pred)
                else:
                    score, p_value = np.nan, np.nan
                # Convert pearsonr function to a simple metric function for null distribution
                metric_func = lambda y_true, y_pred: pearsonr(y_true, y_pred)[0]
            
            # Run null distribution test (shuffle test patient's targets)
            if not np.isnan(score) and len(y_test) > 1:
                # Use the existing calculate_null_distribution function
                null_values, actual_value, perm_p_value = calculate_null_distribution(
                    X_test, y_test, y_pred, metric_func, model_func, is_classification=is_binary
                )
                
                # Plot null distribution
                plot_title = (f"Statistical Significance - Leave One Patient Out\n"
                             f"Test Patient: {test_patient_id} | {INTERNAL_STATE} | "
                             f"{METHOD_DISPLAY_MAP.get(method, method)} | {time_window} min")
                
                plot_null_distribution(
                    null_values, score, perm_p_value, metric_name,
                    plot_title, os.path.join(lopo_folder, f"null_dist_patient_{test_patient_id}_time_{time_window}.png")
                )
            else:
                perm_p_value = np.nan
            
            # Store results
            results.append({
                'patient_id': test_patient_id,
                'time_window': time_window,
                'metric_value': score,
                'metric_name': metric_name,
                'p_value': perm_p_value,
                'significant': perm_p_value < 0.05 if not np.isnan(perm_p_value) else False
            })
        
        # Save results for this time window
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(lopo_folder, f"lopo_results_time_{time_window}.csv"), index=False)
        
        # Create bar chart of per-patient performance for this time window
        plt.figure(figsize=(12, 8))
        
        # Sort patients by performance
        sorted_results = results_df.sort_values(by='metric_value')
        patients = sorted_results['patient_id'].values
        metric_values = sorted_results['metric_value'].values
        significant = sorted_results['significant'].values
        
        # Create color coding based on significance
        bar_colors = [COLORS[0] if sig else COLORS[1] for sig in significant]
        
        # Create bar chart
        bars = plt.bar(patients, metric_values, color=bar_colors)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f"Leave-One-Patient-Out Performance - {INTERNAL_STATE}\n"
                 f"{METHOD_DISPLAY_MAP.get(method, method)} | {time_window} min", fontsize=16)
        plt.xlabel("Patient", fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.xticks(rotation=45)
        
        # Add legend
        plt.legend([
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[0]),
            plt.Rectangle((0, 0), 1, 1, fc=COLORS[1])
        ], ['Significant (p<0.05)', 'Not significant'], fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(lopo_folder, f"per_patient_performance_time_{time_window}.png"), dpi=300)
        plt.close()
        
        # Store average metrics for this time window
        time_window_results[time_window] = {
            'mean_metric': results_df['metric_value'].mean(),
            'std_metric': results_df['metric_value'].std(),
            'significant_patients': results_df['significant'].sum(),
            'total_patients': len(results_df)
        }
        
        all_results.extend(results)
    
    # Create plot comparing performance across time windows
    plt.figure(figsize=(12, 8))
    
    time_windows_list = sorted(time_window_results.keys())
    mean_metrics = [time_window_results[tw]['mean_metric'] for tw in time_windows_list]
    std_metrics = [time_window_results[tw]['std_metric'] for tw in time_windows_list]
    
    plt.errorbar(time_windows_list, mean_metrics, yerr=std_metrics, 
                fmt='o-', color=COLORS[0], capsize=5)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time Window (minutes)', fontsize=14)
    plt.ylabel(f'Mean {metric_name}', fontsize=14)
    plt.title(f"{INTERNAL_STATE} - Leave-One-Patient-Out Performance Across Time Windows\n"
             f"{METHOD_DISPLAY_MAP.get(method, method)}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(lopo_folder, f"performance_across_time_windows.png"), dpi=300)
    plt.close()
    
    # Save all results to CSV
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(lopo_folder, "all_lopo_results.csv"), index=False)
    
    # Save summary results to CSV
    summary_rows = []
    for tw in time_windows_list:
        summary_rows.append({
            'Time_Window': tw,
            f'Mean_{metric_name}': time_window_results[tw]['mean_metric'],
            f'Std_{metric_name}': time_window_results[tw]['std_metric'],
            'Significant_Patients': time_window_results[tw]['significant_patients'],
            'Total_Patients': time_window_results[tw]['total_patients'],
            'Percent_Significant': (time_window_results[tw]['significant_patients'] / 
                                   time_window_results[tw]['total_patients']) * 100
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(lopo_folder, "summary_results.csv"), index=False)
    
    return summary_df

# Function to run concatenated time analysis
def analyze_concatenated_times(all_patient_data, time_windows, method, output_folder, 
                              is_limited=False, is_binary=False):
    print(f"\nRunning concatenated time window analysis with method {METHOD_DISPLAY_MAP.get(method, method)}")
    
    # Set up subfolder
    suffix = ""
    if is_limited:
        suffix += "_LIMITED"
    if is_binary:
        suffix += "_BINARY"
    concat_folder = os.path.join(output_folder, f"{method}{suffix}", "ConcatenatedTimeWindows")
    os.makedirs(concat_folder, exist_ok=True)
    
    # Create concatenated dataframes for each patient
    concatenated_data = {}
    
    for patient_id, patient_windows in all_patient_data.items():
        patient_df = None
        
        for time_window in sorted(patient_windows.keys()):
            df = patient_windows[time_window]
            
            if is_limited:
                df = filter_limited_features(df)
                
            if is_binary:
                binarized_df = binarize_mood(df)
                if binarized_df is None:
                    print(f"  Skipping binary analysis for patient {patient_id} at time window {time_window} - invalid binarization")
                    continue

                df = binarized_df
            
            # Rename columns to include time window
            feature_cols = df.columns[:-1]
            target_col = df.columns[-1]
            
            df_renamed = df.copy()
            df_renamed.columns = [f"{col}_time_{time_window}" if col in feature_cols else col 
                                 for col in df.columns]
            
            if patient_df is None:
                patient_df = df_renamed
            else:
                # Merge on target column to maintain alignment
                patient_df = pd.merge(patient_df, df_renamed, on=target_col, how='inner')
        
        if patient_df is not None and not patient_df.empty:
            concatenated_data[patient_id] = patient_df
    
    # Run analysis on concatenated data
    # Similar to previous individual patient analysis but using concatenated features
    results = []
    
    for patient_id, df in concatenated_data.items():
        print(f"  Processing patient {patient_id} with concatenated features")
        
        # Extract features and target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Handle NaN values
        if np.isnan(X).any():
            print(f"  WARNING: NaNs found in features for patient {patient_id}. Filling with 0s.")
            X = np.nan_to_num(X, nan=0.0)
            
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Run leave-one-out cross-validation
        loo = LeaveOneOut()
        predictions = []
        actuals = []
        
        # Define model based on classification or regression
        if is_binary:
            model = LogisticRegressionCV(
                Cs=1/np.array(ALPHAS), 
                cv=loo,  # Use leave-one-out instead of fixed 5-fold CV
                penalty='l1', 
                solver='liblinear',
                random_state=42
            )
            model_func = lambda: LogisticRegressionCV(
                Cs=1/np.array(ALPHAS), 
                cv=loo,  # Use leave-one-out instead of fixed 5-fold CV
                penalty='l1', 
                solver='liblinear',
                random_state=42
            )
        else:
            model = LassoCV(
                alphas=ALPHAS, 
                cv=loo,  # Use leave-one-out instead of fixed 5-fold CV
                random_state=42
            )
            model_func = lambda: LassoCV(
                alphas=ALPHAS, 
                cv=loo,  # Use leave-one-out instead of fixed 5-fold CV
                random_state=42
            )
        
        # Perform leave-one-out cross-validation
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            
            if is_binary:
                y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                
            predictions.append(y_pred[0])
            actuals.append(y_test[0])
        
        # Calculate metrics
        if is_binary:
            metric_func = roc_auc_score
            metric_name = "AUC"
            if len(np.unique(actuals)) > 1:
                score = metric_func(actuals, predictions)
            else:
                score = np.nan
        else:
            metric_func = pearsonr
            metric_name = "Pearson r"
            if len(actuals) > 1:
                score, p_value = metric_func(actuals, predictions)
            else:
                score, p_value = np.nan, np.nan
            # Convert pearsonr function to a simple metric function for null distribution
            metric_func = lambda y_true, y_pred: pearsonr(y_true, y_pred)[0]
        
        # Run null distribution test
        if not np.isnan(score) and len(actuals) > 1:
            # Use the existing calculate_null_distribution function
            null_values, actual_value, perm_p_value = calculate_null_distribution(
                X, y, predictions, metric_func, model_func, is_classification=is_binary
            )
            
            # Plot null distribution
            plot_title = (f"Statistical Significance - Concatenated Time Windows\n"
                         f"Patient: {patient_id} | {INTERNAL_STATE} | "
                         f"{METHOD_DISPLAY_MAP.get(method, method)}")
            
            plot_null_distribution(
                null_values, actual_value, perm_p_value, metric_name,
                plot_title, os.path.join(concat_folder, f"null_dist_patient_{patient_id}.png")
            )
        else:
            perm_p_value = np.nan
        
        # Create scatter plot of predictions vs actuals
        plt.figure(figsize=(10, 6))
        
        if is_binary:
            # For binary classification, create a confusion matrix
            if len(np.unique(actuals)) > 1:
                binary_preds = np.array(predictions) > 0.5
                conf_matrix = confusion_matrix(actuals, binary_preds)
                
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f"Concatenated Time Windows - Patient {patient_id}\n"
                         f"AUC: {score:.3f}" + 
                         (f"\np: {perm_p_value:.3f}*" if perm_p_value < 0.05 else f"\np: {perm_p_value:.3f}"))
            else:
                plt.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                plt.title(f"Concatenated Time Windows - Patient {patient_id}\nNo data")
        else:
            # For regression, create a scatter plot
            plt.scatter(actuals, predictions)
            
            # Add regression line
            if len(actuals) > 1:
                z = np.polyfit(actuals, predictions, 1)
                p = np.poly1d(z)
                plt.plot(sorted(actuals), p(sorted(actuals)), 'r--')
                
                # Add identity line (perfect prediction)
                min_val = min(min(actuals), min(predictions))
                max_val = max(max(actuals), max(predictions))
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4)
                
                plt.title(f"Concatenated Time Windows - Patient {patient_id}\n"
                         f"r: {score:.3f}" + 
                         (f"\np: {perm_p_value:.3f}*" if perm_p_value < 0.05 else f"\np: {perm_p_value:.3f}"))
            else:
                plt.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                plt.title(f"Concatenated Time Windows - Patient {patient_id}\nNo data")
        
        plt.tight_layout()
        plt.savefig(os.path.join(concat_folder, f"predictions_patient_{patient_id}.png"), dpi=300)
        plt.close()
        
        # Store results
        results.append({
            'patient_id': patient_id,
            'metric_value': score,
            'metric_name': metric_name,
            'p_value': perm_p_value,
            'significant': perm_p_value < 0.05 if not np.isnan(perm_p_value) else False,
            'num_features': X.shape[1],
            'num_samples': len(actuals)
        })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(concat_folder, "concatenated_results.csv"), index=False)
    
    # Create bar chart of per-patient performance
    plt.figure(figsize=(12, 8))
    
    # Sort patients by performance
    sorted_results = results_df.sort_values(by='metric_value')
    patients = sorted_results['patient_id'].values
    metric_values = sorted_results['metric_value'].values
    significant = sorted_results['significant'].values
    
    # Create color coding based on significance
    bar_colors = [COLORS[0] if sig else COLORS[1] for sig in significant]
    
    # Create bar chart
    bars = plt.bar(patients, metric_values, color=bar_colors)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Concatenated Time Windows Performance - {INTERNAL_STATE}\n"
             f"{METHOD_DISPLAY_MAP.get(method, method)}", fontsize=16)
    plt.xlabel("Patient", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend([
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[0]),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[1])
    ], ['Significant (p<0.05)', 'Not significant'], fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(concat_folder, "per_patient_performance.png"), dpi=300)
    plt.close()
    
    # Create summary statistics
    summary = {
        'mean_metric': results_df['metric_value'].mean(),
        'std_metric': results_df['metric_value'].std(),
        'significant_patients': results_df['significant'].sum(),
        'total_patients': len(results_df),
        'percent_significant': (results_df['significant'].sum() / len(results_df)) * 100 if len(results_df) > 0 else 0
    }
    
    # Save summary to CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(concat_folder, "summary_results.csv"), index=False)
    
    # ADDITION: Leave-one-patient-out analysis with concatenated features
    print(f"\nRunning leave-one-patient-out analysis with concatenated features")
    
    # Create subfolder for leave-one-patient-out with concatenated features
    lopo_concat_folder = os.path.join(concat_folder, "LeaveOneOut")
    os.makedirs(lopo_concat_folder, exist_ok=True)
    
    # Only proceed if we have enough patients
    if len(concatenated_data) < 2:
        print("  Not enough patients for leave-one-patient-out analysis")
        return results_df
    
    # Initialize results for leave-one-patient-out
    lopo_results = []
    
    # For each patient as test set
    for test_patient_id in concatenated_data.keys():
        print(f"  Using patient {test_patient_id} as test set")
        
        test_df = concatenated_data[test_patient_id]
        
        # Combine all other patients' data for training
        train_dfs = [concatenated_data[pid] for pid in concatenated_data.keys() if pid != test_patient_id]
        
        # Skip if no training data
        if not train_dfs:
            print(f"  No training data available for test patient {test_patient_id}")
            continue
            
        train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
        
        # Extract features and targets
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values
        
        # Handle NaN values
        if np.isnan(X_train).any():
            print(f"  WARNING: NaNs found in training features. Filling with 0s.")
            X_train = np.nan_to_num(X_train, nan=0.0)
        if np.isnan(X_test).any():
            print(f"  WARNING: NaNs found in test features. Filling with 0s.")
            X_test = np.nan_to_num(X_test, nan=0.0)
            
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Define model based on classification or regression
        if is_binary:
            model = LogisticRegressionCV(
                Cs=1/np.array(ALPHAS), 
                cv=5, 
                penalty='l1', 
                solver='liblinear',
                random_state=42
            )
            model_func = lambda: LogisticRegressionCV(
                Cs=1/np.array(ALPHAS), 
                cv=5, 
                penalty='l1', 
                solver='liblinear',
                random_state=42
            )
        else:
            model = LassoCV(
                alphas=ALPHAS, 
                cv=5,
                random_state=42
            )
            model_func = lambda: LassoCV(
                alphas=ALPHAS, 
                cv=5,
                random_state=42
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on test patient
        if is_binary:
            y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        if is_binary:
            metric_func = roc_auc_score
            metric_name = "AUC"
            if len(np.unique(y_test)) > 1:
                score = metric_func(y_test, y_pred)
            else:
                score = np.nan
        else:
            metric_func = pearsonr
            metric_name = "Pearson r"
            if len(y_test) > 1:
                score, p_value = metric_func(y_test, y_pred)
            else:
                score, p_value = np.nan, np.nan
            # Convert pearsonr function to a simple metric function for null distribution
            metric_func = lambda y_true, y_pred: pearsonr(y_true, y_pred)[0]
        
        # Run null distribution test (shuffle test patient's targets)
        if not np.isnan(score) and len(y_test) > 1:
            # Use the existing calculate_null_distribution function
            null_values, actual_value, perm_p_value = calculate_null_distribution(
                X_test, y_test, y_pred, metric_func, model_func, is_classification=is_binary
            )
            
            # Plot null distribution
            plot_title = (f"Statistical Significance - Leave One Patient Out (Concatenated Features)\n"
                         f"Test Patient: {test_patient_id} | {INTERNAL_STATE} | "
                         f"{METHOD_DISPLAY_MAP.get(method, method)}")
            
            plot_null_distribution(
                null_values, score, perm_p_value, metric_name,
                plot_title, os.path.join(lopo_concat_folder, f"null_dist_patient_{test_patient_id}.png")
            )
        else:
            perm_p_value = np.nan
        # Create scatter plot of predictions vs actuals
        plt.figure(figsize=(10, 6))
        
        if is_binary:
            # For binary classification, create a confusion matrix
            if len(np.unique(y_test)) > 1:
                binary_preds = np.array(y_pred) > 0.5
                conf_matrix = confusion_matrix(y_test, binary_preds)
                
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f"Leave-One-Patient-Out - Test Patient {test_patient_id}\n"
                         f"AUC: {score:.3f}" + 
                         (f"\np: {perm_p_value:.3f}*" if perm_p_value < 0.05 else f"\np: {perm_p_value:.3f}"))
            else:
                plt.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                plt.title(f"Leave-One-Patient-Out - Test Patient {test_patient_id}\nNo data")
        else:
            # For regression, create a scatter plot
            plt.scatter(y_test, y_pred)
            
            # Add regression line
            if len(y_test) > 1:
                z = np.polyfit(y_test, y_pred, 1)
                p = np.poly1d(z)
                plt.plot(sorted(y_test), p(sorted(y_test)), 'r--')
                
                # Add identity line (perfect prediction)
                min_val = min(min(y_test), min(y_pred))
                max_val = max(max(y_test), max(y_pred))
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4)
                
                plt.title(f"Leave-One-Patient-Out - Test Patient {test_patient_id}\n"
                         f"r: {score:.3f}" + 
                         (f"\np: {perm_p_value:.3f}*" if perm_p_value < 0.05 else f"\np: {perm_p_value:.3f}"))
            else:
                plt.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                plt.title(f"Leave-One-Patient-Out - Test Patient {test_patient_id}\nNo data")
        
        plt.tight_layout()
        plt.savefig(os.path.join(lopo_concat_folder, f"predictions_patient_{test_patient_id}.png"), dpi=300)
        plt.close()
        
        # Store results
        lopo_results.append({
            'test_patient_id': test_patient_id,
            'metric_value': score,
            'metric_name': metric_name,
            'p_value': perm_p_value,
            'significant': perm_p_value < 0.05 if not np.isnan(perm_p_value) else False,
            'num_train_patients': len(train_dfs),
            'num_train_samples': len(y_train),
            'num_test_samples': len(y_test)
        })
    
    # Save LOPO results to CSV
    lopo_results_df = pd.DataFrame(lopo_results)
    lopo_results_df.to_csv(os.path.join(lopo_concat_folder, "lopo_results.csv"), index=False)
    
    # Create bar chart of per-patient LOPO performance
    plt.figure(figsize=(12, 8))
    
    # Sort patients by performance
    sorted_lopo_results = lopo_results_df.sort_values(by='metric_value')
    patients = sorted_lopo_results['test_patient_id'].values
    metric_values = sorted_lopo_results['metric_value'].values
    significant = sorted_lopo_results['significant'].values
    
    # Create color coding based on significance
    bar_colors = [COLORS[0] if sig else COLORS[1] for sig in significant]
    
    # Create bar chart
    bars = plt.bar(patients, metric_values, color=bar_colors)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Leave-One-Patient-Out Performance - {INTERNAL_STATE}\n"
             f"{METHOD_DISPLAY_MAP.get(method, method)}", fontsize=16)
    plt.xlabel("Test Patient", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend([
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[0]),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS[1])
    ], ['Significant (p<0.05)', 'Not significant'], fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(lopo_concat_folder, "per_patient_lopo_performance.png"), dpi=300)
    plt.close()
    
    # Create summary statistics for LOPO
    lopo_summary = {
        'mean_metric': lopo_results_df['metric_value'].mean(),
        'std_metric': lopo_results_df['metric_value'].std(),
        'significant_patients': lopo_results_df['significant'].sum(),
        'total_patients': len(lopo_results_df),
        'percent_significant': (lopo_results_df['significant'].sum() / len(lopo_results_df)) * 100 if len(lopo_results_df) > 0 else 0
    }
    
    # Save LOPO summary to CSV
    lopo_summary_df = pd.DataFrame([lopo_summary])
    lopo_summary_df.to_csv(os.path.join(lopo_concat_folder, "lopo_summary_results.csv"), index=False)

# Main execution code
def main():
    # Load all data files
    print("Loading data files...")
    all_patient_data = {}
    
    # Iterate through patient subfolders
    for patient_folder in os.listdir(FEATURE_SAVE_FOLDER):
        patient_folder_path = os.path.join(FEATURE_SAVE_FOLDER, patient_folder)
        
        # Skip if not a directory
        if not os.path.isdir(patient_folder_path):
            continue
        
        # Extract patient ID from folder name
        patient_id = patient_folder
        
        # Process files within patient folder
        patient_data_loaded = False
        patient_meets_criteria = False
        
        for filename in os.listdir(patient_folder_path):
            if filename.endswith('.csv') and INTERNAL_STATE in filename:
                for method in METHODS:
                    if method in filename:
                        internal_state, time_window, prefix = parse_filename(filename)
                        
                        # Load data
                        file_path = os.path.join(patient_folder_path, filename)
                        df = pd.read_csv(file_path)
                        
                        # Check inclusion criteria only once per patient
                        if not patient_data_loaded:
                            # Get mood scores (last column)
                            mood_scores = df.iloc[:, -1]
                            patient_meets_criteria = inclusion_criteria(mood_scores)
                            patient_data_loaded = True
                            
                            if not patient_meets_criteria:
                                print(f"Patient {patient_id} does not meet inclusion criteria. Skipping.")
                                break
                        
                        # Skip this patient entirely if they don't meet criteria
                        if not patient_meets_criteria:
                            break
                        
                        # Remove duplicates if any
                        df = remove_duplicate_features(df)
                        
                        # Initialize patient data structure if needed
                        if patient_id not in all_patient_data:
                            all_patient_data[patient_id] = {}
                        
                        if time_window not in all_patient_data[patient_id]:
                            all_patient_data[patient_id][time_window] = {}
                        
                        all_patient_data[patient_id][time_window] = df
                
                # If patient doesn't meet criteria, skip processing more files
                if patient_data_loaded and not patient_meets_criteria:
                    break
    
    print(f"Loaded data for {len(all_patient_data)} patients who meet inclusion criteria")
    
    # Run all analyses
    for method in METHODS:
        # # Individual patient analysis (standard features)
        # for patient_id, patient_data in all_patient_data.items():
        #     analyze_single_patient(patient_id, patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH)
            
        # # Individual patient analysis (limited features)
        # for patient_id, patient_data in all_patient_data.items():
        #     analyze_single_patient(patient_id, patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, is_limited=True)
            
        # Individual patient analysis (binary classification)
        for patient_id, patient_data in all_patient_data.items():
            # Debug: Skip to problematic patient
            if patient_id != "S23_207":
                continue
            analyze_single_patient(patient_id, patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, is_binary=True)
            
        # Individual patient analysis (limited features, binary classification)
        for patient_id, patient_data in all_patient_data.items():
            analyze_single_patient(patient_id, patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, 
                                    is_limited=True, is_binary=True)
            
        # Leave-one-patient-out analysis
        analyze_leave_one_patient_out(all_patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH)
        analyze_leave_one_patient_out(all_patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, is_limited=True)
        analyze_leave_one_patient_out(all_patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, is_binary=True)
        analyze_leave_one_patient_out(all_patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, 
                                        is_limited=True, is_binary=True)
        
        # Concatenated time window analysis
        analyze_concatenated_times(all_patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH)
        analyze_concatenated_times(all_patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, is_limited=True)
        analyze_concatenated_times(all_patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, is_binary=True)
        analyze_concatenated_times(all_patient_data, TIME_WINDOWS, method, RESULTS_OUTPUT_PATH, 
                                    is_limited=True, is_binary=True)
    
    print("Analysis complete!")

# Run the main function if executed as a script
if __name__ == "__main__":
    main()