"""
Assumes main_NoBootstraps_Supplement.py has already been run.

This script will re-plot the results for the given method and internal state, with better style.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configuration
INPUT_BASE_PATH = '/Users/jaygopal/NotiCloudDrive/all_plots'
OUTPUT_BASE_PATH = os.path.join(INPUT_BASE_PATH, 'updated_plots')
PREDICTION_TYPES = ['MoodPrediction', 'AnxietyPrediction', 'DepressionPrediction']
TIME_WINDOWS = list(range(30, 241, 30))

# Method display names
METHOD_DISPLAY_NAMES = {
    'OF_L_': 'OpenFace',
    'OGAUHSE_L_': 'FaceDx'
}

# Set up the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def create_time_window_colors():
    """Create a gradient of grayscale colors for time windows"""
    n_windows = len(TIME_WINDOWS)
    # Create a gradient from light gray to dark gray (starting at 0.3 instead of 0)
    colors = [f'#{int(255 * (0.3 + 0.7 * i/n_windows)):02x}{int(255 * (0.3 + 0.7 * i/n_windows)):02x}{int(255 * (0.3 + 0.7 * i/n_windows)):02x}' 
              for i in range(n_windows)]
    return dict(zip(TIME_WINDOWS, colors))

def plot_bar_with_significance(ax, data, time_windows, colors, title, ylabel):
    """Plot a single bar plot with significance indicators"""
    # Plot bars
    bars = ax.bar([str(tw) for tw in time_windows], 
                  data['mean_score'],
                  yerr=data['sem_score'],
                  color=[colors[tw] for tw in time_windows],
                  capsize=5)
    
    # Add significance bars for p < 0.05
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]
    significance_height = y_min + 0.9 * (y_max - y_min)
    
    significant_bars = 0
    for i, pval in enumerate(data['mean_pval']):
        if pval < 0.05:
            ax.plot([i-0.4, i+0.4], [significance_height, significance_height], 
                   color='red', linewidth=2)
            significant_bars += 1
    
    ax.set_title(title)
    ax.set_xlabel("Time Window (min)")
    ax.set_ylabel(ylabel)
    
    # Ensure y-axis includes negative values
    current_ylim = ax.get_ylim()
    ax.set_ylim(min(current_ylim[0], -0.1), max(current_ylim[1], 0.1))
    
    return significant_bars

def create_legend_figure(colors):
    """Create a separate figure with the time window color legend"""
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Create a simple box for each time window
    for i, (tw, color) in enumerate(colors.items()):
        # Create a rectangle for the color
        rect = plt.Rectangle((0.1, 0.9 - i*0.1), 0.2, 0.08, 
                           facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        # Add the time window label
        ax.text(0.35, 0.9 - i*0.1 + 0.04, f'{tw} min', 
                verticalalignment='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    plt.tight_layout()
    return fig

def get_plot_title(file_path, pred_type):
    """Generate appropriate title based on file name and prediction type"""
    filename = os.path.basename(file_path).replace('_scores.csv', '')
    
    # Extract components
    is_binary = 'binary' in filename
    is_limited = 'limited' in filename
    
    # Get the method name
    method = None
    for m in METHOD_DISPLAY_NAMES:
        if m in filename:
            method = METHOD_DISPLAY_NAMES[m]
            break
    
    # Get the state from the prediction type folder
    state = pred_type.replace('Prediction', '')
    
    # Build the title
    title_parts = []
    if method:
        title_parts.append(method)
    title_parts.append(state)  # Always include the state
    if is_binary:
        title_parts.append("Binary")
    if is_limited:
        title_parts.append("Smile Features")
    
    return " - ".join(title_parts)

def get_lopo_plot_title(file_path, pred_type):
    """Generate title for LOPO plots: method, state, and 'Leave One Patient Out'"""
    filename = os.path.basename(file_path).replace('_scores.csv', '').replace('_lopo', '')
    is_binary = 'binary' in filename
    is_limited = 'limited' in filename

    # Get the method name
    method = None
    for m in METHOD_DISPLAY_NAMES:
        if m in filename:
            method = METHOD_DISPLAY_NAMES[m]
            break

    # Get the state from the prediction type folder
    state = pred_type.replace('Prediction', '')

    title_parts = []
    if method:
        title_parts.append(method)
    title_parts.append(state)
    if is_binary:
        title_parts.append("Binary")
    if is_limited:
        title_parts.append("Smile Features")
    title_parts.append("Leave One Patient Out")
    return " - ".join(title_parts)

def process_csv_file(file_path, colors, output_path, pred_type):
    """Process a single CSV file and create the replotted figure"""
    # Read the data
    data = pd.read_csv(file_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Determine if this is a binary (AUC) or continuous (R) plot
    is_binary = 'binary' in file_path
    ylabel = "AUC" if is_binary else "Pearson r"
    
    # Get the title
    title = get_plot_title(file_path, pred_type)
    
    # Plot the data and get number of significant bars
    significant_bars = plot_bar_with_significance(ax, data, TIME_WINDOWS, colors, title, ylabel)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return significant_bars

def process_lopo_csv_file(file_path, colors, output_path, pred_type):
    """Process a single LOPO CSV file and create the replotted figure"""
    # Read the LOPO scores data (matrix format: rows=patients, columns=time_windows)
    scores_data = pd.read_csv(file_path, index_col='patient_id')
    
    # Read the corresponding p-values file
    pvals_file_path = file_path.replace('_scores.csv', '_pvals.csv')
    if os.path.exists(pvals_file_path):
        pvals_data = pd.read_csv(pvals_file_path, index_col='patient_id')
    else:
        print(f"Warning: P-values file not found: {pvals_file_path}")
        pvals_data = None
    
    # Calculate means and standard errors across patients for each time window
    mean_scores = scores_data.mean(axis=0, skipna=True)
    sem_scores = scores_data.std(axis=0, skipna=True) / np.sqrt(scores_data.notna().sum(axis=0))
    
    # Calculate mean p-values across patients for each time window
    if pvals_data is not None:
        mean_pvals = pvals_data.mean(axis=0, skipna=True)
    else:
        mean_pvals = pd.Series([np.nan] * len(TIME_WINDOWS), index=TIME_WINDOWS)
    
    # Create the data structure expected by plot_bar_with_significance
    plot_data = pd.DataFrame({
        'mean_score': mean_scores.values,
        'sem_score': sem_scores.values,
        'mean_pval': mean_pvals.values
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Determine if this is a binary (AUC) or continuous (R) plot
    is_binary = 'binary' in file_path
    ylabel = "AUC" if is_binary else "Pearson r"
    
    # Get the title
    title = get_lopo_plot_title(file_path, pred_type)
    
    # Plot the data and get number of significant bars
    significant_bars = plot_bar_with_significance(ax, plot_data, TIME_WINDOWS, colors, title, ylabel)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return significant_bars

def main():
    # Create output directory structure
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    for pred_type in PREDICTION_TYPES:
        os.makedirs(os.path.join(OUTPUT_BASE_PATH, pred_type), exist_ok=True)
    
    # Create color mapping for time windows
    time_window_colors = create_time_window_colors()
    
    # Create and save the legend figure
    legend_fig = create_legend_figure(time_window_colors)
    legend_fig.savefig(os.path.join(OUTPUT_BASE_PATH, 'time_window_legend.png'), dpi=300, bbox_inches='tight')
    plt.close(legend_fig)
    
    # Track total significant bars
    total_significant_bars = 0
    
    # Process each prediction type
    for pred_type in PREDICTION_TYPES:
        input_path = os.path.join(INPUT_BASE_PATH, pred_type)
        output_path = os.path.join(OUTPUT_BASE_PATH, pred_type)
        
        # Find all CSV files (excluding LOPO files)
        csv_files = [f for f in os.listdir(input_path) 
                    if f.endswith('_scores.csv') and 'lopo' not in f.lower()]
        
        for csv_file in csv_files:
            input_file_path = os.path.join(input_path, csv_file)
            output_file_path = os.path.join(output_path, f'replotted_{csv_file.replace(".csv", ".png")}')
            
            # Process and save the figure, track significant bars
            significant_bars = process_csv_file(input_file_path, time_window_colors, output_file_path, pred_type)
            total_significant_bars += significant_bars
            print(f"Processed: {csv_file}")
    
    print(f"\nTotal number of significant bars (p < 0.05) across all plots: {total_significant_bars}")

def main_lopo():
    # Create output directory structure for LOPO plots
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    for pred_type in PREDICTION_TYPES:
        os.makedirs(os.path.join(OUTPUT_BASE_PATH, pred_type), exist_ok=True)
    
    # Create color mapping for time windows
    time_window_colors = create_time_window_colors()
    
    # Track total significant bars
    total_significant_bars = 0

    # Process each prediction type
    for pred_type in PREDICTION_TYPES:
        input_path = os.path.join(INPUT_BASE_PATH, pred_type)
        output_path = os.path.join(OUTPUT_BASE_PATH, pred_type)
        
        # Find all LOPO CSV files
        csv_files = [f for f in os.listdir(input_path) 
                    if f.endswith('_scores.csv') and 'lopo' in f.lower()]
        
        for csv_file in csv_files:
            input_file_path = os.path.join(input_path, csv_file)
            output_file_path = os.path.join(output_path, f'replotted_{csv_file.replace(".csv", ".png")}')
            
            # Process and save the figure, track significant bars
            significant_bars = process_lopo_csv_file(input_file_path, time_window_colors, output_file_path, pred_type)
            total_significant_bars += significant_bars
            print(f"Processed LOPO: {csv_file}")
    
    print(f"\nTotal number of significant bars (p < 0.05) across all LOPO plots: {total_significant_bars}")

if __name__ == "__main__":
    #main() 
    main_lopo()
