import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Paths
BASE_RESULTS_PATH = '/home/jgopal/NAS/Analysis/AudioFacialEEG/Results_Apr_2025/'
GROUP_OUTPUT_PATH = os.path.join(BASE_RESULTS_PATH, 'Group')
os.makedirs(GROUP_OUTPUT_PATH, exist_ok=True)

# Get list of patient folders (excluding Bootstrap folders)
patient_folders = [f for f in os.listdir(BASE_RESULTS_PATH) 
                  if os.path.isdir(os.path.join(BASE_RESULTS_PATH, f))
                  and not f.endswith('Bootstrap')
                  and not f == 'Group']

# Method mapping
METHOD_MAPPING = {
    'HSE': 'FaceDx Emo',
    'OGAU': 'FaceDx AU', 
    'OGAUHSE': 'FaceDx Complete',
    'OF': 'OpenFace'
}

# Plot colors and styles
PLOT_COLORS = {
    'FaceDx Emo': '#2ecc71',      # Green
    'FaceDx AU': '#e74c3c',       # Red
    'FaceDx Complete': '#3498db',  # Blue
    'OpenFace': '#9b59b6'         # Purple
}

PLOT_STYLES = {
    'FaceDx Emo': '-o',
    'FaceDx AU': '-s', 
    'FaceDx Complete': '-^',
    'OpenFace': '-D'
}

# Data structures to store results
time_window_results = defaultdict(lambda: defaultdict(list))
best_results = defaultdict(list)
patient_time_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
feature_correlation_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Process each patient's data
for patient in patient_folders:
    mood_path = os.path.join(BASE_RESULTS_PATH, patient, 'Mood', 'CSV_Results')
    if not os.path.exists(mood_path):
        continue
        
    csv_files = [f for f in os.listdir(mood_path) 
                 if f.startswith('Mood') and 'performance_over_time' in f]
    
    for csv_file in csv_files:
        # Determine method using METHOD_MAPPING
        method = None
        if 'OGAUHSE' in csv_file:
            method = METHOD_MAPPING['OGAUHSE']
        elif 'OGAU' in csv_file:
            method = METHOD_MAPPING['OGAU']
        elif 'HSE' in csv_file:
            method = METHOD_MAPPING['HSE']
        elif 'OF' in csv_file:
            method = METHOD_MAPPING['OF']
            
        if method is None:
            continue
        # Read and process CSV
        df = pd.read_csv(os.path.join(mood_path, csv_file))
        
        # Store results per time window
        for _, row in df.iterrows():
            pearson_r = max(0, row['Pearson_R'])  # Set negative correlations to 0
            time_window_results[method][row['Time_Window']].append(pearson_r)
            patient_time_results[patient][method][row['Time_Window']] = pearson_r
        
        # Store best result
        best_results[method].append(df['Pearson_R'].max())
    
    # Process feature correlation files
    for method_key, method_name in METHOD_MAPPING.items():
        for time_window in range(15, 241, 30):  # Assuming same time windows as main analysis
            corr_file = f"Mood_{method_key}_L__time_{time_window}_feature_correlations.csv"
            corr_path = os.path.join(mood_path, corr_file)
            
            if os.path.exists(corr_path):
                try:
                    corr_df = pd.read_csv(corr_path)
                    for _, row in corr_df.iterrows():
                        feature = row.get('Feature', '')
                        feature_type = row.get('Type', '')
                        correlation = row.get('Correlation', 0)
                        
                        # Convert empty correlations to 0
                        if pd.isna(correlation) or correlation == '':
                            correlation = 0
                        else:
                            correlation = float(correlation)
                        
                        # Store feature correlation data
                        if feature_type not in feature_correlation_data[method_name]:
                            feature_correlation_data[method_name][feature_type] = []
                        
                        feature_correlation_data[method_name][feature_type].append({
                            'Patient': patient,
                            'Feature': feature,
                            'Correlation': correlation,
                            'Time_Window': time_window
                        })
                except Exception as e:
                    print(f"Error processing {corr_path}: {e}")

# Plot 1: Time window performance curves
plt.figure(figsize=(12, 8))
for method in time_window_results:
    time_windows = sorted(time_window_results[method].keys())
    means = [np.mean(time_window_results[method][tw]) for tw in time_windows]
    sems = [np.std(time_window_results[method][tw])/np.sqrt(len(time_window_results[method][tw])) 
            for tw in time_windows]
    
    plt.plot(time_windows, means, PLOT_STYLES[method], label=method, color=PLOT_COLORS[method])
    plt.fill_between(time_windows, 
                     [m-s for m,s in zip(means,sems)],
                     [m+s for m,s in zip(means,sems)], 
                     alpha=0.2,
                     color=PLOT_COLORS[method])

plt.xlabel('Time Window (minutes)')
plt.ylabel('Pearson Correlation (r)')
plt.title('Group-Level Decoding Performance Over Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(GROUP_OUTPUT_PATH, 'group_performance_over_time.png'))
plt.close()

# Plot 2: Box plot of best performances
plt.figure(figsize=(10, 6))
data = []
labels = []
for method in best_results:
    data.append(best_results[method])
    labels.extend([method] * len(best_results[method]))

df_plot = pd.DataFrame({
    'Method': labels,
    'Best Pearson R': [item for sublist in data for item in sublist]
})

sns.boxplot(data=df_plot, x='Method', y='Best Pearson R', 
            palette=PLOT_COLORS)
plt.xticks(rotation=45)
plt.title('Distribution of Best Performance Across Patients')
plt.tight_layout()
plt.savefig(os.path.join(GROUP_OUTPUT_PATH, 'group_best_performance_dist.png'))
plt.close()

# Plot 3: Violin plot with individual points
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_plot, x='Method', y='Best Pearson R', inner=None,
               palette=PLOT_COLORS)
sns.stripplot(data=df_plot, x='Method', y='Best Pearson R', color='red', alpha=0.3)
plt.xticks(rotation=45)
plt.title('Distribution of Best Performance with Individual Patients')
plt.tight_layout()
plt.savefig(os.path.join(GROUP_OUTPUT_PATH, 'group_best_performance_violin.png'))
plt.close()

# Plot 4: Individual patient panels
n_patients = len(patient_time_results)
n_cols = 3
n_rows = (n_patients + n_cols - 1) // n_cols

plt.figure(figsize=(15, 4*n_rows))
for idx, (patient, methods) in enumerate(patient_time_results.items()):
    plt.subplot(n_rows, n_cols, idx+1)
    
    for method in methods:
        time_windows = sorted(methods[method].keys())
        values = [methods[method][tw] for tw in time_windows]
        
        plt.plot(time_windows, values, PLOT_STYLES[method], 
                label=method, color=PLOT_COLORS[method], markersize=4)
    
    plt.title(f'Patient {patient}')
    plt.xlabel('Time Window (minutes)')
    plt.ylabel('Pearson Correlation (r)')
    plt.grid(True)
    if idx == 0:  # Only show legend for first subplot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

plt.savefig(os.path.join(GROUP_OUTPUT_PATH, 'individual_patient_performance.png'), 
            bbox_inches='tight')
plt.close()

# Process feature correlation data for plots 5 and 6
# Convert feature correlation data to DataFrames
top_features_data = []
bottom_features_data = []

for method, type_data in feature_correlation_data.items():
    for feature_type, features in type_data.items():
        for feature_info in features:
            if feature_type == 'Top':
                top_features_data.append({
                    'Method': method,
                    'Feature': feature_info['Feature'],
                    'Correlation': feature_info['Correlation'],
                    'Patient': feature_info['Patient'],
                    'Time_Window': feature_info['Time_Window']
                })
            elif feature_type == 'Bottom':
                bottom_features_data.append({
                    'Method': method,
                    'Feature': feature_info['Feature'],
                    'Correlation': feature_info['Correlation'],
                    'Patient': feature_info['Patient'],
                    'Time_Window': feature_info['Time_Window']
                })

top_df = pd.DataFrame(top_features_data)
bottom_df = pd.DataFrame(bottom_features_data)

# Plot 5: Top features across methods
if not top_df.empty:
    # Aggregate top features by method
    top_feature_counts = top_df.groupby(['Method', 'Feature']).size().reset_index(name='Count')
    top_feature_corrs = top_df.groupby(['Method', 'Feature'])['Correlation'].mean().reset_index()
    
    # Merge counts and correlations
    top_feature_summary = pd.merge(top_feature_counts, top_feature_corrs, on=['Method', 'Feature'])
    
    # Get top 10 features by count for each method
    top_features_by_method = {}
    for method in top_feature_summary['Method'].unique():
        method_data = top_feature_summary[top_feature_summary['Method'] == method]
        top_features_by_method[method] = method_data.nlargest(10, 'Count')
    
    # Plot
    plt.figure(figsize=(15, 10))
    for i, (method, data) in enumerate(top_features_by_method.items()):
        plt.subplot(2, 2, i+1)
        
        # Sort by count
        data = data.sort_values('Count', ascending=False)
        
        # Create horizontal bar chart
        bars = plt.barh(data['Feature'], data['Count'], color=PLOT_COLORS.get(method, 'blue'))
        
        # Add correlation values as text
        for j, (_, row) in enumerate(data.iterrows()):
            plt.text(row['Count'] + 0.1, j, f"{row['Correlation']:.2f}", 
                     va='center', fontsize=8)
        
        plt.title(f'Top Predictive Features: {method}')
        plt.xlabel('Frequency Across Patients & Time Windows')
        plt.tight_layout()
    
    plt.suptitle('Most Frequently Selected Top Predictive Features by Method', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(GROUP_OUTPUT_PATH, 'top_features_by_method.png'), bbox_inches='tight')
    plt.close()

# Plot 6: Feature correlation heatmap across time windows
if not top_df.empty:
    # Select the most common features across all methods
    all_top_features = top_df.groupby('Feature').size().reset_index(name='Count')
    most_common_features = all_top_features.nlargest(15, 'Count')['Feature'].tolist()
    
    # Filter data to include only the most common features
    filtered_top_df = top_df[top_df['Feature'].isin(most_common_features)]
    
    # Create a pivot table: features x time windows with correlation values
    pivot_data = filtered_top_df.pivot_table(
        index='Feature', 
        columns='Time_Window', 
        values='Correlation',
        aggfunc='mean'
    )
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
                linewidths=.5, cbar_kws={'label': 'Mean Correlation'})
    plt.title('Feature Correlation Strength Across Time Windows')
    plt.xlabel('Time Window (minutes)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(GROUP_OUTPUT_PATH, 'feature_correlation_heatmap.png'))
    plt.close()
