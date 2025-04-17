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
            time_window_results[method][row['Time_Window']].append(row['Pearson_R'])
            
        # Store best result
        best_results[method].append(df['Pearson_R'].max())

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
